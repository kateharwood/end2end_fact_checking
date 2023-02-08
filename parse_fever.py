import json
import jsonlines
from random import choice

def create_train_chunks(train_or_dev='train'):
    if train_or_dev == 'dev':
        with open('shared_task_dev.jsonl', 'r') as f:
            lines = [json.loads(jline) for jline in f.read().splitlines()]
    else:
        with open('train.jsonl', 'r') as f:
            lines = [json.loads(jline) for jline in f.read().splitlines()]

    wiki_dict = {}
    num_titles = 0
    for line in lines:
        for i,evidence in enumerate(line['evidence']):
            for e in evidence:
                title = e[2]
                if title != None:
                    if title not in wiki_dict:
                        num_titles = num_titles + 1
                        wiki_dict[title] = {e[3]: {line['id']}} # add wikipedia page title
                    else:
                        if e[3] not in wiki_dict[title]:
                            wiki_dict[title][e[3]] = {line['id']} # add sentence index to title dict
                        else:
                            wiki_dict[title][e[3]].add(line['id']) # add claim id to sentence index set
                
    # Put the not enough info examples in
    for line in lines:
        for evidence in line['evidence']:
            for e in evidence:
                title = e[2]
                if title == None:
                    random_title = choice(list(wiki_dict))
                    random_index = choice(list(wiki_dict[random_title]))
                    wiki_dict[random_title][random_index].add(line['id'])
                    
    print(num_titles)
    # get highest sent index in all claim examples
    largest = 0
    for line in lines:
        for evidence in line['evidence']:
            for e in evidence:
                if e[3] and e[3] > largest:
                    largest = e[3]
    print(largest)

    # chunk sentence indices into sets of 4
    sentences = []
    i = 0
    for i in range(0, largest+4, 4):
        sentences.append((i, i+1, i+2, i+3))
    # print(sentences)

    train_dict = {}
    title_counts = {}
    multi_chunks = 0
    num_sentence_chunks = 0
    num_claims = 0
    for title in wiki_dict.keys():
        title_counts[title] = 0
        for sent_index in wiki_dict[title].keys():
            for claim_id in wiki_dict[title][sent_index]:
                for sentence_chunk in sentences:
                    if sent_index in sentence_chunk:
                        num_sentence_chunks = num_sentence_chunks + 1
                        if claim_id not in train_dict:
                            num_claims += 1
                            title_counts[title] += 1                                
                            train_dict[claim_id] = {title: [sentence_chunk]}
                        else:
                            if title not in train_dict[claim_id]:
                                title_counts[title] += 1
                                train_dict[claim_id][title] = [sentence_chunk]
                            elif sentence_chunk not in train_dict[claim_id][title]:
                                train_dict[claim_id][title].append(sentence_chunk)
    train_dict_list = []
    for key in train_dict.keys():
        train_dict_list.append({key: train_dict[key]})
    print(num_sentence_chunks)
    # print(sorted( ((v,k) for k,v in title_counts.items())))
    print(num_claims)

    if train_or_dev == "dev":
        with jsonlines.open('dev_chunks.jsonl', mode='w') as writer:
            writer.write_all(train_dict_list)
        writer.close()
    else:
        with jsonlines.open('train_chunks.jsonl', mode='w') as writer:
            writer.write_all(train_dict_list)
        writer.close()




def create_trainset(train_or_dev='train', one_chunk=False):
    if train_or_dev == 'dev':
        with open('dev_chunks.jsonl', 'r') as f:
            train_lines = [json.loads(jline) for jline in f.read().splitlines()]
        with open('shared_task_dev.jsonl', 'r') as f:
            train_orig = [json.loads(jline) for jline in f.read().splitlines()]
    else:
        with open('train_chunks.jsonl', 'r') as f:
            train_lines = [json.loads(jline) for jline in f.read().splitlines()]
        with open('train.jsonl', 'r') as f:
            train_orig = [json.loads(jline) for jline in f.read().splitlines()]


    train_examples = []
    for j, line in enumerate(train_lines):
        # if j < 1000:
        claim = list(line.keys())[0]
        if one_chunk:
            article = line.values()
            if len(line[claim]) == 1:
                article = line[claim]
                chunk = list(article.values())[0]
                if len(chunk) == 1:
                    article_title = list(article.keys())[0]
                for label_line in train_orig:
                    if claim == str(label_line["id"]):
                        train_examples.append({
                            'claim_id': claim,
                            'wiki_title': article_title,
                            'sentences': chunk[0],
                            'label': label_line['label']
                        })
        else:
            for articles in line.values():
                for article in articles.keys():
                        for chunk in line[claim][article]:
                            for label_line in train_orig:
                                if claim == str(label_line["id"]):
                                    train_examples.append({
                                        'claim_id': claim,
                                        'wiki_title': article,
                                        'sentences': chunk,
                                        'label': label_line['label']
                                    })
    if one_chunk:   
        if train_or_dev == 'dev':
            with jsonlines.open('dev_examples_one_chunk.jsonl', mode='w') as writer:
                writer.write_all(train_examples)
            writer.close()
        else:
            with jsonlines.open('train_examples_one_chunk.jsonl', mode='w') as writer:
                writer.write_all(train_examples)
            writer.close()
    else:
        if train_or_dev == 'dev':
            with jsonlines.open('dev_examples.jsonl', mode='w') as writer:
                writer.write_all(train_examples)
            writer.close()
        else:
            with jsonlines.open('train_examples.jsonl', mode='w') as writer:
                writer.write_all(train_examples)
            writer.close()


def merge_trainset(train_or_dev='train', one_chunk=False):
    if train_or_dev == 'dev':
        if one_chunk:
            examples = 'dev_examples_one_chunk.jsonl'
        else:
            examples = 'dev_examples.jsonl'
        with open(examples, 'r') as f:
            train_file = [json.loads(jline) for jline in f.read().splitlines()]
        with open('wiki_articles_train_dev.jsonl', 'r') as f:
            corpus = [json.loads(jline) for jline in f.read().splitlines()]
        with open('shared_task_dev.jsonl', 'r') as f:
            orig_train = [json.loads(jline) for jline in f.read().splitlines()]
    else:
        if one_chunk:
            examples = 'train_examples_one_chunk.jsonl'
        else:
            examples = 'train_examples.jsonl'
        with open(examples, 'r') as f:
            train_file = [json.loads(jline) for jline in f.read().splitlines()]
        with open('wiki_articles_train_dev.jsonl', 'r') as f:
            corpus = [json.loads(jline) for jline in f.read().splitlines()]
        with open('train.jsonl', 'r') as f:
            orig_train = [json.loads(jline) for jline in f.read().splitlines()]


    train_examples = []
    for j, data in enumerate(train_file):
        # if j < 1000:
        claim = str(data["claim_id"])
        title = str(data['wiki_title'])

        # get actual claim text
        for example in orig_train:
            if str(example['id']) == claim:
                claim = example['claim']

        # get actual article text
        for article in corpus:
            if article['title'] == title:
                sentences = []
                # make sure all sentences in chunk are within article length
                out_of_range = data['sentences'][-1] - len(article['text'])+1
                if out_of_range > 0: # TODO, this means we could have some repeat chunks, or parts of chunks for same claim article pairs
                    data['sentences'] = [sentence - out_of_range for sentence in data['sentences']]
                abstract = []
                for i in data['sentences']:
                    if i > 0: # might have shorter chunks for articles that are less than chunk_length long
                        abstract.append(article['text'][i].strip())
                train_examples.append({
                    'claim': claim,
                    'title': title,
                    'abstract': abstract,
                    'label': data["label"]  
                })

        if one_chunk:
            if train_or_dev == 'dev':
                with jsonlines.open('dev_examples_parsed_one_chunk.jsonl', mode='w') as writer:
                    writer.write_all(train_examples)
                writer.close()
            else: 
                with jsonlines.open('train_examples_parsed_one_chunk.jsonl', mode='w') as writer:
                    writer.write_all(train_examples)
                writer.close()    
        else:
            if train_or_dev == 'dev':
                with jsonlines.open('dev_examples_parsed.jsonl', mode='w') as writer:
                    writer.write_all(train_examples)
                writer.close()
            else: 
                with jsonlines.open('train_examples_parsed.jsonl', mode='w') as writer:
                    writer.write_all(train_examples)
                writer.close()


from iteration_utilities import unique_everseen
def remove_dups():
    with open('dev_examples_parsed_one_chunk.jsonl', 'r') as f:
        dev_file = [json.loads(jline) for jline in f.read().splitlines()]
    dev_file = list(unique_everseen(dev_file))
    with jsonlines.open('dev_examples_parsed_one_chunk_dedup.jsonl', mode='w') as writer:
        writer.write_all(dev_file)
    writer.close()

    # with open('train_examples_parsed_one_chunk.jsonl', 'r') as f:
    #     train_file = [json.loads(jline) for jline in f.read().splitlines()]
    # train_file = list(unique_everseen(train_file))
    # with jsonlines.open('train_examples_parsed_one_chunk_dedup.jsonl', mode='w') as writer:
    #     writer.write_all(train_file)
    # writer.close()




# create_train_chunks()
# create_trainset(one_chunk=True)

# create_train_chunks('dev')
# create_trainset('dev', one_chunk=True)

# merge_trainset(one_chunk=True)
# merge_trainset('dev', one_chunk=True)

remove_dups()






# with jsonlines.open('/home/user/data/json_lines2.jl', 'w') as writer:
#     writer.write_all(json_data)

# writer = jsonlines.Writer(default=set_default)


# go through train.jsonl
#    create dict of title: [sentence1: [claim ids sentence1 goes with], sentence2: [claim ids sentence2 goes with], etc]
#    if come upon same title, add the sentences and claim, or add the claim if the sentences are already there

# go through dict
#   for each title, chunk 4 sentences around each sentence (making sure not duplicate parts of chunks)
#       for each sentence
#           add to training data dict {claim_id: {title: [chunk1, chunk2,etc]}, {title2: [asdfasdf,asdfasdf,asffs]}, {claim_id2: [chunk1, chunk2, etc]}