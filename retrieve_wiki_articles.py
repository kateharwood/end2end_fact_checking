import json
import os
import jsonlines
import nltk


with open('train_chunks.jsonl', 'r') as f:
    lines = [json.loads(jline) for jline in f.read().splitlines()]

titles = set()
for line in lines:
    for articles in line.values():
        for title in articles.keys():
            titles.add(title)

with open('dev_chunks.jsonl', 'r') as f:
    lines = [json.loads(jline) for jline in f.read().splitlines()]

for line in lines:
    for articles in line.values():
        for title in articles.keys():
            titles.add(title)


# Find titles in wiki corpus
wiki_articles = []
for filename in os.listdir('wiki-pages'):
    wiki_file = os.path.join('wiki-pages', filename)
    with open(wiki_file, 'r') as f:
        lines = [json.loads(jline) for jline in f.read().splitlines()]
    for line in lines:
        if line['id'] in titles:
            tokenized_text = nltk.sent_tokenize(line['text'])
            wiki_articles.append({'title': line['id'], 'text': tokenized_text})

with jsonlines.open('wiki_articles_2.jsonl', mode='w') as writer:
    writer.write_all(wiki_articles)
writer.close()
