import json
import jsonlines

with open('train_20k_no_nei.jsonl', 'r') as f:
    lines = [json.loads(jline) for jline in f.read().splitlines()]

unique_abstracts = []
abstracts = []
for line in lines:
    if line['abstract'] not in abstracts:
        unique_abstracts.append(line)
        abstracts.append(line['abstract'])

with jsonlines.open('train_20k_unique.jsonl', mode='w') as writer:
    writer.write_all(unique_abstracts)
writer.close()



