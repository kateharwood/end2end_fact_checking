import json
import jsonlines

with open('dev_examples_parsed_one_chunk_dedup.jsonl', 'r') as f:
    lines = [json.loads(jline) for jline in f.read().splitlines()]

non_nei = []
support = 0
refute = 0
for line in lines:
    if line['label'] != 'NOT ENOUGH INFO':
        non_nei.append(line)
    if line['label'] == 'SUPPORTS':
        support +=1
    elif line['label'] == 'REFUTES':
        refute += 1

print(support)
print(refute)

with jsonlines.open('dev_one_chunk_non_nei.jsonl', mode='w') as writer:
    writer.write_all(non_nei)
writer.close()



