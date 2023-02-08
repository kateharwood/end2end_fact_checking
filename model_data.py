import argparse
import torch
import jsonlines
import random
import os
import logging
import json

from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

from torch.utils.data import Dataset, DataLoader
# from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, get_cosine_schedule_with_warmup
from label_model import LabelModel


parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, required=True, help='Path to processed train file')
parser.add_argument('--dev', type=str, required=False, help='Path to processed dev file')
parser.add_argument('--dest', type=str, required=True, help='Folder to save the weights')
parser.add_argument('--model', type=str, default='roberta-large')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size-gpu', type=int, default=8, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=256, help='The batch size for each gradient update')

args = parser.parse_args()

# Setup logging
logger = logging.getLogger(__name__)
if not os.path.exists(args.dest):
    os.makedirs(args.dest)
logging.basicConfig(
    filename=os.path.join(args.dest,"logging.txt"),
    filemode='w',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device "{device}"')


class FeverLabelPredictionDataset(Dataset):
    def __init__(self, train_file):
        self.samples = []

        labels = {'SUPPORTS': 2, 'NOT ENOUGH INFO': 1, 'NOTENOUGHINFO': 1, 'REFUTES': 0}

        for j, data in enumerate(jsonlines.open(train_file)):
            if j < 1000:
                claim = str(data["claim"])
                title = str(data['title'])
                abstract = ' '.join(data['abstract'])
                label = labels[data['label']]

                self.samples.append({
                    'claim': claim,
                    'title': title,
                    'abstract': title + ": " + abstract,
                    'label': label  
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

trainset = FeverLabelPredictionDataset(args.train)
devset = FeverLabelPredictionDataset(args.dev)

# could maybe use same base model here to be able to get batch size higher
claim_tokenizer = RobertaTokenizer.from_pretrained("roberta-large") # TODO get rid of this one, just use abstract one
abstract_tokenizer = RobertaTokenizer.from_pretrained("roberta-large") # use smaller model for beginning, also put on different GPUs
abstract_model = RobertaModel.from_pretrained("roberta-large", cache_dir="cache").to(device)

# MLP to concat claim and abstract embeddings to decide label
# TODO can maybe pass richer representation than CLS to this model
label_model = LabelModel().to(device)


optimizer = torch.optim.Adam([
    # If you are using non-roberta based models, change this to point to the right base
    {'params': abstract_model.parameters(), 'lr': 5e-5}, # TODO use a different learning rate for the classifier layer
    {'params': label_model.parameters(),  'lr': 1e-5}, # TODO is this right?
    # {'params': model.classifier.parameters()} #, 'lr': args.lr_linear}
])
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)


def encode(to_encode, claim_or_abstract='claim'):
    if claim_or_abstract == 'abstract': # TODO just one tokenizer
        tokenizer = abstract_tokenizer
    else:
        assert claim_or_abstract == 'claim'
        tokenizer = claim_tokenizer
    encoded = tokenizer.batch_encode_plus(
        to_encode,
        pad_to_max_length=True,
        return_tensors='pt')
    max_length = 512
    # if args.use_longformer:
    #     max_length = 4096
    if encoded['input_ids'].size(1) > max_length: # 4096 for longformer, 512 for roberta
        # Too long for the model. Truncate it
        encoded = tokenizer.batch_encode_plus(
            to_encode,
            max_length=max_length, # 4096 for longformer, 512 for roberta
            truncation_strategy='only_first',
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt')
    encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
    return encoded

def evaluate_label_model(dataset):
    label_model.eval()
    abstract_model.eval() 
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            print(batch['claim'])
            print(batch['abstract'])
            encoded_claims = encode(batch['claim'], 'claim')
            encoded_abstracts = encode(batch['abstract'], 'abstract')
            claim_cls = abstract_model(**encoded_claims)[0][:,0,:]
            abstract_cls = abstract_model(**encoded_abstracts)[0][:,0,:]

            logits = label_model(torch.cat((claim_cls, abstract_cls), 1)) 
            targets.extend(batch['label'].tolist()) #TODO add float()?
            outputs.extend(logits.argmax(dim=1).tolist())

    return {
        'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
        'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None))
    }

def evaluate_abstract_model(dataset):
    label_model.eval()
    abstract_model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_claims = encode(batch['claim'], 'claim')
            encoded_abstracts = encode(batch['abstract'], 'abstract')
            claim_cls = abstract_model(**encoded_claims)[0][:,0,:]
            abstract_cls = abstract_model(**encoded_abstracts)[0][:,0,:]
            # TODO that ^ is inefficient, we are doing it twice

            scores = torch.matmul(claim_cls, abstract_cls.t())
            assert scores.shape == (len(claim_cls), len(abstract_cls))

            # only evaluate non NEI examples
            labels = batch['label'].tolist()
            non_nei_examples = [i for i in range(len(labels)) if labels[i] == 0 or labels[i] == 2]
            scores = scores[non_nei_examples] 
            scores = scores[:,non_nei_examples]
            assert scores.shape == (len(non_nei_examples), len(non_nei_examples))

            outputs.extend(scores.argmax(dim=1).to("cpu"))
            targets.extend(torch.eye(scores.shape[0]).argmax(dim=1))
    print(outputs)
    return {
        'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
        'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None))
    }


# Train
best_epoch = 0
best_dev_mF1 = 0
best_dir = ""

for e in range(args.epochs):
    abstract_model.train()
    label_model.train()
    t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=True))
    for i, batch in enumerate(t):
        abstract_loss = 0
        encoded_claims = []
        encoded_abstracts = []

        # encode claims and abstracts
        encoded_claims = encode(batch['claim'], 'claim')
        encoded_abstracts = encode(batch['abstract'], 'abstract')
        
        # get claim and abstract embeddings
        claim_cls = abstract_model(**encoded_claims)[0][:,0,:]
        abstract_cls = abstract_model(**encoded_abstracts)[0][:,0,:]

        scores = torch.matmul(claim_cls, abstract_cls.t())
        assert scores.shape == (len(claim_cls), len(abstract_cls))

        # only score the non NEI examples
        labels = batch['label'].tolist()
        non_nei_examples = [i for i in range(len(labels)) if labels[i] == 0 or labels[i] == 2]
        scores = scores[non_nei_examples] 
        scores = scores[:,non_nei_examples]
        assert scores.shape == (len(non_nei_examples), len(non_nei_examples))
        # print(scores)

        # get noise contrastive loss on all non NEI examples
        loss_fn = torch.nn.CrossEntropyLoss()
        abstract_loss = loss_fn(scores, torch.eye(scores.shape[0]).to(device))

        # feed all claims and correct abstracts into simple label model
        # logits = label_model(torch.cat((claim_cls, abstract_cls), 1)) 
        # print(logits)
        # print(batch['label'])
        # label_loss = loss_fn(logits, batch['label'].to(device))

        # loss = torch.add(abstract_loss,label_loss)
        loss = abstract_loss
        loss.backward()

        # announce loss and step gradient
        if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
            optimizer.step()
            optimizer.zero_grad()
            t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
    # scheduler.step()

    # Eval abstract model
    train_score = evaluate_abstract_model(trainset)
    logger.info(f'Abstract Model: Epoch {e} train score:')
    logger.info(train_score)
    dev_score = evaluate_abstract_model(devset)
    logger.info(f'Abstract Model: Epoch {e} dev score:')
    logger.info(dev_score)

    # Eval label model
    # train_score = evaluate_label_model(trainset)
    # logger.info(f'Label Model: Epoch {e} train score:')
    # logger.info(train_score)
    # dev_score = evaluate_label_model(devset)
    # logger.info(f'Label Model: Epoch {e} dev score:')
    # logger.info(dev_score)

    
    # Check if better than previous best score
    if dev_score["macro_f1"] > best_dev_mF1:
        save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score["macro_f1"] * 1e4)}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        abstract_tokenizer.save_pretrained(save_path) # TODO also need to save label model
        abstract_model.save_pretrained(save_path)

        # Remove old best checkpoint
        if os.path.exists(best_dir):
            for root, subdirs, files in os.walk(best_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
            os.rmdir(best_dir)

        # Save new best checkpoint
        best_epoch = e
        best_dev_mF1 = dev_score["macro_f1"]
        best_dir = save_path
        with open(os.path.join(args.dest, "best_model_path.txt"), "w") as f:
            f.write(best_dir)
    
    # Early stopping
    # if e == args.epochs-1 or e-best_epoch>=6:
    #     # save last checkpoint
    #     save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score["macro_f1"] * 1e4)}')
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     break















# TODO
# create NOTENOUGHINFO examples from chunks in random articles (or do we want to make it harder by making it come from relevant article)
# need to sample from all examples some X number for each claim
#   this includes the relevant one as well as X-1 irrelevant ones
# use an "Abstract" encoder like SciBert to embed the abstracts
#   can experiment with different models here
# use claim encoder to embed claims
#   what model for this? maybe also RoBERTa? can experiment
# compare sim between claim and and each abstract (noise contrastive learning)
# pick the most similar abstract and some decider model will choose a label for the claim,abstract pair
#   this could be RoBERTa / RoBERTaLarge like we were using before
# error propagates back through decider model and abstract encoder and claim encoder

# will be easy to experiment with different models (depending on this training time though) 
# if we just use them from huggingface and switch them in and out







        # get cosine sim between all claims and abstracts
        # claim_cls_repeat = claim_cls.repeat_interleave(args.batch_size_gpu, dim=0)
        # abstract_cls_repeat = abstract_cls.repeat(args.batch_size_gpu, 1)
        # sim_scores = torch.nn.functional.cosine_similarity(claim_cls_repeat, abstract_cls_repeat, dim=1) # TODO just matrix multiplication between the two claim_cls abstract_cls matrices
        # sim_scores = sim_scores.reshape(args.batch_size_gpu, args.batch_size_gpu)
