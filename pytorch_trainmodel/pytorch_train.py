import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_model import *
from pytorch_clean import *
from sklearn.metrics import classification_report
from torch.optim import optimizer
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from vncorenlp.vncorenlp import VnCoreNLP
from transformers import AdamW

#calling pretrained model
phobert=AutoModel.from_pretrained('vinai/phobert-base')
tokenizer=AutoTokenizer.from_pretrained('vinai/phobert-base')
rdrsegmenter=VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

#get data
sentences,labels=get_data(['toxic_dataset.json','normal_dataset.json'])
sentences_segment(sentences, rdrsegmenter)
padded,labels=shuffle_and_tokenize(sentences,labels,check_maxlen(sentences), tokenizer)
X_train,X_val,X_test, y_train,y_val, y_test=split_data(padded, labels)
train_dataloader, val_dataloader= Data_Loader(X_train,X_val,y_train,y_val)

#freeze all the parameters
for param in phobert.parameters():
  param.requires_grad=False


#loss  
cross_entropy=nn.NLLLoss()
model=classify(phobert,2)
optimizer=AdamW(model.parameters(),lr=1e-5)

def train():
  model.train()
  total_loss,acc=0,0
  total_preds=[]
  for step , batch in enumerate(train_dataloader):
    if step%50==0 and step!=0:
      print("BATCH {} of {}".format(step, len(train_dataloader)))
   
    input,labels=batch
    model.zero_grad()
    preds=model(input)
    loss=cross_entropy(preds, labels)
    total_loss=total_loss+loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    preds=preds.detach().numpy()
    total_preds.append(preds)
  avg_loss=total_loss/len(train_dataloader)
  total_preds=np.concatenate(total_preds,axis=0)
  return avg_loss, total_preds

def evaluate():
  model.eval()
  total_loss,acc=0,0
  total_preds=[]
  for step, batch in enumerate(val_dataloader):
    if step%50==0 and step!=0:
      print("BATCH {} of {}".format(step, len(val_dataloader)))
    
    input,labels=batch
    with torch.no_grad():
      preds=model(input)
      loss=cross_entropy(preds, labels)
      total_loss+=loss.item()
      preds=preds.detach().numpy()
      total_preds.append(preds)
  avg_loss=total_loss/len(val_dataloader)
  total_preds=np.concatenate(total_preds,axis=0)
  return avg_loss, total_preds

def run(epochs):
  best_valid_loss=float("inf")
  train_losses=[]
  valid_losses=[]
  for epoch in range(epochs):
    print("EPOCH {}/{}".format(epoch,epochs))
    train_loss,_ =train()
    valid_loss,_ =evaluate()
    if valid_loss<best_valid_loss:
      best_valid_loss=valid_loss
      torch.save(model.state_dict(),"pytorch_trainmodel/save_weights.pt")
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(train_loss)
    print(valid_loss)

print("======TRAINING=======")
run(10)

print("======CHECKING=======")
path = 'save_weights.pt'
model.load_state_dict(torch.load(path))
sentence=input("Your sentence you want to preidct: ")
def result(sentence):
  tokens=rdrsegmenter.tokenize(sentence)
  statement=""
  for token in tokens:
    statement+=" ".join(token)
  sentence=statement
  sequence=tokenizer.encode(sentence)
  while(len(sequence)<check_maxlen(sentences)):
    sequence.insert(0,0)
  padded=torch.tensor([sequence])
  with torch.no_grad():
    preds=model(padded)
  preds=np.argmax(preds,axis=1)
  return preds
print(result(sentence))

#check test
with torch.no_grad():
  preds=model(X_test)
  preds=preds.detach().numpy()

preds=np.argmax(preds,axis=1)
print(classification_report(y_test, preds))