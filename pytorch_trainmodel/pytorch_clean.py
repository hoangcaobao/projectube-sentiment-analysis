from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from sklearn.utils import shuffle
import json

def get_data(all_path):
  sentences=[]
  labels=[]
  for i in all_path:
    with open(i,"r") as f:
      datastore=json.load(f)
    for item in datastore:
      sentences.append(item["sentences"])
      labels.append(item["sacarism"])
  return sentences, labels

def sentences_segment(sentences, rdrsegmenter):
  for i in range(len(sentences)):
    tokens=rdrsegmenter.tokenize(sentences[i])
    statement=""
    for token in tokens:
      statement+=" ".join(token)
    sentences[i]=statement

def shuffle_and_tokenize(sentences,labels,maxlen, tokenizer):
  sentences,labels=shuffle(sentences,labels)
  sequences=[tokenizer.encode(i) for i in sentences]
  labels=[int(i) for i in labels]
  padded=pad_sequences(sequences, maxlen=maxlen, padding="pre")
  return padded, labels

def check_maxlen(sentences):
  sentences_len=[len(i.split()) for i in sentences]
  return max(sentences_len)

def split_data(padded, labels):
  padded=torch.tensor(padded).type(torch.LongTensor)
  labels=torch.tensor(labels).type(torch.LongTensor)
  X_train,X_,y_train,y_=train_test_split(padded, labels,random_state=2018, train_size=0.8, stratify=labels)
  X_val,X_test, y_val, y_test=train_test_split(X_, y_, random_state=2018, train_size=0.5, stratify=y_)
  return X_train,X_val,X_test, y_train,y_val, y_test

def Data_Loader(X_train,X_val,y_train,y_val):
  train_data=TensorDataset(X_train,y_train)
  train_sampler=RandomSampler(train_data)
  train_dataloader=DataLoader(train_data, sampler=train_sampler,batch_size=2)
  val_data=TensorDataset(X_val,y_val)
  val_sampler=RandomSampler(val_data)
  val_dataloader=DataLoader(val_data, sampler=val_sampler,batch_size=2)
  return train_dataloader, val_dataloader