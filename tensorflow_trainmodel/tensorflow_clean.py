from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

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

def shuffle_and_tokenize(sentences, labels, maxlen, tokenizer):
  sentences,labels=shuffle(sentences,labels)
  sequences=[tokenizer.encode(i) for i in sentences]
  labels=[int(i) for i in labels]
  padded=pad_sequences(sequences, maxlen=maxlen, padding="pre")
  return sentences, padded, labels

def check_maxlen(sentences):
  sentences_len=[len(i.split()) for i in sentences]
  return max(sentences_len)

def split_data(padded, labels):
  padded=np.array(padded)
  labels=np.array(labels)
  X_train,X_val,y_train,y_val=train_test_split(padded, labels, random_state=2021, train_size=0.8, stratify=labels)
  return X_train,X_val, y_train, y_val