import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.callbacks import *
from vncorenlp import VnCoreNLP
from tensorflow_model import *
from tensorflow.keras.optimizers import *
from tensorflow_clean import *

#call pretrained model
rdrsegmenter=VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
phobert = TFAutoModel.from_pretrained("vinai/phobert-base")
tokenizer=AutoTokenizer.from_pretrained('vinai/phobert-base')

#get data
sentences,labels=get_data(['toxic_dataset.json','normal_dataset.json'])
sentences_segment(sentences, rdrsegmenter)
sentences, padded,labels=shuffle_and_tokenize(sentences,labels,check_maxlen(sentences), tokenizer)
X_train, X_val, y_train, y_val=split_data(padded, labels)

#train model
model=my_model(phobert,  2)
model.compile(optimizer=Adam(3e-5), loss="sparse_categorical_crossentropy", metrics=["acc"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=2)
model.save_weights("tensorflow_trainmodel/my_model/my_model.ckpt")

#predict
model.load_weights("tensorflow_trainmodel/my_model/my_model.ckpt")
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
  padded=[sequence]
  preds=np.argmax(model.predict(padded),axis=1)
  return preds
print(result(sentence))


