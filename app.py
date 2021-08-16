from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import json
from tensorflow_trainmodel.tensorflow_model import *
from vncorenlp import VnCoreNLP
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
from tensorflow_trainmodel.tensorflow_model import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

app=Flask(__name__)
cors=CORS(app)
app.config["CORS_HEADERS"] = 'Content-Type'

@app.route("/",  methods=["GET", "POST", "PUT"])
@cross_origin()
def index():
    re=json.loads(request.data)
    return str(result(re["sentence"]))
    
if __name__=="__main__":
    rdrsegmenter=VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
    tokenizer=AutoTokenizer.from_pretrained('vinai/phobert-base')
    phobert = TFAutoModel.from_pretrained("vinai/phobert-base")
    model= my_model(phobert,2)
    model.load_weights("tensorflow_trainmodel/my_model/my_model.ckpt")
    def result(sentence):
        tokens=rdrsegmenter.tokenize(sentence)
        statement=""
        for token in tokens:
            statement+=" ".join(token)
        sentence=statement
        sequence=tokenizer.encode(sentence)
        sequence=np.array([sequence])
        padded=pad_sequences(sequence, maxlen=16, padding="pre")
        preds=model.predict(padded)
        if(preds[0][1]>=0.7):
            return 1
        else:
            return 0
    app.run(host="0.0.0.0", debug=True)
