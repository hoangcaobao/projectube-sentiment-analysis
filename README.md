# Vietnamese Text Toxic Classify
I train this model using PyTorch and TensroFlow to detect toxic of a comment for Projectube (projectube.org)

I use VNCoreNLP for preprocess the raw Vietnamese sentences data and PhoBERT to train model to classify text. I use these technology at https://github.com/VinAIResearch/PhoBERT.

## Use my code

### 1. Git clone my repository:
```
git clone https://github.com/hoangcaobao/projectube-sentiment-analysis.git
```

### 2. Change directory to my folder and install VNCoreNLP:
```
cd projectube-nlp
pip3 install vncorenlp
mkdir -p vncorenlp/models/wordsegmenter
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
mv VnCoreNLP-1.1.1.jar vncorenlp/ 
mv vi-vocab vncorenlp/models/wordsegmenter/
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```
### 3. Add more data in 2 json files
### 4. Install necessary libraries
```
pip install -r requirements.txt
```
### 5. Run training file:
#### If you want to train Pytorch
```
python3 pytorch_trainmodel/pytorch_train.py
```
### If you want to train TensroFlow
```
python3 pytorch_trainmodel/pytorch_train.py
```
## Optional
### You can even create Flask server to predict toxic level of sentences after finish training
### Normal use
```
python3 app.py
```
### Docker
```
docker build -t app . && docker run -p 5000:5000 app
```
After that you can post requrest to localhost:5000 with data is json {"sentence":"Your sentence"}

---
### HOANG CAO BAO
