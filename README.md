# Vietnamese Text Toxic Classifier
I train this model using PyTorch and TensorFlow to detect toxic comments for [Projectube](https://www.projectube.org/)

I use VNCoreNLP for preprocessing the raw Vietnamese sentence data and PhoBERT to train models to classify text. I use these technologies from https://github.com/VinAIResearch/PhoBERT.

## Use my code

### 1. Git clone my repository:
```
git clone https://github.com/hoangcaobao/projectube-sentiment-analysis.git
```

### 2. Change the directory to my folder and install VNCoreNLP:
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
### 3. Add more data in 2 JSON files
### 4. Install necessary libraries
```
pip install -r requirements.txt
```
### 5. Run training file:
#### If you want to train PyTorch
```
python3 pytorch_trainmodel/pytorch_train.py
```
### If you want to train TensorFlow
```
python3 pytorch_trainmodel/pytorch_train.py
```
## Optional
### You can even create Flask server to predict the toxic level of sentences after finishing training
### Normal use
```
python3 app.py
```
### Docker
```
docker build -t app . && docker run -p 5000:5000 app
```
After that you can post a request to localhost:5000 with data is JSON {"sentence":"Your sentence"}

---
### HOANG CAO BAO
