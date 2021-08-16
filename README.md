# VietnameseTextToxicClassify
I train this model using PyTorch to detect toxic of a comment for Projectube

I use VNCoreNLP for preprocess the raw Vietnamese sentences data and PhoBERT to train model to classify text. I use these technology at https://github.com/VinAIResearch/PhoBERT.

## Use my code

### 1. Git clone my repository:
```
git clone https://github.com/hoangcaobao/Vietnamese_Text_Toxic_Classify.git
```

### 2. Change directory to my folder and install VNCoreNLP:
```
cd VietnameseTextToxicClassify
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

### 4. Run training file:
```
python3 training.py
```
---
### HOANG CAO BAO
