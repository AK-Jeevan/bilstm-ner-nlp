# 🔎 BiLSTM-based Named Entity Recognition (NER)

This project implements a **token-level Named Entity Recognition (NER)** system using:
- NLTK for text preprocessing  
- Word2Vec for vector embeddings  
- Bi-directional LSTM + TimeDistributed dense output layer for sequence labeling  

The model predicts BIO-tagged entity labels such as person, organization, location, geopolitical entity, etc.

---

## ✨ Features
✅ Token-level NER  
✅ BiLSTM architecture  
✅ Word2Vec embeddings  
✅ NLTK preprocessing (tokenize, stem, lemmatize)  
✅ Masking + padded sequences  
✅ Early stopping during training  

---

## 🧠 Pipeline
1) Load dataset (sentences + corresponding tags)  
2) NLTK cleaning (stopwords, lemmatization)  
3) Train Word2Vec embeddings  
4) Encode tokens + tags  
5) Pad sequences  
6) Train BiLSTM model  
7) Predict BIO tags  

---

## 📦 Installation

git clone https://github.com/<your-username>/bilstm-ner-nlp
cd bilstm-ner-nlp
pip install -r requirements.txt

## ▶️ Training

python train.py

## 📁 Model Architecture

Embedding (Word2Vec)
→ Masking
→ BiLSTM (return_sequences=True)
→ TimeDistributed(Dense → softmax)

## ✅ Improvements

Replace Word2Vec with contextual embeddings (BERT)

Add CRF output layer

Add support for more tag formats

Deploy via API

## 📄 License
MIT
