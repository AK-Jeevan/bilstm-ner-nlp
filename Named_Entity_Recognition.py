'''To implement a token-level Named Entity Recognition (NER) system using a Bi-directional Long Short-Term Memory (BiLSTM) neural network, 
leveraging NLTK for preprocessing and Word2Vec embeddings for semantic representation, 
in order to classify each word in a sentence into BIO-tagged entity categories such as person, location, organization, and geopolitical entity e.t.c'''

# Step 2: Imports
import tokenize
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Masking
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Step 3: Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 4: Load dataset
df = pd.read_csv(r"C:\Users\akjee\Documents\AI\NLP\NLP - DL\LSTM-RNN\ner.csv", encoding="latin1")
tokenizer = RegexpTokenizer(r'\w+')
df['Sentence'] = df['Sentence'].apply(lambda x: tokenizer.tokenize(x))
df['Tag'] = df['Tag'].apply(lambda x: tokenizer.tokenize(x))

sentences = df['Sentence'].tolist()
tags = df['Tag'].tolist()

# Step 5: NLTK preprocessing (stemming + lemmatization)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    cleaned = []
    for word in sentence:
        word = word.lower()
        if word not in stop_words:
            stemmed = stemmer.stem(word)
            lemma = lemmatizer.lemmatize(stemmed)
            cleaned.append(lemma)
    return cleaned

sentences = [preprocess(s) for s in sentences]

# Step 6: Train Word2Vec
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
word_vectors = w2v_model.wv #.wv gives you the actual word meanings as numbers. You use it to get the vector for a word

# Step 7: Create tag dictionary
unique_tags = sorted(set(tag for tag_list in tags for tag in tag_list))
tag2idx = {tag: i for i, tag in enumerate(unique_tags)}
idx2tag = {i: tag for tag, i in tag2idx.items()}

# Step 8: Encode sentences and tags
def encode(sentence, tag_seq):
    X = [word_vectors[word] if word in word_vectors else np.zeros(100) for word in sentence]
    y = [tag2idx[tag] for tag in tag_seq[:len(sentence)]]
    return X, y

X_data, y_data = zip(*[encode(s, t) for s, t in zip(sentences, tags)])

# Step 9: Pad sequences
max_len = max(len(x) for x in X_data)
X_padded = pad_sequences(X_data, maxlen=max_len, dtype='float32', padding='post')
y_padded = pad_sequences(y_data, maxlen=max_len, padding='post')
y_padded = np.expand_dims(y_padded, -1)

# Step 10: Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_padded, y_padded, test_size=0.2)

# Step 11: Build BiLSTM model
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(max_len, 100)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2idx), activation='softmax')))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 12: Add EarlyStopping callback to stop training when validation loss stops improving.
early_stopping = EarlyStopping(
    monitor='val_loss',        # monitor validation loss
    patience=3,                # number of epochs with no improvement before stopping
    restore_best_weights=True, # restore model weights from the epoch with the best val_loss
    verbose=1
)

# Step 13: Train model with EarlyStopping
model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20,                     # you can increase/decrease as needed
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]     # <-- pass the callback here
)

# Step 14: Evaluate on validation set
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Step 15: Predict function
def predict_tags(raw_sentence):
    # Preprocess the sentence
    cleaned = preprocess(raw_sentence)
    
    # Convert words to vectors
    X = [word_vectors[word] if word in word_vectors else np.zeros(100) for word in cleaned]
    
    # Pad to max_len
    X_padded = pad_sequences([X], maxlen=max_len, dtype='float32', padding='post')
    
    # Predict
    y_pred = model.predict(X_padded)
    y_pred = np.argmax(y_pred, axis=-1)[0]  # Get tag indices
    
    # Decode tags
    predicted_tags = [idx2tag[idx] for idx in y_pred[:len(cleaned)]]
    
    return list(zip(raw_sentence, predicted_tags))

test_sentence = ["London", "is", "the", "capital", "of", "England"]
print(predict_tags(test_sentence))
