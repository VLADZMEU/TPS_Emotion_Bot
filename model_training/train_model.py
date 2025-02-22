import re
import nltk
import numpy as np
import pandas as pd
import keras
import pickle
import matplotlib.pyplot as plt
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

stop_words = stopwords.words("english")

df_train = pd.read_csv('d:/Files/Otbor_TPS/train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('d:/Files/Otbor_TPS/val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('d:/Files/Otbor_TPS/test.txt', names=['Text', 'Emotion'], sep=';')

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ

    elif tag.startswith('V'):
        return wordnet.VERB

    elif tag.startswith('N'):
        return wordnet.NOUN

    elif tag.startswith('R'):
        return wordnet.ADV

    else:
        return wordnet.NOUN

def normalize_text(input_data):
    text = re.sub(r'https?://\S+|www\.\S+', '', input_data)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text.split()) < 3:
        text = np.nan
    text = " ".join(i for i in str(text).split() if i not in stop_words)
    text = text.split()
    pos_tags = pos_tag(text)
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(text)
def remove_nan(data):
    index = data[data['Text'] == 'nan'].index
    data.drop(index, axis = 0, inplace = True)
    data.reset_index(inplace=True, drop = True)

label_encoder = LabelEncoder()
label_encoder.fit(df_train['Emotion'].values)

X_train = df_train['Text'].values
y_train = df_train['Emotion'].values
X_test = df_test['Text'].values
y_test = df_test['Emotion'].values
X_val = df_val['Text'].values
y_val = df_val['Emotion'].values

encoder = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
y_val = encoder.transform(y_val)

unique_classes = encoder.categories_[0]
print("Unique classes in y_train:", unique_classes)
print("Number of unique classes:", len(unique_classes))
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)

max_length = 40
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')

model = Sequential()
model.add(Input(shape=(max_length,)))
model.add(Embedding(10000, 200, trainable=True))
model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(len(unique_classes), activation='softmax'))
model.summary()

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

history = model.fit(X_train, y_train, epochs=20, verbose=1, batch_size=8, validation_data=(X_val, y_val))
model.save('emotion_model.h5')

# Проверочка
input_text = "I feel hurt because he cheated on me with his best friend"

input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
prediction = model.predict(padded_input_sequence)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
print(predicted_label)