import re
import nltk
import numpy as np
import pandas as pd
import keras
import joblib
import os
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

# Загрузка данных NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
stop_words = stopwords.words("english")
# Загрузка данных
df_train = pd.read_csv('train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('test.txt', names=['Text', 'Emotion'], sep=';')

# Функция для нормализации текста
def normalize_text(input_data):
    text = re.sub(r'https?://\S+|www\.\S+', '', input_data)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text.split()) < 3:
        return np.nan
    text = " ".join(i for i in str(text).split() if i not in stop_words)
    text = text.split()
    pos_tags = pos_tag(text)
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(text)

# Функция для определения части речи
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

# Применение нормализации к данным
df_train['Text'] = df_train['Text'].apply(normalize_text)
df_val['Text'] = df_val['Text'].apply(normalize_text)
df_test['Text'] = df_test['Text'].apply(normalize_text)

# Убедимся, что все значения в столбце 'Text' являются строками
df_train['Text'] = df_train['Text'].astype(str)
df_val['Text'] = df_val['Text'].astype(str)
df_test['Text'] = df_test['Text'].astype(str)

# Удаление строк с NaN значениями
df_train.dropna(subset=['Text'], inplace=True)
df_val.dropna(subset=['Text'], inplace=True)
df_test.dropna(subset=['Text'], inplace=True)

# Сброс индексов после удаления строк
df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

# Кодирование меток
encoder = LabelEncoder()
y_train = encoder.fit_transform(df_train['Emotion'].values)
y_test = encoder.transform(df_test['Emotion'].values)
y_val = encoder.transform(df_val['Emotion'].values)

# Токенизация текста
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df_train['Text'].values)

X_train = tokenizer.texts_to_sequences(df_train['Text'].values)
X_test = tokenizer.texts_to_sequences(df_test['Text'].values)
X_val = tokenizer.texts_to_sequences(df_val['Text'].values)

# Добавление padding
max_length = 40
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')

# Создание модели
model = Sequential()
model.add(Input(shape=(max_length,)))
model.add(Embedding(10000, 200, trainable=True))
model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(len(encoder.classes_), activation='softmax'))

# Компиляция модели
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Callback для уменьшения learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[lr_reduction]
)

# Сохранение модели и артефактов
model.save("emotion_model.h5")  # Сохранение модели
with open("../telegram_bot/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)  # Сохранение токенизатора
with open("../telegram_bot/encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)  # Сохранение LabelEncoder

print("Модель и артефакты сохранены.")


# Пример предсказания
input_text = "I feel hurt because he cheated on me with his best friend"
input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
prediction = model.predict(padded_input_sequence)
predicted_label = encoder.inverse_transform([np.argmax(prediction[0])])
print(f"Предсказанная эмоция: {predicted_label[0]}")