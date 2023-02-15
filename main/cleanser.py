__author__ = 'GalihSuyoga'

from main.model.text_processing import Abusive, KamusAlay, TextLog, AlayAbusiveLog, FileTextLog, AlayAbusiveFileLog, RawText, ProcessedText
import numpy as np
import re
import pandas as pd
import numpy as np
import pickle
from main.model import db

# Untuk Text processing
import re
import nltk
nltk.download('punkt') # untuk punctuation
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

# Machine Learning model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier # neural network
from sklearn.neighbors import KNeighborsClassifier

# Untuk metrics
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix, plot_decision_regions



# Supaya tidak ada warnings yg mengganggu
import warnings

from keras.datasets import imdb
# from keras.layers import LSTM, Embedding, Dense
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf


warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler



nltk.download('stopwords')
listStopword = list(stopwords.words('indonesian'))

__sklearn_regresion = "MLModel/sklearn_regression.pkl"
__sklearn_mlp = "MLModel/sklearn_MPL.pkl"
__sklearn_naive_bayes = "MLModel/sklearn_naive_bayes.pkl"
__sklearn_knn = "MLModel/sklearn_knn.pkl"
__sklearn_tensor_neural_network = "MLModel/tensor_neural_network"

# Platinum =======================================================================================================
def replace_alay_word(text, df_alay):
    cleaned = []
    for word in text.split(' '):
        meaning = word
        alay = df_alay[df_alay['word'].isin([word])]

        if len(alay) > 0:
            # print(alay.Abusive)
            meaning = list(alay['meaning'])[0]
        cleaned.append(meaning)
    return " ".join(word for word in cleaned)


def cleanser_string_step(text, step: int):
    text = str(text)
    if step >= 1:
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in (listStopword)])
        text = re.sub(r'[^\w\s]|[0-9]', ' ', text)
        text = text.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')
        print(text)
    if step >= 2:
        df_alay = pd.read_sql_query(
            sql=db.select([KamusAlay.word, KamusAlay.meaning]),
            con=db.engine
        )
        text = replace_alay_word(text=text, df_alay=df_alay)
    if step >= 3:
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        text = stemmer.stem(text)
    return text

def text_normalization_on_db_raw_data():

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df = pd.read_sql_query(
        sql=db.select([RawText.kalimat, RawText.sentimen]),
        con=db.engine
    )
    df_alay = pd.read_sql_query(
        sql=db.select([KamusAlay.word, KamusAlay.meaning]),
        con=db.engine
    )
    # clean stopwords
    df['kalimat_bersih'] = df['kalimat'].str.lower()

    df['kalimat_bersih'] = df['kalimat_bersih'].apply(lambda x: ' '.join([word for word in x.split() if word not in (listStopword)]))

    df['jumlah_kalimat'] = df['kalimat_bersih'].apply(lambda x: len(sent_tokenize(x)))
    df['jumlah_kata'] = df['kalimat_bersih'].apply(lambda x: len(word_tokenize(x)))
    df['kalimat_bersih'] = df['kalimat_bersih'].apply(lambda x: re.sub(r'[^\w\s]|[0-9]', ' ', x))
    df['kalimat_bersih'] = df['kalimat_bersih'].str.replace('    ', ' ').str.replace('   ', ' ').str.replace('  ', ' ')

    df['kalimat_bersih_v2'] = df['kalimat_bersih'].apply(lambda x: replace_alay_word(text=x, df_alay=df_alay))
    df['kalimat_bersih_v3'] = df['kalimat_bersih_v2'].apply(lambda x: stemmer.stem(x))
    df['kalimat_bersih_v4'] = df['kalimat_bersih'].apply(lambda x: stemmer.stem(x))
    print(df['kalimat_bersih_v3'][:5])

    df.to_sql('processed_text', con=db.engine, if_exists='replace', index_label='id')

    return 'finish'

def training_model_evaluate():
    df = pd.read_sql_query(
        sql=db.select([ProcessedText.kalimat, ProcessedText.sentimen, ProcessedText.kalimat_bersih, ProcessedText.kalimat_bersih_v2, ProcessedText.kalimat_bersih_v3, ProcessedText.jumlah_kata, ProcessedText.jumlah_kalimat]),
        con=db.engine
    )
    over = RandomOverSampler()
    print(df.head())
    # imbalance
    # positif 6383 negatif 3412 netral 1138
    print(f' {len(df[df["sentimen"]=="positive"])} {len(df[df["sentimen"]=="negative"])} {len(df[df["sentimen"]=="neutral"])  }')
    count_vect = CountVectorizer()
    count_vect.fit(df['kalimat_bersih'])

    # jumlah unique words
    word_features = count_vect.get_feature_names_out()
    print(f"jumlah unique words: {len(word_features)}")

    # Hasil Transformasi
    transformed = count_vect.transform(df['kalimat_bersih'])
    X = transformed.toarray()
    print(f"Hasil transformasi array shape: {X.shape}")


    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(df['kalimat_bersih'])

    # Hasil Transformasi
    transformed = tfidf_vect.transform(df['kalimat_bersih'])
    X = transformed.toarray()
    print(f"Hasil transformasi array shape: {X.shape}")
    print(X)

    y = df['sentimen']
    y = y.factorize()[0]

    # oversampling


    X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.1, random_state=0)

    print(X_train[:10])
    print(y_train.shape)
    #modeling
    X_train, y_train = over.fit_resample(X, y)
    # # regresi
    # model_1 = LogisticRegression()
    # model_1.fit(X_train, y_train)
    # model_1_pred = model_1.predict(X_test)
    # print(f"Accuracy model regresi: {metrics.accuracy_score(y_test, model_1_pred) * 100:.2f}%")
    #
    # # gausian naive bayes
    # model_2 = GaussianNB()
    # model_2.fit(X_train, y_train)
    # model_2_pred = model_2.predict(X_test)
    # print(f"Accuracy model naive bayes: {metrics.accuracy_score(y_test, model_2_pred) * 100:.2f}%")

    # regresi
    model_1 = LogisticRegression()
    model_1.fit(X_train, y_train)
    model_1_pred = model_1.predict(X_test)
    print(f"Accuracy model regresi: {metrics.accuracy_score(y_test, model_1_pred) * 100:.2f}%")

    # gausian naive bayes
    model_2 = GaussianNB()
    model_2.fit(X_train, y_train)
    model_2_pred = model_2.predict(X_test)
    print(f"Accuracy model naive bayes: {metrics.accuracy_score(y_test, model_2_pred) * 100:.2f}%")


    # MLP(multi-layer perception)/ neural network
    model_3 = MLPClassifier(100)
    model_3.fit(X_train, y_train)
    model_3_pred = model_3.predict(X_test)
    print(f"Accuracy model MLP: {metrics.accuracy_score(y_test, model_3_pred) * 100:.2f}%")

    # knn
    # model_4 = KNeighborsClassifier()
    # model_4.fit(X_train, y_train)
    # model_4_pred = model_4.predict(X_test)
    #
    # print(f"Accuracy model KNN: {metrics.accuracy_score(y_test, model_4_pred) * 100:.2f}%")

    # d = {"prep": count_vect, "model": model_1}
    # with open(__sklearn_regresion, 'wb') as f:
    #     pickle.dump(d, f)
    #
    # d = {"prep": count_vect, "model": model_2}
    # with open(__sklearn_naive_bayes, 'wb') as f:
    #     pickle.dump(d, f)
    model_4 = KNeighborsClassifier()
    model_4.fit(X_train, y_train)
    model_4_pred = model_4.predict(X_test)

    print(f"Accuracy model KNN: {metrics.accuracy_score(y_test, model_4_pred) * 100:.2f}%")

    # d = {"prep": count_vect, "model": model_1}
    # with open(__sklearn_regresion, 'wb') as f:
    #     pickle.dump(d, f)
    #
    # d = {"prep": count_vect, "model": model_2}
    # with open(__sklearn_naive_bayes, 'wb') as f:
    #     pickle.dump(d, f)

    d = {"prep": count_vect, "model": model_3}
    with open(__sklearn_mlp, 'wb') as f:
        pickle.dump(d, f)

    # d = {"prep": count_vect, "model": model_4}
    # with open(__sklearn_knn, 'wb') as f:
    #     pickle.dump(d, f)

    # d = {"prep": count_vect, "model": model_4}
    # with open(__sklearn_knn, 'wb') as f:
    #     pickle.dump(d, f)




    return ""

def training_model_evaluate_tensor():
    df = pd.read_sql_query(
        sql=db.select([ProcessedText.kalimat, ProcessedText.sentimen, ProcessedText.kalimat_bersih,
                       ProcessedText.kalimat_bersih_v2, ProcessedText.kalimat_bersih_v3, ProcessedText.jumlah_kata,
                       ProcessedText.jumlah_kalimat]),
        con=db.engine
    )
    max_features = 100000
    over = RandomOverSampler()
    print(df.head())
    # imbalance
    # positif 6383 negatif 3412 netral 1138
    print(f' {len(df[df["sentimen"] == "positive"])} {len(df[df["sentimen"] == "negative"])} {len(df[df["sentimen"] == "neutral"])}')
    count_vect = CountVectorizer()
    count_vect.fit(df['kalimat_bersih'])

    # jumlah unique words
    word_features = count_vect.get_feature_names_out()
    print(f"jumlah unique words: {len(word_features)}")

    # Hasil Transformasi
    # transformed = count_vect.transform(df['kalimat_bersih'])
    # X = transformed.toarray()
    # print(f"Hasil transformasi array shape: {X.shape}")

    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(df['kalimat_bersih'])
    word_features = tfidf_vect.get_feature_names_out()
    #
    # # Hasil Transformasi
    transformed = tfidf_vect.transform(df['kalimat_bersih'])
    X = transformed.toarray()
    print(f"Hasil transformasi array shape: {X.shape}")
    print(X)

    y = df['sentimen']
    y = y.factorize()[0]

    # tokenizer =f tf.keras.preprocessing.text.Tokenizer(num_words=len(word_features))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    num_classes = max(y_train) + 1

    # oversampling
    print(X_train[0])
    X_train, y_train = over.fit_resample(X_train, y_train)
    print(np.max(X_train[0]))
    print(str(y_train))

    # tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
    # X_train = tokenizer.sequences_to_matrix(X_train, mode='weight')
    # X_test = tokenizer.sequences_to_matrix(X_test, mode='weight')

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, 100, input_length=X.shape[1]))
    model.add(tf.keras.layers.LSTM(64, dropout=0.2))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.summary()

    # embed_dim = 100
    # units = 64
    # #
    # model = tf.keras.Sequential()
    # model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
    # model.add(LSTM(units, dropout=0.2))
    # model.add(Dense(num_classes, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    model.compile(loss="categorical_crossentropy",
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.fit(X_train, y_train,
    #           epochs=20,
    #           batch_size=32,
    #           validation_split=0.1, validation_data=(X_test, y_test))

    es =tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0)
    model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

    # d = {"prep": count_vect, "model": model}
    # with open(__sklearn_tensor_neural_network, 'wb') as f:
    #     pickle.dump(d, f)

    # model.save(__sklearn_tensor_neural_network)

    return ""


def predict_text(text):
    # with open(__sklearn_regresion, 'rb') as f:
    #     package = pickle.load(f)
    #

    # prediksi_reg = package["model"].predict(kalimat_array)[0]
    #
    # with open(__sklearn_naive_bayes, 'rb') as f:
    #     package = pickle.load(f)
    #
    # prediksi_nb = package["model"].predict(kalimat_array)[0]
    #
    # with open(__sklearn_knn, 'rb') as f:
    #     package = pickle.load(f)
    #
    # prediksi_knn = package["model"].predict(kalimat_array)[0]

    with open(__sklearn_mlp, 'rb') as f:
        package = pickle.load(f)
    kalimat_array = package["prep"].transform([text]).toarray()
    prediksi_mlp = package["model"].predict(kalimat_array)[0]

    lstm_tensor = tf.keras.models.load_model(__sklearn_tensor_neural_network)
    a = list(lstm_tensor.predict(kalimat_array)[0])
    tensor_prediction = (a.index(max(a)))
    label = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    data = {
        'mlp': label[prediksi_mlp],
        'LSTM': label[tensor_prediction]
    }

    return data

def predict_neural_network_text(text):
    with open(__sklearn_mlp, 'rb') as f:
        package = pickle.load(f)
    kalimat_array = package["prep"].transform([text]).toarray()
    prediksi_mlp = package["model"].predict(kalimat_array)[0]

    label = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']

    return label[prediksi_mlp]

def predict_LSTM(text):
    with open(__sklearn_mlp, 'rb') as f:
        package = pickle.load(f)
    kalimat_array = package["prep"].transform([text]).toarray()

    label = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    lstm_tensor = tf.keras.models.load_model(__sklearn_tensor_neural_network)
    a = list(lstm_tensor.predict(kalimat_array)[0])
    tensor_prediction = (a.index(max(a)))

    return label[tensor_prediction]

def test_LSTM():
    df = pd.read_sql_query(
        sql=db.select([ProcessedText.kalimat, ProcessedText.sentimen, ProcessedText.kalimat_bersih,
                       ProcessedText.kalimat_bersih_v2, ProcessedText.kalimat_bersih_v3, ProcessedText.jumlah_kata,
                       ProcessedText.jumlah_kalimat]),
        con=db.engine
    )

    over = RandomOverSampler()
    print(df.head())
    # imbalance
    # positif 6383 negatif 3412 netral 1138
    print(f' {len(df[df["sentimen"] == "positive"])} {len(df[df["sentimen"] == "negative"])} {len(df[df["sentimen"] == "neutral"])}')

    print(df.sentimen.value_counts())

    neg = df.loc[df['sentimen'] == 'negative'].kalimat_bersih.tolist()
    neu = df.loc[df['sentimen'] == 'neutral'].kalimat_bersih.tolist()
    pos = df.loc[df['sentimen'] == 'positive'].kalimat_bersih.tolist()

    neg_label = df.loc[df['sentimen'] == 'negative'].sentimen.tolist()
    neu_label = df.loc[df['sentimen'] == 'neutral'].sentimen.tolist()
    pos_label = df.loc[df['sentimen'] == 'positive'].sentimen.tolist()

    total_data = pos + neu + neg
    labels = pos_label + neu_label + neg_label

    print("Pos: %s, Neu: %s, Neg: %s" % (len(pos), len(neu), len(neg)))
    print("Total data: %s" % len(total_data))

    max_features = 100000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features, split=' ', lower=True)
    tokenizer.fit_on_texts(total_data)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("tokenizer.pickle has created!")

    X = tokenizer.texts_to_sequences(total_data)

    vocab_size = len(tokenizer.word_index)
    maxlen = max(len(x) for x in X)

    X = pad_sequences(X)
    with open('x_pad_sequences.pickle', 'wb') as handle:
        pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("x_pad_sequences.pickle has created!")

    Y = pd.get_dummies(labels)
    Y = Y.values

    with open('y_labels.pickle', 'wb') as handle:
        pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("y_labels.pickle has created!")

    file = open("x_pad_sequences.pickle", 'rb')
    X = pickle.load(file)
    file.close()

    file = open("y_labels.pickle", 'rb')
    Y = pickle.load(file)
    file.close()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    embed_dim = 100
    units = 64
    X_train, y_train = over.fit_resample(X_train, y_train)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, embed_dim, input_length=X.shape[1]))
    model.add(tf.keras.layers.LSTM(units, dropout=0.2))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test), verbose=1,
                        callbacks=[es])

    predictions = model.predict(X_test)
    y_pred = predictions
    matrix_test = metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print("Testing selesai")
    print(matrix_test)
    return ""