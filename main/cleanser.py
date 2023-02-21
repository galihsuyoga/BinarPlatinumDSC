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
from sklearn.model_selection import cross_validate
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

emoticons_happy = [
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ]
emoticons_sad = [
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ]
emoticons = emoticons_happy + emoticons_sad
# adding text need to remove
emoticons.append("http")
emoticons.append("url")

# Platinum =======================================================================================================
def clean_text(text):
    # cek apakah string ataukah afloat
    if type(text) == np.float:
        return ""
    # list dari pattern emoji
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+"
        , flags=re.UNICODE)

    # menghapus spesial karakter diakhir string karena error UnicodeDecodeError: 'unicodeescape' codec can't decode byte 0x5c in position 209: \ at end of string
    # temp = re.sub(f"[ ,;:/]*$", " ", tweet)

    # hapus karakter yang dimulai dengan \x (gak jadi pake. karena kalo nempel kata dibelakang jadi kehapus)
    # temp = re.sub("\\\\x[a-z0-9_]+", " ", temp)

    try:
        # sub karakter \n
        temp = re.sub(f"\\\\n", " ", text)
        # ubah ke utf, ganti \\ dengan \ lalu ganti \u dengan \u000 lalu balikan ke unicode_escape
        temp = temp.encode(
            'unicode_escape').decode('utf-8').replace('\\\\', '\\').replace('\\u', '\\U000').encode('latin-1').decode(
            'unicode-escape')
    except:

        # jika error karena UnicodeDecodeError: 'unicodeescape' codec can't decode byte 0x5c in position 209: \ at end of string
        # split ke kata2
        array_split = text.split()
        # hapus kata atau simbol byte terakhir dengan harapan tak ada lagi unicodeescape
        temp = text.replace(array_split[-1], " ")
        # sub karakter \n
        temp = re.sub(f"\\\\n", " ", temp)
        # ubah ke utf, ganti \\ dengan \ lalu ganti \u dengan \u000 lalu balikan ke unicode_escape
        temp = temp.encode(
            'unicode_escape').decode('utf-8').replace('\\\\', '\\').replace('\\u', '\\U000').encode('latin-1').decode(
            'unicode-escape')

    # merubah byte patern emoji jadi emoji
    temp = EMOJI_PATTERN.sub(r'', temp)

    # hapus mention @
    temp = re.sub("@[A-Za-z0-9_]+", "", temp)
    # hapus hashtag
    # temp = re.sub("#[A-Za-z0-9_]+","", temp) kayaknya gak perlu
    # hapus amp
    temp = re.sub("&amp"," ", temp)
    # hapus tanda baca ()!?
    temp = re.sub('[()!?]', ' ', temp)
    # hapus karakter yang tidak dalam range a-z0-9x (karena emojinya udah jadi bentuk, bukan huruf, emoji bisa tersingkir)
    temp = re.sub("[^A-Za-z0-9]", " ", temp)
    # lowercase huruf
    temp = temp.lower()
    # # hapus kata duplikasi yang berurutan
    temp = re.sub(r'\b(\w+)( \1\b)+', r'\1', temp)

    return temp
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

        text = clean_text(text)
        text = ' '.join([word for word in text.split() if word not in (listStopword)])
        text = re.sub(r'[^\w\s]|[0-9]', ' ', text)
        text = text.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')
        # print(text)
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
    # print(df['kalimat_bersih_v3'][:5])

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
    count_vect.fit(df['kalimat_bersih_v3'])

    # jumlah unique words
    word_features = count_vect.get_feature_names_out()
    print(f"jumlah unique words: {len(word_features)}")

    # Hasil Transformasi
    transformed = count_vect.transform(df['kalimat_bersih_v3'])
    X = transformed.toarray()
    print(f"Hasil transformasi array shape: {X.shape}")


    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(df['kalimat_bersih_v3'])

    # Hasil Transformasi
    transformed = tfidf_vect.transform(df['kalimat_bersih_v3'])
    X = transformed.toarray()
    print(f"Hasil transformasi array shape: {X.shape}")
    print(X)

    y = df['sentimen']
    y = y.factorize()[0]

    # oversampling

    # split with stratiy on
    X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.1, random_state=0, stratify=y)

    print(X_train[:10])
    print(y_train.shape)
    # create model
    model_3 = MLPClassifier(100)
    model_3.fit(X_train, y_train)
    model_3_pred = model_3.predict(X_test)
    print(f"Accuracy model MLP: {metrics.accuracy_score(y_test, model_3_pred) * 100:.2f}%")

    # save model
    d = {"prep": count_vect, "model": model_3}
    with open(__sklearn_mlp, 'wb') as f:
        pickle.dump(d, f)

    print(cross_validation(model=model_3, _X=X_test, _y=y_test))
    return ""

def training_model_evaluate_tensor():
    # vanilla model. not used
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
    num_classes = max(y_train) + 1

    # oversampling
    print(X_train[0])
    X_train, y_train = over.fit_resample(X_train, y_train)
    print(np.max(X_train[0]))
    print(str(y_train))


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
    # not used
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
    # MLP prediction rom saved model
    with open(__sklearn_mlp, 'rb') as f:
        package = pickle.load(f)
    kalimat_array = package["prep"].transform([text]).toarray()
    prediksi_mlp = package["model"].predict(kalimat_array)[0]

    label = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']

    return label[prediksi_mlp]

def predict_LSTM(text):
    #LSTM prediction rom saved model
    file = open("x_pad_sequences.pickle", 'rb')
    sequence = pickle.load(file)
    # file.close()
    #
    # file = open("y_labels.pickle", 'rb')
    # Y = pickle.load(file)

    file = open('tokenizer.pickle', 'rb')
    tokenizer = pickle.load(file)
    file.close()

    cleaned_text = cleanser_string_step(text=text, step=3)
    tokenize_text = tokenizer.texts_to_sequences([cleaned_text])
    # print(tokenize_text)
    text_padding = tf.keras.preprocessing.sequence.pad_sequences(tokenize_text, 500)
    # print(text_padding)
    label = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    lstm_tensor = tf.keras.models.load_model(__sklearn_tensor_neural_network)
    lll = lstm_tensor.predict(text_padding)
    # print(lll)
    a = list(list(lll[0]))
    tensor_prediction = a.index(max(a))
    # print(a)

    return label[tensor_prediction]

def test_LSTM():
    # create model and save
    df = pd.read_sql_query(
        sql=db.select([ProcessedText.kalimat, ProcessedText.sentimen, ProcessedText.kalimat_bersih,
                       ProcessedText.kalimat_bersih_v2, ProcessedText.kalimat_bersih_v3, ProcessedText.jumlah_kata,
                       ProcessedText.jumlah_kalimat, ProcessedText.kalimat_bersih_v4]),
        con=db.engine
    )

    over = RandomOverSampler()
    print(df.head())
    # imbalance
    # positif 6383 negatif 3412 netral 1138
    print(f' {len(df[df["sentimen"] == "positive"])} {len(df[df["sentimen"] == "negative"])} {len(df[df["sentimen"] == "neutral"])}')
    xlabel='kalimat_bersih_v3'

    print(df.sentimen.value_counts())

    neg = df.loc[df['sentimen'] == 'negative'][xlabel].tolist()
    neu = df.loc[df['sentimen'] == 'neutral'][xlabel].tolist()
    pos = df.loc[df['sentimen'] == 'positive'][xlabel].tolist()

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

    print(f'vofcab size {vocab_size}, max length of text is {maxlen}')
    sequence_length = 500
    # X = pad_sequences(X)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=sequence_length)
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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)

    embed_dim = 100
    units = 64
    # resample X_train and ytrain
    # X_train, y_train = over.fit_resample(X_train, y_train)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, embed_dim, input_length=X.shape[1]))
    model.add(tf.keras.layers.LSTM(units, dropout=0.2))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    adam = tf.keras.optimizers.Nadam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test), verbose=1,
                        callbacks=[es])
    model.save(__sklearn_tensor_neural_network)
    predictions = model.predict(X_test)
    y_pred = predictions
    matrix_test = metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print("Testing selesai")
    print(matrix_test)

    return ""


def cross_validation(model, _X, _y, _cv=5):
    '''Function to perform 5 Folds Cross-Validation
     Parameters
     ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    _X: array
         This is the matrix of features.
    _y: array
         This is the target variable.
    _cv: int, default=5
        Determines the number of folds for cross-validation.
     Returns
     -------
     The function returns a dictionary containing the metrics 'accuracy', 'precision',
     'recall', 'f1' for both training set and validation set.
    '''
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=_X,
                             y=_y,
                             cv=_cv,
                             scoring=_scoring,
                             return_train_score=True)

    return {"Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean() * 100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean() * 100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }