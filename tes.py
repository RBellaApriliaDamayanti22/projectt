import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import nltk
nltk.download('punkt')
import string 
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import pickle
import ast
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("Informatika Pariwisata A ")
st.write("### Dosen Pengampu : Ika Oktavia Suzanti, S.Kom., M.Cs.")
st.write("##### Kelompok 3")
st.write("##### R.Bella Aprilia Damayanti - 200411100082")
st.write("#####  Triasmi Dwi Farawati - 200411100186 ")


#Navbar
data_set_description, upload_data, preprocessing, ekstraksifitur, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Preprocessing", "Tf-Idf", "Modeling", "Implementation"])
dataset = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/Data.csv")

#data_set_description
with data_set_description:
    st.write("###### Judul : Analisis Sentiment Review Tempat Pariwisata dengan Automated Lexicon Word2Vec dan Naive Bayes ")
    st.write("""###### Penjelasan Prepocessing Data : """)
    st.write("""1. Case Folding :
    
    Case folding adalah proses dalam pemrosesan teks yang mengubah semua huruf dalam teks menjadi huruf kecil atau huruf besar. Tujuan dari case folding adalah untuk mengurangi variasi yang disebabkan oleh perbedaan huruf besar dan kecil dalam teks, sehingga mempermudah pemrosesan teks secara konsisten.
    
    Dalam case folding, biasanya semua huruf dalam teks dikonversi menjadi huruf kecil dengan menggunakan metode seperti lowercasing. Dengan demikian, perbedaan antara huruf besar dan huruf kecil tidak lagi diperhatikan dalam analisis teks, sehingga memungkinkan untuk mendapatkan hasil yang lebih konsisten dan mengurangi kompleksitas dalam pemrosesan teks.
    """)
    st.write("""2. Tokenize :

    Tokenisasi adalah proses pemisahan teks menjadi unit-unit yang lebih kecil yang disebut token. Token dapat berupa kata, frasa, atau simbol lainnya, tergantung pada tujuan dan aturan tokenisasi yang digunakan.

    Tujuan utama tokenisasi dalam pemrosesan bahasa alami (Natural Language Processing/NLP) adalah untuk memecah teks menjadi unit-unit yang lebih kecil agar dapat diolah lebih lanjut, misalnya dalam analisis teks, pembentukan model bahasa, atau klasifikasi teks.
    """)
    st.write("""3. Filtering (Stopword Removal) :

    Filtering atau Stopword Removal adalah proses penghapusan kata-kata yang dianggap tidak memiliki makna atau kontribusi yang signifikan dalam analisis teks. Kata-kata tersebut disebut sebagai stop words atau stopwords.

    Stopwords biasanya terdiri dari kata-kata umum seperti “a”, “an”, “the”, “is”, “in”, “on”, “and”, “or”, dll. Kata-kata ini sering muncul dalam teks namun memiliki sedikit kontribusi dalam pemahaman konten atau pengambilan informasi penting dari teks.

    Tujuan dari Filtering atau Stopword Removal adalah untuk membersihkan teks dari kata-kata yang tidak penting sehingga fokus dapat diarahkan pada kata-kata kunci yang lebih informatif dalam analisis teks. Dengan menghapus stopwords, kita dapat mengurangi dimensi data, meningkatkan efisiensi pemrosesan, dan memperbaiki kualitas hasil analisis.
    """)
    st.write("""4. Stemming :

    Stemming adalah proses mengubah kata ke dalam bentuk dasarnya atau bentuk kata yang lebih sederhana, yang disebut sebagai “stem”. Stemming bertujuan untuk menghapus infleksi atau imbuhan pada kata sehingga kata-kata yang memiliki akar kata yang sama dapat diidentifikasi sebagai bentuk yang setara.
    """)
    
    st.write("""###### Penjelasan Ekstraksi Fitur : """)
    st.write("""TF-IDF :""")
    st.write("""Ditahap akhir dari text preprocessing adalah term-weighting .Term-weighting merupakan proses pemberian bobot term pada dokumen. Pembobotan ini digunakan nantinya oleh algoritma Machine Learning untuk klasifikasi dokumen. Ada beberapa metode yang dapat digunakan, salah satunya adalah TF-IDF (Term Frequency-Inverse Document Frequency).""")
    st.write("""TF (Term Frequency) :""")
    st.write("""TF (Term Frequency) adalah ukuran yang menggambarkan seberapa sering sebuah kata muncul dalam suatu dokumen. Menghitung TF melibatkan perbandingan jumlah kemunculan kata dengan jumlah kata keseluruhan dalam dokumen.""")
    st.write("""Perhitungan TF (Term Frequency) :
    
    TF(term) = (Jumlah kemunculan term dalam dokumen) / (Jumlah kata dalam dokumen)
    """)
    st.write("""DF (Document Frequency) :""")
    st.write("""DF (Document Frequency) adalah ukuran yang menggambarkan seberapa sering sebuah kata muncul dalam seluruh koleksi dokumen. DF menghitung jumlah dokumen yang mengandung kata tersebut.""")
    st.write("""Perhitungan DF (Document Frequency) :
    
    DF(term) = Jumlah dokumen yang mengandung term
    """)
    st.write("""IDF (Inverse Document Frequency) :""")
    st.write("""IDF (Inverse Document Frequency) adalah ukuran yang menggambarkan seberapa penting sebuah kata dalam seluruh koleksi dokumen. IDF dihitung dengan mengambil logaritma terbalik dari rasio total dokumen dengan jumlah dokumen yang mengandung kata tersebut. Tujuan IDF adalah memberikan bobot yang lebih besar pada kata-kata yang jarang muncul dalam seluruh koleksi dokumen.""")
    st.write("""Perhitungan IDF (Inverse Document Frequency) :
    
    IDF(term) = log((Total jumlah dokumen) / (DF(term)))
    """)
    st.write("""TF-IDF (Term Frequency-Inverse Document Frequency) :""")
    st.write("""TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode yang menggabungkan informasi TF dan IDF. TF-IDF memberikan bobot yang lebih tinggi pada kata-kata yang sering muncul dalam dokumen tertentu (TF tinggi) dan jarang muncul dalam seluruh koleksi dokumen (IDF tinggi). Metode ini digunakan untuk mengevaluasi kepentingan relatif suatu kata dalam konteks dokumen.""")
    st.write("""Perhitungan TF-IDF (Term Frequency-Inverse Document Frequency) :
    
    TF-IDF(term, document) = TF(term, document) * IDF(term)
    """)
    st.write("""Dalam perhitungan TF-IDF, TF(term, document) adalah nilai TF untuk term dalam dokumen tertentu, dan IDF(term) adalah nilai IDF untuk term di seluruh koleksi dokumen.""")
    st.write("""Mengubah representasi teks ke dalam vektor :
    """)
    
    st.write("###### Aplikasi ini untuk : ")
    st.write("""Analisis Sentiment Review Tempat Pariwisata dengan Automated Lexicon Word2Vec dan Naive Bayes """)
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/RBellaApriliaDamayanti22/projectt")


#Uploud data
with upload_data:
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        # view dataset asli
        st.header("Dataset")
        st.dataframe(df)

with preprocessing:
    st.subheader("Preprocessing Data")
    
    st.write("Case Folding:")
    dataset['ulasan'] = dataset['ulasan'].str.lower()
    st.write("Case Folding Result:")
    st.write(dataset['ulasan'].head())

    st.write("Tokenize:")
    def hapus_tweet_khusus(text):
        text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
        text = text.encode('ascii', 'replace').decode('ascii')
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
        return text.replace("http://", " ").replace("https://", " ")
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_tweet_khusus)
    
    def hapus_nomor(text):
        return re.sub(r"\d+", "", text)
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_nomor)
    
    def hapus_tanda_baca(text):
        return text.translate(str.maketrans("", "", string.punctuation))
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_tanda_baca)
    
    def hapus_whitespace_LT(text):
        return text.strip()
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_whitespace_LT)
    
    def hapus_whitespace_multiple(text):
        return re.sub('\s+', ' ', text)
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_whitespace_multiple)
    
    def hapus_single_char(text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_single_char)
    
    def word_tokenize_wrapper(text):
        tokenizer = RegexpTokenizer(r'dataran\s+tinggi|jawa\s+tengah|[\w\']+')
        tokens = tokenizer.tokenize(text)
        return tokens
    
    dataset['ulasan_tokens'] = dataset['ulasan'].apply(word_tokenize_wrapper)
    st.write(dataset['ulasan_tokens'].head())
    
    def freqDist_wrapper(text):
        return FreqDist(text)
    
    dataset['ulasan_tokens_fdist'] = dataset['ulasan_tokens'].apply(freqDist_wrapper)
    st.write(dataset['ulasan_tokens_fdist'].head())
    
    st.write("Filtering (Stopword Removal):")
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                        '&amp', 'yah'])
    
    txt_stopword = pd.read_csv("https://raw.githubusercontent.com/RBellaApriliaDamayanti22/projectt/main/list_stopwords.csv", names=["stopwords"], header=None)
    
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
    list_stopwords = set(list_stopwords)
    
    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    dataset['ulasan_tokens_WSW'] = dataset['ulasan_tokens'].apply(stopwords_removal)
    st.write(dataset['ulasan_tokens_WSW'].head())

    st.write("Stemming:")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in dataset['ulasan_tokens_WSW']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term, ":", term_dict[term])

    print(term_dict)
    print("------------------------")

    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    dataset['ulasan_tokens_stemmed'] = dataset['ulasan_tokens_WSW'].apply(get_stemmed_term)
    st.write(dataset['ulasan_tokens_stemmed'].head())

with modeling:
    st.write("Menyimpan data hasil preprocessing ke pickle")
    with open('data.pickle', 'wb') as file:
        pickle.dump(dataset, file)

    # Memuat data dari file pickle
    with open('data.pickle', 'rb') as file:
        loaded_data = pickle.load(file)
    Data_ulasan = pd.DataFrame(loaded_data, columns=["label", "ulasan"])
    Data_ulasan.head()
    ulasan = Data_ulasan['ulasan']
    sentimen = Data_ulasan['label']
    X_train, X_test, y_train, y_test = train_test_split(ulasan, sentimen, test_size=0.2, random_state=42)

    def convert_text_list(texts):
        try:
            texts = ast.literal_eval(texts)
            if isinstance(texts, list):
                return texts
            else:
                return []
        except (SyntaxError, ValueError):
            return []

    Data_ulasan["ulasan_list"] = Data_ulasan["ulasan"].apply(convert_text_list)
    print(Data_ulasan["ulasan_list"][90])
    print("\ntype: ", type(Data_ulasan["ulasan_list"][90]))

    # Ekstraksi fitur menggunakan TF-IDF
    def calculate_tf(corpus):
        tf_dict = {}
        for document in corpus:
            words = document.split()
            for word in words:
                if word not in tf_dict:
                    tf_dict[word] = 1
                else:
                    tf_dict[word] += 1
        total_words = sum(tf_dict.values())
        for word in tf_dict:
            tf_dict[word] = tf_dict[word] / total_words
        return tf_dict

    def calculate_df(corpus):
        df_dict = {}
        for document in corpus:
            words = set(document.split())
            for word in words:
                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] += 1
        return df_dict

    def calculate_idf(corpus):
        idf_dict = {}
        N = len(corpus)
        df_dict = calculate_df(corpus)
        for word in df_dict:
            idf_dict[word] = np.log(N / df_dict[word])
        return idf_dict

    def calculate_tfidf(tf_dict, idf_dict):
        tfidf_dict = {}
        for word in tf_dict:
            if word in idf_dict:
                tfidf_dict[word] = tf_dict[word] * idf_dict[word]
            else:
                tfidf_dict[word] = 0
        return tfidf_dict

    tf_train = calculate_tfidf(calculate_tf(X_train), calculate_idf(X_train))
    tf_test = calculate_tfidf(calculate_tf(X_test), calculate_idf(X_train))

    print("Term Frequency-Inverse Document Frequency (TF-IDF):")
    for i, document in enumerate(X_train):
        tfidf_dict = calculate_tfidf(calculate_tf([document]), calculate_idf(X_train))
        print(f"Document {i+1}:")
        for word, tfidf in tfidf_dict.items():
            print(f"{word}: {tfidf}")
        print()

    def text_to_vector(text, tfidf_dict):
        words = text.split()
        vector = np.zeros(len(tfidf_dict))
        for i, word in enumerate(tfidf_dict):
            if word in words:
                vector[i] = tfidf_dict[word]
        return vector

    X_train_vectors = [text_to_vector(document, tf_train) for document in X_train]
    X_test_vectors = [text_to_vector(document, tf_test) for document in X_test]

    print("Vector Representation (TF-IDF):")
    for i, vector in enumerate(X_train_vectors):
        print(f"Document {i+1}: {vector}")
    print()

    # Klasifikasi menggunakan KNN
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train_vectors, y_train)

# Implementasi dengan Streamlit
with implementation:
    st.title("Klasifikasi Sentimen Ulasan Menggunakan KNN")
    st.write("Masukkan ulasan di bawah ini:")
    input_text = st.text_input("Ulasan")

    if st.button("Prediksi"):
        # Mengubah input ulasan menjadi vektor
        input_vector = text_to_vector(input_text, tf_train)

        # Melakukan prediksi pada input ulasan
        predicted_label = knn_classifier.predict([input_vector])

        # Menampilkan hasil prediksi
        st.write("Hasil Prediksi:")
        st.write(f"Ulasan: {input_text}")
        st.write(f"Label: {predicted_label[0]}")

    # Menghitung akurasi pada data uji
    y_pred = knn_classifier.predict(X_test_vectors)
    accuracy = accuracy_score(y_test, y_pred)

    # Menampilkan akurasi
    st.write("Akurasi: {:.2f}%".format(accuracy * 100))

    # Menampilkan label prediksi
    st.write("Label Prediksi:")
    for i, (label, ulasan) in enumerate(zip(y_pred, X_test)):
        st.write(f"Data Uji {i+1}:")
        st.write(f"Ulasan: {ulasan}")
        st.write(f"Label: {label}")
        st.write()
