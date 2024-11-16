import pandas as pd
import numpy as np
from spacy.lang.es.stop_words import STOP_WORDS
from sklearn.decomposition import PCA
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
import stanza
from nltk.corpus import stopwords
import en_core_web_lg
from sklearn.decomposition import PCA
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from gensim import corpora, models
from pprint import pprint
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import spacy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta mensajes de información y advertencias
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactiva las optimizaciones de oneDNN
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input,Concatenate
from sklearn.metrics import accuracy_score, roc_auc_score



def cargar_datos():
    path = os.path.join(os.getcwd(), "Ejercicios Quiz")
    archivo_tsv = "training_set_rel3.tsv"
    df = pd.read_csv(path + archivo_tsv, delimiter='\t', encoding='latin1')
    datos=  pd.read_pickle(path + 'training_features_NLP_Ensayos.pkl')

    return datos, path,df


def preparacion_datos(datos, path):
  datos = datos.get(["essay_id","corrected","tokens","essay_set",
           "corrections","token_count","unique_token_count",
           "nostop_count","sent_count","ner_count","comma","question",
           "exclamation", "quotation", "organization", "caps", "person",
           "location", "money", "time", "date", "percent", "noun", "adj", "pron",
           "verb", "cconj", "adv", "det", "propn", "num", "part", "intj"])
  datos = datos.rename(columns = {'corrected':'essay'})
  datos = datos.drop(columns=['essay_set'])
  datos['essay'].to_csv(path + 'ensayos_NLP_Ensayos.csv')
  return datos


## MODELO LDA

def lemmatize_stemming(text):
    ps = PorterStemmer()
    return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text): #  gensim.utils.simple_preprocess tokeniza el texto
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def modelo_LDA(datos, path):
  data = datos.get(['essay','essay_id'])
  nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
  stop_nltk = stopwords.words('english')
  nlp = en_core_web_lg.load()
  stop_spacy = nlp.Defaults.stop_words
  stop_todas = list(stop_spacy.union(set(stop_nltk)))
  data=  pd.read_pickle(path + 'training_procesed_text.pkl')
  data["essay"] = data["processed_text"].copy()
  data = data.drop(["processed_text"],axis = 1)
  documents = data
  doc_sample = documents[documents['essay_id'] == 10].values[0][0]
  words = []
  for word in doc_sample.split(' '):
    words.append(word)
  processed_docs = documents['essay'].map(preprocess)
  dictionary = gensim.corpora.Dictionary(processed_docs)
  count = 0
  for k, v in dictionary.iteritems():
    count += 1
    if count > 10:
        break
  dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=500)
  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
  tfidf = models.TfidfModel(bow_corpus)
  corpus_tfidf = tfidf[bow_corpus]
  return dictionary, bow_corpus, tfidf, corpus_tfidf,documents,data


def LDA_wo(bow_corpus, data,dictionary):
  lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=8, id2word=dictionary, passes=2, workers=2)
  ind_without_tfidf = lda_model[bow_corpus]
  topics_wo = []
  for y in range(data.shape[0]):
    if len(ind_without_tfidf[y]) > 0:
      valid_sublist = [sublist for sublist in ind_without_tfidf[y] if len(sublist) > 1]
      if len(valid_sublist) > 0:
        max_index = np.argmax([sublist[1] for sublist in valid_sublist])
        topics_wo.append(valid_sublist[max_index][0])
      else:
        topics_wo.append(None)
    else:
      topics_wo.append(None)
  data["topic"] = topics_wo
  return data

def LDA_TFIDF(tfidf, data,dictionary,corpus_tfidf, bow_corpus):
  lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary, passes=2, workers=4)
  ind_with_tfidf = lda_model_tfidf[bow_corpus]
  topics_with = []
  for y in range(data.shape[0]):
    if len(ind_with_tfidf[y]) > 0:
      valid_sublist = [sublist for sublist in ind_with_tfidf[y] if len(sublist) > 1]
      if len(valid_sublist) > 0:
        max_index = np.argmax([sublist[1] for sublist in valid_sublist])
        topics_with.append(valid_sublist[max_index][0])
      else:
        topics_with.append(None)
    else:
      topics_with.append(None)
  data["topic_tfidf"] = topics_with
  data_LDA = data.copy()
  return data_LDA

def embeddings(path):
  data=  pd.read_pickle(path + 'training_procesed_text_embeddings_finaaaall.pkl')
  embeddings = np.stack(data['bert_embedding'].values)
  pca = PCA(n_components=2)
  embeddings_pca = pca.fit_transform(embeddings)
  df_pca = pd.DataFrame(embeddings_pca, columns=['x', 'y'])
  n_max = 10  
  silhouette_coefficients = []
  scaler = StandardScaler()
  scaled_pcs = scaler.fit_transform(embeddings_pca)
  for k in range(2, n_max + 1):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_pcs)
    score = silhouette_score(scaled_pcs, kmeans.labels_)
    silhouette_coefficients.append(score)
  kmeans = KMeans(n_clusters=8, random_state=0).fit(embeddings)
  data['cluster'] = kmeans.labels_
  return data, n_max

def fast_text(path,data):
  doc_embedding = pd.read_csv(path + 'Doc_Embedding_300_NLP_Ensayos.csv',index_col=0)
  pca = PCA(n_components=30, random_state=0)
  pcs = pca.fit_transform(doc_embedding.values)
  scaler = StandardScaler()
  scaled_pcs = scaler.fit_transform(pcs)
  silhouette_coefficients = []
  kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42}
  for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_pcs)
    score = silhouette_score(scaled_pcs, kmeans.labels_)
    silhouette_coefficients.append(score)
  K_ = 8
  km = KMeans(n_clusters=K_, random_state=0)
  km.fit_transform(scaled_pcs)
  cluster_labels = km.labels_
  cluster_labels = pd.DataFrame(cluster_labels, columns=['Grupo'])
  data["FAST"] = km.labels_
  return data

def compar_models(data,data_LDA,df):
  data = data.drop(["Unnamed: 0"],axis = 1)
  data["essay_id"] = data_LDA["essay_id"].copy()
  data_LDA = data_LDA.drop(["essay"],axis = 1)
  data = pd.merge(data,data_LDA,on = ["essay_id"], how = "left")
  data["essay_set"] = df["essay_set"].copy()
  crosstabb =pd.crosstab(data["essay_set"],data["topic_tfidf"])
  crosstab1 = pd.crosstab(data["essay_set"],data["topic"])
  crosstab2 = pd.crosstab(data["essay_set"],data["FAST"])
  crosstab3 = pd.crosstab(data["essay_set"],data["cluster"])
  return crosstabb,crosstab1,crosstab2,crosstab3,data

def reorganizar_matriz(crosstab):
    crosstab = crosstab.apply(pd.to_numeric, errors='coerce')
    available_cols = list(crosstab.columns)
    row_order = []
    col_order = []

    for row in crosstab.index:
        max_col = crosstab.loc[row, available_cols].idxmax()
        row_order.append(row)
        col_order.append(max_col)
        available_cols.remove(max_col)
        
    crosstab_sorted = crosstab.loc[row_order, col_order]
    matriz = crosstab_sorted.values
    return matriz

def calcular_accuracy(matriz):
    return np.diag(matriz).sum() * 100 / np.sum(matriz)

def accuracy_global(data_con_fast, data_con_tema_tfidf, df):
    crosstabb, crosstab1, crosstab2, crosstab3, data_comparada = compar_models(data_con_fast, data_con_tema_tfidf, df)
    matrices = [crosstabb, crosstab1, crosstab2, crosstab3]
    matrices_reorganizadas = [reorganizar_matriz(mat) for mat in matrices]
    accuracies = [calcular_accuracy(mat) for mat in matrices_reorganizadas]
    
    ac_tfidf, ac_topic, ac_fast, ac_bert = accuracies
    
    print(f"Accuracy TF-IDF: {ac_tfidf}%")
    print(f"Accuracy Topic: {ac_topic}%")
    print(f"Accuracy Fast: {ac_fast}%")
    print(f"Accuracy BERT: {ac_bert}%")
    return ac_tfidf, ac_topic, ac_fast, ac_bert, data_comparada

def tabla_roc_scores(path,data):
  resultados_roc = {}
  columnas = ['cluster', 'topic', 'topic_tfidf', 'FAST']
  for col in columnas:
    roc_scores = []
    for Clase in range(1, 9):
      prediccion = Clase - 1  
      data_final_copia = data.copy()
      data_final_copia.loc[data_final_copia["essay_set"] != Clase, "essay_set"] = -1
      data_final_copia.loc[data_final_copia["essay_set"] == Clase, "essay_set"] = 1
      data_final_copia.loc[data_final_copia["essay_set"] == -1, "essay_set"] = 0

      data_final_copia.loc[data_final_copia[col] != prediccion, col] = -1
      data_final_copia.loc[data_final_copia[col] == prediccion, col] = 1
      data_final_copia.loc[data_final_copia[col] == -1, col] = 0

      ytrue = list(data_final_copia["essay_set"])
      yest = list(data_final_copia[col])
      roc_score = roc_auc_score(ytrue, yest)
      roc_scores.append(roc_score)
    resultados_roc[col] = roc_scores
  resultados_df = pd.DataFrame(resultados_roc, index=range(1, 9))
  resultados_df.index.name = 'essay_set'
  resultados_df.to_csv(path + 'roc_scores.csv')
  return resultados_df

# RED NEURONAL 

def inic_red(path):
  doc_embedding = pd.read_csv(path + 'Doc_Embedding_300_NLP_Ensayos.csv',index_col=0)
  datos=  pd.read_pickle(path + 'training_red_neuronal.pkl')
  X_provisional = datos.get(["corrections","token_count"]) # dos variables al azar
  y_provisional = datos.get(["essay_set"]).to_numpy()
  X_train, X_test, y_train, y_test = train_test_split(X_provisional,y_provisional,test_size = 0.10,stratify = y_provisional)
  index_train = X_train.index
  index_test = X_test.index
  features = [x for x in datos.columns if x not in ["essay_set","essay","tokens","essay_id"]]
  X = datos.get(features).copy()
  y = datos.get(["essay_set"]).copy()
  return index_train, index_test, X, y, X_train, X_test, y_train, y_test,doc_embedding

def proc_red(X_train, X_test, y_train, y_test,index_train,index_test,doc_embedding,X,y):
  features_train = X.iloc[index_train,]
  embeddings_train = doc_embedding.iloc[index_train,]
  y_train = y.iloc[index_train,] # supervisada

  # Test
  features_test = X.iloc[index_test,]
  embeddings_test = doc_embedding.iloc[index_test,]
  y_test = y.iloc[index_test,] # supervisada

  y_train = y_train - 1
  y_test = y_test - 1
  return features_train, embeddings_train, y_train, features_test, embeddings_test, y_test

def estandariza(features_train,features_test):
  scaler = MinMaxScaler((-1.0,1.0))
  features_train_scaled = pd.DataFrame(scaler.fit_transform(features_train))
  features_train_scaled.columns = features_train.columns
  features_test_scaled = pd.DataFrame(scaler.fit_transform(features_test))
  features_test_scaled.columns = features_test.columns
  return features_train_scaled, features_test_scaled

def capas(features_train_scaled):
  embedding_vector_length = 300
  
  x1 = Input(shape=(embedding_vector_length,), name='Input_Embedding')
  x2 = Input(shape=(features_train_scaled.shape[1],), name='Input_Features')

  # Capa entrada
  x = Concatenate(name='Concatenar')([x1, x2])
  x = Dropout(0.25)(x)

  # capas ocultas
  x = Dense(64, activation='elu', name='Capa_Densa_1')(x)
  x = Dropout(0.25)(x)
  x = Dense(32, activation='elu', name='Capa_Densa_2')(x)
  x = Dropout(0.25)(x)
  x = Dense(16, activation='elu', name='Capa_Densa_3')(x)
  x = Dropout(0.25)(x)
  x = Dense(8, activation='softmax', name='Output')(x)
  model = Model(inputs=[x1, x2], outputs=x)
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def red_neuronal(embeddings_train,features_train_scaled,y_train,embeddings_test,features_test_scaled,y_test,model):
  history = model.fit(x = [embeddings_train,features_train_scaled],
                    y = y_train,
                    validation_data = ([embeddings_test,features_test_scaled],y_test),
                    epochs=100,
                    batch_size=32,verbose=1)
  return history

def pred_train(embeddings_train,features_train_scaled,y_train,model):
  y_pred = model.predict([embeddings_train,features_train_scaled])
  y_true = y_train
  y_pred = (y_pred >= 0.5).astype(int)
  y_pred_classes = np.argmax(y_pred, axis=1)
  accuracy = accuracy_score(y_true, y_pred_classes)
  return accuracy

def pred_test(embeddings_test,features_test_scaled,y_test,model):
  y_pred = model.predict([embeddings_test,features_test_scaled])
  y_true = y_test
  y_pred = (y_pred>=0.5).astype(int)
  y_pred_classes1 = np.argmax(y_pred, axis=1)
  accuracy1 = accuracy_score(y_true, y_pred_classes1)
  return accuracy1

def ejecutar( embeddings_train, features_train_scaled, y_train,
                               embeddings_test, features_test_scaled, y_test, model):
    history = red_neuronal(embeddings_train, features_train_scaled, y_train, 
                           embeddings_test, features_test_scaled, y_test, model)
    accuracy_train = pred_train(embeddings_train, features_train_scaled, y_train, model)
    accuracy_test = pred_test(embeddings_test, features_test_scaled, y_test, model)
    print(f"Accuracy en entrenamiento: {accuracy_train * 100:.2f}%")
    print(f"Accuracy en prueba: {accuracy_test * 100:.2f}%")
    return accuracy_train, accuracy_test, history


def cargar_y_preparar_datos():
    datos,path, df = cargar_datos()
    datos_procesados = preparacion_datos(datos, path)
    return datos_procesados, path,df
def crear_modelo_LDA(datos_procesados, path):
    dictionary, bow_corpus, tfidf, corpus_tfidf, documents, data = modelo_LDA(datos_procesados, path)
    data_con_tema_wo = LDA_wo(bow_corpus, data, dictionary)
    data_con_tema_tfidf = LDA_TFIDF(tfidf, data, dictionary, corpus_tfidf, bow_corpus)
    return data_con_tema_wo, data_con_tema_tfidf, data
def integrar_embeddings_y_modelos(path, data):
    data_con_embeddings, n_max = embeddings(path)
    data_con_fast = fast_text(path, data_con_embeddings)
    return data_con_fast
def comparar_modelos_y_calcular_accuracy(data_con_fast, data_con_tema_tfidf, df):
    crosstabb, crosstab1, crosstab2, crosstab3, data_comparada = compar_models(data_con_fast, data_con_tema_tfidf, df)
    ac_tfidf, ac_topic, ac_fast, ac_bert, data_comparada = accuracy_global(data_con_fast, data_con_tema_tfidf, df)
    return data_comparada
def generar_tabla_roc(path, data_comparada):
    resultados_roc = tabla_roc_scores(path, data_comparada)
    return resultados_roc

def parte1():
    datos_procesados, path, df = cargar_y_preparar_datos()
    data_con_tema_wo, data_con_tema_tfidf, data = crear_modelo_LDA(datos_procesados, path)
    data_con_fast = integrar_embeddings_y_modelos(path, data)
    data_comparada = comparar_modelos_y_calcular_accuracy(data_con_fast, data_con_tema_tfidf, df)
    resultados_roc = generar_tabla_roc(path, data_comparada)
    print(resultados_roc)
    return resultados_roc

def preparar_ejecutar_red_neuronal():
    datos_procesados, path, df = cargar_y_preparar_datos()
    index_train, index_test, X, y, X_train, X_test, y_train, y_test, doc_embedding = inic_red(path)
    features_train, embeddings_train, y_train, features_test, embeddings_test, y_test = proc_red(
    X_train, X_test, y_train, y_test, index_train, index_test, doc_embedding, X, y)
    features_train_scaled, features_test_scaled = estandariza(features_train, features_test)
    model = capas(features_train_scaled)
    accuracy_train, accuracy_test, history = ejecutar(
       embeddings_train, features_train_scaled, y_train,
        embeddings_test, features_test_scaled, y_test, model)
    return accuracy_train, accuracy_test, history
