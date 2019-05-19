#importando a base de dados
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


df_positivo = pd.read_csv(r"Diretorio\sentimentopositivo.csv", usecols=['text'])
df_negativo = pd.read_csv(r"Diretorio\sentimentonegativo.csv", usecols=['text']) 

df_positivo['label'] = 1
df_negativo['label'] = 0
print("save data frame")

#define dataset
join_frames = [df_positivo, df_negativo]
df_final_dataset = pd.concat(join_frames)
print("definiu o data set")
#Divide o que vai ser treino e teste
X_train, X_test, y_train, y_test = train_test_split(df_final_dataset['text'], 
                                                    df_final_dataset['label'], 
                                                    random_state=1,
                                                    test_size=0.30
                                                   )
print("dividiu em treino e teste")
# Transforma em matriz
import nltk
tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
count_vector = CountVectorizer(tokenizer=tokenizer.tokenize, max_features=1000)
doc_array = count_vector.fit_transform(X_train)

print("transformou em matriz")

# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1,max_depth=1, random_state=0)
# clf.fit(doc_array,y_train)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(doc_array, y_train)
print("treinou o modelo")

#Cria o dataset

print("acuracia")
dataset_test = count_vector.transform(X_test).toarray()

#Acuracia
# print (clf.score(dataset_test, y_test))
print(clf.score(dataset_test, y_test))

model_persist = {
    'model': clf,
    'vector': count_vector
}
import pickle 
pickle.dump(model_persist, open("model_persist.pkl", "wb"))

#lixao

# import re
# df_positivoTratado = pd.DataFrame()
# df_negativoTratado = pd.DataFrame()
# # print(df_negativo.text[1])
# for i in range(10):    
#   textoP = re.sub(r'[-./?!,":;()\'@]',' ',df_positivo.text[i])
#   df_positivoTratado.at[i,'text'] = textoP
# print("parou o positivo")
# for i in range(10):    
#   textoN = re.sub(r'[-./?!,":;()\'@]',' ',df_negativo.text[i])
#   df_negativoTratado.at[i,'text'] = textoN
# print("parou o negativo")
# df_positivoTratado['label'] = 1
# df_negativoTratado['label'] = 0

# print(df_positivoTratado,"\n\n")
# print(df_positivoTratado)
  
# for i in df_negativo:
#   df_negativoTratado.append(re.sub(r'[-./?!,":;()\'@]',' ',df_negativo[text][i]))
# #print (df_positivoTratado)
# df_positivoTratado.head()

# from sklearn.svm import SVC
# clf =SVC(gamma='scale')#kernel="rbf",C=0.025
# clf.fit(doc_array,y_train)

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='sgd', alpha=1e-5,
#                      hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(doc_array,y_train)

#Implementa e treina
# naive_bayes = MultinomialNB()
# naive_bayes.fit(doc_array, y_train)