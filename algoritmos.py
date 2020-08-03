import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier # neural network
from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score
from urllib.request import Request,urlopen #para abrir la dirección
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.figure import Figure
from time import time #importamos la función time para capturar tiempos
from io import StringIO
import matplotlib.pyplot as plt

############################# DATASET IRIS#######################################
def cargar_iris():
    iris_obj = load_iris()
    iris = DataFrame(iris_obj.data, columns=iris_obj.feature_names,index=pd.Index([i for i in range(iris_obj.data.shape[0])])).join(DataFrame(iris_obj.target, columns=pd.Index(["species"]), index=pd.Index([i for i in range(iris_obj.target.shape[0])])))
    iris2 = DataFrame(iris_obj.data, columns=iris_obj.feature_names,index=pd.Index([i for i in range(iris_obj.data.shape[0])])).join(DataFrame(iris_obj.target, columns=pd.Index(["species"]), index=pd.Index([i for i in range(iris_obj.target.shape[0])])))
    iris.species.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)
    return iris, iris2
iris, iris2 =cargar_iris()

x = np.array(iris.drop(['species'], 1))
y = np.array(iris['species'])

############################# DATASET EXTERNO ###################################
def cargar_externo():
    url = Request("https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data", headers={"User-Agent": "Chrome/18.0.1025.133"})
    pag = urlopen(url)
    archivo = pag.read().decode('ISO-8859-1')#UTF-8
    colnames = ['X1', 'X2', 'X3', 'Y']
    dataset = pd.read_csv(StringIO(archivo), names=colnames, header=None)
    return dataset
dataset = cargar_externo()

x2=np.array(dataset.drop(['Y'],1)) #variables independientes
y2=np.array(dataset['Y']) #Variable dependiente

######################CLASIFICACION DE VARIABLES#########################
def resultado_variables():
    x_train, x_test, y_train, y_test=var_train_test(x,y)
    x2_train, x2_test, y2_train, y2_test=var_train_test(x2,y2)
    return x_train, x_test, y_train, y_test,x2_train, x2_test, y2_train, y2_test

###################### metricas de clasificacion###############################
def var_train_test(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=12)
    return(x_train, x_test, y_train, y_test)

def fun_accuracy(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    accuracy=round(accuracy,2)
    return accuracy

def fun_recall(y_true, y_pred):
    recall=recall_score(y_true, y_pred,average=None)
    recall=np.round(recall,2)
    return recall

def fun_precision(y_true, y_pred):
    precision=precision_score(y_true, y_pred, average=None)
    precision = np.round(precision, 2)
    return precision

def fun_f1(y_true, y_pred):
    f1=f1_score(y_true, y_pred, average=None)
    f1 = np.round(f1, 2)
    return f1
###############################################################################
def dispersion_externo():
    data = cargar_externo()
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot()
    data[data.Y == 1].plot(kind='scatter', x='X1', y='X3', color='blue', label='patient survived 5 years', ax=ax)
    data[data.Y == 2].plot(kind='scatter', x='X1', y='X3', color='green', label='patient died within 5 year', ax=ax)
    ax.set_title('Survival Dataset')
    return fig

def dispersion_iris():
    d,data = cargar_iris()
    fig=Figure(figsize=(8,4))
    ax=fig.add_subplot()
    ancho = data[data.species == 0]

    data[data.species == 0].plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)', color='blue', label='Setosa', ax=ax)
    data[data.species == 1].plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)', color='green', label='Versicolor', ax=ax)
    data[data.species == 2].plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)', color='red', label='Virginica', ax=ax)
    ax.set_title('Iris Dataset')
    return fig

def regresion_logistica(x_train,y_train,x_test,y_test):
    tiempo_inicial = time()
    algoritmo = LogisticRegression()
    algoritmo.fit(x_train, y_train)
    Y_pred = algoritmo.predict(x_test)
    accuracy=fun_accuracy(y_test, Y_pred)
    recall=fun_recall(y_test,Y_pred)
    pres=fun_precision(y_test,Y_pred)
    f1= fun_f1(y_test,Y_pred)
    tiempo_final = time()
    tiempo_ejecucion = round((tiempo_final - tiempo_inicial),3)
    return Y_pred,accuracy,recall,pres,f1,tiempo_ejecucion

def m_soporte_vectorial(x_train,y_train,x_test,y_test):
    tiempo_inicial = time()
    algoritmo = SVC()
    algoritmo.fit(x_train, y_train)
    Y_pred = algoritmo.predict(x_test)
    accuracy=fun_accuracy(y_test, Y_pred)
    recall=fun_recall(y_test,Y_pred)
    pres=fun_precision(y_test,Y_pred)
    f1= fun_f1(y_test,Y_pred)
    tiempo_final = time()
    tiempo_ejecucion = round((tiempo_final - tiempo_inicial),3)
    return Y_pred,accuracy,recall,pres,f1,tiempo_ejecucion

def red_neuronal(x_train,y_train,x_test,y_test):
    tiempo_inicial = time()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)
    clf.fit(x_train,y_train)
    prediction = clf.predict(x_test)
    accuracy=fun_accuracy(y_test, prediction)
    recall=fun_recall(y_test,prediction)
    pres=fun_precision(y_test,prediction)
    f1= fun_f1(y_test,prediction)
    tiempo_final = time()
    tiempo_ejecucion = round((tiempo_final - tiempo_inicial),3)
    return prediction,accuracy,recall,pres,f1,tiempo_ejecucion

def knn(x_train,y_train,x_test,y_test):
    tiempo_inicial = time()
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy=fun_accuracy(y_test, y_pred)
    recall=fun_recall(y_test,y_pred)
    pres=fun_precision(y_test,y_pred)
    f1= fun_f1(y_test,y_pred)
    tiempo_final = time()
    tiempo_ejecucion = round((tiempo_final - tiempo_inicial),3)
    return y_pred,accuracy,recall,pres,f1,tiempo_ejecucion

def random(x_train,y_train,x_test,y_test):
    tiempo_inicial = time()
    clf=RandomForestClassifier(n_estimators=150)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accuracy=fun_accuracy(y_test, y_pred)
    recall=fun_recall(y_test,y_pred)
    pres=fun_precision(y_test,y_pred)
    f1= fun_f1(y_test,y_pred)
    tiempo_final = time()
    tiempo_ejecucion = round((tiempo_final - tiempo_inicial),3)
    return y_pred,accuracy,recall,pres,f1,tiempo_ejecucion

def nbayes(x_train,y_train,x_test,y_test):
    tiempo_inicial = time()
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy=fun_accuracy(y_test, y_pred)
    recall=fun_recall(y_test,y_pred)
    pres=fun_precision(y_test,y_pred)
    f1= fun_f1(y_test,y_pred)
    tiempo_final = time()
    tiempo_ejecucion = round((tiempo_final - tiempo_inicial), 3)
    return y_pred,accuracy,recall,pres,f1,tiempo_ejecucion

def metricas_reg_log(accuracy_log,recall_log,pres_log,f1_log):
    metricas_rl = []
    metricas_rl.append(accuracy_log)
    metricas_rl.append(recall_log)
    metricas_rl.append(pres_log)
    metricas_rl.append(f1_log)
    return metricas_rl

def tiempos(tiempo_knn,tiempo_ran,tiempo_nb,tiempo_reg,tiempo_vec,tiempo_neu):
    tiempos = []
    tiempos.append(tiempo_knn)
    tiempos.append(tiempo_ran)
    tiempos.append(tiempo_nb)
    tiempos.append(tiempo_reg)
    tiempos.append(tiempo_vec)
    tiempos.append(tiempo_neu)
    return tiempos

def pred_real_iris(y_test,pred_reg_log_iris,pred_soporte_vectorial_iris,pred_red_neurona_iris,pred_knn_iris,pred_ran_iris,pred_nb1):
    reg_log_iris = []
    s_vectorial=[]
    red=[]
    knn=[]
    ran=[]
    nb=[]
    for e in range(len(y_test)):
        reg_log_iris.append([y_test[e], pred_reg_log_iris[e]])
        s_vectorial.append([y_test[e],pred_soporte_vectorial_iris[e]])
        red.append([y_test[e], pred_red_neurona_iris[e]])
        knn.append([y_test[e],pred_knn_iris[e]])
        ran.append([y_test[e],pred_ran_iris[e]])
        nb.append([y_test[e],pred_nb1[e]])
    return  reg_log_iris,s_vectorial,red,knn,ran,nb
