import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import kdeplot,scatterplot
from sklearn.metrics import confusion_matrix,cohen_kappa_score

def plot_distributie(z,y,k=0):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Distributie in axa discriminanta "+str(k+1),fontsize=16,color="m")
    kdeplot(x=z[:,k],hue=y,fill=True,ax=ax)

def show():
    plt.show()

def scatterplot_g(z,y,clase,k1=0,k2=1):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1,aspect=1)
    ax.set_title("Plot instante in axele discriminante",fontsize=16,color="m")
    ax.set_xlabel("z"+str(k1+1))
    ax.set_ylabel("z"+str(k2+1))
    scatterplot(x=z[:,k1],y=z[:,k2],hue=y,hue_order=clase,ax=ax)

def calcul_metrici(y,y_,clase):
    c = confusion_matrix(y,y_)
    tabel_c = pd.DataFrame(c,clase,clase)
    tabel_c["Acuratete"] = np.round(np.diag(c)*100/np.sum(c,axis=1) ,3)
    acuratete_medie = tabel_c["Acuratete"].mean()
    acuratete_globala = np.round( sum(np.diag(c))*100/len(y),3)
    index_CK = cohen_kappa_score(y,y_)
    acuratete = pd.DataFrame(data={
        "Acuratete globala":[acuratete_globala],
        "Acuratete medie":[acuratete_medie],
        "Index Cohen-Kappa":[index_CK]
    })
    return tabel_c,acuratete

def salvare_erori(y,y_,tinta,nume_instante,model):
    tabel = pd.DataFrame(
        data={
            tinta:y,
            "Predictie":y_
        }, index=nume_instante
    )
    tabel[y!=y_].to_csv("err_"+model+".csv")
