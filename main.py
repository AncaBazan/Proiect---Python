import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from functii import *
from sklearn.naive_bayes import GaussianNB
import scipy.stats as sts

tabel_invatare_testare = pd.read_csv("cancer_set1.csv", index_col=0)
variabile = list(tabel_invatare_testare)
predictori = variabile[:-1]
tinta = variabile[-1]

# Divizare in set de invatare si set de testare
x_train, x_test, y_train, y_test = train_test_split(tabel_invatare_testare[predictori], tabel_invatare_testare[tinta],
                                                    test_size=0.4)

# Construire model liniar
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train, y_train)

clase = model_lda.classes_
q = len(clase)

# Calcul putere discriminare predictori
n = len(x_train)
g = model_lda.means_ - np.mean(x_train.values, axis=0)
dg = np.diag(model_lda.priors_)
ssb = n * g.T @ dg @ g
sst = n * np.cov(x_train.values, rowvar=False, bias=True)
ssw = sst - ssb
r = (n - q) / (q - 1)
f = r * np.diag(ssb) / np.diag(ssw)
p_value = 1 - sts.f.cdf(f, q - 1, n - q)
tabel_predictori = pd.DataFrame(
    data={
        "Putere discriminare":f,
        "p_values":p_value
    }, index=predictori
)
tabel_predictori.to_csv("Predictori.csv")

# Preluare scoruri discriminante
z = model_lda.transform(x_test)
m = q - 1  # Numar functii discriminante
for i in range(m):
    # plot_distributie(z,y_train,i)
    plot_distributie(z, y_test, i)
for i in range(m - 1):
    for j in range(i + 1, m):
        # scatterplot_g(z,y_train,clase,i,j)
        scatterplot_g(z, y_test, clase, i, j)
# Testare
predictie_lda_test = model_lda.predict(x_test)
metrici_lda = calcul_metrici(y_test, predictie_lda_test, clase)
metrici_lda[0].to_csv("MatC_LDA.csv")
metrici_lda[1].to_csv("Acuratete_LDA.csv", index=False)
salvare_erori(y_test,predictie_lda_test,tinta,x_test.index,"LDA")
# Aplicare model
x_apply = pd.read_csv("cancer_set2.csv", index_col=0)
predictie_lda = model_lda.predict(x_apply[predictori])
tabel_predictii = pd.DataFrame(
    data={
        "Predictie LDA": predictie_lda
    }, index=x_apply.index
)


tabel_predictii.to_csv("Predictii.csv")

show()
