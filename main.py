from sklearn import neural_network, datasets
import pandas as pd
import numpy as np

# Pentru incarcarea datelor, folosim pandas si numpy.
# Preluam datele si etichetele din fisiere de tip csv cu functia read_csv
# si convertim DataFrame-urile in array-uri, folosind functia to_numpy.

citire1 = pd.read_csv('DATA.csv')
date = citire1.to_numpy()
citire2 = pd.read_csv('ETICHETE.csv')
etichete = citire2.to_numpy()
# Avem: 2310 instante, 19 atribute, 7 clase


# Impartim datele si etichetele in date / etichete de train si date / etichete de test
# astfel: 75% train si 25% test

date_train = date[:1732, :]
etichete_train = etichete[:1732]

date_test = date[1732:, :]
etichete_test = etichete[1732:]

# Creem reteaua MLP.

clf = neural_network.MLPClassifier(hidden_layer_sizes=(150, 150), learning_rate_init=0.01)

# Antrenam reteaua.

clf.fit(date_train, etichete_train)

# Obtinem predictiile.

predictii = clf.predict(date_test)

predictii_reusite = 0

for i in range(len(etichete_test)):
    if etichete_test[i] == predictii[i]:
        predictii_reusite += 1

# Obtinem acuratetea ca metrica de masurare a performantei.

acuratete = predictii_reusite / len(etichete_test)

print('Acuratete = ' + str(acuratete))

