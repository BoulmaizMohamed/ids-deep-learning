

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ctgan import CTGANSynthesizer
from numpy import savetxt

data = pd.read_csv('/content/drive/MyDrive/memoire/2222.csv')
encoder = LabelEncoder()

data['proto'] = encoder.fit_transform(data['proto'])
data['service'] = encoder.fit_transform(data['service'])
data['state'] = encoder.fit_transform(data['state'])
data['attack_cat'] = encoder.fit_transform(data['attack_cat'])

print(data.columns)

categorical_features = ['proto','service','state','attack_cat']


ctgan = CTGANSynthesizer(verbose=True)
ctgan.fit(data,categorical_features,epochs = 20)

samples = ctgan.sample(10000)

print(samples.head())

savetxt('/content/drive/MyDrive/memoire/Generated.csv', samples, delimiter=',')
