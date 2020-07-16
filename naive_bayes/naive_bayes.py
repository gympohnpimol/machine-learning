import pandas as pd 
import numpy as np
import time
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('/Users/gympohnpimol/Desktop/machine-learning/naive_bayes/naivebayes.csv')

df['Flu'] = np.where(df['Flu'] == 'Y', 1, 0)
df['Chills'] = np.where(df['Chills'] == 'Y', 1, 0)
df['runny nose'] = np.where(df['runny nose'] == 'Y', 1 ,0)
df['head ache'] = np.where(df['head ache'] == 'No', 0,
                    np.where(df['head ache'] == 'Mild', 1, 2))
df['fever'] = np.where(df['fever'] == 'Y', 1, 0)
# print(df)

x_train, x_test = train_test_split(df, test_size = 0.5, random_state = int(time.time()))

clf = GaussianNB()
features = ['Chills', 'runny nose', 'head ache', 'fever']
clf.fit(x_train[features].values, x_train['Flu'])
y_pred = clf.predict(x_test[features])

# print('Number of mislabled points out of a total {} points: {}, performance {:05.2f}%'.format(
#     x_test.shape[0],
#     (x_test['Flu'] != y_pred).sum(),
#     100*(1-(x_test['Flu'] != y_pred).sum()/x_test.shape[0])
# ))

""" result :
    Number of mislabled points out of a total 4 points: 2, performance 50.00% """

mean_flu = np.mean(x_train['Flu'])
mean_not_flu = 1-mean_flu

# print('Flu prob = {:03.2f}%, Not flu prob = {:03.2f}%'
#     .format(100*mean_flu,100*mean_not_flu))

""" result : 
    Flu prob = 75.00%, Not flu prob = 25.00% """

