import sys
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris_data = load_iris()
print(iris_data)
print(iris_data.feature_names)
#分割数据
x_train, x_test, y_train, y_test = train_test_split(iris_data['data'], iris_data['target'], random_state=0)
#iris_dataframe = pd.DataFrame(x_train, columns = iris_data.feature_names)
#graph = pd.scatter_matrix(iris_dataframe, c= y_train, figsize =(15,15),marker = 'o',hist_kwds={'bin':20}, s= 60,alpha = .8)

knc = KNeighborsClassifier(n_neighbors = 1)
knc.fit(x_train, y_train)

y_pre = knc.predict(x_test)
print(y_pre)
print(y_test)

