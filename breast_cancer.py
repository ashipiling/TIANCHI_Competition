from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

cancer_data = load_breast_cancer()
#绘图
fig, axes = plt.subplots(15,2,figsize= (10,20))
print(cancer_data.data[cancer_data.target == 0])
fig.tight_layout()


x_train, x_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, random_state=0)
Lr = LogisticRegression().fit(x_train, y_train)
forest = RandomForestClassifier(n_estimators = 100, random_state = 0).fit(x_train, y_train)
print("LogisticRegression on train_data", Lr.score(x_train, y_train))
print("LogisticRegression on test_data", Lr.score(x_test, y_test))
print("RandomForestClassifier on train_data", forest.score(x_train, y_train))
print("RandomForestClassifier on train_data", forest.score(x_test, y_test))