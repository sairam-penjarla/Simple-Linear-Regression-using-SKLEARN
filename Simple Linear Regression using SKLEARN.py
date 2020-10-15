import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('/Users/sai/Desktop/everything/Programming/Multiple-Linear-Regression-master/50_Startups.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]
states = pd.get_dummies(x['State'],drop_first = True)
x=x.drop('State',axis = 1)
x=pd.concat([x,states],axis = 1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=10)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
ypred = model.predict(x_test)
from sklearn.metrics import r2_score
score = (r2_score(y_test,ypred))*100
print('accuracy = ',int(score),"%")

