# ML-Report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

cov = pd.read_csv('worldwide-aggregate.csv')

cov.head()

cov.dropna()

sns.regplot(x=cov["Deaths"], y=cov["Confirmed"], line_kws={"color":"r","alpha":0.7,"lw":5})
sns.plt.show()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X= cov.iloc[:,:-1].values
Y=cov.iloc[:,2].values

labelencoder =LabelEncoder()
X[:,0]=labelencoder.fit_transform(X[:,0])

print(X)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
Regressor= LinearRegression()
Regressor.fit(X_train,Y_train)

pred_y = Regressor.predict(X_test)
print(pred_y)

print(Regressor.coef_)

print(Regressor.intercept_)

from sklearn.metrics import r2_score
r2_score(Y_test,pred_y)

plt.scatter(y=Y_test, x=pred_y)
plt.xlabel('Y predictor', size=12)
plt.ylabel('Y test', size=12);

cov.Confirmed.plot(kind='hist',color='blue',edgecolor='black',figsize=(10,7))
plt.title('Distribution of the Cases', size=24)

cov.Deaths.plot(kind='hist',color='blue',edgecolor='black',figsize=(10,7))
plt.title('Distribution of the Deaths', size=24)
