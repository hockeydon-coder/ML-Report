

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

df=pd.read_csv('train.csv')

df.head()


plt.scatter(df.clock_speed, df.price_range)
plt.title('clock speed v price_range')
plt.show()

plt.scatter(df.battery_power, df.price_range)
plt.title('battery_power v price_range')
plt.show()

print(df)


plt.scatter(df.blue, df.price_range)
plt.title('Blue v price_range')
plt.show()

plt.scatter(df.ram, df.price_range)
plt.title('battery_power v price_range')
plt.show()

plt.scatter(df.price_range, df.battery_power)
plt.show()

plt.scatter(df.wifi, df.price_range,)
plt.show()

plt.scatter(pop.battery_power, pop.ram)
plt.show()

plt.scatter(df.ram, df.price_range)
plt.title('ram v price_range')
plt.show()

df.price_range.plot(kind='hist',color='blue',edgecolor='black',figsize=(10,7))
plt.title('Price', size=24)
plt.xlabel('Values', size=18)
plt.ylabel('Frequency', size=18)


df.battery_power.plot(kind='hist',color='blue',edgecolor='black',figsize=(10,7))
plt.title('Price', size=24)
plt.xlabel('Values', size=18)
plt.ylabel('Frequency', size=18)



df.ram.plot(kind='hist',color='blue',edgecolor='black',figsize=(10,7))
plt.title('Price', size=24)
plt.xlabel('Values', size=18)
plt.ylabel('Frequency', size=18)


ML= df.plot(kind='scatter', x='ram',y='price_range', color='blue',alpha=0.5, figsize=(10,7))
FL=df.plot(kind='scatter', x='battery_power',y='price_range', color='magenta',alpha=0.5, figsize=(10,7),ax=ML)
plt.legend(labels=['ram','battery power'])
plt.title('Relationship between ram and battery power with the price', size=18)
plt.xlabel('Price', size=14)
plt.ylabel('Frequency', size=14);


 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

df=pd.read_csv('train.csv')

df.head()

	

ax = sns.heatmap(df)

# Here I transform the price_range variable as required

df_=df['price_range'] = df['price_range'].replace(['1','2','3'],'1')

df.loc[df.price_range == "2", "price_range"] = "1"


df.replace(to_replace =["2", "3"],  value ="1") 


#Here I remove the target variable

df.drop(columns=['price_range'])


# I now perform the Logistic Regression classifiaction
# Firstly, I divide the columns into two types of variables; dependent
#(target) and independent (feature) variables.

from sklearn.model_selection import train_test_split
feature_df = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
X=df[feature_df]
Y=df.battery_power

# Here I split the dataset. Then I print out the differnt training and test
# shapes

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

print(X_train.shape)


print(Y_train.shape)


print(X_test.shape)
print(Y_test.shape)

# Here I import the Logistic regression module and create a Logistic-
# Regression classifier objet. The I fit the model on the train set and -
#then I perform the prediction on the test ste.

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,Y_train)

#
Y_pred=logreg.predict(X_test)

# Here I train the logistic regression model with the training set. Then I
# print it out

logisticMod=LogisticRegression()
logisticMod.fit(X_train,Y_train)


LogisticRegression()

print(logisticMod.predict(X_test[0:10]))


print(logisticMod.predict(X_test[0:10]))


Y_pred=logisticMod.predict(X_test)

score = logisticMod.score(X_test,Y_test)
print(score)

# Confusion matrix construction

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_pred)
confusion_matrix(Y_test, Y_pred)



from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
cnf_matrix


# Below is the Logistic Regression for the test dataset. I repeat the above 
# procedure

dft=pd.read_csv('test.csv')

dft.head()

from sklearn.model_selection import train_test_split
feature_dft = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
X=dft[feature_dft]
Y=dft.battery_power

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

print(X_train.shape)


print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

from sklearn.linear_model import LogisticRegression
logisticMod=LogisticRegression()
logisticMod.fit(X_train,Y_train)

LogisticRegression()

print(logisticMod.predict(X_test[0:10]))


print(logisticMod.predict(X_test[0:10]))


Y_pred=logisticMod.predict(X_test)

score = logisticMod.score(X_test,Y_test)
print(score)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_pred)
confusion_matrix(Y_test, Y_pred)


 

 



