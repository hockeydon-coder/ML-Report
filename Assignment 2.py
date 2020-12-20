

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline 
    from sklearn.cluster import KMeans
    from sklearn import datasets
    
    iris = datasets.load_iris()
  
    df=pd.DataFrame(iris['data'])
    print(df.head())


    iris['target']


    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(12,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

km = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
km.fit(X)


KMeans(n_clusters=3, n_jobs=4, random_state=21)

centers = km.cluster_centers_
print(centers)

new_labels = km.labels_
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)


plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Sepa1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)



from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target']

colours = ['red', 'orange', 'blue']
species = ['I. setosa', 'I. versicolor', 'I. virginica']

for i in range(0, 3):    
    species_df = iris_df[iris_df['species'] == i]    
    plt.scatter(        
        species_df['petal length (cm)'],        
        species_df['petal width (cm)'],
        color=colours[i],        
        alpha=0.5,        
        label=species[i]   
    )

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Iris dataset: petal width vs petal length')
plt.legend(loc='best')

plt.show()

from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target']

colours = ['red', 'orange', 'blue']
species = ['I. setosa', 'I. versicolor', 'I. virginica']

for i in range(0, 3):    
    species_df = iris_df[iris_df['species'] == i]    
    plt.scatter(        
        species_df['sepal length (cm)'],        
        species_df['sepal width (cm)'],
        color=colours[i],        
        alpha=0.5,        
        label=species[i]   
    )

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Iris dataset: sepal width vs sepal length')
plt.legend(loc='best')

plt.show()

from sklearn.cluster import KMeans
iris = datasets.load_iris()

df = pd.read_csv('iris.csv')

df.head(10)

x = df.iloc[:,[0,1,2,3]].values

kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans5)

kmeans3.cluster_centers_


kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)
                
kmeans3.cluster_centers_

Error =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,11), Error)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('Error')
plt.show()

plt.scatter(x[:,0], x[:,1], c=y_kmeans3, cmap = 'rainbow')

plt.title('Iris dataset')

Text(0.5, 1.0, 'Iris dataset')

from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print (iris.data)
print (iris.target)


x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

plt.figure(figsize=(12,3))

<Figure size 864x216 with 0 Axes>

<Figure size 864x216 with 0 Axes>

colors = np.array(['red', 'green', 'blue'])

plt.subplot(1, 2, 1)

<matplotlib.axes._subplots.AxesSubplot at 0x20762f13f40>

plt.scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=40)
plt.title('Sepal Length vs Sepal Width')

Text(0.5, 1.0, 'Sepal Length vs Sepal Width')

plt.subplot(1,2,2)
plt.scatter(x['Petal Length'], x['Petal Width'], c= colors[y.Target], s=40)
plt.title('Petal Length vs Petal Width')

Text(0.5, 1.0, 'Petal Length vs Petal Width')

model = KMeans(n_clusters=3)
model.fit(x)

KMeans(n_clusters=3)

print (model.labels_)

plt.figure(figsize=(12,3))

<Figure size 864x216 with 0 Axes>

<Figure size 864x216 with 0 Axes>

colors = np.array(['red', 'green', 'blue'])

predictedY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

plt.subplot(1, 2, 1)
plt.scatter(x['Petal Length'], x['Petal Width'], c=[y['Target']], s=40)
plt.title('Before classification')

Text(0.5, 1.0, 'Before classification')

plt.subplot(1, 2, 2)
plt.scatter(x['Petal Length'], x['Petal Width'], c=[predictedY], s=40)
plt.title("Model's classification")

Text(0.5, 1.0, "Model's classification")

Error =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,11), Error)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('Error')
plt.show()

 

 


