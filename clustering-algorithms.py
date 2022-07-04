#!/usr/bin/env python
# coding: utf-8

# ### Author ~ Saurabh Kumar

# # **Lets First UnderStand Clustering**

# # **What is Clustering?**

# ### Defination:
# * **Clustering is Unsupervised Machine Learning Techinique (Unsupervised ~ Unlabled data) where we are trying grouping of data points.These groups are called Cluster.Data points that are in the same group should have similar properties and/or features, while data points in different groups should have highly dissimilar properties and/or features.**
# *  **The aim is to segregate groups with similar traits and assign them into clusters.**
# 
# ### Simple Use Case: 
# > Suppose, you are the head of a rental store and wish to understand preferences of your costumers to scale up your business. Is it possible for you to look at details of each costumer and devise a unique business strategy for each one of them? Definitely not. But, what you can do is to cluster all of your costumers into say 10 groups based on their purchasing habits and use a separate strategy for costumers in each of these 10 groups. And this is what we call clustering.
# 

# # **Why we use Clustering? or How its helps in Data Science???**

# * Clustering is used for things like **feature engineering** or **pattern discovery**.
# * We can use clustering analysis to gain some valuable insights from our data by seeing what groups the data points   falls.
# * Statistical Analysis of Data.
# 
# ### Real-time Use-cases:
# * One of the most importat application is related to image processing. detecting distinct kinds of pattern in image data. This can be very effective in biology research, distinguishing objects and identifying patterns.
# 
# * The personal data combined with shopping, location, interest, actions and a infinite number of indicators, can be analysed with this methodology, providing very important information and trends. Examples of this are the market research, marketing strategies, web analytics, and a lot of others.
# 
# * Other types of applications based on clustering algorithms are climatology, robotics, recommender systems, mathematical and statistical analysis, providing a broad spectrum of utilization.
# 
# ### Possible Applications
# 
# * Marketing: Finding groups of customers with similar behavior given a large database of customer data containing their properties and past buying records;
# 
# * Biology: Classification of plants and animals given their features;
# 
# * Libraries: Book ordering;
#  
# * Insurance: Identifying groups of motor insurance policy holders with a high average claim cost; identifying frauds;
#  
# * City-planning: Identifying groups of houses according to their house type, value and geographical location;
#  
# * Earthquake studies: Clustering observed earthquake epicenters to identify dangerous zones;
#  
# * WWW document classification: Clustering weblog data to discover groups of similar access patterns.

# ## **Clustering Visualization**
# ![image.png](attachment:ea25274c-c0b8-4f37-82ec-1103cf0b658f.png))

# ## **Types of Clustering**
# 
# ### Based on the area of overlap
# 1. **Hard Clustering** 
# - Clusters don’t overlap:k-means, k-means++.Each data point either belongs to a cluster completely or not.
# 
# 2. **Soft Clustering** 
# - Clusters overlap:Fuzzy c-means, EM. A data object can exist in more than one cluster with a certain probability or degree of membership.
# or
# - Instead of putting each data point into a separate cluster, a probability or likelihood of that data point to be in those clusters is assigned.
# 

# ## A Comprehensive Survey of Clustering Algorithms.
# ##### Link : https://link.springer.com/article/10.1007/s40745-015-0040-1
# ![image.png](attachment:e93e142e-7a27-4058-980c-ff6a04aa293c.png)

# ### **Various Clustering Algorithms:**
# ![image](https://miro.medium.com/max/875/1*S_Pkf-cK1gUE0OPbDM81ww.png)
# 

# ## **Lets Discuss diffrerent type of method used for clustering**

# ## Clustering Algorithm Based on Distribution
# 
# The basic idea is that the data, generated from the same distribution, belongs to the same cluster if there exists several distributions in the original data.
# The typical algorithms are DBCLASD and GMM. The core idea of DBCLASD, a dynamic incremental algorithm, is that if the distance between a cluster and its nearest data point satisfies the distribution of expected distance which is generated from the existing data points of that cluster, the nearest data point should belong to this cluster. The core idea of GMM is that GMM consists of several Gaussian distributions from which the original data is generated and the data, obeying the same independent Gaussian distribution, is considered to belong to the same cluster.
# 
# #### Analysis
# ![image.png](attachment:848647f1-1146-4097-97b7-85abd4769634.png)

# ### Advantages:
# More realistic to give the probability of belonging, relatively high scalability by changing the distribution, number of clusters and so on, and supported by the well developed statistical science;
# 
# ### Disadvantages: 
# The premise not completely correct, involved in many parameters which have a strong influence on the clustering result and relatively high time complexity.

# ## Clustering Based on Density:
# The basic idea of this kind of clustering algorithms is that the data which is in the region with high density of the data space is considered to belong to the same cluster.
# The typical ones include DBSCAN, OPTICS and Mean-shift.
# * DBSCAN is the most well known density-based clustering algorithm, which is generated from the basic idea of this kind of clustering algorithms directly. 
# * OPTICS is an improvement of DBSCAN and it overcomes the shortcoming of DBSCAN that being sensitive to two parameters, the radius of the neighborhood and the minimum number of points in a neighborhood.
# * In the process of Mean-shift, the mean of offset of current data point is calculated at first, the next data point is figured out based on the current data point and the offset then, and last, the iteration will be continued until some criteria are met.
# #### Analysis :
# ![DBSCAN](https://miro.medium.com/max/844/1*tc8UF-h0nQqUfLC8-0uInQ.gif)
# * ### Mean-shift
# ![Mean-shift](https://miro.medium.com/max/405/1*bkFlVrrm4HACGfUzeBnErw.gif)

# ### Advantages: 
# Clustering in high efficiency and suitable for data with arbitrary shape.
# 
# ### Disadvantages:
# Resulting in a clustering result with low quality when the density of data space isn’t even, a memory with big size needed when the data volume is big, and the clustering result highly sensitive to the parameters.
# 
# ### Analysis
# ![image.png](attachment:71f3a206-b36d-4cae-a38c-2a33f7ff4b5a.png)

# ## Clustering Algorithm Based on Hierarchy
# Hierarchical clustering, as the name suggests is an algorithm that builds hierarchy of clusters. This algorithm starts with all the data points assigned to a cluster of their own. Then two nearest clusters are merged into the same cluster. In the end, this algorithm terminates when there is only a single cluster left.
# 
# ![image.png](attachment:d9fba235-6abe-4f23-909d-d1d3f46c69e3.png)
# 
# ### Advantages: 
# Suitable for the data set with arbitrary shape and attribute of arbitrary type, the hierarchical relationship among clusters easily detected, and relatively high scalability in general;
# 
# ### Disadvantages: 
# Relatively high in time complexity in general, the number of clusters needed to be preset.
# 
# ### Analysis
# ![image.png](attachment:5b7cabc3-0e79-4578-a523-84b4525497ff.png)

# ## Clustering Algorithm Based on Fuzzy Theory
# 
# The basic idea of this kind of clustering algorithms is that the discrete value of belonging label, {0, 1}, is changed into the continuous interval [0,1], in order to describe the belonging relationship among objects more reasonably. Typical algorithms of this kind of clustering include FCM, FCS and MM. The core idea of FCM is to get membership of each data point to every cluster by optimizing the object function. FCS, different from the traditional fuzzy clustering algorithms, takes the multidimensional hypersphere as the prototype of each cluster, so as to cluster with the distance function based on the hypersphere. MM, based on the Mountain Function, is used to find the center of cluster
# 
# ### Advantages: 
# More realistic to give the probability of belonging, relatively high accuracy of clustering;
# 
# ### Disadvantages: 
# Relatively low scalability in general, easily drawn into local optimal, the clustering result sensitive to the initial parameter values, and the number of clusters needed to be preset.
# 
# ![image.png](attachment:9a11da8d-0114-465d-9385-035da0b2581b.png)
# 
# ![image.png](attachment:220eb66c-1104-4c97-83ec-4b6749d99e75.png)

# ## **Now Lets Jumps to Implementation Section**
# ### Famaous Clustering Algorithms:

# ### 1. [K-means clustering algorithm](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)
# ### 2. [DBSCAN clustering algorithm](https://www.analyticsvidhya.com/blog/2021/06/understand-the-dbscan-clustering-algorithm/)
# ### 3. [Gaussian Mixture Model algorithm](https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/)
# ### 4. [BIRCH algorithm]()
# ### 5. Affinity Propagation clustering algorithm
# ### 6. Mean-Shift clustering algorithm
# ### 7. OPTICS algorithm
# ### 8. Agglomerative Hierarchy clustering algorithm

# ## K-means clustering algorithm
# 
# K-means clustering is the most commonly used clustering algorithm. It's a centroid-based algorithm and the simplest unsupervised learning algorithm.
# 
# This algorithm tries to minimize the variance of data points within a cluster. It's also how most people are introduced to unsupervised machine learning.
# 
# K-means is best used on smaller data sets because it iterates over all of the data points. That means it'll take more time to classify data points if there are a large amount of them in the data set.
# 
# ![image.png](attachment:ab85f069-08c7-433e-b829-a52a0693de29.png)

# ### General Implementation

# In[ ]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
kmeans_model = KMeans(n_clusters=2)

# assign each data point to a cluster
kmeans_result = kmeans_model.fit_predict(training_data)

# get all of the unique clusters
kmeans_clusters = unique(kmeans_result)

# plot the kmeans clusters
for kmeans_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = where(kmeans_result == kmeans_cluster)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the kmeans plot
pyplot.show()


# ### Lets Implement K-means Clustering as per our dataset:

# In[1]:


#importing Library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import  LabelEncoder, RobustScaler , MinMaxScaler ,StandardScaler


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation ,Dropout
from tensorflow.keras import layers

import plotly.express as px
from scipy import signal
import scipy
#to supress warning
import warnings
warnings.filterwarnings('ignore')


#to make shell more intractive
from IPython.display import display

# setting up the chart size and background
plt.rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')
sns.set_style("darkgrid")


# In[2]:


os.listdir()


# In[3]:


pwd


# In[4]:


path='E:\\DataScience\\Clustering-Algorithms\\data'


# In[5]:


os.listdir(path)


# In[6]:


#import dataset
df =pd.read_csv(path+"\\data.csv")
sample =pd.read_csv(path+"\\sample_submission.csv")


# In[7]:


df.head(10).style.background_gradient(cmap = 'rocket_r')


# In[8]:


df.shape


# In[9]:


df.drop("id", axis=1, inplace=True)


# In[10]:


df.describe().T.style.background_gradient(cmap = 'RdYlGn')


# In[11]:


df.info()


# In[12]:


# Correlation matrix
df.corr().style.background_gradient(cmap = 'rocket_r')


# In[13]:


# Correlation matrix
corrMatrix =df.corr(method='pearson', min_periods=1)
corrMatrix.style.background_gradient(cmap = 'rocket_r')


# In[14]:


#Classifying Catogorical and Numerical
cat_columns = [x for (x, y) in df.dtypes.items() if y == "int64"]
num_columns = [x for (x, y) in df.dtypes.items() if y == "float64"]

num_data = df[num_columns]
cat_data = df[cat_columns]
print('Categorical columns -> ', cat_columns)
print('\nNumerical columns -> ', num_columns)


# In[15]:


def mix_plot(feature):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    feature.plot(kind = 'hist')
    plt.title(f'{feature.name} histogram plot')
    #mean = feature.describe().mean()
    plt.subplot(1, 3, 2)
    mu, sigma = scipy.stats.norm.fit(feature)
    sns.distplot(feature) 
    #plt.legend({'--': mu, 'sigma': sigma})
    plt.axvline(mu, linestyle = '--', color = 'green', )
    plt.axvline(sigma, linestyle = '--', color = 'red')
    plt.title(f'{feature.name} distribution plot')
    plt.subplot(1, 3, 3)
    sns.boxplot(feature)
    plt.title(f'{feature.name} box plot')
    plt.show()


# In[16]:


for i in cat_data.columns:
    mix_plot(cat_data[i])


# In[17]:


import statsmodels.api as sm
fig, axs = plt.subplots(7, 2,
                        figsize=(13, 26))
for i, col in enumerate(cat_columns):
    sns.histplot(cat_data[col], ax=axs[i, 0],color='#800080')
    sm.qqplot(cat_data[col], line="s", ax=axs[i, 1], color='#800080')


# In[18]:


from sklearn.cluster import KMeans
X = df.values
# 20 clusters
n_clusters = 20
# Train K-Means.
kmeans = KMeans(n_clusters = n_clusters, n_init = 20).fit(X)


# In[19]:


kmeans.labels_


# In[20]:


target = kmeans.labels_


# In[21]:


pd.DataFrame(target)[0].value_counts()


# In[22]:


from sklearn.preprocessing import StandardScaler,RobustScaler
sc = StandardScaler()
rb = RobustScaler()

##Scaling of Data
rb.fit(df)
df = rb.transform(df)
df.shape, target.shape


# In[23]:


#creating_neural_network
model =Sequential()
#1st_hidden_layers
model.add(Dense(units=256, activation='relu', use_bias=True, kernel_initializer='he_uniform',input_shape=(29,)))
#2nd_hidden_layers
model.add(Dense(units=128,activation='relu',kernel_initializer='he_uniform'))
model.add(layers.Dropout(0.25))
#3rd_hidden_layers
model.add(Dense(units=128,activation='relu',kernel_initializer='he_uniform'))
model.add(layers.Dropout(0.25))
#4nd_hidden_layers
model.add(Dense(units=64,activation='relu',kernel_initializer='he_uniform'))
#5th_hidden_layers
model.add(Dense(units=32,activation='relu',kernel_initializer='he_uniform'))
#6th_hidden_layer
model.add(Dense(units=20,activation='softmax',kernel_initializer='glorot_uniform'))
#model complile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy',])


# In[24]:


model.summary()


# In[25]:


model.layers


# In[26]:


save_best = tf.keras.callbacks.ModelCheckpoint("Model.h5", verbose=1, monitor='val_accuracy', save_best_only=True)


# In[27]:


#model traning
model.fit(df, target, validation_split=0.2, epochs=30, batch_size=256, shuffle=True, callbacks=[save_best])


# In[28]:


history =model.fit(df, target, validation_split=0.2, epochs=30, batch_size=256, shuffle=True, callbacks=[save_best])


# In[29]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(30)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[30]:


#Model Prediction
pred = model.predict(df, verbose=1)
pred = np.argmax(pred, axis=1)


# In[31]:


#Predicting of Clusters
pd.DataFrame(pred)[0].value_counts()


# In[32]:


sample['Predicted'] = kmeans.labels_
sample.to_csv('submission.csv', index=False)


# In[33]:


sample


#  ### Working in other clustering algorithms....

# In[ ]:




