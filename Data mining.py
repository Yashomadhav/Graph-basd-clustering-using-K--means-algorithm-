# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 23:37:12 2023

@author: Yashomadhav Mudgal
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import pandas as pd 
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv") # reading data set 
print(df)
print(df.shape)

# Performing Pre data processing (Principal Component Analysis)

features = ['Normalized 10','Normalized 20','Normalized 30','Normalized 40']


# Separating out the features
x = df.loc[:,features].values

# Separating out the target
y = df.loc[:,['Product_Code']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca1 = PCA(n_components=4)

X_std = pca1.fit_transform(x)

feature = X_std.T
cov_matrix = np.cov(feature) # calculating covariance matrix
print(cov_matrix)

# Calculating eigen values and eigen vectors of covariance matrix
eignvalues, eignvectors = np.linalg.eig(cov_matrix) 
print(eignvalues)
print(eignvectors)

# Finding variance
explained_variances = []
for i in range(len(eignvalues)):
    explained_variances.append(eignvalues[i] / np.sum(eignvalues))
 
print(np.sum(explained_variances), '\n', explained_variances)

# Scree Plot 
percent_variance = np.round(pca1.explained_variance_ratio_* 100, decimals =2)
columns = ['PC1', 'PC2', 'PC3', 'PC4']
plt.figure()
plt.bar(x= range(1,5), height=percent_variance, tick_label=columns)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.show()


# Variation per principal component 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['Product_Code']]], axis = 1)

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# 2 Component PCA (Scatter plot)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['P200', 'P400', 'P600','P800'] # Taking 4 product codes
colors = ['r', 'g', 'b', 'c']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Product_Code'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()



# Graph based clustering using K- means 
def drawclusters(ax):
    for i in range(ncluster):
        points = X[y == i]
        ax.scatter(points[:, 0], points[:, 1], s=100, c=col[i], label=f'Cluster {i + 1}')
        hull = ConvexHull(points)
        vert = np.append(hull.vertices, hull.vertices[0])  # Close the polygon by appending the first point at the end
        ax.plot(points[vert, 0], points[vert, 1], '--', c=col[i]) # Scatter plot after clustering 
        ax.fill(points[vert, 0], points[vert, 1], c=col[i], alpha=0.2)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids', marker='x') 


X = df.loc[:,['W19', 'W20']].values # Taking data for 2 weeks 


col = ['blue', 'green']
ncluster = 2
kmeans = KMeans(n_clusters=ncluster, max_iter=500).fit(X)
y = kmeans.labels_
centroids = kmeans.cluster_centers_
fig, ax = plt.subplots(1, figsize=(7, 5))
drawclusters(ax)
ax.legend()
plt.tight_layout()
plt.show()


