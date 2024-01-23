import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

#k-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')#cmap为颜色映射
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')#绘制了聚类中心（centroids）
plt.title('K-Means Clustering')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.legend()
plt.show()

#基于密度的聚类
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.4, min_samples=5)#eps为半径
dbscan.fit(X)

y_dbscan = dbscan.labels_

plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis')
plt.title('Density-Based Clustering')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.show()

#基于层次的聚类
from sklearn.cluster import AgglomerativeClustering

agg_clustering = AgglomerativeClustering(n_clusters=3)
y_agg = agg_clustering.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_agg, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.show()

#实际分类
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Iris Dataset - Actual Classifications')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()