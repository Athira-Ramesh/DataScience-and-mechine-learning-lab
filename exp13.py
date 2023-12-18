from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = load_iris()
x = iris.data
kmeans = KMeans(n_clusters=3,random_state=39)
kmeans.fit(x)
cluster_label = kmeans.labels_
print("cluster labels\n",cluster_label)
centroid = kmeans.cluster_centers_
print("centroids\n",centroid)
plt.scatter(x[:,0],x[:,1],c = cluster_label,cmap = 'viridis',marker = 'o',edgecolors='black')
plt.scatter(centroid[:,0],centroid[:,1],marker = '*',s=200,c='red',label='centroid')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Kmeans cluster')
plt.legend()
plt.show()