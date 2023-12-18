from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
bc = load_breast_cancer()
x = bc.data
kmeans = KMeans(n_clusters=3,random_state=40)
kmeans.fit(x)
cluster = kmeans.labels_
print("cluster labels\n",cluster)
centroid = kmeans.cluster_centers_
print("centroid\n",centroid)
plt.scatter(x[:,0],x[:,1],c=cluster,cmap='viridis',marker = 'o',edgecolor = 'red')
plt.scatter(centroid[:,0],centroid[:,1],s=100,c='blue',label='centroid')
plt.xlabel(bc.feature_names[0])
plt.ylabel(bc.feature_names[1])
plt.legend()
plt.show()