from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
iris=load_iris()
x=digits.data
y=digits.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
v=knn.predict(x_test)
res=accuracy_score(y_test,v)
print("accuracy:",res)