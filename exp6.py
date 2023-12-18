from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
naive_bayes = GaussianNB()
naive_bayes.fit(x_train,y_train)
v=naive_bayes.predict(x_test)
print(v)
result=accuracy_score(y_test,v)
print("result",result)