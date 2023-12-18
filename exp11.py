from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score,classification_report
from matplotlib import pyplot as plt
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=39)
dt=DecisionTreeClassifier(max_depth=2)
dt.fit(x_train,y_train)
v=dt.predict(x_test)
print(v)
result=accuracy_score(y_test,v)
print("result",result)
report = classification_report(y_test,v,target_names=iris.target_names)
print("classification_report\n",report)
plt.figure(figsize=(10, 8))
features = iris.feature_names
classes = iris.target_names
plot_tree(dt, feature_names=features,class_names=classes,rounded=True,
filled=True,proportion=True)
plt.title("Decision tree")
plt.show()