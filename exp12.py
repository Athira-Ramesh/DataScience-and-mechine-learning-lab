from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score,classification_report
bc=load_breast_cancer()
x=bc.data
y=bc.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=39)
dt=DecisionTreeClassifier(max_depth=3)
dt.fit(x_train,y_train)
v=dt.predict(x_test)
result=accuracy_score(y_test,v)
print("result",result)
report = classification_report(y_test,v,target_names=bc.target_names)
print("classification_report\n",report)
plt.figure(figsize=(20, 18))
features = bc.feature_names
classes =bc.target_names
plot_tree(dt, feature_names=features,class_names=classes,rounded=True,filled=True,
proportion=True)
plt.title("Decision tree breast_cancer")
plt.show()