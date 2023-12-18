from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
breast_cancer=load_breast_cancer()
x=breast_cancer.data
y=breast_cancer.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
bc=GaussianNB()
bc.fit(x_train,y_train)
v=bc.predict(x_test)
print(v)
result = accuracy_score(y_test, v)
print("result", result)
report = classification_report(y_test,v,target_names=breast_cancer.target_names)
print("classification_report\n",report)