import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,classification_report
import matplotlib.pyplot as plt
data=pd.read_csv('Salary_Data.csv')
x=data['YearsExperience'].values.reshape(-1,1)
y=data['Salary'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
v=lr.predict(x_test)
result = r2_score(y_test,v)
print("result",result)
plt.scatter(x_test,y_test,color='black',label='datapoints')
plt.plot(x_test,v,color='blue', linewidth=3,label='Regression lines')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.legend()
plt.show()