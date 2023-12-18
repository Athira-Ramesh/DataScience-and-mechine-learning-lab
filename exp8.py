import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
x_value=np.array([64,75,68,73,78,82,76,85,71,88]).reshape(-1,1)
y_value=np.array([17,27,15,24,39,44,30,48,19,47])
lr=LinearRegression()
lr.fit(x_value,y_value)
slope=lr.coef_[0]
intercept=lr.intercept_
print("slope",slope)
print("intercept",intercept)