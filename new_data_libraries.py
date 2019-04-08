import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
data = pd.read_csv("Admission_Predict.csv",sep=',')
# data.plot(x='GRE Score',y='Chance of Admit ',style='o')
# plt.show()
# Training set!
X_train = data.iloc[1:350,1:-1].values
Y_train = data.iloc[1:350,8].values
# Test set!
X_test = data.iloc[351:400,1:-1].values
Y_test = data.iloc[351:400,8].values
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train,Y_train)
y_pred = linear_regression.predict(X_test)
results = pd.DataFrame({'Actual':Y_test,'Model prediction':y_pred})
print(results)

# https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
# HELPED ME TO PREAPER AND TRANSFORM ALL DATA TO AN SUITED SHAPE!!!!

