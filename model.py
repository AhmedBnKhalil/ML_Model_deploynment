import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart.csv')

# print first 5 rows of the dataset
print(heart_data.head())

# split the dataset
X = heart_data.iloc[:, :-1].values
y = heart_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print x   and y   values
print(X.shape, X_train.shape, X_test.shape)

# create a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# print the accuracy of the model
print(model.score(X_test, y_test))

input_data = (37,1,2,130,250,0,1,187,0,3.5,0,0,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

# save the  trained model as pkl
pickle.dump(model,open("heart_model.pkl",'wb'))
