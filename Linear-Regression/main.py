import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("Linear-Regression/student-dataset/student-mat.csv", sep = ";") #load dataset
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] #trim dataset to selected attributes

predict = "G3" #Set the label(the thing to predict using attributes)

x = np.array(data.drop([predict], 1)) #making an array of the attributes (exluding the label)
y = np.array(data[predict]) #making an array of the label's real values

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
#splitting x and y to use one portion for training and the other for testing (in this case 10%)
#this is needed so that the program hasn't already seen the data it is trying to predict after training

linear = linear_model.LinearRegression() #free model for LR, thanks sklearn
linear.fit(x_train, y_train) #plots the best fit line in multidimensional space using the values given
acc = linear.score(x_test, y_test) #uses the best fit line to determine the label and returns the accuracy
print(f'Accuracy: {acc*100}%')
print(f'Coefficients: {linear.coef_}') #.coef_ returns a list of the coefficients that make the best fit line
print(f'Intercept: {linear.intercept_}') #.intercept_ returns the intercept used in the best fit line

predictions = linear.predict(x_test) #returns a list of all the predictions made by the model for given data
for i in range(y_test.size):
    print(f'Data: {x_test[i]}, Prediction: {predictions[i]}, Actual score: {y_test[i]}')