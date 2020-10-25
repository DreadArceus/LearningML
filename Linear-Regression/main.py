import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

data = pd.read_csv("Linear-Regression/student-dataset/student-mat.csv", sep = ";") #load dataset
data = data[["G1", "G2", "G3", "famrel", "schoolsup"]] #trim dataset to selected attributes
# data[["sex"]] = data[["sex"]].replace("M", 1)
# data[["sex"]] = data[["sex"]].replace("F", 0)
# data[["school"]] = data[["school"]].replace("GP", 0)
# data[["school"]] = data[["school"]].replace("MS", 1)
# data[["address"]] = data[["address"]].replace("R", 1)
# data[["address"]] = data[["address"]].replace("U", 0)
# data[["famsize"]] = data[["famsize"]].replace("LE3", 1)
# data[["famsize"]] = data[["famsize"]].replace("GT3", 0)
for attrib in ["schoolsup"]:
    data[[attrib]] = data[[attrib]].replace("yes", 1)
    data[[attrib]] = data[[attrib]].replace("no", 0)

predict = "G3" #Set the label(the thing to predict using attributes)

x = np.array(data.drop([predict], 1)) #making an array of the attributes (exluding the label)
y = np.array(data[predict]) #making an array of the label's real values

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
# i = 0
# best = 0
# while True:
#     i += 1
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
#     #splitting x and y to use one portion for training and the other for testing (in this case 10%)
#     #this is needed so that the program hasn't already seen the data it is trying to predict after training

#     linear = linear_model.LinearRegression() #free model for LR, thanks sklearn
#     linear.fit(x_train, y_train) #plots the best fit line in multidimensional space using the values given
#     acc = linear.score(x_test, y_test) #uses the best fit line to determine the label and returns the accuracy
#     if i % 100000 == 0:
#         print(f'Just crossed {i}')
#     if acc > best:
#         print(f'Accuracy: {acc*100}%')
#         best = acc
#         with open("stud_model.pickle", "wb") as f: #saving the model
#             pickle.dump(linear, f)
#         if acc > 0.99:
#             break
#     elif acc > 0.98:
#         print("Another one!", i)
# print(i)
inp = open("stud_model.pickle", "rb")
linear = pickle.load(inp) #load the model

print(f'Coefficients: {linear.coef_}') #.coef_ returns a list of the coefficients that make the best fit line
print(f'Intercept: {linear.intercept_}') #.intercept_ returns the intercept used in the best fit line

predictions = linear.predict(x_test) #returns a list of all the predictions made by the model for given data
for i in range(y_test.size):
    print(f'Data: {x_test[i]}, Prediction: {predictions[i]}, Actual score: {y_test[i]}')

p = "G2"
style.use("ggplot") #makes the graph look good lol
pyplot.scatter(data[p], data[predict]) #plots stuff (x, y) making a scatter plot graph
pyplot.xlabel(p) #names the x-axis
pyplot.ylabel("Final Grade") #names the y-axis
pyplot.show() #shows the graph in a window