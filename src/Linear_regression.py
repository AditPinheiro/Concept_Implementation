import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

# read data
data = pd.read_csv("student-mat.csv", sep=";")

# clean data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# remove the label from the data (in this case score)
X = np.array(data.drop(columns="G3"))
# make a new array for just the label/score
y = np.array(data["G3"])

# converting the data without label from numpy array to a dataframe
X_df = pd.DataFrame({"G1": X[:, 0], "G2": X[:, 1], "studytime": X[:, 2], "failures": X[:, 3], "absences": X[:, 4]})

# splitting the data into two sets: One for training + One for testing
# x_train --> features of training set : y_train --> labels of training set
# x_test  --> features of test set     : y_test  --> labels of test set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25)

# deciding on model to be used. Here it is LinearRegression
linear = linear_model.LinearRegression()

# finding the best fit line for the given model with the given data ie; x_train & y_train
# aka fitting the data
linear.fit(x_train,y_train)

# finding the accuracy of this model using unused data ie; x_text, y_test
# aka checking how good the prediction is
acc = linear.score(x_test, y_test)

# x_test1 = [[12, 11, 2, 1, 16]]
# y_test1 = [11]

# storing just the predicted values using the x_test data
# aka the predicted scores using the model without looking at the y_test data
predictions = linear.predict(x_test)


# printing the accuracy
print("Accuracy of Project: ", acc)

# Showing which attributes cause major change
coeffecients = pd.DataFrame(linear.coef_, X_df.columns)
coeffecients.columns = ["Coeffecient"]
print(coeffecients)


# printing the predicted scores, the used features (x_test data) and the actual score
for x in range(len(predictions)):
    print("Predicted Score: ", "{:< 10d}".format(int(predictions[x])),
          "Values: ", x_test[x],
          "\t\tActual Score: ", "{:< 10d}".format(int(y_test[x])))


