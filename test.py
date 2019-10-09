import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
from matplotlib import style
import matplotlib.pyplot as pyplot

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data)

predict="G3"
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.3)


best=0
try:
    pickle_in = open("studentmodel.pickel", "rb")
    linear = pickle.load(pickle_in)
except:
    for i in range(30):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.3)

        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)
        acc= linear.score(x_test, y_test)
        #print("accuracy:", acc)
        if best<acc:
            best=acc
            with open("studentmodel.pickel", "wb") as f:
                pickle.dump(linear, f)
print('accuracy: ', linear.score(x_test, y_test))
prediction = linear.predict(x_test)

for i in range(len(prediction)):
    print(prediction[i], x_test[i], y_test[i])

p='G3'
style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('final Grade')
pyplot.show()