from sklearn.datasets import load_iris
import numpy as np
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0])

X = [0,50,100]

#spltting of dataset to training and testing set
xtrain = np.delete(iris.data, X , axis=0) #delete  rows from matrix
ytrain = np.delete(iris.target, X)

#testing data
xtest = iris.data[X]
ytest = iris.target[X]
 
clf = DecisionTreeClassifier()
clf.fit(xtrain , ytrain)

print(ytest)
print("Prediction = ", clf.predict(xtest))
