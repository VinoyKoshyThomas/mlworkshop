from scipy.spatial import distance

def eucli(a,b):
	return distance.euclidean(a,b)

class myKNN():
	#fit method of Classifier
	def fit(self,X_train,Y_train):
		self.X_train = X_train
		self.Y_train = Y_train

	#predict method of Classifier
	def predict(self,X_test):
		predictions = []
		for row in X_test:
			labels = self.closest(row)
			predictions.append(labels)
		return predictions

	#Closest Distance
	def closest(self,row):
		best_dist = eucli(row,self.X_train[0])
		best_index = 0
		for i in range(1,len(self.X_train)):
			dist=eucli(row,self.X_train[i])
			#finding out best distance
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.Y_train[best_index]

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris =load_iris()
features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(features,labels,train_size=.7)

clf=myKNN()
clf.fit(X_train,Y_train)
p=clf.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy = ',accuracy_score(Y_test,p))
