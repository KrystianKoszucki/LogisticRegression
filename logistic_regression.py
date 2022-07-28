import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time


#building a class of logistic regression
class Logistic:
    def __init__(self, tolerance, iterations):
        self.tolerance= tolerance
        self.iterations= iterations
        self.weights= None
        self.bias= None

#defining method for fitting data to the model    
    def fit_model(self, X, y):
        n_samples, n_features= X.shape
        self.weights= np.zeros(n_features)
        self.bias= 0

#gradient descent
        for _ in range(self.iterations):
            linear= np.dot(X, self.weights)+ self.bias
            y_predicted= self.sigmoid(linear)
            dw= 2*((1/ n_samples)* np.dot(X.T, (y_predicted-y)))
            db= 2*((1/ n_samples)* np.sum(y_predicted-y))
            self.weights -= self.tolerance* dw
            self.bias -= self.tolerance* db

#defining method fot predicting new points of data
    def predict(self, X):
        linear=np.dot(X, self.weights)+ self.bias
        y_predicted= self.sigmoid(linear)
        y_predicted_classes= [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_classes
#defining method for sigmoid function (it is used in hypothesis of logistic regression)
    def sigmoid(self, x):
        return np.exp(x)/(1+ np.exp(x))

#defining function for computing accuracy 
def built_score(y_true, y_pred):
    score= np.sum(y_true==y_pred)/len(y_true)
    return score

#defining dataset and sharing it for data and target parts
data= load_breast_cancer()
X, y= data.data, data.target

#spliting data for training and test parts
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.21, random_state=0)


#starting counting the time execution
starting_time= time.time()

#defining logistic regression model using previously built class for logistic regression
built_model= Logistic(tolerance= 0.0001, iterations=10000)

#fitting and predicting data
built_model.fit_model(X_train, y_train)
predictions= built_model.predict(X_test)

#the end point of the time measurement and computing time of execution of the built model
ending_time= time.time()
built_time=ending_time- starting_time

#presenting an accuracy and the time execution of the built model
print(f"\n Accuracy of the built logistic regression model= {built_score(y_test, predictions)}.\n The execution time: {round(built_time, 3)}s.")

#starting counting time execution for scikit- learn logisctic regression model
start_time=time.time()

#defining and fitting logistic regression model from scikit- learn library
model= LogisticRegression(tol=0.0001 ,max_iter=10000)
model.fit(X_train, y_train)

#the end point of the time measurement and computing the time of the execution of the model implemented in scikit- learn library
end_time= time.time()
library_time=end_time- start_time

#presenting an accuracy and the time execution of the scikit- learn model
print(f"\n Accuracy of the logistic regression model from the scikit- learn library: {model.score(X_test, y_test)}.\n The execution time: {round(library_time, 3)}s.")
