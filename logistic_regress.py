from sklearn import  datasets,svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
import matplotlib.pyplot as plt
import math

#Extracting data for classification
#Only 2 classes are imported !!!
features=datasets.load_iris()
df_features=pd.DataFrame(features.data[:,0:2])
df_target=pd.DataFrame(features.target)
X=np.array(df_features)
y=np.array(df_target)
X_train =  X[:-50]
Y_train= y[:-50]
X_train_n = np.append([[1 for _ in range(0,len(X_train))]], X_train.T,0).T


def logistic_func(value):
   value_of_sigmoid = float(1.0 / float((1.0 + math.exp(-1.0*value))))
   return value_of_sigmoid

def calc_hypothesis(theta, x):
   H_theta_X = 0
   for i in range(len(theta)):
       H_theta_X += x[i]*theta[i]
   return logistic_func(H_theta_X )


def partial_derivate_cost(X,Y,theta,j,m,alpha):
   errors_sum = 0
   for i in range(m):
      xi = X[i]
      xij = xi[j]
      hi = calc_hypothesis(theta,X[i])
      error = (hi - Y[i])*xij # partial derivative w.r.t xij
      errors_sum += error
   m = len(Y)
   constant = float(alpha)/float(m)
   J = constant * errors_sum
   return J


def gradient_descent(X,Y,theta,m,alpha):
   theta_new = []
   for pos_i in range(len(theta)):
      PDerivative = partial_derivate_cost(X,Y,theta,pos_i,m,alpha)
      updated_theta = theta[pos_i] - PDerivative
      theta_new.append(updated_theta)
   return theta_new


def logistic_regression(X,Y,alpha,theta,num_iters):
   m = len(Y)
   for x in range(num_iters):
      new_theta = gradient_descent(X,Y,theta,m,alpha)
      theta = new_theta
   return theta

#Implementing through Gradient Discent
initial_theta =[0]*(len(df_features.columns)+1)# Initial values for parameters
alpha = 0.1 # learning rate
iterations = 1000 # Number of iterations
optimal_theta = np.array(logistic_regression(X_train_n,Y_train,alpha,initial_theta,iterations))
slope_g=-optimal_theta[1]/optimal_theta[2]
intercept_g=-optimal_theta[0]/optimal_theta[2]
ans=[0]*len(X_train)

for i in range(len(X_train_n)):
    zz = np.dot(optimal_theta.T, X_train_n[i])
    ui=logistic_func(zz)
    if ui>=0.5:
        ans[i]=1
    else:
        ans[i]=0

#Sklearn implementation(using SVM)
regr = svm.LinearSVC()
regr.fit(X_train,np.array(Y_train).ravel())
coef_=regr.coef_.ravel()
intercept=regr.intercept_
slope_s=-coef_[0]/coef_[1]
intercept_s=-intercept/coef_[1]
y_pred=regr.predict(X_train)

#Comparing accuracy and loss by both models !!!
print("Accuracy and loss by Gradient_discent")
print(accuracy_score(Y_train,ans),log_loss(Y_train,ans))
print("Accuracy and loss by Sklearn")
print(accuracy_score(Y_train,y_pred),log_loss(Y_train,y_pred))

#Plotting HYPERPLANES for both models(only valid when 2 features are selected)
#for more than 2 features comment plotting part!!!
g = np.array([4, 5, 6, 7])
plt.scatter(X[:50,0],X[:50,1],color='black',label='class 1')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',label='class 2')
plt.plot(g, g*slope_g + intercept_g, color='green',linewidth=1,label='Gradient_discent')
plt.plot(g, g*slope_s + intercept_s, color='red',linewidth=1,label='Sklearn(SVM)')
plt.xticks()
plt.yticks()
plt.title("Hyperplanes of both model")
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.show()


