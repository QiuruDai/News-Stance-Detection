import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve


def preprocess_X(X):
    m,n = X.shape
    norm = X
    mu = np.mean(X,axis=0)
    s = np.max(X,axis=0)-np.min(X,axis=0)
    norm = (norm-mu)/s
    norm = np.concatenate((np.ones([m,1]),norm),axis=1)
    return norm

#x(m,n+1);y(m,1);theta(n+1,1)
def cost_linear(X,y,theta):
    m = len(y)
    cost = 1/2/m*np.sum((X@theta-y)**2)
    return cost

def gradient_linear(X,y,ini_theta,learning_rate,iters):
    m = len(y)
    cost = []
    theta = ini_theta
    for i in range(iters):
        theta = theta - learning_rate/m*(X.T@(X@theta - y))
        cost.append(cost_linear(X,y,theta))
    
    return theta,cost


def sigmoid(x):
    return 1/(np.exp(-x)+1)

def cost_gradient_lr(X, y, theta, lam = 0):
    m = len(y)
    n = len(theta)
    cost = (-y)*np.log(sigmoid(X@theta)) - (1-y)*np.log(1-sigmoid(X@theta))
    cost = 1/m*np.sum(cost)
    #reglarization
    cost = cost + lam/2/m*np.sum(theta[1:n]**2)
    
    #gradient
    grad = (X.T@(sigmoid(X@theta)-y))/m
    temp = grad[0]
    grad = grad + lam/m*theta
    grad[0] = temp
    
    return cost,grad


def gradient_lr(X,y,ini_theta,learning_rate,iters, lam = 0):
    m = len(y)
    costs = []
    theta = ini_theta
    for i in range(iters):
        cost, grad = cost_gradient_lr(X, y, theta, lam)
        theta = theta - learning_rate*grad
        costs.append(cost)
    
    return theta,costs




def evaluate(y_class, y_pred_class):
    print(confusion_matrix(y_class, y_pred_class))
    print ("Accuracy:",accuracy_score(y_class, y_pred_class))
    print ("Recall:",recall_score(y_class, y_pred_class))
    print ("Precision:",precision_score(y_class, y_pred_class))
    print ("F1 score:", f1_score(y_class, y_pred_class))
    print("AUC:",roc_auc_score(y_class, y_pred_class))

def roc(y_class, y_pred):
    fpr, tpr, thresh = roc_curve(y_class, y_pred)
    auc = roc_auc_score(y_class, y_pred)
    print("AUC:",auc)
    
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.grid(True)
    plt.show()
