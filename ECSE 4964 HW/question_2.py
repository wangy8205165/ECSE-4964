"""
ECSE 4964 Homework
Author: Yixiang Wang
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\75703\Desktop\2024 Spring\ECSE 4964 Distributed Optimization & Learning\homework\ionosphere.data'
data = pd.read_csv(file_path, header=None)  # Assuming the file does not have a header row

# Split the data into features and labels
X = data.iloc[:, :-1].values  # All rows, all columns except the last
y = data.iloc[:, -1].values  # All rows, last column

# Convert labels from 'g' and 'b' to numerical values: 'g' -> 1, 'b' -> 0
y = np.where(y == 'g', 1, 0)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)


def compute_loss(X, y, theta, lambda_reg):
    """
    Compute the logistic regression loss.

    Parameters:
    - X: Features matrix for all instances (with an added column of ones for the intercept).
    - y: Labels vector, with values 1 or 0.
    - theta: Parameter vector (including the intercept term).
    - lambda_reg: Regularization constant.

    Returns:
    - The computed loss value.
    """
    # Ensure X includes the intercept term
    #X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Initialize the total loss
    total_loss = 0
    
    # Compute the loss for each instance and sum
    for n in range(X.shape[0]):
        xn = X[n]
        yn = y[n]
        z = np.dot(xn, theta)
        instance_loss = np.log(1 + np.exp(-yn * z))
        total_loss += instance_loss
    
    # Compute the regularization term
    regularization = (lambda_reg / 2) * np.sum(theta[1:]**2)  # Typically, theta[0] is not regularized
    
    # Add the regularization term to the total loss
    total_loss += regularization
    
    return total_loss

def logistic_regression_gradient_descent(X, y, lr=0.001, iterations=500, lambda_reg=0.01):
    '''
    Compute the full-batch gradient descent
    X: Features matrix for all instances (with an added column of ones for the intercept).
    y: Labels vector, with values 1 or 0.
    theta: Parameter vector (including the intercept term).
    lambda_reg: Regularization constant.
    
    Returns: The loss history, cpu time history, theta and iteration times. 
    '''
    # Initialize theta to zeros (n+1 for the intercept)
    theta = np.zeros(X.shape[1]+1)
    # Add intercept term to X
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    loss_history = []
    cpu_times = []
    start_time = time.time()
    iterat = np.empty(0,np.float64)
    for i in range(iterations):
        gradients = np.zeros(theta.shape)
        for n in range(X.shape[0]):
            xn = X[n]
            yn = y[n]
            #compute the gradient
            z = yn * np.dot(xn, theta)
            exp_term = np.exp(-z)
            fraction = -yn * xn * exp_term / (1 + exp_term)
            gradients += fraction
        #gradients /= X.shape[0]
        gradients += lambda_reg * theta  # Regularization term
        # Update theta
        theta -= lr * gradients
        #Compute the loss
        loss = compute_loss(X,y,theta,lambda_reg)
        loss_history.append(loss)
        #Compute the cpu time for each iteration. 
        end_time = time.time()
        cpu_time = end_time - start_time
        cpu_times.append(cpu_time)#Store the cpu time.
        iterat = np.append(iterat, i)
    return theta,loss_history, cpu_times,iterat

def stochastic_regression_gradient_descent(X, y, lr=0.001, iterations=500, lambda_reg=0.01):
    '''
    Compute the stochastic gradient descent
    X: Features matrix for all instances (with an added column of ones for the intercept).
    y: Labels vector, with values 1 or 0.
    idx: random index produced between 0 and 351
    xn: the idx_th entry of X.
    yn: the idx_th entry of y. 
    
    theta: Parameter vector (including the intercept term).
    lambda_reg: Regularization constant.
    
    
    
    Returns: The loss history, cpu time history, theta and iteration times. 
    ''' 
    # Initialize theta to zeros (n+1 for the intercept)
    theta = np.zeros(X.shape[1]+1)
    # Add intercept term to X
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    loss_history = []
    cpu_times = []
    start_time = time.time()
    iterat = np.empty(0,np.float64)
    
    n_samples, n_features = X.shape
    for i in range(iterations):
        gradients = np.zeros(theta.shape)
        for n in range(1):
            idx = np.random.randint(0, n_samples)
            xn = X[idx]
            yn = y[idx]
            z = yn * np.dot(xn, theta)
            exp_term = np.exp(-z)
            fraction = -yn * xn * exp_term / (1 + exp_term)
            gradients += fraction
        
        gradients += lambda_reg * theta  # Regularization term
        # Update theta
        theta -= lr * gradients
        #Compute the loss
        loss = compute_loss(X,y,theta,lambda_reg)
        loss_history.append(loss)
        #Compute the cpu time for each iteration. 
        end_time = time.time()
        cpu_time = end_time - start_time
        cpu_times.append(cpu_time)#Store the cpu time.
        iterat = np.append(iterat, i)
    return theta,loss_history, cpu_times,iterat

'''
'''
def minibatch_regression_gradient_descent(X, y, lr=0.001, iterations=500, lambda_reg=0.01):
    '''
    Compute the stochastic gradient descent
    X: Features matrix for all instances (with an added column of ones for the intercept).
    y: Labels vector, with values 1 or 0.
    random_batch: 20 random index selected between 0 and 351
    xn: 20 random entries of X.
    yn: 20 random entries of y. 
    
    theta: Parameter vector (including the intercept term).
    lambda_reg: Regularization constant.
    
    
    
    Returns: The loss history, cpu time history, theta and iteration times. 
    ''' 
    # Initialize theta to zeros (n+1 for the intercept)
    theta = np.zeros(X.shape[1]+1)
    # Add intercept term to X
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    loss_history = []
    cpu_times = []
    start_time = time.time()
    iterat = np.empty(0,np.float64)
    
    n_samples, n_features = X.shape
    batch_size = 20
    for i in range(iterations):
        gradients = np.zeros(theta.shape)
        
        random_batch = np.random.choice(X.shape[0],batch_size)
        X_random = X[random_batch]
        y_random = y[random_batch]
        for n in range(len(random_batch)):
            xn = X_random[n]
            yn = y_random[n]
            z = yn * np.dot(xn, theta)
            exp_term = np.exp(-z)
            fraction = -yn * xn * exp_term / (1 + exp_term)
            gradients += fraction 
        gradients += lambda_reg * theta  # Regularization term
        # Update theta
        theta -= lr * gradients
        #Compute the loss
        loss = compute_loss(X,y,theta,lambda_reg)
        loss_history.append(loss)
        #Compute the cpu time for each iteration. 
        end_time = time.time()
        cpu_time = end_time - start_time
        cpu_times.append(cpu_time)#Store the cpu time.
        iterat = np.append(iterat, i)
    return theta,loss_history, cpu_times,iterat



#Plot all the graphs 
lambda_reg = 0  # Example regularization constant
# Assuming X and y are already defined as per previous steps
theta,loss_history_g,cpu_times_g,iterat= logistic_regression_gradient_descent(X, y, lr=0.001, iterations=2000, lambda_reg=0)
theta,loss_history_s,cpu_times_s,iterat= stochastic_regression_gradient_descent(X, y,lr=0.001, iterations=2000, lambda_reg=0)
theta,loss_history_m,cpu_times_m,iterat= minibatch_regression_gradient_descent(X, y, lr=0.001, iterations=2000, lambda_reg=0)

plt.plot(iterat, loss_history_g,"red",label="Gradient Descent",linestyle="-")
plt.plot(iterat, loss_history_s,"blue",label="Stochastic Descent",linestyle="-.") 
plt.plot(iterat, loss_history_m,"cyan",label="minibatch_gd",linestyle="--")
plt.title("loss vs iterations"+ " (\u03BB = " + str(lambda_reg)+")")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.plot(cpu_times_g,loss_history_g,"red",label="Gradient Descent",linestyle="-")
plt.plot(cpu_times_s,loss_history_s,"blue",label="Stochastic Descent",linestyle="-.") 
plt.plot(cpu_times_m,loss_history_m,"cyan",label="minibatch_gd",linestyle="--")
plt.title("loss vs cpu time"+ " (\u03BB = " + str(lambda_reg)+")")
plt.xlabel("cpu time")
plt.ylabel("loss")
plt.legend()
plt.show()
    


