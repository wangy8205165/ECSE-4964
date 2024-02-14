"""
ECSE 4964 Homework
Author: Yixiang Wang
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Define the mapping of workclass categories to numeric values
mapping_1 = {
    'Private': 1,
    'Self-emp-not-inc': 2,
    'Self-emp-inc': 3,
    'Federal-gov': 4,
    'Local-gov': 5,
    'State-gov': 6,
    'Without-pay': 7,
    'Never-worked': 8
}
mapping_2 = {
    'Bachelors':1,
    'Some-college':2, 
    '11th':3,
    'HS-grad':4,
    'Prof-school':5, 
    'Assoc-acdm':6, 
    'Assoc-voc':7, 
    '9th':8, 
    '7th-8th':9, 
    '12th':10, 
    'Masters':11, 
    '1st-4th':12, 
    '10th':13, 
    'Doctorate':14, 
    '5th-6th':15, 
    'Preschool':16
    }

mapping_3={
    'Married-civ-spouse':1, 
    'Divorced':2, 
    'Never-married':3, 
    'Separated':4, 
    'Widowed':5, 
    'Married-spouse-absent':6, 
    'Married-AF-spouse':7 
    }
mapping_4={
    'Tech-support':1, 
    'Craft-repair':2, 
    'Other-service':3, 
    'Sales':4, 
    'Exec-managerial':5, 
    'Prof-specialty':6, 
    'Handlers-cleaners':7, 
    'Machine-op-inspct':8, 
    'Adm-clerical':9, 
    'Farming-fishing':10, 
    'Transport-moving':11, 
    'Priv-house-serv':12, 
    'Protective-serv':13, 
    'Armed-Forces':14}

mapping_5={
    'Wife':1, 
    'Own-child':2, 
    'Husband':3, 
    'Not-in-family':4, 
    'Other-relative':5, 
    'Unmarried':6
    }

mapping_6={
    'White':1, 
    'Asian-Pac-Islander':2, 
    'Amer-Indian-Eskimo':3, 
    'Other':4, 
    'Black':5
    }

mapping_7={
    'Female':1,
    'Male':2
    }
mapping_8={
    'United-States':1, 
    'Cambodia':2, 
    'England':3, 
    'Puerto-Rico':4, 
    'Canada':5, 
    'Germany':6, 
    'Outlying-US(Guam-USVI-etc)':7, 
    'India':8, 
    'Japan':9, 
    'Greece':10,
    'South':11, 
    'China':12, 
    'Cuba':13, 
    'Iran':14, 
    'Honduras':15, 
    'Philippines':16, 
    'Italy':17, 
    'Poland':18, 
    'Jamaica':19, 
    'Vietnam':20, 
    'Mexico':21, 
    'Portugal':22, 
    'Ireland':23, 
    'France':24, 
    'Dominican-Republic':25, 
    'Laos':26, 
    'Ecuador':27, 
    'Taiwan':28, 
    'Haiti':29, 
    'Columbia':30, 
    'Hungary':31, 
    'Guatemala':32, 
    'Nicaragua':33, 
    'Scotland':34, 
    'Thailand':35, 
    'Yugoslavia':36, 
    'El-Salvador':37, 
    'Trinadad&Tobago':38, 
    'Peru':39, 
    'Hong':40, 
    'Holand-Netherlands':41
    }

mapping_9={
    '>50K':0, 
    '<=50K':1.
    }

# Load the dataset
file_path =r'C:\Users\75703\Desktop\2024 Spring\ECSE 4964 Distributed Optimization & Learning\homework\adult.data'
# Assuming the dataset does not have a header and the workclass is the second column
df = pd.read_csv(file_path, header=None)

#strip off all spaces in each column
df[1] = df[1].str.strip()
df[3]= df[3].str.strip()
df[5]= df[5].str.strip()
df[6]= df[6].str.strip()
df[7]= df[7].str.strip()
df[8]= df[8].str.strip()
df[9]= df[9].str.strip()
df[13]= df[13].str.strip()
df[14]= df[14].str.strip()


    
# Replace the workclass categories with numeric values
df[1] = df[1].map(mapping_1)
df[1] = df[1].fillna(10)

df[3]= df[3].map(mapping_2)
df[3] = df[3].fillna(10)

df[5]= df[5].map(mapping_3)
df[3] = df[3].fillna(10)

df[6]= df[6].map(mapping_4)
df[6] = df[6].fillna(10)

df[7]= df[7].map(mapping_5)
df[7]= df[7].fillna(10)

df[8]= df[8].map(mapping_6)
df[8]= df[8].fillna(10)

df[9]= df[9].map(mapping_7)
df[9]= df[9].fillna(10)

df[13]= df[13].map(mapping_8)
df[13] = df[13].fillna(10)

df[14]= df[14].map(mapping_9)
df[14] = df[14].fillna(10)

X = df.iloc[:, :-1].values  # All rows, all columns except the last
y = df.iloc[:, -1].values  # All rows, last column
X = np.hstack([np.ones((X.shape[0], 1)), X])


M = 10 #number of workers for the distributed gradient descent. 
splits = np.array_split(df, M)
M1, M2, M3, M4, M5, M6, M7, M8, M9, M10 = splits

#for each data sample, separate the features from the lables.
M1x, M1y = M1.iloc[:, :-1].values, M1.iloc[:, -1].values
M2x, M2y = M2.iloc[:, :-1].values, M2.iloc[:, -1].values
M3x, M3y = M3.iloc[:, :-1].values, M3.iloc[:, -1].values
M4x, M4y = M4.iloc[:, :-1].values, M4.iloc[:, -1].values
M5x, M5y = M5.iloc[:, :-1].values, M5.iloc[:, -1].values
M6x, M6y = M6.iloc[:, :-1].values, M6.iloc[:, -1].values
M7x, M7y = M7.iloc[:, :-1].values, M7.iloc[:, -1].values
M8x, M8y = M8.iloc[:, :-1].values, M8.iloc[:, -1].values
M9x, M9y = M9.iloc[:, :-1].values, M9.iloc[:, -1].values
M10x, M10y = M10.iloc[:, :-1].values, M10.iloc[:, -1].values

Xs = [M1x, M2x, M3x, M4x, M5x, M6x, M7x, M8x, M9x, M10x]
ys = [M1y, M2y, M3y, M4y, M5y, M6y, M7y, M8y, M9y, M10y]

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
        if np.isnan(total_loss):
            print("Break point is {}".format(n))
    # Compute the regularization term
    regularization = (lambda_reg / 2) * np.sum(theta[1:]**2)  # Typically, theta[0] is not regularized
    
    # Add the regularization term to the total loss
    total_loss += regularization
    
    return total_loss

def compute_gradient(X, y, theta, lambda_reg):
    gradients = np.zeros(theta.shape)
    for n in range(X.shape[0]): 
        xn = X[n]
        yn = y[n]
        z = yn * np.dot(xn, theta)
        exp_term = np.exp(-z)
        fraction = -yn * xn * exp_term / (1 + exp_term)
        gradients += fraction
    #gradients /= X.shape[0]
    gradients += lambda_reg * theta  # Regularization term
    return gradients

def distributed_gradient_descent(Xs, ys, lr=0.01, iterations=1000, lambda_reg=0.01, decay = 0.1):
    # Xs and ys are lists of feature matrices and label vectors for each worker
    n_workers = len(Xs)
    n_features = Xs[0].shape[1] + 1  # Assuming all Xs have the same number of features, adding 1 for intercept
    theta = np.zeros(n_features)
    loss_history = []
    cpu_times = []
    start_time = time.time()
    iterat = np.empty(0,np.float64)
    communications = np.empty(0,np.float64)
    for it in range(iterations):
        gradients_sum = np.zeros(n_features)
        
        # Each worker computes its gradient
        for i in range(n_workers):
            Xi = np.hstack([np.ones((Xs[i].shape[0], 1)), Xs[i]])  # Add intercept term to X
            gradient = compute_gradient(Xi, ys[i], theta, lambda_reg)
            gradients_sum += gradient
        
        # Average the gradients
        gradients_avg = gradients_sum / n_workers
        
        # Update theta
        lr_t = lr / (1 + decay * it) #update the learning rate based on the decay constant the iteration times
        theta -= lr_t * gradients_avg
        loss = compute_loss(X,y,theta,lambda_reg)/1e7
        print("full-batch loss is {}".format(loss))
        loss_history.append(loss)
        #Compute the cpu time for each iteration. 
        end_time = time.time()
        cpu_time = end_time - start_time
        cpu_times.append(cpu_time)#Store the cpu time.
        iterat = np.append(iterat, it)  
        communications = np.append(communications,10*it)# record the number of communications
    return theta,loss_history,cpu_times,iterat, communications

def stochastic_distributed_gradient_descent(Xs, ys, lr=0.1, iterations=1000, lambda_reg=0.01,decay=0.1):
    n_workers = len(Xs)
    n_features = Xs[0].shape[1] + 1  # Include intercept term
    theta = np.zeros(n_features)
    loss_history = []
    cpu_times = []
    start_time = time.time()
    iterat = np.empty(0,np.float64)
    for it in range(iterations):
        gradients_sum = np.zeros(n_features)

        for i in range(n_workers):
            # Randomly select one sample from the ith worker's dataset
            sample_index = np.random.randint(0, Xs[i].shape[0])
            #pick the corresponding pair of data. 
            Xi_sample = np.hstack([np.ones(1), Xs[i][sample_index]])  # Add intercept term
            Xi_sample = Xi_sample.reshape(1, -1)  # Correctly reshape Xi_sample to (1, 15)
            yi_sample = ys[i][sample_index]
            yi_sample = np.array([yi_sample]).reshape(1, 1)
            
            # Compute gradient for the selected sample
            gradient = compute_gradient(Xi_sample,np.array(yi_sample), theta, lambda_reg)
            gradients_sum += gradient
        
        # Average the gradients
        gradients_avg = gradients_sum / n_workers
        # Update theta
        lr_t = lr / (1 + decay * it) #update the learning rate based on the decay constant the iteration times
        theta -= lr_t * gradients_avg
        loss = (compute_loss(X,y,theta,lambda_reg))
        print("Stochastic loss is {}".format(loss))
        loss_history.append(loss)
        #Compute the cpu time for each iteration. 
        end_time = time.time()
        cpu_time = end_time - start_time
        cpu_times.append(cpu_time)#Store the cpu time.
        iterat = np.append(iterat, it) 
        gradients_sum = np.zeros(n_features)
    return theta, loss_history, cpu_times, np.arange(iterations)



def mini_batch_distributed_gradient_descent(Xs, ys, lr=0.01, iterations=1000, lambda_reg=0.01, batch_size=100, decay =0.1):
    n_workers = len(Xs)
    n_features = Xs[0].shape[1] + 1  # Include intercept term
    theta = np.zeros(n_features)
    loss_history = []
    cpu_times = []
    start_time = time.time()

    for it in range(iterations):
        gradients_sum = np.zeros(n_features)

        for i in range(n_workers):
            # Randomly select a mini-batch from the ith worker's dataset
            indices = np.random.choice(Xs[i].shape[0], size=min(batch_size, Xs[i].shape[0]), replace=False)
            Xi_batch = np.hstack([np.ones((len(indices), 1)), Xs[i][indices]])  # Add intercept term
            yi_batch = ys[i][indices]
            
            # Compute gradient for the selected mini-batch
            gradient = compute_gradient(Xi_batch, yi_batch, theta, lambda_reg)
            gradients_sum += gradient

        # Average the gradients
        gradients_avg = gradients_sum / n_workers

        # Update theta
        lr_t = lr / (1 + decay * it) #update the learning rate based on the decay constant the iteration times
        theta -= lr_t * gradients_avg
        loss = compute_loss(X,y,theta,lambda_reg)/1e4
        print("mini-batch loss is {}".format(loss))
        loss_history.append(loss)
        # Compute the cpu time for this iteration
        cpu_times.append(time.time() - start_time)

    return theta, loss_history, cpu_times, np.arange(iterations)



lambda_reg=0.01

theta_g,loss_history_g,cpu_times_g,iterat, communications=distributed_gradient_descent(Xs, ys, lr=0.1, iterations=300, lambda_reg=0.01)
theta_s,loss_history_s,cpu_times_s,iterat= stochastic_distributed_gradient_descent(Xs, ys,lr=0.1, iterations=300, lambda_reg=0.01)
theta_m,loss_history_m,cpu_times_m,iterat= mini_batch_distributed_gradient_descent(Xs, ys,lr=0.1,iterations=300, lambda_reg=0.01)


init = loss_history_g[0]
final = loss_history_g[-1]
print(init,final)
init=loss_history_s[0]
final=loss_history_s[-1]
print(init,final)
init = loss_history_m[0]
final = loss_history_m[-1]
print(init,final)

plt.plot(iterat, loss_history_m,"red",label="Gradient Descent",linestyle="-")
plt.plot(iterat, loss_history_s,"blue",label="Stochastic Descent",linestyle="-.") 
plt.plot(iterat, loss_history_g,"cyan",label="minibatch_gd",linestyle="--")
plt.title("loss vs iterations"+ " (\u03BB = " + str(lambda_reg)+")")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(communications, loss_history_m,"red",label="Gradient Descent",linestyle="-")
plt.plot(communications, loss_history_s,"blue",label="Stochastic Descent",linestyle="-.") 
plt.plot(communications, loss_history_g,"cyan",label="minibatch_gd",linestyle="--")
plt.title("loss vs communications"+ " (\u03BB = " + str(lambda_reg)+")")
plt.xlabel("communications")
plt.ylabel("Loss")
plt.legend()
plt.show()

