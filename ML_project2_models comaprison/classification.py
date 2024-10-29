import numpy as np
import xlrd
import matplotlib.pyplot as plt
from scipy.stats import binom, beta
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sys

###### DATA LOADING AND PREPARATION ######

# Function to format a number to four decimal places
def format_to_four_decimals(number):
    return round(number, 4)

# Path to the datafile
filename = "./concrete_dataset.xls"

# Load xls sheet with data
doc = xlrd.open_workbook(filename).sheet_by_index(0)

# Extract attribute names
attribute_names = doc.row_values(rowx=0, start_colx=0, end_colx=8)

# Extract attribute values and output (compressive strength) values into arrays
attribute_values = np.array([doc.col_values(colx=j, start_rowx=1, end_rowx=1030) for j in range(8)])
y_values = np.array(doc.col_values(colx=8, start_rowx=1, end_rowx=1030))

# Transpose the attribute_values matrix to get a matrix where each row is one sample
X = attribute_values.T
y = np.array(y_values)

# Scale data standarising
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Classify output values by average (two classes)
y_mean = np.mean(y_values)
y_binary = np.array([0 if value <= y_mean else 1 for value in y_values])


# Number of samples
N = len(y)

###### CROSS VALIDATION FOR EVERY CLASSIFICATION METHOD ######

K = 10 # Number of folds
k_neighbors = 20  # Maximum number of neighbors in KNN
lambdas = 20 # Number of lambdas to iterate

# Building cross validation
CV = KFold(n_splits=K, shuffle=True, random_state=42)

# Initialize the array to store predictions and actual values
all_y_test_BL = np.array([])  # For storing all test labels for BL
all_y_test_KNN = np.array([])  # For storing all test labels for KNN
all_y_test_LR = np.array([])  # For storing all test labels for LR
all_y_est_BL = np.array([])   # For storing all predictions for BL
all_y_est_KNN = np.array([])  # For storing all test labels for KNN
all_y_est_LR = np.array([])   # For storing all predictions for LR

# Error matrixes
errors_BL = np.zeros((K, K))
errors_KNN = np.zeros((K, K, k_neighbors))
errors_LR = np.zeros((K, K, lambdas))


# Loop through each cross-validation 

k1 = 0

# First level cross-validation
for train_index_1, test_index_1 in CV.split(X_scaled, y_binary):

    # Extract training and test sets for the current CV fold
    X_train_1, X_test_1 = X_scaled[train_index_1, :], X_scaled[test_index_1, :]
    y_train_1, y_test_1 = y_binary[train_index_1], y_binary[test_index_1]

    k2 = 0

    for train_index_2, test_index_2 in CV.split(X_train_1, y_train_1):

        # Extract training and test sets for the current CV fold
        X_train_2, X_test_2 = X_train_1[train_index_2, :], X_train_1[test_index_2, :]
        y_train_2, y_test_2 = y_train_1[train_index_2], y_train_1[test_index_2]
        all_y_test_BL = np.concatenate((all_y_test_BL, y_test_2))

        ###### BASELINE METHOD ######

        # Estimation assume that every test-data belongs to the biggest class
        values, counts = np.unique(y_train_2, return_counts=True) 
        max_count_index = np.argmax(counts)
        biggest_class = values[max_count_index]

        # Identify every element as of the biggest class
        y_est_BL = [biggest_class for i in range(0, len(y_test_2))]
        all_y_est_BL = np.concatenate((all_y_est_BL, y_est_BL))

        # Calculate error
        errors_BL[k1, k2] = np.sum(y_est_BL != y_test_2) / len(y_test_2) * 100


        ###### KNN METHOD ######

        # Fit the classifier and classify the test points for 1 to L neighbors
        k_interval = np.linspace(1, k_neighbors, k_neighbors, dtype=int)
       
        for k in k_interval:
            knclassifier = KNeighborsClassifier(n_neighbors=k)
            knclassifier.fit(X_train_2, y_train_2)

            # Evaluate the model
            y_est_KNN = knclassifier.predict(X_test_2)

            # Calculate error
            errors_KNN[k1, k2, k - 1] = np.sum(y_est_KNN != y_test_2) / len(y_test_2) * 100


        ###### LOGISTIC REGRESSION METHOD ######  

        # Definition of a set of lambda values
        lambda_interval = np.logspace(-8, 2, lambdas) 

        # Train the model
        for l in range(len(lambda_interval)):
            mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[l], max_iter=10000)
            mdl.fit(X_train_2, y_train_2)
            
            # Evaluate the model
            y_est_LR = mdl.predict(X_test_2)
            
            # Calculate error
            errors_LR[k1, k2, l] = np.sum(y_est_LR != y_test_2) / len(y_test_2) * 100

        k2+=1
        
    k1+=1  



# Minimum errors for K1
min_errors_KNN = np.zeros(K)  
min_errors_LR = np.zeros(K)
min_errors_BL = np.zeros(K)

optimal_k = []  # Minimum optimal k for each K1
optimal_lambdas = []  # Minimum optimal lambda for each K1
all_optimal_k = [] # All optimal k for each K1
all_optimal_lambdas = [] # All optimal lambdas for each K1

# Calculation of min errors for each outter fold
for k1 in range(K):
    #Baseline
    min_error = np.min(errors_BL[k1, :])
    min_errors_BL[k1] = min_error

    #KNN
    min_error = np.min(errors_KNN[k1, :, :])
    min_errors_KNN[k1] = min_error
    k2, k_idx = np.where(errors_KNN[k1, :, :] == min_error)
    all_optimal_k.append(k_interval[k_idx])
    optimal_k.append(k_interval[k_idx[0]])

    #Logistic regression
    min_error = np.min(errors_LR[k1, :, :])
    min_errors_LR[k1] = min_error
    k2, lambda_idx = np.where(errors_LR[k1, :, :] == min_error)
    all_optimal_lambdas.append(lambda_interval[lambda_idx])
    optimal_lambdas.append(lambda_interval[lambda_idx[0]])

optimal_k = [array.tolist() for array in optimal_k]
optimal_lambdas = [array.tolist() for array in optimal_lambdas]



###### PRINT RESULTS ######

print("Minimum errors of BL for each K1:", min_errors_BL)
print("Minimum errors of KNN for each K1:", min_errors_KNN)
print("Optimal k values of KNN for each K1:", optimal_k)
print("All optimal k values of KNN for each K1:", all_optimal_k)
print("Minimum errors of LR for each K1:", min_errors_LR)
print("Minimum optimal lambda values of LR for each K1:", optimal_lambdas)
print("All optimal lambda values of LR for each K1:", all_optimal_lambdas)


# Initialize a new array to store the minimum errors for each K1
min_errors_LR_per_lambda = np.zeros((K, lambdas))
min_errors_KNN_per_k = np.zeros((K, k_neighbors))

# Calculate the minimum errors for each parameter across K2 iterations within each K1
for k1 in range(K):
    for l in range(lambdas):
        min_errors_LR_per_lambda[k1, l] = np.min(errors_LR[k1, :, l])

for k1 in range(K):
    for k in range(k_neighbors):
        min_errors_KNN_per_k[k1, k] = np.min(errors_KNN[k1, :, k])

# Plot the minimum errors for each lambda for each K1
plt.figure(figsize=(12, 8))
for k1 in range(K):
    plt.semilogx(lambda_interval, min_errors_LR_per_lambda[k1, :], label=f'Fold {k1+1}')
plt.title('Minimum cross-validation error by lambda value for each K1')
plt.xlabel('Lambda value (log scale)')
plt.ylabel('Minimum classification error (%)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the trend lines of minimum errors for each k for each K1
plt.figure(figsize=(12, 8))
for k1 in range(K):
    coefficients = np.polyfit(k_interval, min_errors_KNN_per_k[k1, :], 1)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(k_interval)

    plt.plot(k_interval, ys, label=f'Trend Line for Fold {k1+1}')

plt.title('Trend lines for minimum cross-validation error by k value for each K1')
plt.xlabel('k')
plt.ylabel('Minimum Classification Error (%)')
plt.legend()
plt.grid(True)
plt.show()



###### MCNEMAR'S TEST ######

def mcnemar_test(y_test_A, y_test_B, y_est_A, y_est_B):

    # Calculate c^A_i and c^B_i
    c_A = (y_est_A == y_test_A).astype(int)
    c_B = (y_est_B == y_test_B).astype(int)

    # Form the 2x2 matrix
    n11 = np.sum(c_A * c_B)
    n12 = np.sum(c_A * (1 - c_B))
    n21 = np.sum((1 - c_A) * c_B)
    n22 = np.sum((1 - c_A) * (1 - c_B))

    # Calculate the confidence interval for the difference in performance
    n = np.float64(n12 + n21)

    if n <= 5:
        sys.exit("n is too low to perform the test")
    
    E_theta = (n12 - n21) / n
    Q = (n**2 * (n + 1) * (E_theta + 1) * (1 - E_theta)) / (n * (n12 + n21) - (n12 - n21)**2)
    f = (E_theta + 1) * 0.5 * (Q - 1)
    g = (1 - E_theta) * 0.5 * (Q - 1)
    alpha = 0.05
    ci_lower = 2 * beta.ppf(alpha / 2, f, g) - 1
    ci_upper = 2 * beta.ppf(1 - alpha / 2, f, g) - 1

    # Perform McNemar's test
    m = min(n12, n21)
    p_value = 2 * binom.cdf(m, n, 0.5)


    return p_value, (ci_lower, ci_upper)


# Calculation of optimal sets of y_values to compare 

k1 = 0

# First level cross-validation
for train_index_1, test_index_1 in CV.split(X_scaled, y_binary):

    # Extract training and test sets for the current CV fold
    X_train_1, X_test_1 = X_scaled[train_index_1, :], X_scaled[test_index_1, :]
    y_train_1, y_test_1 = y_binary[train_index_1], y_binary[test_index_1]

    k2 = 0

    for train_index_2, test_index_2 in CV.split(X_train_1, y_train_1):

        # Extract training and test sets for the current CV fold
        X_train_2, X_test_2 = X_train_1[train_index_2, :], X_train_1[test_index_2, :]
        y_train_2, y_test_2 = y_train_1[train_index_2], y_train_1[test_index_2]


        ###### KNN METHOD ######

        # Train the model
        knclassifier = KNeighborsClassifier(n_neighbors=optimal_k[k1])
        knclassifier.fit(X_train_2, y_train_2)

        # Evaluate the model
        y_est_KNN = knclassifier.predict(X_test_2)
        all_y_test_KNN = np.concatenate((all_y_test_KNN, y_test_2))
        all_y_est_KNN = np.concatenate((all_y_est_KNN, y_est_KNN))


        ###### LOGISTIC REGRESSION METHOD ######  

        # Train the model
        mdl = LogisticRegression(penalty="l2", C=1 / optimal_lambdas[k1], max_iter=10000)
        mdl.fit(X_train_2, y_train_2)
        
        # Evaluate the model
        y_est_LR = mdl.predict(X_test_2)
        all_y_test_LR = np.concatenate((all_y_test_LR, y_test_2))
        all_y_est_LR = np.concatenate((all_y_est_LR, y_est_LR))

        k2+=1
        
    k1+=1  

p_value_knn_bl = mcnemar_test(all_y_test_KNN, all_y_test_BL, all_y_est_KNN, all_y_est_BL)  
p_value_lr_bl = mcnemar_test(all_y_test_LR, all_y_test_BL, all_y_est_LR, all_y_est_BL)
p_value_knn_lr = mcnemar_test(all_y_test_KNN, all_y_test_LR, all_y_est_KNN, all_y_est_LR)

print(f"P-value para BL vs KNN: {p_value_knn_bl}")
print(f"P-value para BL vs LR: {p_value_lr_bl}")
print(f"P-value para KNN vs LR: {p_value_knn_lr}")


###### COEFFICIENTS COMPARISON ######

coefficients = mdl.coef_[0] #We can use the last optimal model trained because it happens that is the most optimal

# Print the coefficients
for i, coef in enumerate(coefficients):
    print(f"Coefficient for {attribute_names[i]}: {coef}")

