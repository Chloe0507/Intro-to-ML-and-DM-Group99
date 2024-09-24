# Import necessary libraries
import importlib_resources
import numpy as np
import xlrd
import matplotlib.pyplot as plt
import csv

# Function to format a number to four decimal places
def format_to_four_decimals(number):
    return round(number, 4)

# Path to the datafile
filename = "./concrete_dataset.xls"

# Load xls sheet with data
doc = xlrd.open_workbook(filename).sheet_by_index(0)

# Extract attribute names
attribute_names = doc.row_values(rowx=0, start_colx=0, end_colx=8)

# Extract column values for different attributes
# Each attribute's values are extracted into separate lists
c_values = doc.col_values(colx=0, start_rowx=1, end_rowx=1030)
bfs_values = doc.col_values(colx=1, start_rowx=1, end_rowx=1030)
fa_values = doc.col_values(colx=2, start_rowx=1, end_rowx=1030)
w_values = doc.col_values(colx=3, start_rowx=1, end_rowx=1030)
s_values = doc.col_values(colx=4, start_rowx=1, end_rowx=1030)
ca_values = doc.col_values(colx=5, start_rowx=1, end_rowx=1030)
fag_values = doc.col_values(colx=6, start_rowx=1, end_rowx=1030)
a_values = doc.col_values(colx=7, start_rowx=1, end_rowx=1030)

# Extract output values (Compressive strength)
y_values = doc.col_values(colx=8, start_rowx=1, end_rowx=1030)

# Organize values into a matrix for easier iteration
values_matrix = np.array([c_values, bfs_values, fa_values, w_values, s_values, ca_values, fag_values, a_values])

### BASIC STATISTICS

# Initialize lists to store statistical measures
mean = []
variance = []
stddev = []
median = []
max_values = []
min_values = []

# Calculate statistical measures for each attribute
for column in values_matrix:
    mean.append(np.mean(column))
    variance.append(np.var(column))
    stddev.append(np.std(column, ddof=1))
    median.append(np.median(column))
    max_values.append(max(column))
    min_values.append(min(column))

# Transposing the matrix
values_matrix = np.array(values_matrix).T

# Define the filename for the CSV file
output_filename = "./computed_basic_statistics.csv"

# Create a list of headers for the CSV file
headers = ["", "Mean", "Variance", "Standard Deviation", "Median", "Max Value", "Min Value"]

# Combine attribute names and their corresponding statistics into a list of tuples
attribute_stats = list(zip(attribute_names, mean, variance, stddev, median, max_values, min_values))

# Write the data to the CSV file
with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    # Apply formatting to numeric values before writing to CSV
    for row in attribute_stats:
        formatted_row = [row[0]] + [format_to_four_decimals(value) if isinstance(value, (int, float)) else value for value in row[1:]]
        writer.writerow(formatted_row)

# Initialize list to store letter classifications
y_letters = []

# Determine which kind of classification the user wants to do
choice = input("Do you want to classify the outputs by quantiles (two or more classes) or by average (two classes)? (Q / A): ")
if choice == "q" or choice == "Q":

    # Ask for how many classes the user wants to divide the outputs
    intervals_n = int(input("How many classes do you want to divide the outputs into? (two classes are recommended): "))

    # Computing the median
    y_max = max(y_values)
    y_min = min(y_values)
    y_intervals = (y_max - y_min) / intervals_n

    # Initialize lists to store classifications
    y_classification = []

    # Create classifications for output values (by quantiles)
    for interval in range(intervals_n):
        y_classification.append([y_min + (y_intervals * interval), y_min + y_intervals * (interval + 1), chr(97 + interval)])

    # Classify output values by quantiles (two or more classes)
    for value in y_values:
        for interval in y_classification:
            if interval[0] <= value <= interval[1]:
                y_letters.append(interval[2])

elif choice == "a" or choice == "A":

    # Classify output values by average (two classes)
    for value in y_values:
        if value <= np.mean(y_values):
            y_letters.append("a")
        else:
            y_letters.append("b")
    intervals_n = 2

# Extract unique classes for outputs, assign numeric values to each class
y_classNames = sorted(set(y_letters))
y_classDict = dict(zip(y_classNames, range(intervals_n)))

# Convert letter classifications to numeric values
y_y = np.asarray([y_classDict[value] for value in y_letters])

# Create subplots for plots against y_values
fig_y, axs_y = plt.subplots(2, 4, figsize=(16, 8))
fig_y.canvas.manager.set_window_title('Plots against y_values')

# Plot each attribute against y_values on separate subplots with red dots
for i, ax in enumerate(axs_y.flat):
    if i == len(axs_y.flat)-1:  # Se è l'ultimo plot
        ax.scatter(values_matrix[:, i], y_values, color='red', s=1)
        ax.set_title(f'{attribute_names[i]} vs Compressive Strength', fontsize=10)
        ax.set_xlabel('Days', fontsize=10)
        ax.set_ylabel('MPa', fontsize=10)
    else:
        ax.scatter(values_matrix[:, i], y_values, color='red', s=1)
        ax.set_title(f'{attribute_names[i]} vs Compressive Strength', fontsize=10)
        ax.set_xlabel('kg/m^3', fontsize=10)
        ax.set_ylabel('MPa', fontsize=10)


# Adjust layout to avoid overlap
plt.tight_layout()

### PCA

# Center the data and standardize them
standardized_matrix = np.zeros_like(values_matrix)
for i in range(len(values_matrix)):
    for j, value in enumerate(values_matrix[i]):
        standardized_matrix[i, j] = (value - mean[j]) / stddev[j]

# Plot boxplots for input values
plt.figure(figsize=(16, 8))

# Customizing fliers (outliers)
flierprops = dict(marker='x', color='red', markersize=8, linestyle='none')
plt.boxplot(standardized_matrix, labels=attribute_names, flierprops=flierprops)

plt.title('Boxplots of Input Values')
plt.xlabel('Attributes')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Perform Singular Value Decomposition (SVD)
U, S, V = np.linalg.svd(standardized_matrix)
Vt = V.T

# Create headers for the CSV file
pca_headers = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]

# Write the data to the CSV file
pca_output_filename = "./computed_pca_components.csv"
with open(pca_output_filename, mode='w', newline='') as pca_file:
    pca_writer = csv.writer(pca_file)

    # Write headers
    pca_writer.writerow([""] + pca_headers)

    # Apply formatting to numeric values before writing to CSV
    for i, attribute in enumerate(attribute_names):
        formatted_values = [format_to_four_decimals(value) for value in Vt[i]]
        pca_writer.writerow([attribute] + formatted_values)

# Calculate explained variance
rho = (S * S) / (S * S).sum()

# Plot variance explained
plt.figure(figsize=(16, 8))
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative"])
plt.grid()

# Project the centered data onto principal component space
Z = standardized_matrix @ Vt

# Plot PCA of the data for outputs using principal components 1 and 2
plt.figure(figsize=(16, 8))
plt.title("PCA Compressive Strength")
for letter in y_classNames:
    class_mask = y_y == y_classDict[letter]
    plt.plot(Z[class_mask, 0], Z[class_mask, 1], ".", alpha=1, markersize=5)
plt.legend(["Lower Compressive Strength Values","Higher Compressive Strength Values"])
plt.xlabel("PC1")
plt.ylabel("PC2")

# Plot PCA of the data for outputs using different components
plt.figure(figsize=(16, 8))
plt.title("PCA Compressive Strength")
for letter in y_classNames:
    class_mask = y_y == y_classDict[letter]
    plt.plot(Z[class_mask, 2], Z[class_mask, 3], ".", alpha=1, markersize=5)
plt.legend(["Lower Compressive Strength Values","Higher Compressive Strength Values"])
plt.xlabel("PC3")
plt.ylabel("PC4")

# Plot PCA of the data for outputs using different components
plt.figure(figsize=(16, 8))
plt.title("PCA Compressive Strength")
for letter in y_classNames:
    class_mask = y_y == y_classDict[letter]
    plt.plot(Z[class_mask, 2], Z[class_mask, 5], ".", alpha=1, markersize=5)
plt.legend(["Lower Compressive Strength Values","Higher Compressive Strength Values"])
plt.xlabel("PC3")
plt.ylabel("PC6")

# Show plots
plt.show()
