import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy().reshape(-1, 1)
x_train = train_data['Weight'].to_numpy().reshape(-1, 1)

y_test = test_data['MPG'].to_numpy().reshape(-1, 1)
x_test = test_data['Weight'].to_numpy().reshape(-1, 1)


# TODO: calculate closed-form solution
theta_best = [0, 0]

x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))

theta_best = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train  # to z tego wzoru i z specyfikacji lib. numpy

y_pred = theta_best[0] + theta_best[1] * x_train

# TODO: calculate error
print(len(x_train))

mse = sum((y_train - y_pred) ** 2)/len(x_train)
print(f'MSE: {mse}')
print(f'theta[0]: {theta_best[0]}')
print(f'theta[1]: {theta_best[1]}')


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x  # robi predykcje
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
x_train_std = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_std = (y_train - np.mean(y_train)) / np.std(y_train)
x_test_std = (x_test - np.mean(x_train)) / np.std(x_train)
y_test_std = (y_test - np.mean(y_train)) / np.std(y_train)

# TODO: calculate theta using Batch Gradient Descent
theta_best = np.zeros((2, 1))  
l_rate = 0.1
epoch = 1000

m = x_train.shape[0]
print(theta_best.shape)

for i in range(epoch):
    grad = 2/m * x_train_std.T @ (x_train_std @ theta_best - y_train_std)
    theta_best = theta_best - l_rate * grad

# TODO: calculate error
y_pred_gd = theta_best[0] + theta_best[1] * x_train
y_train_original = (y_train * np.std(y_test)) + np.mean(y_test)
y_pred_gd_original = y_pred_gd * np.std(y_train) + np.mean(y_train)
mse_gd = sum((y_pred_gd_original - y_train_original) ** 2) / len(y_train_original)

print(f'MSE_GD: {mse_gd}')
print(f'theta[0]: {theta_best[0]}')
print(f'theta[1]: {theta_best[1]}')

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()