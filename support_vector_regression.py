import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling

# Reshaping y into a 2D array
# reshape first argument is the shape wanted
# We give the length and the number of columns
y = y.reshape(len(y), 1)

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Training a model with SVR
regressor = SVR(kernel="rbf")
regressor.fit(x, y)

# Predicting a new result
prediction = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1))

# Visualizing the SVR results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)),
         color="blue")
plt.title("Support Vector Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the SVR results with a higher resolution and smoother curve
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1, 1)),
         color="blue")
plt.title("Support Vector Regression Model (Smoother Curve)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
