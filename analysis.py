import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# import the train and test data
train = pd.read_csv("data/ca_train.csv")
test = pd.read_csv("data/ca_test.csv")

### clean & normalize data ###
# drop unnecessary columns
train = train.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)
test = test.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)

# drop unnecessary rows
train = train.drop(np.arange(0, 47))

### Visualize train data ###
plt.style.use('fivethirtyeight')
plt.plot(train['Date'], train['ConfirmedCases'])
plt.title("Confirmed Cases over time")
plt.xlabel("Dates")
plt.ylabel("# of Confirmed Cases")
plt.show()

### try out different models ###

# model 1: mathematical modeling with R0

# model 2: Arima

# model 3: linear regression
LinearRegression.fit(train['Id'], train['ConfirmedCases'])

