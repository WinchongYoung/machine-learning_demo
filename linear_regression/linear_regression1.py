import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../data/challenge_dataset.txt', names=['X', 'Y'])
sns.regplot(x='X', y='Y', data=df, fit_reg=False)
X_train, X_test, y_train, y_test = np.asarray(train_test_split(df['X'], df['Y'], test_size=0.1), dtype=object)

reg = LinearRegression()
reg.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

x_line = np.arange(5, 25).reshape(-1, 1)
sns.regplot(x=df['X'], y=df['Y'], data=df, fit_reg=False)
plt.plot(x_line, reg.predict(x_line))
plt.show()
