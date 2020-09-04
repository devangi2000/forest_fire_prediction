import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

data = pd.read_csv('forest_fires.csv')
data = np.array(data)

x = data[1:, 1:-1]
y = data[1:, -1]

y = y.astype('int')
x = x.astype('int')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)

pickle.dump(log_reg, open('model.pkl', 'wb'))