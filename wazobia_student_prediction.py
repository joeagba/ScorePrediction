import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/soul/Desktop/Book1.csv", header=None)
df.columns = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V']

X = df[['C','D','E','F','G','H','I','J','R','S','T','U']].values
y = df[['V']].values

#standardizing the features
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


from sklearn.preprocessing import PolynomialFeatures

quadratic = PolynomialFeatures(degree=2)
#X_quad = quadratic.fit_transform(X)
X_log = np.log(0.0000000001+X)
sc_x = StandardScaler()
X_std = sc_x.fit_transform(X_log)

#regr = LinearRegression()
#regr.fit(X_std, y)

lasso = Lasso(alpha=0.1)
lasso.fit(X_std, y)

r = pd.read_csv("C:/Users/soul/Desktop/Book2.csv", header=None)
r.columns = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U']
t = r[['C','D','E','F','G','H','I','J','R','S','T','U']].values

#t_quad = quadratic.fit_transform(t)
t_log = np.log(0.0000000001+t)
sc_t = StandardScaler()
t_std = sc_t.fit_transform(t_log)


prediction = (lasso.predict(t_std))

print (prediction)

np.savetxt("C:/Users/soul/Desktop/submission2.csv",prediction,delimiter=',')
