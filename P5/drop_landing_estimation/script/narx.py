import numpy as np
from sklearn.linear_model import LinearRegression
from fireTS.models import NARX

x = np.random.randn(100, 1)
y = np.random.randn(100)
mdl = NARX(LinearRegression(), auto_order=2, exog_order=[2])
mdl.fit(x, y)
y_forecast = mdl.forecast(x, y, step=10, X_future=np.random.randn((9, 1)))
