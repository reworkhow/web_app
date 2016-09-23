import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame,Series
import math
from sklearn.ensemble import RandomForestRegressor

def plot(fromUser  = 'Default', births = []):
    df = np.random.rand(3,2)
    X, y = df[:, 1:], df[:, 0]


    forest = RandomForestRegressor(n_estimators=100,
                                  random_state=0,
                                  n_jobs=-1)
    forest.fit(X, y)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print importances[indices[f]]

    result = 10
    if fromUser != 'Default':
        return importances
    else:
        return 'check your input'
