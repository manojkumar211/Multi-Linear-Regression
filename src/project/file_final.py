from joblib import dump
from joblib import load
import pandas as pd
import pickle
from models import Linear_regression


with open('file_linear.pickle','wb') as f:
    pickle.dump(Linear_regression.linear_model,f) # type: ignore

