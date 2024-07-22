import pandas as pd
import numpy as np 
import re
import requests
import json
from sklearn.preprocessing import OneHotEncoder, RobustScaler 
from sklearn.linear_model import ElasticNet
from scipy import stats
from scipy.stats.mstats import winsorize
import pickle
from car_data_prep import prepare_data

df = pd.read_csv("dataset.csv")



df =prepare_data(df, fit=True)

X = df.drop(["Price"], axis=1) 
y = df["Price"]


en_model =ElasticNet(alpha=0.1,l1_ratio=0.9)
en_model.fit(X, y)
pickle.dump(en_model, open("trained_model.pkl","wb"))

