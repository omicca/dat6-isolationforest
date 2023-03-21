import pickle as pk
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest

with open('data/Credit Card Fraud Detection_train.pkl', "rb") as file:
    data = pk.load(file)
dataframe = pd.DataFrame(data)
dataframe.to_csv(r'csv-data/file.csv')

datafrane = pd.read_csv('csv-data/file.csv')

print(datafrane.info('csv-data/file.csv'))