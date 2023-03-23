import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
import visualization as vs

# convert to csv
with open('data/Credit Card Fraud Detection_train.pkl', "rb") as file:
    data = pk.load(file)
dataframe = pd.DataFrame(data)
dataframe.to_csv(r'csv-data/file.csv')
dataframe = pd.read_csv('csv-data/file.csv')

vs.visualize_data()
