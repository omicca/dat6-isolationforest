import pandas as pd
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

#data = pd.read_csv("csv-data/file.csv")

def isoforest(dataframe):
    feature_input = ['0', '1']
    isomodel = IsolationForest(contamination=float(0.1), random_state=42)
    isomodel.fit(dataframe[feature_input])

    dataframe['score'] = isomodel.decision_function(dataframe[feature_input])
    dataframe['prediction'] = isomodel.predict(dataframe[feature_input])
    print(dataframe.loc[:, ['0', '1', 'score', 'prediction']])

def dataset_analytics(dataframe):
    # average transaction
    mean = (dataframe['28'].mean())
    print("Average transaction amount: " + str(mean))

    median = (dataframe['28'].median())
    print("Middle value: " + str(median))

    print(dataframe.info())


    # contains null
    if (dataframe.isnull == True):
        print("Null values: True\n\n")
    else:
        print("Null values: False\n")

def pkl_to_csv(datafile):
    # convert to csv
    with open('data/Credit Card Fraud Detection_' + datafile + '.pkl', "rb") as file:
        data = pk.load(file)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(r'csv-data/file.csv')
    dataframe = pd.read_csv('csv-data/file.csv')

    return dataframe
