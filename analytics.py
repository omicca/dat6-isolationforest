import pandas as pd
import pickle as pk
import seaborn as sns

data = pd.read_csv("csv-data/file.csv")
def dataset_analytics():
    #average
    average = (data['28'].mean())
    print("Average transaction amount: " + str(average))

    #contains null
    if (data.isnull == True):
        print("data contains null values")
    else:
        print("data contains no null values")

def pkl_to_csv(datafile):
    # convert to csv
    with open('data/Credit Card Fraud Detection_train.pkl', "rb") as file:
        data = pk.load(file)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(r'csv-data/file.csv')
    dataframe = pd.read_csv('csv-data/file.csv')