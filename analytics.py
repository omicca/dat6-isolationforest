import pandas as pd
import pickle as pk
import seaborn as sns

#data = pd.read_csv("csv-data/file.csv")


def dataset_analytics(dataframe):
    # average transaction
    mean = (dataframe['28'].mean())
    print("Average transaction amount: " + str(mean))

    median = (dataframe['28'].median())
    print("Middle value: " + str(median))


    # contains null
    if (dataframe.isnull == True):
        print("Null values: True\n\n")
    else:
        print("Null values: False\n\n")




def pkl_to_csv(datafile):
    # convert to csv
    with open('data/Credit Card Fraud Detection_' + datafile + '.pkl', "rb") as file:
        data = pk.load(file)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(r'csv-data/file.csv')
    dataframe = pd.read_csv('csv-data/file.csv')

    return dataframe
