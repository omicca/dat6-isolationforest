import pandas as pd
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

#data = pd.read_csv("csv-data/file.csv")

def isoforest(dataframe):
    feature_input = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    isomodel = IsolationForest(contamination=float(0.1), random_state=42) #0.00126
    isomodel.fit(dataframe[feature_input])

    dataframe['score'] = isomodel.decision_function(dataframe[feature_input])
    dataframe['prediction'] = isomodel.predict(dataframe[feature_input])

    print(dataframe.loc[:, ['0', '1', 'score', 'prediction']])
    dataframe.to_csv(r'csv-data/result.csv')
    dataframe = pd.read_csv('csv-data/result.csv')

    count_ones = (dataframe['prediction'] == 1).sum()
    count_ones_neg = (dataframe['prediction'] == -1).sum()
    print(f"Predictions\nFraud: {count_ones_neg} \nNon-fraud: {count_ones}")

    return dataframe

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
        print("Null values: False\n")

def pkl_to_csv(datafile):
    # convert to csv
    with open('data/Credit Card Fraud Detection_' + datafile + '.pkl', "rb") as file:
        data = pk.load(file)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(r'csv-data/' + datafile + 'file.csv')
    dataframe = pd.read_csv('csv-data/' + datafile + 'file.csv')

    return dataframe

def transform_to_test(df1, df2):
    df1 = pd.read_csv('csv-data/testfile.csv')
    df2 = pd.read_csv('csv-data/testlabelfile.csv')
    column_to_add = df2['0']

    new_df = pd.concat([df1, column_to_add], axis=1)
    rename_index = new_df.columns.get_loc('0', 1)
    new_df = new_df.rename(columns={new_df.columns[rename_index]: 'Class'})
    new_df.to_csv('testlabel.csv', index=False)

    return new_df
