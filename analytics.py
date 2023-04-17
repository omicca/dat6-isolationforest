import pandas as pd
import pickle as pk
import visualization as vs
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, make_scorer

#data = pd.read_csv("csv-data/file.csv")

def isoforest(train, test):

    isomodel = IsolationForest(n_estimators=100, max_samples=5000, contamination='auto', max_features=1.0, random_state=42)  # 0.00159

    feature_input = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                     '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                     '21', '22', '23', '24', '25', '26', '27']
    isomodel.fit(train[feature_input])

    test['score'] = isomodel.decision_function(test[feature_input])
    test['prediction'] = isomodel.predict(test[feature_input])

    print(test.loc[:, ['0', '1', 'score', 'prediction']])
    test.to_csv(r'csv-data/result.csv')
    test = pd.read_csv('csv-data/result.csv')

    count_ones = (test['prediction'] == 1).sum()
    count_ones_neg = (test['prediction'] == -1).sum()
    print(f"Predictions\nFraud: {count_ones_neg} \nNon-fraud: {count_ones}")

    test.drop(test.columns[0:31], axis=1, inplace=True)

    transform_to_test(test)
 
    return test

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

    max_values = dataframe.max()
    print(max_values)
    min_values = dataframe.min()
    print(min_values)


def pkl_to_csv(datafile):
    # convert to csv
    with open('data/Credit Card Fraud Detection_' + datafile + '.pkl', "rb") as file:
        data = pk.load(file)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(r'csv-data/' + datafile + 'file.csv')
    dataframe = pd.read_csv('csv-data/' + datafile + 'file.csv')

    return dataframe

def transform_to_test():
    df1 = pd.read_csv('csv-data/testfile.csv')
    df2 = pd.read_csv('csv-data/testlabelfile.csv')

    new_df = pd.merge(df1, df2[['Unnamed: 0', '0']], on='Unnamed: 0', how='left')
    new_df.to_csv(r'testing.csv')



