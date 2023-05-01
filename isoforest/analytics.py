import pandas as pd
import numpy as np
import pickle as pk
from sklearn.ensemble import IsolationForest
from . import visualization as vs

#data = pd.read_csv("csv-data/file.csv")

def isoforest(train, test):

    isomodel = IsolationForest(n_estimators=20, max_samples='auto', contamination='auto',
                               max_features=1, bootstrap=False, n_jobs=None,
                               verbose=0, warm_start=False)  # 0.01

    feature_input = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
     '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
     '21', '22', '23', '24', '25', '26', '27'] #'11', '13', '16', '17'

    #['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    # '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
   #  '21', '22', '23', '24', '25', '26', '27']



    isomodel.fit(train[feature_input])

    test['score'] = isomodel.decision_function(test[feature_input])
    test['prediction'] = isomodel.predict(test[feature_input])
    results = test.drop(['Unnamed: 0'], axis=1)

    results.to_csv(r'csv-data/result.csv')
    test = pd.read_csv('csv-data/result.csv')

    count_ones = (test['prediction'] == 1).sum()
    count_ones_neg = (test['prediction'] == -1).sum()
    print(f"Predictions\nFraud: {count_ones_neg} \nNon-fraud: {count_ones}\n")

    try:
        testlabel = pd.read_csv('csv-data/finaltestlabel.csv')
    except FileNotFoundError:
        print("Failed to perform further tests: missing merged test label file\n")

    accuracy, precision, recall, f1, cfmatrix = metrics(test, testlabel)
    vs.confusion_matrix(cfmatrix, accuracy, precision, recall, f1)

    print(f'P: {precision}\nR: {recall}\nF1: {f1}')



#def matrices():



def metrics(test, testlabel):
    def merge_dataframes(prediction_val, class_val):
        return pd.merge(test[test['prediction'] == prediction_val], testlabel[testlabel['Class'] == class_val],
                        how='inner', left_index=True, right_index=True)

    # True positives
    merged_df = merge_dataframes(-1, 1.0)
    tp = len(merged_df.index)

    # False positives
    merged_df = merge_dataframes(-1, 0.0)
    fp = len(merged_df.index)

    # False negatives
    merged_df = merge_dataframes(1, 1.0)
    fn = len(merged_df.index)

    # True negatives
    merged_df = merge_dataframes(1, 0.0)
    tn = len(merged_df.index)

    cf_matrix = np.array([[tn, fp],
                         [fn, tp]])

    accuracy = ((tp+tn)/(tp+tn+fp+fn))

    precision = (tp / (tp + fp))

    recall = (tp / (tp + fn))

    f1 = 2 * ((precision*recall)/(precision+recall))

    return accuracy, precision, recall, f1, cf_matrix


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
    dataframe.to_csv(r'csv-data/' + datafile + '.csv')
    dataframe = pd.read_csv('csv-data/' + datafile + '.csv')

    return dataframe

def transform_to_test():
    df1 = pd.read_csv('csv-data/test.csv')
    df2 = pd.read_csv('csv-data/label.csv')
    df2 = df2.drop(df2.index[0])

    new_df = pd.merge(df1, df2[['Unnamed: 0', '0']], on='Unnamed: 0', how='left')
    new_df = new_df.drop(['Unnamed: 0'], axis=1)
    new_df = new_df.rename(columns={'0_x': '0', '0_y': 'Class'})

    new_df.to_csv(r'csv-data/finaltestlabel.csv', index=True)



