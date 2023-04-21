import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#data = pd.read_csv("csv-data/trainfile.csv")

def visualize_data(dataframe):
    if dataframe is None:
        raise ValueError("Dataframe is none")
    sns.kdeplot(data=dataframe, x="28")
    plt.xlabel("Transaction amount")
    plt.savefig('train_trans_amount.png')
    plt.show()

    fraud_df = dataframe[dataframe["class"] == 1]
    non_fraud_df = dataframe[dataframe["class"] == 0]

    print(dataframe[dataframe.index.duplicated()])

def variance_plot(variances):
    sns.barplot(x=variances.index, y=variances.values)
    plt.xticks(rotation=90)
    plt.title("Feature Variances")
    plt.show()

def corr_matrix():
    dataset = pd.read_csv('csv-data/finaltestlabel.csv')
    correlation_m = dataset.corr()
    sns.heatmap(correlation_m, cmap='coolwarm_r')
    plt.show()

def confusion_matrix(matrixinput):
    sns.heatmap(matrixinput, annot=True, cmap='Blues', fmt='g')
    plt.show()


