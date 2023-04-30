import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from . import analytics as an
from sklearn.metrics import confusion_matrix
import itertools

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

def confusion_matrix(matrixinput, acc, pre, rec, f1):
    ax = sns.heatmap(matrixinput, annot=True, cmap='Blues', square=True, vmin=0, vmax=0, linewidths=0.5, linecolor='k'
                                , fmt="g", cbar=False)
    sns.despine(left=False, right=False, top=False, bottom=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    #plt.text(2.05, 0.01, f'A: %.10s' % acc, fontsize=12)
    plt.text(2.05, 0.09, f'P: %.10s' % pre, fontsize=10)
    plt.text(2.05, 0.17, f'R: %.10s' % rec, fontsize=10)
    plt.text(2.05, 0.25, f'F1: %.10s' % f1, fontsize=10)

    plt.show()

def boxplot():
    data = pd.read_csv('csv-data/test.csv')
    data = data['18']
    subsample = data.sample(n=100, random_state=42)

    sns.boxplot(y=subsample)
    sns.stripplot(y=subsample, color="black", size=4) #jitter

    plt.show()


def splot():
    scatter = pd.read_csv('csv-data/finaltestlabel.csv')
    scatter = scatter.drop('Unnamed: 0', axis=1)
    combinations = list(itertools.combinations(scatter.columns, 2))
    #sns.relplot(data = scatter, x='15', y='17', hue='Class')

    for features in combinations:
        fig = sns.relplot(data=scatter, x=scatter[features[0]], y=scatter[features[1]], hue='Class')
        plt.show()
        #plt.savefig(f"images/{features[0]}_&_{features[1]}.png")
        fig.savefig(f"images/{features[0]}_&_{features[1]}.png")
        plt.close()





