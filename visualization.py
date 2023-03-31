import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("csv-data/trainfile.csv")

def visualize_data(dataframe):
    if dataframe is None:
        raise ValueError("Dataframe is none")
    #sns.kdeplot(data=dataframe, x="28")
    #plt.xlabel("Transaction amount")
   # plt.savefig('train_trans_amount.png')
    #plt.show()

   # fraud_df = dataframe[dataframe["class"] == 1]
  #  non_fraud_df = dataframe[dataframe["class"] == 0]

    print(dataframe[dataframe.index.duplicated()])