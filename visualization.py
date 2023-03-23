import pandas as pd
import seaborn as sns


def visualize_data():
    sns.set_theme()
    data = pd.read_csv("csv-data/file.csv")
    #print(data.head(), data.info())
    average = (data['28'].mean())
    print(average)