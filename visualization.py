import pandas as pd
import seaborn as sns

data = pd.read_csv("csv-data/file.csv")

def visualize_data():
    sns.set_theme()
    #print(data.head(), data.info())
