import os
import pandas as pd
from isoforest import analytics as an
from isoforest import visualization as vs

def run():
    select = input("1. Convert .pkl\n2. Dataset analytics\n3. Data visualization\n4. Merge test & labelled\nSelection: ")
    csvfolder = r'csv-data'
    if not os.path.exists(csvfolder):
        os.makedirs(csvfolder)

    while (True):
        if (select == "1"):
            x = input("1. test and labelled\n2. train\nSelection: ")
            match x:
                case "1":
                    an.pkl_to_csv("test")
                    an.pkl_to_csv("label")

                case "2":
                    an.pkl_to_csv("train")

                case other:
                    print("Invalid selection")

            print(".pkl converted successfully.\n\n")

        elif (select == "2"):
            try:
                train = pd.read_csv('csv-data/train.csv')
                test = pd.read_csv('csv-data/test.csv')
                an.isoforest(train, test)
                # an.dataset_analytics(df_train)
            except FileNotFoundError:
                print("ERROR: Convert .pkl before running analytics\n")

        elif (select == "3"):
            #vs.corr_matrix()
            vs.boxplot()

        elif (select == "4"):
            an.transform_to_test()
            print("Merge successful\n")

        elif (select == "5"):
            an.normalize_data()

        else:
            print("Invalid selection")
            exit()

        select = input(
            "1. Convert .pkl\n2. Dataset analytics\n3. Data visualization\n4. Merge test and labelled\nSelection: ")
