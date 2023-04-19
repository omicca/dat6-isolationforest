import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import visualization as vs
import analytics as an

select = input("1. Convert .pkl\n2. Dataset analytics\n3. Data visualization\n4. Merge test & labelled\nSelection: ")

while (True):
    if (select == "1"):
        x = input("1. test\n2. testlabel\n3. train\nSelection: ")
        match x:
            case "1":
                an.pkl_to_csv("test")

            case "2":
                an.pkl_to_csv("testlabel")

            case "3":
                an.pkl_to_csv("train")

            case other:
                print("Invalid selection")


        print(".pkl converted successfully.\n\n")

    elif (select == "2"):
        try:
            train = pd.read_csv('csv-data/trainfile.csv')
            test = pd.read_csv('csv-data/testfile.csv')
            an.isoforest(train, test)
            #an.dataset_analytics(df_train)
        except FileNotFoundError:
            print("ERROR: Convert .pkl before running analytics\n")

    elif (select == "3"):
        vs.corr_matrix()

    elif (select == "4"):
        an.transform_to_test()

    else:
        print("Invalid selection")
        exit()

    select = input("1. Convert .pkl\n2. Dataset analytics\n3. Data visualization\n4. Merge test and labelled\nSelection: ")


