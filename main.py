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
                df_test = an.pkl_to_csv("test")

            case "2":
                df_label = an.pkl_to_csv("testlabel")

            case "3":
                df_train = an.pkl_to_csv("train")

            case other:
                print("Invalid selection")
                exit()


        print(".pkl converted successfully.\n\n")

    elif (select == "2"):
        try:
            an.dataset_analytics(df)
            #an.isoforest(df)
        except NameError:
            print("ERROR: Convert .pkl file before accessing dataframe\n")

    elif (select == "3"):
        df_test = df_test.reset_index(drop=True)
        vs.visualize_data(df_test)

    elif (select == "4"):
        df_test = an.transform_to_test(df_label, df_test)

    else:
        print("Invalid selection")
        exit()

    select = input("1. Convert .pkl\n2. Dataset analytics\n3. Data visualization\n4. Merge test and labelled\nSelection: ")


