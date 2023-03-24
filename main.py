import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
import visualization as vs
import analytics as an

x = input("1. test\n2. testlabel\n3. train\nSelection: ")

an.dataset_analytics()