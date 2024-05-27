import numpy as np
import pandas as pd
from datetime import date

import warnings
warnings.filterwarnings("ignore")
"""
    路径规划
"""
dftrain = pd.read_csv('D:\\Data\\opodata\\tabel3\\ccf_offline_stage1_train.csv',
                             header=0, keep_default_na=False)
dftest = pd.read_csv('D:\\Data\\opodata\\tabel1\\ccf_offline_stage1_test_revised.csv',
                            header=0, keep_default_na=False)
#特征提取后数据存放路径
afterPath = r'D:\Data\opodata'