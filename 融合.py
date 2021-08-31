import datetime
import lightgbm
import numba
import numpy
import pandas
import pickle
import random
import torch
import warnings
import zipfile
warnings.simplefilter(action="ignore", category=pandas.errors.PerformanceWarning)

姫哀預測表 = pandas.read_csv("ja.csv", header=0, names=["文章", "姫哀預測"])
花小姐預測表 = pandas.read_csv("hxj9414.csv", header=0, names=["文章", "花小姐預測"])

預測表 = 姫哀預測表.merge(花小姐預測表, on="文章")
預測表["預測"] = 0.6 * 預測表.姫哀預測 + 0.4 * 預測表.花小姐預測

提交表 = 預測表.loc[:, ["文章", "預測"]]
提交表.columns = ["article_id", "orders_3h_15h"]
提交表.to_csv("result.csv", index=False)
壓縮档案 = zipfile.ZipFile("result.zip", mode="w")
壓縮档案.write("result.csv")
壓縮档案.close()
