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

姫哀预测表 = pandas.read_csv("ja.csv", header=0, names=["文章", "姫哀预测"])
花小姐预测表 = pandas.read_csv("hxj.csv", header=0, names=["文章", "花小姐预测"])

预测表 = 姫哀预测表.merge(花小姐预测表, on="文章")
预测表["预测"] = 0.6 * 预测表.姫哀预测 + 0.4 * 预测表.花小姐预测

提交表 = 预测表.loc[:, ["文章", "预测"]]
提交表.columns = ["article_id", "orders_3h_15h"]
提交表.to_csv("result.csv", index=False)
压缩文件 = zipfile.ZipFile("result.zip", mode="w")
压缩文件.write("result.csv")
压缩文件.close()
