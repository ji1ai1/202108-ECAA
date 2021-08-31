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


def 統計特征(某表, 鍵, 統計字典, 前綴=""):
	if not isinstance(鍵, list):
		鍵 = [鍵]
	某統計表 = 某表.groupby(鍵).aggregate(統計字典)
	某統計表.columns = ["%s%s之%s%s" % (前綴, "".join(鍵), 欄名, 丑) for 欄名, 函式 in 統計字典.items() for 丑 in (函式 if isinstance(函式, list) else [函式])]
	return 某統計表
pandas.DataFrame.統計特征 = 統計特征

訓練表 = pandas.read_csv("train.csv", header=0, names=["文章", "日", "一小時商品", "價格", "價格差", "作者", "一級品類", "二級品類", "三級品類", "四級品類", "品牌", "商城", "鏈接", "一小時評論數", "一小時値數", "一小時不値數", "一小時收藏數", "一小時下單數", "二小時商品", "二小時評論數", "二小時値數", "二小時不値數", "二小時收藏數", "二小時下單數", "標籤"])
訓練表 = 訓練表.loc[訓練表.文章 < 1957244].reset_index(drop=True)
訓練表["第二標籤"] = (訓練表.標籤 / (1 + 訓練表.標籤 + 訓練表.一小時下單數 + 訓練表.二小時下單數)).fillna(0)
訓練表["列號"] = range(len(訓練表))
測試表 = pandas.read_csv("test.csv", header=0, names=["文章", "日", "一小時商品", "價格", "價格差", "作者", "一級品類", "二級品類", "三級品類", "四級品類", "品牌", "商城", "鏈接", "一小時評論數", "一小時値數", "一小時不値數", "一小時收藏數", "一小時下單數", "二小時商品", "二小時評論數", "二小時値數", "二小時不値數", "二小時收藏數", "二小時下單數"]).assign(標籤=-1, 第二標籤=-1)
測試表["列號"] = range(len(訓練表), len(訓練表) + len(測試表))
測試日 = 117

測訓表 = pandas.concat([測試表, 訓練表], ignore_index=True)
測訓表["曜日"] = 1 + (1 + 測訓表.日) % 7
測訓表["一小時値不値數"] = 測訓表.一小時値數 + 測訓表.一小時不値數
測訓表["二小時値不値數"] = 測訓表.二小時値數 + 測訓表.二小時不値數
for 甲 in ["評論", "收藏", "下單", "値", "不値", "値不値"]:
	測訓表["總二小時%s數" % 甲] = 測訓表["一小時%s數" % 甲] + 測訓表["二小時%s數" % 甲]
最大日表 = 測訓表.groupby("日").aggregate(日之最大文章=("文章", "max")).reset_index()
最大日表["後日"] = 1 + 最大日表.日
測訓表 = 測訓表.merge(最大日表.loc[:, ["日", "日之最大文章"]], on="日")
測訓表 = 測訓表.merge(最大日表.loc[:, ["後日", "日之最大文章"]].rename({"後日": "日", "日之最大文章": "日之最小文章"}, axis=1), on="日", how="left")
測訓表["日之最小文章"] = 測訓表.日之最小文章.fillna(0).astype("int")
測訓表["日文章"] = 測訓表.文章 - 測訓表.日之最小文章
測訓表["日文章比"] = 測訓表.日文章 / (測訓表.日之最大文章 - 測訓表.日之最小文章)
時段數 = [6, 12, 24]
for 甲, 甲時段數 in enumerate(時段數):
	測訓表["時段%s" % 甲] = 測訓表.日文章比 // (1 / 甲時段數)
測訓表["總日文章比"] = 測訓表.日 + 測訓表.日文章比
測訓表.loc[測訓表.日文章比 < 0, ["總日文章比"]] = numpy.nan
測訓表 = 測訓表.drop(["日之最小文章", "日之最大文章"], axis=1)

測訓表 = 測訓表.sort_values("總日文章比", ignore_index=True)
for 甲 in ["作者", "品牌", "商城", "鏈接", "一級品類", "二級品類", "三級品類", "四級品類", "二小時商品"]:
	測訓表["%s後差" % 甲] = 測訓表["總日文章比"] - 測訓表.groupby(甲).總日文章比.shift(-1)
	測訓表["%s前差" % 甲] = 測訓表["總日文章比"] - 測訓表.groupby(甲).總日文章比.shift(1)

測試表 = 測訓表.loc[測訓表.日 >= 測試日].reset_index(drop=True)
訓練表 = 測訓表.loc[測訓表.日 < 測試日].reset_index(drop=True)

統計表清單 = []
for 甲 in [
	"作者", "品牌", "商城", "鏈接", "作者", "品牌", "四級品類", "三級品類", "二級品類", "一級品類", "時段0", "時段1", "時段2"
	, ["作者", "商城"], ["商城", "品牌"]
	, ["作者", "四級品類"], ["作者", "三級品類"], ["作者", "二級品類"], ["作者", "一級品類"]
	, ["商城", "四級品類"], ["商城", "三級品類"], ["商城", "二級品類"], ["商城", "一級品類"]
	, ["品牌", "四級品類"], ["品牌", "三級品類"], ["品牌", "二級品類"], ["品牌", "一級品類"]
]:
	統計表清單.append((甲, 測訓表.統計特征(甲, {
		"文章": "count"
		, "價格": ["mean", "min", "max"]
		, "價格差": ["mean", "min", "max"]
		, **{("一小時%s數" % 子): ["mean", "sum"] for 子 in ["評論", "收藏", "下單", "値", "不値", "値不値"]}
		, **{("總二小時%s數" % 子): ["mean", "sum"] for 子 in ["評論", "收藏", "下單", "値", "不値", "値不値"]}
		, **{"%s後差" % 子: "mean" for 子 in ["作者", "品牌", "商城", "鏈接", "四級品類", "二小時商品"]}
		, **{"%s前差" % 子: "mean" for 子 in ["作者", "品牌", "商城", "鏈接", "四級品類", "二小時商品"]}
		, "日文章比": "mean"
	}).reset_index()))


def 取得資料表(某表, 某特征表, 測試日):
	某特征表["一小時下單數比"] = 某特征表.一小時下單數 / 某特征表.標籤
	某特征表["二小時下單數比"] = 某特征表.二小時下單數 / 某特征表.標籤
	某特征表["總二小時下單數比"] = 某特征表.總二小時下單數 / 某特征表.標籤

	某統計表清單 = []
	for 甲 in [
		"作者", "品牌", "商城", "鏈接", "四級品類", "二小時商品", "曜日"
		, ["作者", "鏈接"]
	]:
		某統計表清單.append((甲, 某特征表.統計特征(甲, {
			"標籤": "mean"
			, "第二標籤": "mean"
			, "一小時下單數比": "mean"
			, "二小時下單數比": "mean"
			, "總二小時下單數比": "mean"
		}).reset_index()))
		
	for 甲 in [
		"日", "作者", "品牌", "商城", "鏈接", "四級品類", "二小時商品"
		, ["作者", "鏈接"]
	]:
		某統計表清單.append((甲, 某表.統計特征(甲, {
			"標籤": "count"
			, "價格": ["mean", "min", "max"]
			, "價格差": ["mean", "min", "max"]
			, **{("一小時%s數" % 子): ["mean", "sum"] for 子 in ["評論", "收藏", "下單", "値", "不値", "値不値"]}
			, **{("總二小時%s數" % 子): ["mean", "sum"] for 子 in ["評論", "收藏", "下單", "値", "不値", "値不値"]}
			, "日文章比": "mean"
		}, "當日").reset_index()))
	
	某資料表 = 某表.loc[:, ["文章", "日", "標籤", "第二標籤", "二小時商品", "價格", "價格差", "曜日", "作者", "日文章比", "一級品類", "二級品類", "三級品類", "四級品類", "品牌", "商城", "鏈接", "時段0", "時段1", "時段2"
			, "一小時評論數", "一小時値數", "一小時不値數", "一小時收藏數", "一小時下單數", "一小時値不値數"
			, "二小時評論數", "二小時値數", "二小時不値數", "二小時收藏數", "二小時下單數", "二小時値不値數"
			, "總二小時評論數", "總二小時收藏數", "總二小時下單數", "總二小時値數", "總二小時不値數", "總二小時値不値數"
		]
		+ ["%s後差" % 子 for 子 in ["作者", "品牌", "商城", "鏈接", "四級品類", "二小時商品"]]
		+ ["%s前差" % 子 for 子 in ["作者", "品牌", "商城", "鏈接", "四級品類", "二小時商品"]]
	].copy()

	某資料表["一小時收藏下單數比"] = 某資料表.一小時收藏數 / (1 + 某資料表.一小時下單數)
	某資料表["總二小時收藏下單數比"] = 某資料表.總二小時收藏數 / (1 + 某資料表.總二小時下單數)
	for 甲 in ["評論", "收藏", "下單", "値", "不値"]:
		某資料表["%s比" % 甲] = 某資料表["一小時%s數" % 甲] / (1 + 某資料表["總二小時%s數" % 甲])
		
	for 甲, 甲表 in 某統計表清單 + 統計表清單:
		某資料表 = 某資料表.merge(甲表, on=甲, how="left")
		
	某資料表 = 某資料表.loc[:, ["文章", "日", "標籤", "第二標籤"] + [子 for 子 in 某資料表.columns if 子 not in ["文章", "日", "標籤", "第二標籤"]]]
	某資料表 = 某資料表.drop(["作者", "一級品類", "二級品類", "三級品類", "四級品類", "商城", "品牌", "鏈接", "時段0", "時段1", "時段2"], axis=1)

	return 某資料表

訓練資料表 = None
for 甲 in set(訓練表.日):
	print(str(datetime.datetime.now()) + "\t%s折" % 甲)

	甲標籤表 = 訓練表.loc[訓練表.日 == 甲]
	甲特征表 = 訓練表.loc[訓練表.日 != 甲]

	訓練資料表 = pandas.concat([訓練資料表, 取得資料表(甲標籤表.copy(), 甲特征表.copy(), 甲)], ignore_index=True)

訓練資料表.to_pickle("資料/訓練資料表")

輕模型 = lightgbm.train(
	train_set=lightgbm.Dataset(訓練資料表.iloc[:, 4:], label=訓練資料表.標籤), callbacks=[lambda 子: None if 子.iteration % 256 != 0 else print(str(datetime.datetime.now()) + "\t第%s輪" % 子.iteration)]
	, num_boost_round=4096, params={"objective": "regression", "learning_rate": 0.03, "lambda_l2": 10, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
)
with open("資料/輕模型", "wb") as 档案:
	pickle.dump(輕模型, 档案)
第二輕模型 = lightgbm.train(
	train_set=lightgbm.Dataset(訓練資料表.iloc[:, 4:], label=訓練資料表.第二標籤), callbacks=[lambda 子: None if 子.iteration % 256 != 0 else print(str(datetime.datetime.now()) + "\t第%s輪" % 子.iteration)]
	, num_boost_round=4096, params={"objective": "regression", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
)
with open("資料/第二輕模型", "wb") as 档案:
	pickle.dump(第二輕模型, 档案)

多分類輕模型 = lightgbm.train(
	train_set=lightgbm.Dataset(訓練資料表.loc[訓練資料表.日 >= 測試日 - 30].iloc[:, 4:], label=訓練資料表.loc[訓練資料表.日 >= 測試日 - 30].標籤), callbacks=[lambda 子: None if 子.iteration % 256 != 0 else print(str(datetime.datetime.now()) + "\t第%s輪" % 子.iteration)]
	, num_boost_round=2048, params={"objective": "multiclassova", "num_classes": 10, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
)
with open("資料/多分類輕模型" % 甲, "wb") as 档案:
	pickle.dump(多分類輕模型, 档案)


  

# 以下係數來自在線下做線性回歸的係數；由於該步驟較慢，線上直接採用線下的係數。
線性係數 = numpy.array([-3.3182132 , -2.36831045, -1.3350912 , -0.71965236, 0.84751301, 2.175101, 3.0399831, 3.77211737, 1.97724101, 0.46886752])
線性截距 = 3.3228797352171453

測試資料表 = None
for 甲 in set(測試表.日):
	print(str(datetime.datetime.now()) + "\t%s折" % 甲)

	甲標籤表 = 測試表.loc[測試表.日 == 甲]
	測試資料表 = pandas.concat([測試資料表, 取得資料表(甲標籤表.copy(), 訓練表.copy(), 甲)], ignore_index=True)

預測表 = 測試資料表.loc[:, ["文章"]]
預測表["第一預測"] = 輕模型.predict(測試資料表.iloc[:, 4:])
預測表.loc[預測表.第一預測 < 0, "第一預測"] = 0
預測表["第二預測"] = (1 + 測試資料表.總二小時下單數) / (1 / 第二輕模型.predict(測試資料表.iloc[:, 4:]) - 1)
預測表.loc[預測表.第二預測 < 0, "第二預測"] = 0
預測表["第三預測"] = numpy.sum(線性係數 * 多分類輕模型.predict(測試資料表.iloc[:, 4:]), axis=1) + 線性截距
預測表.loc[預測表.第三預測 < 0, "第三預測"] = 0
預測表["預測"] = 0.6 * 預測表.第一預測 + 0.1 * 預測表.第二預測 + 0.3 * 預測表.第三預測

提交表 = 預測表.loc[:, ["文章", "預測"]]
提交表.columns = ["article_id", "orders_3h_15h"]
提交表.to_csv("ja.csv", index=False)
壓縮档案 = zipfile.ZipFile("ja.zip", mode="w")
壓縮档案.write("ja.csv")
壓縮档案.close()
