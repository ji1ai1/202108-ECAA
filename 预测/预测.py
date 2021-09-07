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


def 统计特征(某表, 键, 统计字典, 前缀=""):
	if not isinstance(键, list):
		键 = [键]
	某统计表 = 某表.groupby(键).aggregate(统计字典)
	某统计表.columns = ["%s%s之%s%s" % (前缀, "".join(键), 栏名, 丑) for 栏名, 函式 in 统计字典.items() for 丑 in (函式 if isinstance(函式, list) else [函式])]
	return 某统计表
pandas.DataFrame.统计特征 = 统计特征

训练表 = pandas.read_csv("train.csv", header=0, names=["文章", "日", "一小时商品", "价格", "价格差", "作者", "一级品类", "二级品类", "三级品类", "四级品类", "品牌", "商城", "链接", "一小时评论数", "一小时値数", "一小时不値数", "一小时收藏数", "一小时下单数", "二小时商品", "二小时评论数", "二小时値数", "二小时不値数", "二小时收藏数", "二小时下单数", "标签"])
训练表 = 训练表.loc[训练表.文章 < 1957244].reset_index(drop=True)
训练表["第二标签"] = (训练表.标签 / (1 + 训练表.标签 + 训练表.一小时下单数 + 训练表.二小时下单数)).fillna(0)
训练表["列号"] = range(len(训练表))
测试表 = pandas.read_csv("test.csv", header=0, names=["文章", "日", "一小时商品", "价格", "价格差", "作者", "一级品类", "二级品类", "三级品类", "四级品类", "品牌", "商城", "链接", "一小时评论数", "一小时値数", "一小时不値数", "一小时收藏数", "一小时下单数", "二小时商品", "二小时评论数", "二小时値数", "二小时不値数", "二小时收藏数", "二小时下单数"]).assign(卷标=-1, 第二标签=-1)
测试表["列号"] = range(len(训练表), len(训练表) + len(测试表))
测试日 = 117

测训表 = pandas.concat([测试表, 训练表], ignore_index=True)
测训表["曜日"] = 1 + (1 + 测训表.日) % 7
测训表["一小时値不値数"] = 测训表.一小时値数 + 测训表.一小时不値数
测训表["二小时値不値数"] = 测训表.二小时値数 + 测训表.二小时不値数
for 甲 in ["评论", "收藏", "下单", "値", "不値", "値不値"]:
	测训表["总二小时%s数" % 甲] = 测训表["一小时%s数" % 甲] + 测训表["二小时%s数" % 甲]
最大日表 = 测训表.groupby("日").aggregate(日之最大文章=("文章", "max")).reset_index()
最大日表["后日"] = 1 + 最大日表.日
测训表 = 测训表.merge(最大日表.loc[:, ["日", "日之最大文章"]], on="日")
测训表 = 测训表.merge(最大日表.loc[:, ["后日", "日之最大文章"]].rename({"后日": "日", "日之最大文章": "日之最小文章"}, axis=1), on="日", how="left")
测训表["日之最小文章"] = 测训表.日之最小文章.fillna(0).astype("int")
测训表["日文章"] = 测训表.文章 - 测训表.日之最小文章
测训表["日文章比"] = 测训表.日文章 / (测训表.日之最大文章 - 测训表.日之最小文章)
时段数 = [6, 12, 24]
for 甲, 甲时段数 in enumerate(时段数):
	测训表["时段%s" % 甲] = 测训表.日文章比 // (1 / 甲时段数)
测训表["总日文章比"] = 测训表.日 + 测训表.日文章比
测训表.loc[测训表.日文章比 < 0, ["总日文章比"]] = numpy.nan
测训表 = 测训表.drop(["日之最小文章", "日之最大文章"], axis=1)

测训表 = 测训表.sort_values("总日文章比", ignore_index=True)
for 甲 in ["作者", "品牌", "商城", "链接", "一级品类", "二级品类", "三级品类", "四级品类", "二小时商品"]:
	测训表["%s后差" % 甲] = 测训表["总日文章比"] - 测训表.groupby(甲).总日文章比.shift(-1)
	测训表["%s前差" % 甲] = 测训表["总日文章比"] - 测训表.groupby(甲).总日文章比.shift(1)

测试表 = 测训表.loc[测训表.日 >= 测试日].reset_index(drop=True)
训练表 = 测训表.loc[测训表.日 < 测试日].reset_index(drop=True)

统计表清单 = []
for 甲 in [
	"作者", "品牌", "商城", "链接", "作者", "品牌", "四级品类", "三级品类", "二级品类", "一级品类", "时段0", "时段1", "时段2"
	, ["作者", "商城"], ["商城", "品牌"]
	, ["作者", "四级品类"], ["作者", "三级品类"], ["作者", "二级品类"], ["作者", "一级品类"]
	, ["商城", "四级品类"], ["商城", "三级品类"], ["商城", "二级品类"], ["商城", "一级品类"]
	, ["品牌", "四级品类"], ["品牌", "三级品类"], ["品牌", "二级品类"], ["品牌", "一级品类"]
]:
	统计表清单.append((甲, 测训表.统计特征(甲, {
		"文章": "count"
		, "价格": ["mean", "min", "max"]
		, "价格差": ["mean", "min", "max"]
		, **{("一小时%s数" % 子): ["mean", "sum"] for 子 in ["评论", "收藏", "下单", "値", "不値", "値不値"]}
		, **{("总二小时%s数" % 子): ["mean", "sum"] for 子 in ["评论", "收藏", "下单", "値", "不値", "値不値"]}
		, **{"%s后差" % 子: "mean" for 子 in ["作者", "品牌", "商城", "链接", "四级品类", "二小时商品"]}
		, **{"%s前差" % 子: "mean" for 子 in ["作者", "品牌", "商城", "链接", "四级品类", "二小时商品"]}
		, "日文章比": "mean"
	}).reset_index()))


def 取得数据表(某表, 某特征表, 测试日):
	某特征表["一小时下单数比"] = 某特征表.一小时下单数 / 某特征表.标签
	某特征表["二小时下单数比"] = 某特征表.二小时下单数 / 某特征表.标签
	某特征表["总二小时下单数比"] = 某特征表.总二小时下单数 / 某特征表.标签

	某统计表清单 = []
	for 甲 in [
		"作者", "品牌", "商城", "链接", "四级品类", "二小时商品", "曜日"
		, ["作者", "链接"]
	]:
		某统计表清单.append((甲, 某特征表.统计特征(甲, {
			"标签": "mean"
			, "第二标签": "mean"
			, "一小时下单数比": "mean"
			, "二小时下单数比": "mean"
			, "总二小时下单数比": "mean"
		}).reset_index()))
		
	for 甲 in [
		"日", "作者", "品牌", "商城", "链接", "四级品类", "二小时商品"
		, ["作者", "链接"]
	]:
		某统计表清单.append((甲, 某表.统计特征(甲, {
			"标签": "count"
			, "价格": ["mean", "min", "max"]
			, "价格差": ["mean", "min", "max"]
			, **{("一小时%s数" % 子): ["mean", "sum"] for 子 in ["评论", "收藏", "下单", "値", "不値", "値不値"]}
			, **{("总二小时%s数" % 子): ["mean", "sum"] for 子 in ["评论", "收藏", "下单", "値", "不値", "値不値"]}
			, "日文章比": "mean"
		}, "当日").reset_index()))
	
	某数据表 = 某表.loc[:, ["文章", "日", "标签", "第二标签", "二小时商品", "价格", "价格差", "曜日", "作者", "日文章比", "一级品类", "二级品类", "三级品类", "四级品类", "品牌", "商城", "链接", "时段0", "时段1", "时段2"
			, "一小时评论数", "一小时値数", "一小时不値数", "一小时收藏数", "一小时下单数", "一小时値不値数"
			, "二小时评论数", "二小时値数", "二小时不値数", "二小时收藏数", "二小时下单数", "二小时値不値数"
			, "总二小时评论数", "总二小时收藏数", "总二小时下单数", "总二小时値数", "总二小时不値数", "总二小时値不値数"
		]
		+ ["%s后差" % 子 for 子 in ["作者", "品牌", "商城", "链接", "四级品类", "二小时商品"]]
		+ ["%s前差" % 子 for 子 in ["作者", "品牌", "商城", "链接", "四级品类", "二小时商品"]]
	].copy()

	某数据表["一小时收藏下单数比"] = 某数据表.一小时收藏数 / (1 + 某数据表.一小时下单数)
	某数据表["总二小时收藏下单数比"] = 某数据表.总二小时收藏数 / (1 + 某数据表.总二小时下单数)
	for 甲 in ["评论", "收藏", "下单", "値", "不値"]:
		某数据表["%s比" % 甲] = 某数据表["一小时%s数" % 甲] / (1 + 某数据表["总二小时%s数" % 甲])
		
	for 甲, 甲表 in 某统计表列表 + 统计表列表:
		某数据表 = 某数据表.merge(甲表, on=甲, how="left")
		
	某数据表 = 某数据表.loc[:, ["文章", "日", "标签", "第二标签"] + [子 for 子 in 某数据表.columns if 子 not in ["文章", "日", "标签", "第二标签"]]]
	某数据表 = 某数据表.drop(["作者", "一级品类", "二级品类", "三级品类", "四级品类", "商城", "品牌", "链接", "时段0", "时段1", "时段2"], axis=1)

	return 某数据表

训练数据表 = None
for 甲 in set(训练表.日):
	print(str(datetime.datetime.now()) + "\t%s折" % 甲)

	甲标签表 = 训练表.loc[训练表.日 == 甲]
	甲特征表 = 训练表.loc[训练表.日 != 甲]

	训练数据表 = pandas.concat([训练数据表, 取得数据表(甲标签表.copy(), 甲特征表.copy(), 甲)], ignore_index=True)

训练数据表.to_pickle("资料/训练数据表")

轻模型 = lightgbm.train(
	train_set=lightgbm.Dataset(训练数据表.iloc[:, 4:], label=训练数据表.标签), callbacks=[lambda 子: None if 子.iteration % 256 != 0 else print(str(datetime.datetime.now()) + "\t第%s轮" % 子.iteration)]
	, num_boost_round=4096, params={"objective": "regression", "learning_rate": 0.03, "lambda_l2": 10, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
)
with open("数据/轻模型", "wb") as 档案:
	pickle.dump(轻模型, 档案)
第二轻模型 = lightgbm.train(
	train_set=lightgbm.Dataset(训练数据表.iloc[:, 4:], label=训练数据表.第二标签), callbacks=[lambda 子: None if 子.iteration % 256 != 0 else print(str(datetime.datetime.now()) + "\t第%s轮" % 子.iteration)]
	, num_boost_round=4096, params={"objective": "regression", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
)
with open("数据/第二轻模型", "wb") as 档案:
	pickle.dump(第二轻模型, 档案)

多分类轻模型 = lightgbm.train(
	train_set=lightgbm.Dataset(训练数据表.loc[训练数据表.日 >= 测试日 - 30].iloc[:, 4:], label=训练数据表.loc[训练数据表.日 >= 测试日 - 30].标签), callbacks=[lambda 子: None if 子.iteration % 256 != 0 else print(str(datetime.datetime.now()) + "\t第%s轮" % 子.iteration)]
	, num_boost_round=2048, params={"objective": "multiclassova", "num_classes": 10, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
)
with open("数据/多分类轻模型" % 甲, "wb") as 档案:
	pickle.dump(多分类轻模型, 档案)


  

# 以下系数来自在线下做线性回归的系数；由于该步骤较慢，线上直接采用线下的系数。
线性系数 = numpy.array([-3.3182132 , -2.36831045, -1.3350912 , -0.71965236, 0.84751301, 2.175101, 3.0399831, 3.77211737, 1.97724101, 0.46886752])
线性截距 = 3.3228797352171453

测试数据表 = None
for 甲 in set(测试表.日):
	print(str(datetime.datetime.now()) + "\t%s折" % 甲)

	甲标签表 = 测试表.loc[测试表.日 == 甲]
	测试数据表 = pandas.concat([测试数据表, 取得数据表(甲标签表.copy(), 训练表.copy(), 甲)], ignore_index=True)

预测表 = 测试数据表.loc[:, ["文章"]]
预测表["第一预测"] = 轻模型.predict(测试数据表.iloc[:, 4:])
预测表.loc[预测表.第一预测 < 0, "第一预测"] = 0
预测表["第二预测"] = (1 + 测试数据表.总二小时下单数) / (1 / 第二轻模型.predict(测试数据表.iloc[:, 4:]) - 1)
预测表.loc[预测表.第二预测 < 0, "第二预测"] = 0
预测表["第三预测"] = numpy.sum(线性系数 * 多分类轻模型.predict(测试数据表.iloc[:, 4:]), axis=1) + 线性截距
预测表.loc[预测表.第三预测 < 0, "第三预测"] = 0
预测表["预测"] = 0.6 * 预测表.第一预测 + 0.1 * 预测表.第二预测 + 0.3 * 预测表.第三预测

提交表 = 预测表.loc[:, ["文章", "预测"]]
提交表.columns = ["article_id", "orders_3h_15h"]
提交表.to_csv("ja.csv", index=False)
压缩文件 = zipfile.ZipFile("ja.zip", mode="w")
压缩文件.write("ja.csv")
压缩文件.close()
