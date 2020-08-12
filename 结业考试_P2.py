import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time

#输入数据
product_lists = pd.read_csv('D:\python\Data_Engine_with_Python-master\数据分析训练营-结营考试\ProjectB\订单表.csv',encoding='GBK')
#print(product_lists.head())
product_lists=product_lists[['订单日期','产品名称']]
#处理数据

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

start = time.time()
hot_encoded_df = product_lists.groupby(['订单日期', '产品名称'])['产品名称'].count().unstack().reset_index().fillna(0).set_index('订单日期')
hot_encoded_df = hot_encoded_df.applymap(encode_units)
print(hot_encoded_df)

#计算频繁项集
frequent_itemsets = apriori(hot_encoded_df,  min_support=0.35,use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
print("频繁项集：", frequent_itemsets.sort_values(by="support" , ascending=False))

#计算关联规则
#显示20列设置
pd.options.display.max_columns=20
print("关联规则：", rules.sort_values(by="lift" , ascending=False))

end = time.time()
print("用时：", end-start)
