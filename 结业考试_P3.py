# -*- coding: gb2312 -*-
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
#数据加载
data = pd.read_csv('D:\python\Data_Engine_with_Python-master\数据分析训练营-结营考试\ProjectC\CarPrice_Assignment.csv')
train_x=data[['car_ID','symboling','CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','wheelbase','carlength','carwidth','carheight','curbweight','enginetype','cylindernumber','enginesize','fuelsystem','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']]
print(train_x.shape)
# LabelEncoder/字符串规范化归一化
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
columns=['CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']
for column in columns:
    train_x[column]= le.fit_transform(train_x[column])
#print(train_x)
kmeans = KMeans(n_clusters=10)
# 规范化到[0,1]空间
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
#训练模型/预测模型
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
#可视化
from scipy.cluster.hierarchy import dendrogram, ward
linkage_matrix = ward(train_x)
dendrogram(linkage_matrix)
plt.show()
#输出结果
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'聚类结果'},axis=1,inplace=True)
print(result)
# 将结果导出到CSV文件中
result.to_excel("D:\python\Data_Engine_with_Python-master\数据分析训练营-结营考试\结业考试_P3.xlsx",index=False)
#手肘法预测K的值
sse = []
for k in range(1, 20):
	# kmeans算法
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(train_x)
	# 计算inertia簇内误差平方和
	sse.append(kmeans.inertia_)
x = range(1, 20)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o--')
plt.show()



