# -*- coding: gb2312 -*-
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
#���ݼ���
data = pd.read_csv('D:\python\Data_Engine_with_Python-master\���ݷ���ѵ��Ӫ-��Ӫ����\ProjectC\CarPrice_Assignment.csv')
train_x=data[['car_ID','symboling','CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','wheelbase','carlength','carwidth','carheight','curbweight','enginetype','cylindernumber','enginesize','fuelsystem','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']]
print(train_x.shape)
# LabelEncoder/�ַ����淶����һ��
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
columns=['CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']
for column in columns:
    train_x[column]= le.fit_transform(train_x[column])
#print(train_x)
kmeans = KMeans(n_clusters=10)
# �淶����[0,1]�ռ�
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
#ѵ��ģ��/Ԥ��ģ��
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
#���ӻ�
from scipy.cluster.hierarchy import dendrogram, ward
linkage_matrix = ward(train_x)
dendrogram(linkage_matrix)
plt.show()
#������
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'������'},axis=1,inplace=True)
print(result)
# �����������CSV�ļ���
result.to_excel("D:\python\Data_Engine_with_Python-master\���ݷ���ѵ��Ӫ-��Ӫ����\��ҵ����_P3.xlsx",index=False)
#���ⷨԤ��K��ֵ
sse = []
for k in range(1, 20):
	# kmeans�㷨
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(train_x)
	# ����inertia�������ƽ����
	sse.append(kmeans.inertia_)
x = range(1, 20)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o--')
plt.show()



