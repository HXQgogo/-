# -*- coding: gb2312 -*-
from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
#网页地址
url_path='http://car.bitauto.com/xuanchegongju/?mid=8'

#提取网页函数
def get_url(i):
    if i==0:
        url=url_path
    else:
        url = url_path + '&page='+str(i)
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    html1 = requests.get(url, headers=headers, timeout=10)
    content = html1.text
    bs1 = BeautifulSoup(content, 'html.parser', from_encoding='GBK')
    return bs1
#解析数据
def analysis(bs1):
    df = pd.DataFrame(columns=['名称','最低价格','最高价格','产品图片'])
    car_models = bs1.find('div', class_='search-result-list')
    car_infos = car_models.find_all('div', class_='search-result-list-item')
    #print(car_infos)
    for car_info in car_infos:
        image=car_info.find_all('img')[0].attrs['src']
        name=car_info.find_all('p',class_='cx-name text-hover')[0].text
        #print(name)
        price=car_info.find_all('p',class_='cx-price')[0].text.split('-')
        #print(price)
        temp={}
        if len(price)==2:
            temp['名称'],temp['最低价格'],temp['最高价格'],temp['产品图片']=name,price[0],price[1],image
        else:
            temp['名称'],temp['最低价格'],temp['最高价格'],temp['产品图片']=name,price[0],"",image
        df = df.append(temp, ignore_index=True)
    return df
#构造主函数
def main():
    print('开始爬取...')
    start_time = time.time()
    for page_num in range(0, 3):
        print('正在爬取第%s页' % page_num)
        bs_list=get_url(page_num)
        df_new=analysis(bs_list)
        df_new=df_new.append(df_new)

    df_new.to_csv('D:\python\Data_Engine_with_Python-master\数据分析训练营-结营考试\结业考试_P1.csv',  header=True, index=False, encoding='GBK')
    print('爬取完成')
    end_time = time.time()
    cost_time = end_time - start_time
    print("爬取任务完成，共耗时%s分钟！" % str(round(cost_time / 60, 2)))
    return df_new

if __name__=='__main__':
    main()



