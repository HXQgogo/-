# -*- coding: gb2312 -*-
from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
#��ҳ��ַ
url_path='http://car.bitauto.com/xuanchegongju/?mid=8'

#��ȡ��ҳ����
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
#��������
def analysis(bs1):
    df = pd.DataFrame(columns=['����','��ͼ۸�','��߼۸�','��ƷͼƬ'])
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
            temp['����'],temp['��ͼ۸�'],temp['��߼۸�'],temp['��ƷͼƬ']=name,price[0],price[1],image
        else:
            temp['����'],temp['��ͼ۸�'],temp['��߼۸�'],temp['��ƷͼƬ']=name,price[0],"",image
        df = df.append(temp, ignore_index=True)
    return df
#����������
def main():
    print('��ʼ��ȡ...')
    start_time = time.time()
    for page_num in range(0, 3):
        print('������ȡ��%sҳ' % page_num)
        bs_list=get_url(page_num)
        df_new=analysis(bs_list)
        df_new=df_new.append(df_new)

    df_new.to_csv('D:\python\Data_Engine_with_Python-master\���ݷ���ѵ��Ӫ-��Ӫ����\��ҵ����_P1.csv',  header=True, index=False, encoding='GBK')
    print('��ȡ���')
    end_time = time.time()
    cost_time = end_time - start_time
    print("��ȡ������ɣ�����ʱ%s���ӣ�" % str(round(cost_time / 60, 2)))
    return df_new

if __name__=='__main__':
    main()



