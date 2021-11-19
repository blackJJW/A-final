import pandas as pd
import re
from datetime import datetime, timedelta
from tqdm import tqdm

class Gen_Senti:
    def __init__(self, stock_file, news_file):
        self.stock_file = stock_file
        self.news_file = news_file
    
    def gen_senti(self):
        def get_stock_price(stock_list, p, date):
            while(True):
               # print(stock_list[p]['date'], date)
              if len(stock_list) > p:
                stock_date = datetime.strptime(stock_list[p][0], '%Y-%m-%d')
                new_date = datetime.strptime(str(date), '%Y.%m.%d %H:%M') + timedelta(hours=9) # 15시 이후 데이터 -> 다음 날짜

                diff = stock_date.date()- new_date.date()
                # print(stock_date, new_date, diff.days)
                if  diff.days>=0:
                   return stock_list[p], p
                else:
                   p = p+1
              else:
                return None, p

        data_df_sorted = pd.read_csv('./data/news/sorted_article/'+self.news_file, encoding='utf8')
        data_total_df = pd.read_csv('./data/stock/total_df/'+self.stock_file, encoding='utf8')

        news_list = data_df_sorted.values.tolist()
        stock_list = data_total_df.values.tolist()

        temp_list_1 = []
        temp_list_2 = []

        p = 0

        for item in news_list: 
          ratio, p = get_stock_price(stock_list, p, item[1])
          if ratio is not None: 
            temp_list_1.append(item)
            temp_list_2.append(ratio)

        total_list = list(map(list.__add__, temp_list_1, temp_list_2))

        col_name = ['title', 'dates', 'article', 'apply_date', 'ratio' ,'up/down']

        senti_df = pd.DataFrame(total_list,  columns=col_name)

        index = []

        for i in range(len(senti_df)):
          index.append(i)

        senti_df['index'] = index
        senti_df = senti_df[['index', 'title', 'dates', 'article', 'apply_date', 'ratio' ,'up/down']]

        # csv파일로 저장
        senti_df.to_csv('./data/dict/'+self.news_file+'_senti.csv', index=True, encoding= 'cp949')
