import pandas as pd
import re
from datetime import datetime, timedelta

#---------custom modules---------------
import cust_dates
import cust_math
#--------------------------------------

# --------- 날짜, 시간 데이터 format -----------------
format = ' %Y.%m.%d %H:%M' # ex) ' 2021.10.20 17:20'
format_s = '%Y/%m/%d'      # ex) '2021/10/20'
format_t = '%Y-%m-%d'      # ex) '2021-10-20'
time_format ='%H:%M:%S'    # ex) '17:20:00'
# ----------------------------------------------------

#-----------회사 주식 데이터 ---------------------------------------------------------------------------------------------
def gen_stock_data_df(name):
    # --------회사 주식 데이터 csv파일 open-------------------------------------------------------------------------------
    company_data = pd.read_csv("./data/stock/"+name, encoding="cp949") # 주식 data read
    company_data_df_sorted = company_data.sort_values(by='일자', axis = 0) # '일자'를 기준으로 오름차순으로 정렬하고 저장
    # -------------------------------------------------------------------------------------------------------------------
    return company_data_df_sorted
#------------------------------------------------------------------------------------------------------------------------

#-----------회사 뉴스 데이터----------------------------------------------------------------------------------------------------------
def gen_news_data_df(name):
    # ---news csv파일 open --------------------------------------------
    data_df = pd.read_csv('./data/news/cr_article/'+name, encoding="utf8")
    # ------------------------------------------------------------------

    # ----------------정규식을 이용하여 기사 내용 중 불 필요한 부분 제거-----------------------------------------------------
    data_df['title'] = data_df['title'].apply(lambda x: re.sub(r'[^ a-zA-z ㅣㄱ-ㅣ가-힣]+', " ", x))  
    data_df['article'] = data_df['article'].apply(lambda x: re.sub(r'[^ a-zA-z ㅣㄱ-ㅣ가-힣]+', " ", x))
    data_df['article'] = data_df['article'].apply(lambda x: re.sub(r"오류를 우회하기 위한 함수 추가", " ", x ))
    # ---------------------------------------------------------------------------------------------------------------------
    data_df_sorted = data_df.sort_values(by='date', axis = 0) # 'date'를 기준으로 오름차순 정렬하여 data_df_sorted로 저장

    data_df_sorted.to_csv('./data/news/sorted_article/'+name+'_data_df_sorted.csv', index = False, encoding ="utf-8")
    
    return data_df_sorted
# ---------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------날짜별 등락률 데이터 프레임------------------------------------------------------------------
def gen_total_df(stock_data, prob, name):
    date_stock = [] # 주식데이터의 날짜 빈 리스트 선언
    updown_stock = [] # 주식데이터의 등락률 빈 리스트 선언

    for i in range(len(stock_data)): # 리스트에 data append
      company_datetime = cust_dates.type_date_transform(stock_data.iloc[i]["일자"], format_s) # 주식 '일자' 데이터 형식 변환
      date_stock.append(company_datetime)
      updown_stock.append(stock_data.iloc[i]["등락률"])

    data_total_df = pd.DataFrame({"dates": date_stock, "등락률":updown_stock}) #data_total_df DataFrame 생성

    #--------정규분포에 따른 기준 값 생성----------------------------------------------
    greater_norm = cust_math.norm_dist(stock_data["등락률"], prob)
    smaller_norm = cust_math.norm_dist(stock_data["등락률"], 1 - prob)
    #---------------------------------------------------------------------------------

    #-------data_total_df DataFrame에 result column 추가 (조건)-----------------------------------------------------------------------------------
    data_total_df.loc[data_total_df['등락률'] >= greater_norm, 'result'] = 1 # 정규분포 75% 이상의 경우 1
    data_total_df.loc[(data_total_df['등락률'] < greater_norm) & (data_total_df['등락률'] > smaller_norm), 'result'] = 0 # 정규분포 중간의 경우 0
    data_total_df.loc[data_total_df['등락률'] <= smaller_norm, 'result'] = -1 # 정규분포 25% 이하의 경우 -1
    #---------------------------------------------------------------------------------------------------------------------------------------------

    data_total_df.to_csv('./data/stock/total_df/'+name+'_data_total_df.csv', index = False, encoding ="utf-8")
    
    return data_total_df
#------------------------------------------------------------------------------------------------------------------------------

#---------------------------senti 생성----------------------------------------------------------------------------------------------------------
def gen_senti(file_name, data_df_sorted, data_total_df):
    def get_stock_price(stock_list, p, date):
        while(True):
           # print(stock_list[p]['date'], date)
           stock_date = datetime.strptime(stock_list[p][0], '%Y-%m-%d')
           new_date = datetime.strptime(date, '%Y.%m.%d %H:%M') + timedelta(hours=9) # 15시 이후 데이터 -> 다음 날짜
           
           diff = stock_date.date()- new_date.date()
           # print(stock_date, new_date, diff.days)
           if  diff.days>=0:
              return stock_list[p], p
           else:
              p = p+1
              
    data_df_sorted = pd.read_csv('./data/news/sorted_article/'+data_df_sorted, encoding='utf8')
    data_total_df = pd.read_csv('./data/stock/total_df/'+data_total_df, encoding='utf8')

    news_list = data_df_sorted.values.tolist()
    stock_list = data_total_df.values.tolist()

    temp_list_1 = []
    temp_list_2 = []

    p = 0

    for item in news_list: 
      ratio, p = get_stock_price(stock_list, p, item[1]) 
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
    senti_df.to_csv('./data/dict/'+file_name+'_senti.csv', index=True, encoding= 'cp949')