import pandas as pd
import json
import re
from datetime import datetime, timedelta

#---------custom modules---------------
import cust_dates
import crawling
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
    company_data = pd.read_csv("./data/stock/"+name+".csv", encoding="cp949") # 주식 data read
    company_data_df_sorted = company_data.sort_values(by='일자', axis = 0) # '일자'를 기준으로 오름차순으로 정렬하고 저장
    # -------------------------------------------------------------------------------------------------------------------
    return company_data_df_sorted
#------------------------------------------------------------------------------------------------------------------------

#-----------회사 뉴스 데이터----------------------------------------------------------------------------------------------------------
def gen_news_data_df(name):
    # ---news json파일 open --------------------------------------------
    with open("./data/news/"+name+".json", encoding= "utf-8") as f:
        data = json.load(f)
    # ------------------------------------------------------------------
    # ----- news json파일 data 정제, 정렬, Data Frame화 ------------------------------------------------------
    d_title = [] # 기사 제목 빈 리스트 선언
    d_date = [] # 기사 날짜 빈 리스트 선언
    d_article = [] # 기사 본문 빈 리스트 선언

    for i in range(len(data)): # 리스트에 따라 data append
        try:
            d_title.append(data[str(i)]["title"])
            d_date.append(data[str(i)]["date"])
            d_article.append(data[str(i)]["article"])
        except KeyError: # error 발생시 pass
            pass

    data_df = pd.DataFrame({"title":d_title, "date":d_date, "article":d_article}) # Data Frame 화
    #------------------------------------------------------------------------------------------------------------
    # ----------------정규식을 이용하여 기사 내용 중 불 필요한 부분 제거-----------------------------------------------------
    data_df['title'] = data_df['title'].apply(lambda x: re.sub(r'[^ a-zA-z ㅣㄱ-ㅣ가-힣]+', " ", x))  
    data_df['article'] = data_df['article'].apply(lambda x: re.sub(r'[^ a-zA-z ㅣㄱ-ㅣ가-힣]+', " ", x))
    data_df['article'] = data_df['article'].apply(lambda x: re.sub(r"오류를 우회하기 위한 함수 추가", " ", x ))
    # ---------------------------------------------------------------------------------------------------------------------
    data_df_sorted = data_df.sort_values(by='date', axis = 0) # 'date'를 기준으로 오름차순 정렬하여 data_df_sorted로 저장

    return data_df_sorted
# ---------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------날짜별 등락률 데이터 프레임------------------------------------------------------------------
def gen_total_df(stock_data, dates, prob):
    date_stock = [] # 주식데이터의 날짜 빈 리스트 선언
    updown_stock = [] # 주식데이터의 등락률 빈 리스트 선언

    for i in range(len(stock_data)): # 리스트에 data append
      company_datetime = cust_dates.type_date_transform(stock_data.iloc[i]["일자"], format_s) # 주식 '일자' 데이터 형식 변환
      date_stock.append(company_datetime)
      updown_stock.append(stock_data.iloc[i]["등락률"])

    '''
    for i in range(len(dates)): # 일정 기간 내 모든 주식 데이터의 날짜와 비교하여 없으면 해당 '날짜'와 '등락률 = 0' 삽입
      if dates[i] not in date_stock:
        date_stock.insert(i, dates[i])
        updown_stock.insert(i, 0)
    '''

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

    return data_total_df
#------------------------------------------------------------------------------------------------------------------------------

def gen_senti(company_name, data_df_sorted, data_total_df):
    def get_stock_price(stock_list, p, date):
        while(True):
           # print(stock_list[p]['date'], date)
           stock_date = datetime.strptime(stock_list[p][0], '%Y-%m-%d')
           new_date = datetime.strptime(date, ' %Y.%m.%d %H:%M') + timedelta(hours=9) # 15시 이후 데이터 -> 다음 날짜
           
           diff = stock_date.date()- new_date.date()
           # print(stock_date, new_date, diff.days)
           if  diff.days>=0:
              return stock_list[p], p
           else:
              p = p+1

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
    senti_df.to_csv('./data/dict/'+company_name+'_senti.csv', index=True, encoding= 'cp949')

    return senti_df

'''

#-------------------------------뉴스 날짜에 따른 result값------------------------------------------------------------------
def gen_news_updown(company_news_data_sorted, data_total_df):
    news_updown = []

    a = dict(zip(data_total_df["dates"], data_total_df["result"]))

    for i in range(len(company_news_data_sorted)):
      news_datetime = cust_dates.type_date_transform(company_news_data_sorted.iloc[i]["date"], format)
      news_updown.append(a[news_datetime])

    return news_updown
#----------------------------------------------------------------------------------------------------

# -------------------------------감성 사전 구축--------------------------------------------------------------------------------------------
def gen_sentiDic(company_name, company_news, updown, total_df, start_year, end_year, start_date, end_date):
    sentiDic = pd.DataFrame({"dates" : company_news["date"], 
                         "title" : company_news["title"], 
                         "article" : company_news["article"],
                         "up/down": updown})

    sentiDic["dates"] = pd.to_datetime(sentiDic["dates"])
    sentiDic["date"] = sentiDic["dates"].dt.date
    sentiDic["time"] = sentiDic["dates"].dt.time

    apply_date = [] #등락률이 적용될 날짜 빈 리스트 선언

    for i in range(len(sentiDic)):
      if cust_dates.type_time_transform(str(sentiDic.iloc[i]["time"])) >= cust_dates.type_time_transform("00:00:00") and cust_dates.type_time_transform(str(sentiDic.iloc[i]["time"])) <= cust_dates.type_time_transform("15:30:00"):
        apply_date.append(sentiDic.iloc[i]["date"])
      else:
        apply_date.append(cust_dates.date_plus(sentiDic.iloc[i]["date"], 1))

    sentiDic["apply_date"] = apply_date
    #-------------------날짜 데이터 처리 --------------------------------------------------------------------------------------
    stock_holiday = crawling.stock_holiday_krx(start_year, end_year) # krx에서 해당 년도 휴장일 크롤링

    weekend_days = cust_dates.weekend(start_date, end_date) # 주말 날짜 리스트 

    total_stock_holiday = weekend_days.copy()

    for i in range(len(stock_holiday)): # 휴장일 리스트 중 주말 날짜 데이터 이전의 날짜는 제외
        if stock_holiday[i] > weekend_days[0]: 
            total_stock_holiday.append(stock_holiday[i])
        else:
            pass

    for i in  range(len(stock_holiday)): # 종합 휴장일 리스트 중 지정 날짜 이후의 날짜는 제외
        if stock_holiday[-1 - i] > end_date:
            total_stock_holiday.remove(stock_holiday[-1 - i])
        else:
            pass

    total_stock_holiday = list(set(total_stock_holiday)) #중복 제거 및 정렬
    total_stock_holiday.sort()
    #-----------------------------------------------------------------------------------------------------------------------
    for i in range(len(sentiDic)-1, -1, -1):    # 공휴일, 임시공휴일 -> 평일 
      for holi in range(5):                     # 토요일 -> 월요일             
                                                # 일요일 -> 월요일
        day_temp = sentiDic["apply_date"][i]
        if str(day_temp) in total_stock_holiday:

          day_temp = cust_dates.date_plus(day_temp, 1) 
          sentiDic["apply_date"][i]=day_temp

        else:
          break

    sentiDic.drop('up/down', axis=1, inplace=True) # 적용 날짜 리스트 적용 전 'up/down' 열 삭제

    upzerodown = []
    for i in range(len(sentiDic)):      # 적용 날짜에 따른 등락률 적용
      days_temp = cust_dates.type_date_transform(str(sentiDic.iloc[i]["apply_date"]), format_t) 
      for j in range(len(total_df)):
        if days_temp == total_df["dates"][j]:
            upzerodown.append(total_df["result"][j])

        else:
          pass

    sentiDic["up/down"] = upzerodown

    # csv파일로 저장
    sentiDic.to_csv('./data/dict/'+company_name+'_sentiDic.csv', index=True, encoding= 'cp949')

    return sentiDic
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''