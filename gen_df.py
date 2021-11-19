import pandas as pd
import re
from datetime import datetime, timedelta
from tqdm import tqdm

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
    data_df['title'] = data_df['title'].apply(lambda x: re.sub(r'[^ a-zA-z ㅣㄱ-ㅣ가-힣]+', " ", str(x)))  
    data_df['article'] = data_df['article'].apply(lambda x: re.sub(r'[^ a-zA-z ㅣㄱ-ㅣ가-힣]+', " ", str(x)))
    data_df['article'] = data_df['article'].apply(lambda x: re.sub(r"오류를 우회하기 위한 함수 추가", " ", str(x) ))
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
            
    data_df_sorted = pd.read_csv('./data/news/sorted_article/'+data_df_sorted, encoding='utf8')
    data_total_df = pd.read_csv('./data/stock/total_df/'+data_total_df, encoding='utf8')

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

    a = int(len(senti_df)*0.7)
    senti_df_training = senti_df[:a]
    senti_df_test = senti_df[a:]
    
    # csv파일로 저장
    senti_df.to_csv('./data/dict/'+file_name+'_senti.csv', index=True, encoding= 'cp949')
    senti_df_training.to_csv('./data/dict/'+file_name+'_senti_training.csv', index=True, encoding= 'cp949')
    senti_df_test.to_csv('./data/dict/'+file_name+'_senti_test.csv', index=True, encoding= 'cp949')
    
#--------------주식데이터와 pos, neg 지수 결합---------------------------------------------------------------------
def senti_stock(file_name, st_data, re_data):
  stock_data = pd.read_csv('./data/stock/'+st_data, encoding='cp949')
  stock_data_s = stock_data.sort_values(by='일자', axiss = 0)
  stock_data_d = stock_data_s.reset_index(drop=False, inplace=False)
  
  result_data = pd.read_csv('./data/result'+re_data, encoding='cp949')
  re_date = sorted(list(set(list(result_data['apply_date']))))
  
  l = []
  pos = []
  neg = []
  
  for i in range(len(re_date)):
    l_2 = []
    res_list = list(filter(lambda x: result_data['apply_date'][x] == re_date[i], range(len(result_data))))
    
    for j in res_list:
      pos += result_data['sumPos'][j]
      neg += result_data['sumNeg'][j]
    l_2.append(re_date[i])
    l_2.append(pos/len(res_list))
    l_2.append(neg/len(res_list))
    pos = 0
    neg = 0
    l.append(l_2)
    
  stock_list = list(stock_data_d['일자'])
  
  for i in range(len(stock_list)):
    t = datetime.strptime(stock_list[i], '%Y/%m/%d')
    stock_list[i] = datetime.strftime(t, '%Y-%m-%d')
    
  for j in range(len(l)):
    for i in range(len(stock_list)):
      if l[j][0] not in stock_list:
        stock_data_d.loc[i,'sumPos'] = 10
        stock_data_d.loc[i,'sumNeg'] = 10
      elif l[j][0] == stock_list[i]:
        stock_data_d.loc[i,'sumPos'] = l[j][1]
        stock_data_d.loc[i,'sumNeg'] = l[j][2]
      else: 
        pass 
      
  s = []
  for i in range(len(stock_data_d)):
    s.append(max(stock_data_d['sumPos'][i], stock_data_d['sumNeg'][i]) / (stock_data_d['sumPos'][i] + stock_data_d['sumNeg'][i]))
  stock_data_d["prob"] = s
  
  stock_data_d.to_csv('./data/result/stock_result/'+file_name+'_result_for_AI.csv', encoding='cp949')
  
  #-------------------result data 정제 ------------------------------------------------------------------------------------------------------------
def refine_result():
    
    company_result = pd.read_csv("./data/result/셀트리온_test_1115_result.csv", encoding="cp949")
    company_stock = pd.read_csv("./data/stock/셀트리온_주가_050719_211112.csv", encoding="cp949")
    
    company_stock = company_stock.sort_values(by = '일자', axis = 0)
    s_d = company_stock.reset_index(drop=False, inplace=False)
    
    date_list = list(company_result['dates'])
    
    d_list = []
    for i in tqdm(range(len(date_list))):

      temp = datetime.strptime(date_list[i], '%Y.%m.%d %H:%M')
      temp_1 = datetime.strftime(temp, '%Y/%m/%d')
      d_list.append(temp_1)
      
    company_result_1 = company_result.copy()
    
    company_result_1['date'] = d_list
    
    company_result_2 = company_result_1.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    company_result_2 = company_result_2[['index', 'title', 'article', 'dates', 'date', 'apply_date', 'ratio', 'up/down', 'sumPos', 'sumNeg']]
    c = sorted(list(set(list(company_result_2['date']))))
    
    stock_list = list(s_d['일자'])
    
    for i in tqdm(range(len(stock_list))):
      t = datetime.strptime(stock_list[i], '%Y/%m/%d')
      stock_list[i] = datetime.strftime(t, '%Y-%m-%d')
      
    l = []
    pos = 0
    neg = 0
    for i in tqdm(range(len(c))):
      l_2 = []
      res_list = list(filter(lambda x: company_result_2['date'][x] == c[i], range(len(company_result_2))))
      
      for j in res_list:
        pos += company_result_2['sumPos'][j]
        neg += company_result_2['sumNeg'][j]
        
      l_2.append(c[i])
      l_2.append(pos/len(res_list))
      l_2.append(neg/len(res_list))
      pos = 0
      neg = 0
      l.append(l_2)
    l_df = pd.DataFrame(l, columns=['일자', 'sumPos', 'sumNeg'])
    
    pd_merge  = pd.merge(s_d, l_df, how='left')
    pd_merge_1 = pd_merge.dropna()
    
    s = pd_merge_1.reset_index()
    
    s.to_csv('./data/셀트리온_test_1_result.csv', encoding='cp949')
      


      