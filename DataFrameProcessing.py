from selenium import webdriver
from selenium.webdriver.support.ui import Select
import chromedriver_autoinstaller
import pandas as pd
from tqdm import tqdm
import time
import os
import shutil

from scipy.stats import norm
import numpy as np

import cust_dates

# --------- 날짜, 시간 데이터 format -----------------
format = ' %Y.%m.%d %H:%M' # ex) ' 2021.10.20 17:20'
format_s = '%Y/%m/%d'      # ex) '2021/10/20'
format_t = '%Y-%m-%d'      # ex) '2021-10-20'
time_format ='%H:%M:%S'    # ex) '17:20:00'
# ----------------------------------------------------

# --------- 정규분포 함수 ----------------------------------------------
def norm_dist(list, p): 
    m = list.mean() #평균
    s = list.std() #표준편차

    rv = norm(loc = m, scale = s) # loc : 평균  scale : 표준편차

    return rv.ppf(p)
#-----------------------------------------------------------------------

#----------주식 데이터 정제----------------------------------------------------------------------------------------------------------------------------------
class Get_Stock_DF:
    def __init__(self, company_name, company_code, date1, date2, dir_path, prob):
        self.company_name = company_name # 회사명
        self.company_code = company_code # 회사 종목 코드
        self.date1 = date1 # 주식 데이터 조회 시작 날짜
        self.date2 = date2 # 주식 데이터 조회 끝 날짜
        self.dir_path = dir_path # 다운로드 경로
        self.prob = prob # 정규분포 기준 확률
    
    #-------- selenium을 이용,  KRX사이트를 통해 주식 데이터를 csv파일 형식으로 다운로드 ----------------------------------   
    def download_stock_data(self):
        chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0] #크롬 드라이버 버전 확인

        #-- 'USB : 시스템에 부착된 장치가 작동하지 않습니다' 오류 회피 ----------------- 
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        #----------------------------------------------------------------------------
        options.add_argument('headless') # headless 모드 설정
        options.add_argument("disable-gpu") # gpu 모드 해제
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64 Trident/7.0; rv:11.0) like Gecko")
        options.add_argument("window-size=1440x900")

        #---------다운로드 경로 설정 -------------------------
        options.add_experimental_option("prefs", {
          "download.default_directory": self.dir_path,
          "download.prompt_for_download": False,
          "download.directory_upgrade": True,
          "safebrowsing.enabled": True
        })
        #----------------------------------------------------

        try:
            driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)
        except:
            chromedriver_autoinstaller.install(True) # 크롬 드라이버 자동 설치
            driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)

        driver.implicitly_wait(5) # 5초대기

        driver.get('http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020103')
        driver.implicitly_wait(5)

        driver.get('http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020103')
        driver.implicitly_wait(5)

        input_code = driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[1]/td/div/div/p/input')
        input_code.clear()
        input_code.send_keys(self.company_code)

        find_code = driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[1]/td/div/div/p/img')
        find_code.click()
        driver.implicitly_wait(5)

        date_1 = driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[2]/td/div/div/input[1]')
        date_1.clear()
        date_1.send_keys(self.date1)
        time.sleep(1)

        date_2 = driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[2]/td/div/div/input[2]')
        date_2.clear()
        date_2.send_keys(self.date2)
        time.sleep(1)

        driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[2]/td/a').click()
        time.sleep(2)

        driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[2]/div/p[2]/button[2]/img').click()
        driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[2]/div[2]/div[2]/div/div[2]').click()
        time.sleep(2)

        driver.close()

        filename = max([self.dir_path + "\\" + f for f in os.listdir(self.dir_path)], key=os.path.getctime)
        shutil.move(filename, os.path.join(self.dir_path, self.company_name+'_'+self.date1+'_'+self.date2+'.csv'))
        
    #-----------회사 주식 데이터 ---------------------------------------------------------------------------------------------
    def gen_stock_data_df(self):
        # --------회사 주식 데이터 csv파일 open-------------------------------------------------------------------------------
        company_data = pd.read_csv('./data/stock/'+self.company_name+'_'+self.date1+'_'+self.date2+'.csv', encoding="cp949") # 주식 data read
        company_data_df_sorted = company_data.sort_values(by='일자', axis = 0) # '일자'를 기준으로 오름차순으로 정렬하고 저장
        # -------------------------------------------------------------------------------------------------------------------
        company_data_df_sorted.to_csv('./data/stock/'+self.company_name+'_'+self.date1+'_'+self.date2+'.csv', encoding="cp949")
    #------------------------------------------------------------------------------------------------------------------------
    
    #------------total_df 생성------------------------------------------------------
    def gen_stock_data_total_df(self):
        
        company_data = pd.read_csv('./data/stock/'+self.company_name+'_'+self.date1+'_'+self.date2+'.csv', encoding="cp949")
        
        date_stock = [] # 주식데이터의 날짜 빈 리스트 선언
        updown_stock = [] # 주식데이터의 등락률 빈 리스트 선언

        for i in range(len(company_data)): # 리스트에 data append
          company_datetime = cust_dates.type_date_transform(company_data.iloc[i]["일자"], format_s) # 주식 '일자' 데이터 형식 변환
          date_stock.append(company_datetime)
          updown_stock.append(company_data.iloc[i]["등락률"])

        data_total_df = pd.DataFrame({"dates": date_stock, "등락률":updown_stock}) #data_total_df DataFrame 생성

        #--------정규분포에 따른 기준 값 생성----------------------------------------------
        greater_norm = norm_dist(company_data["등락률"], self.prob)
        smaller_norm = norm_dist(company_data["등락률"], 1 - self.prob)
        #---------------------------------------------------------------------------------

        #-------data_total_df DataFrame에 result column 추가 (조건)-----------------------------------------------------------------------------------
        data_total_df.loc[data_total_df['등락률'] >= greater_norm, 'result'] = 1 # 정규분포 75% 이상의 경우 1
        data_total_df.loc[(data_total_df['등락률'] < greater_norm) & (data_total_df['등락률'] > smaller_norm), 'result'] = 0 # 정규분포 중간의 경우 0
        data_total_df.loc[data_total_df['등락률'] <= smaller_norm, 'result'] = -1 # 정규분포 25% 이하의 경우 -1
        #---------------------------------------------------------------------------------------------------------------------------------------------

        data_total_df.to_csv('./data/stock/total_df/'+self.company_name+'_'+self.date1+'_'+self.date2+'_data_total_df.csv', index = False, encoding ="utf-8")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
        

        
    
    

        
    
    
    

        
    
    
    

        
    

        
        
    

    

    
    