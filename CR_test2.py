import re
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import chromedriver_autoinstaller
from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm
import time
import os
import shutil
import urllib.request
import urllib.parse


class Naver_news:
    def __init__(self, company_name, maxpage, file_name):
        print("네이버 크롤링")
        self.company_name= company_name
        self.maxpage = maxpage
        self.file_name = file_name
    
        self.ni_news_url_crawler()
        #self.ni_article_crawler()


    def ni_news_url_crawler(self):
        link_result =[]
        page = 0
        
        chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0] #크롬 드라이버 버전 확인
        #-- 'USB : 시스템에 부착된 장치가 작동하지 않습니다' 오류 회피 ---------------- 
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        #----------------------------------------------------------------------------
        options.add_argument('headless') # headless 모드 설정
        options.add_argument("disable-gpu")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64 Trident/7.0; rv:11.0) like Gecko")
        
        try:
            driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)
        except:
            chromedriver_autoinstaller.install(True) # 크롬 드라이버 자동 설치
            driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)
        driver.implicitly_wait(180) # 5초대기
        
        
        driver.quit()
        
        
        for page in tqdm(range(int(self.maxpage))): 
            
            url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=' + self.company_name + '&mynews=1&office_type=1&office_section_code=1&news_office_checked=1032&nso=so:r,p:all,a:all&start=' + str(10*page+1)
            driver.get(url)

            links = driver.find_elements_by_class_name('bx')
            
            for i in links:
                aTag = i.find_element_by_tag_name('a')
                hrefs = aTag.get_attribute('href')
                link_result.append(hrefs)
    
            page += 1
    
        driver.quit()
    
        result= {"링크" : link_result} 
        df_result = pd.DataFrame(result)

        df_result.to_csv('./data/news/links/'+self.file_name, mode='w', encoding='utf-8-sig')
        
'''
    def ni_article_crawler(self):

        chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0] #크롬 드라이버 버전 확인

        #-- 'USB : 시스템에 부착된 장치가 작동하지 않습니다' 오류 회피 ---------------- 
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        #----------------------------------------------------------------------------
        options.add_argument('headless') # headless 모드 설정
        options.add_argument("disable-gpu")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64 Trident/7.0; rv:11.0) like Gecko")

        try:
            driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)
        except:
            chromedriver_autoinstaller.install(True) # 크롬 드라이버 자동 설치
            driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)

        driver.implicitly_wait(180) # 10초대기


        title_list = []
        date_list = []
        article_list = []

        df = pd.read_csv('./data/news/links/ni_news/'+self.file_name)

        for i in tqdm(range(len(df))):

            driver.get(df["링크"][i])

            if len(driver.window_handles) >= 2:
                for i in range(len(driver.window_handles)-1):
                    driver.switch_to_window(driver.window_handles[i+1])
                    driver.close()

                driver.switch_to_window(driver.window_handles[0])

            try:
                title = driver.find_element_by_xpath('/html/body/div/div[1]/div[2]/div[1]/div/div[1]/h3').text
                title_list.append(title)

                news_date = driver.find_element_by_xpath('/html/body/div/div[1]/div[2]/div[1]/div/div[1]/div[2]').text
                news_date = news_date.replace(' 게재', '')
                date_list.append(news_date)

                news_article = driver.find_element_by_class_name('article').text
                article_list.append(news_article)
            except:
                pass

        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/ni_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])
        driver.quit()
        
'''

Naver_news('GS건설', 3, 'GS건설.csv')