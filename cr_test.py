from tqdm import tqdm
import pandas as pd

from selenium import webdriver
from selenium.webdriver.support.ui import Select
import chromedriver_autoinstaller
import csv
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os 
import time


link_result =[]
page = 1 

chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0] #크롬 드라이버 버전 확인
#-- 'USB : 시스템에 부착된 장치가 작동하지 않습니다' 오류 회피 ---------------- 
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
#----------------------------------------------------------------------------
try:
    driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)
except:
    chromedriver_autoinstaller.install(True) # 크롬 드라이버 자동 설치
    driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)

driver.implicitly_wait(5)

driver.get('https://finance.naver.com/item/news.naver?code=006360')
driver.implicitly_wait(5)

driver.find_element_by_xpath('/html/body/div/table[2]/tbody/tr/td[12]/a').click()
time.sleep(1)
driver.find_element_by_xpath('/html/body/div/table[2]/tbody/tr/td[12]/a').click()
time.sleep(1)
driver.find_element_by_xpath('/html/body/div/table[2]/tbody/tr/td[12]/a').click()
time.sleep(1)


a = driver.find_element_by_xpath('/html/body/div/table[2]/tbody/tr/td[10]/a')
print(a)

'''    
for page in tqdm(range(int(maxpage))): 
    
    url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(pag
    source_code = requests.get(url).text
    html = BeautifulSoup(source_code, "html.parser")
    # 뉴스 링크
    links = html.select('.title') 
    for link in links: 
        add = 'https://finance.naver.com' + link.find('a')['href']
        link_result.append(add)
    page += 1
result= {"링크" : link_result} 
df_result = pd.DataFrame(result)
    
    
df_result.to_csv('./data/news/links/'+company_name+'_links.csv', mode='w', encoding='utf-8-sig')

'''










'''
chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0] #크롬 드라이버 버전 확인

#-- 'USB : 시스템에 부착된 장치가 작동하지 않습니다' 오류 회피 ---------------- 
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
#----------------------------------------------------------------------------

try:
    driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)
except:
    chromedriver_autoinstaller.install(True) # 크롬 드라이버 자동 설치
    driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)
    
driver.implicitly_wait(5) # 5초대기


title_list = []
date_list = []
article_list = []

df = pd.read_csv("page.csv")

for i in tqdm(range(len(df))):
    
    driver.get(df["링크"][i])
    driver.implicitly_wait(5)

    title = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div[2]/div[1]/div[2]/table/tbody/tr[1]/th/strong').text
    title_list.append(title)

    news_date = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div[2]/div[1]/div[2]/table/tbody/tr[2]/th/span/span').text
    date_list.append(news_date)

    news_article = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div[2]/div[1]/div[2]/table/tbody/tr[3]/td/div[1]').text
    article_list.append(news_article)


news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
news_df.to_csv("news_article.csv", index = False, encoding ="cp949")
'''