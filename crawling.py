from selenium import webdriver
from selenium.webdriver.support.ui import Select
import chromedriver_autoinstaller
import csv
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os 
from tqdm import tqdm


def url_crawler(company_name, company_code, maxpage):
    link_result =[]
    page = 1 
    
    for page in tqdm(range(int(maxpage))): 
    
        url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page+1) 
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

def article_crawler(company_name):

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

    df = pd.read_csv('./data/news/links/'+company_name+'_links.csv')

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
    news_df.to_csv('./data/news/'+company_name+'_news_article.csv', index = False, encoding ="utf-8")


'''
def stock_holiday_krx(year1, year2): # krx에서 휴장일 크롤링
    
    driver.get('https://open.krx.co.kr/contents/MKD/01/0110/01100305/MKD01100305.jsp')
    driver.implicitly_wait(3)

    years = []
    for i in range(year2 - year1 +1):
        years.append(year1 + i)

    date_list = []

    for k in range(len(years)):
        select = Select(driver.find_element_by_xpath('/html/body/div/div[2]/div/div[2]/article/div/fieldset/form/dl/dd/select'))    
        select.select_by_value(str(years[k]))

        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(1)
        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(1)
        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(1)
        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(1)
        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(1)
        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(1)
        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(1)
        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(1)
        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(1)
        driver.find_element_by_id('btnidc4ca4238a0b923820dcc509a6f75849b').click()
        driver.implicitly_wait(300000)

        length = len(driver.find_elements_by_xpath('/html/body/div/div[2]/div/div[2]/article/div/div[1]/div[1]/div[1]/div[2]/div/div/table/tbody/tr'))

        for i in range(length):
            url_data = '/html/body/div/div[2]/div/div[2]/article/div/div[1]/div[1]/div[1]/div[2]/div/div/table/tbody/tr['+ str(i+1) +']/td[1]'
            date_list.append(driver.find_element_by_xpath(url_data).text)
        
    with open('./data/dates/stock_holiday_krx.csv', 'w') as file:
        write = csv.writer(file)
        write.writerow(date_list)

    return date_list
'''    
