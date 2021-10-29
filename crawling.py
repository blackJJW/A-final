from selenium import webdriver
from selenium.webdriver.support.ui import Select
import chromedriver_autoinstaller
from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm
import time

def url_crawler(company_name, company_code, maxpage, file_name):
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
        
        
    df_result.to_csv('./data/news/links/'+file_name+'_links.csv', mode='w', encoding='utf-8-sig')

def article_crawler(file_name):

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

    df = pd.read_csv('./data/news/links/'+file_name)

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
    news_df.to_csv('./data/news/cr_article/'+file_name+'_news_article.csv', index = False, encoding ="utf-8")

    print("첫 날짜 : "+news_df["date"][0])
    print("끝 날짜 : "+news_df["date"][len(news_df)-1])
    driver.close()

#-------------------------주식 데이터 다운로드--------------------------------------------------------
def download_stock_data(stock_code, date1, date2, dir_path):
    chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0] #크롬 드라이버 버전 확인

    #-- 'USB : 시스템에 부착된 장치가 작동하지 않습니다' 오류 회피 ----------------- 
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    #----------------------------------------------------------------------------

    #---------다운로드 경로 설정 -------------------------
    options.add_experimental_option("prefs", {
      "download.default_directory": dir_path,
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

    input_code = driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[1]/td/div/div/p/input')
    input_code.clear()
    input_code.send_keys(stock_code)

    find_code = driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[1]/td/div/div/p/img')
    find_code.click()
    driver.implicitly_wait(5)

    date_1 = driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[2]/td/div/div/input[1]')
    date_1.clear()
    date_1.send_keys(date1)
    time.sleep(1)

    date_2 = driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[2]/td/div/div/input[2]')
    date_2.clear()
    date_2.send_keys(date2)
    time.sleep(1)

    driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[1]/div/table/tbody/tr[2]/td/a').click()
    time.sleep(2)

    driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[2]/div/p[2]/button[2]/img').click()
    driver.find_element_by_xpath('/html/body/div[2]/section[2]/section/section/div/div/form/div[2]/div[2]/div[2]/div/div[2]').click()
    time.sleep(2)

    driver.close()

