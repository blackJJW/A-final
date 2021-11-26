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
import datetime
from datetime import date, datetime

# --------- 날짜, 시간 데이터 format -----------------
format = ' %Y.%m.%d %H:%M' # ex) ' 2021.10.20 17:20'
format_l = '%Y.%m.%d %H:%M'
format_s = '%Y/%m/%d'      # ex) '2021/10/20'
format_t = '%Y-%m-%d'      # ex) '2021-10-20'
time_format ='%H:%M:%S'    # ex) '17:20:00'
format_ni = '%Y-%m-%d %H:%M:%S'
format_at = '%Y. %m. %d. %H:%M'
format_seoul = '%Y-%m-%d %H:%M'
# ----------------------------------------------------

class News_Act:
    def __init__(self, company_name, file_name, maxpage):
        print("NewsArticleDFProcessing - News_Act  Start")
        self.company_name = company_name
        self.maxpage = maxpage
        self.file_name = file_name
        
        KH_news(self.company_name,self.maxpage, self.file_name)
        NI_news(self.company_name,self.maxpage, self.file_name)
        DA_news(self.company_name,self.maxpage, self.file_name)
        MI_news(self.company_name,self.maxpage, self.file_name)
        Seoul_news(self.company_name, self.maxpage, self.file_name)
        AT_news(self.company_name, self.maxpage, self.file_name)
        HG_news(self.company_name, self.maxpage, self.file_name)
        
        print("NewsArticleDFProcessing - News_Act  Done")
    
    def selenium_set(self):
        print("------ setting selenium  Start -----")
        chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0] #크롬 드라이버 버전 확인

        #-- 'USB : 시스템에 부착된 장치가 작동하지 않습니다' 오류 회피 ---------------- 
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        #----------------------------------------------------------------------------
        options.add_argument('headless') # headless 모드 설정
        options.add_argument("disable-gpu")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64 Trident/7.0; rv:11.0) like Gecko")
        options.add_argument("window-size=1440x900")
        prefs = {'profile.default_content_setting_values': {'cookies' : 2, 'images': 2, 'plugins' : 2, 'popups': 2, 
                                                            'geolocation': 2, 'notifications' : 2, 'auto_select_certificate': 2, 
                                                            'fullscreen' : 2, 'mouselock' : 2, 'mixed_script': 2, 'media_stream' : 2, 
                                                            'media_stream_mic' : 2, 'media_stream_camera': 2, 'protocol_handlers' : 2, 
                                                            'ppapi_broker' : 2, 'automatic_downloads': 2, 'midi_sysex' : 2, 'push_messaging' : 2, 
                                                            'ssl_cert_decisions': 2, 'metro_switch_to_desktop' : 2, 'protected_media_identifier': 2, 
                                                            'app_banner': 2, 'site_engagement' : 2, 'durable_storage' : 2}}   
        options.add_experimental_option('prefs', prefs)        
    
        try:
            driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)
        except:
            chromedriver_autoinstaller.install(True) # 크롬 드라이버 자동 설치
            driver = webdriver.Chrome(f'./chromedriver/{chrome_ver}/chromedriver.exe', options = options)

        driver.implicitly_wait(180) # 5초대기
        print("------ setting selenium  Done -----")
        
        return driver
      
class KH_news:
    def __init__(self, company_name, maxpage, file_name):
        print("경향신문 크롤링")
        self.company_name= company_name
        self.maxpage = maxpage
        self.file_name = file_name
    
        self.kh_news_url_crawler()
        self.kh_article_crawler()


    def kh_news_url_crawler(self):
        link_result =[]
        page = 0

        for page in tqdm(range(int(self.maxpage))): 
        
            url = 'https://search.khan.co.kr/search.html?stb=khan&q=' + self.company_name + '&pg=' + str(page+1) +'&sort=1'
            source_code = requests.get(url).text
            html = BeautifulSoup(source_code, "html.parser")

            # 뉴스 링크
            links = html.find_all('dl', 'phArtc')

            for link in links: 
                add = link.find('a')['href']
                link_result.append(add)


            page += 1
    
        result= {"링크" : link_result} 
        df_result = pd.DataFrame(result)


        df_result.to_csv('./data/news/links/kh_news/'+self.file_name, mode='w', encoding='utf-8-sig')

    def kh_article_crawler(self):

        driver = News_Act.selenium_set(self)


        title_list = []
        date_list = []
        article_list = []

        df = pd.read_csv('./data/news/links/kh_news/'+self.file_name)

        for i in tqdm(range(len(df))):

            driver.get(df["링크"][i])
            driver.implicitly_wait(10)
 
            if len(driver.window_handles) >= 2:
                for i in range(len(driver.window_handles)-1):
                    try:
                        driver.switch_to_window(driver.window_handles[i+1])
                        driver.close()
                    except:
                        pass
                driver.switch_to_window(driver.window_handles[0])
            else:
                driver.switch_to_window(driver.window_handles[0])

            try:
                title = driver.find_element_by_xpath('/html/body/div/div[3]/div[2]/div[1]/h1').text

                news_date = driver.find_element_by_xpath('/html/body/div/div[3]/div[2]/div[2]/div[1]/div/em').text
                news_date = news_date.replace('입력 : ', '')

                news_article = driver.find_elements_by_xpath('/html/body/div/div[3]/div[3]/div[1]/p')
                l = []
                for i in news_article:
                    l.append(i.text)

            except:
                title = None
                news_date = None
                l = None
                
            title_list.append(title)
            date_list.append(news_date)
            article_list.append(l)


        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/kh_news/kh_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])
        driver.quit()


    def __init__(self, company_name, maxpage, file_name):
        print("국민일보 크롤링")
        self.company_name= company_name
        self.maxpage = maxpage
        self.file_name = file_name
    
        self.km_news_url_crawler()
        self.km_article_crawler()


    def km_news_url_crawler(self):
        link_result =[]
        page = 0
        a = self.company_name.encode('unicode_escape')
        re_a = str(a).replace('\\', '%')
        re_a = re_a.replace('b', '')
        re_a = re_a.replace("'", '')
        
        driver = News_Act.selenium_set(self)

        for page in tqdm(range(int(self.maxpage))): 
            
            url = 'http://www.kmib.co.kr/search/searchResult.asp?searchWord=' + re_a + '&pageNo=' + str(page+1) +'&period='
            driver.get(url)

            links = driver.find_elements_by_xpath('/html/body/div[1]/div[3]/div/div/div/div[2]/div[1]/div[3]/div/dl/dt')
            
            for i in links:
                aTag = i.find_element_by_tag_name('a')
                hrefs = aTag.get_attribute('href')
                link_result.append(hrefs)
    
            page += 1
    
        driver.quit()
    
        result= {"링크" : link_result} 
        df_result = pd.DataFrame(result)

        df_result.to_csv('./data/news/links/km_news/'+self.file_name, mode='w', encoding='utf-8-sig')

    def km_article_crawler(self):

        driver = News_Act.selenium_set(self)


        title_list = []
        date_list = []
        article_list = []

        df = pd.read_csv('./data/news/links/km_news/'+self.file_name)

        for i in tqdm(range(len(df))):

            driver.get(df["링크"][i])
            if len(driver.window_handles) >= 2:
                for i in range(len(driver.window_handles)-1):
                    try:
                        driver.switch_to_window(driver.window_handles[i+1])
                        driver.close()
                    except:
                        pass
                driver.switch_to_window(driver.window_handles[0])
            else:
                driver.switch_to_window(driver.window_handles[0])

            try:
                title = driver.find_element_by_xpath('/html/body/div[1]/div[3]/div[1]/div/div[1]/div/div[2]/h3').text
                title_list.append(title)

                news_date = driver.find_element_by_xpath('/html/body/div[1]/div[3]/div[1]/div/div[1]/div/div[2]/div/div[1]/span').text
                date_list.append(news_date)

                news_article = driver.find_element_by_xpath('/html/body/div[1]/div[3]/div[1]/div/div[2]/div[1]/div[1]/div[1]').text
                article_list.append(news_article)
            except:
                pass

        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/km_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])
        driver.quit()

class NI_news:
    def __init__(self, company_name, maxpage, file_name):
        print("내일신문 크롤링")
        self.company_name= company_name
        self.maxpage = maxpage
        self.file_name = file_name
    
        self.ni_news_url_crawler()
        self.ni_article_crawler()


    def ni_news_url_crawler(self):
        link_result =[]
        page = 0
        
        driver = News_Act.selenium_set(self)

        for page in tqdm(range(int(self.maxpage))): 
            
            url = 'https://www.naeil.com/search/?tsearch=' + self.company_name + '&gubun=body&tpage=' + str(page+1)
            driver.get(url)

            links = driver.find_elements_by_xpath('/html/body/div/div[1]/div[2]/div[1]/div/div/div/dl/dt')
            
            for i in links:
                aTag = i.find_element_by_tag_name('a')
                hrefs = aTag.get_attribute('href')
                link_result.append(hrefs)
    
            page += 1
    
        driver.quit()
    
        result= {"링크" : link_result} 
        df_result = pd.DataFrame(result)

        df_result.to_csv('./data/news/links/ni_news/'+self.file_name, mode='w', encoding='utf-8-sig')

    def ni_article_crawler(self):

        driver = News_Act.selenium_set(self)


        title_list = []
        date_list = []
        article_list = []

        df = pd.read_csv('./data/news/links/ni_news/'+self.file_name)

        for i in tqdm(range(len(df))):

            driver.get(df["링크"][i])
            
            if len(driver.window_handles) >= 2:
                for i in range(len(driver.window_handles)-1):
                    try:
                        driver.switch_to_window(driver.window_handles[i+1])
                        driver.close()
                    except:
                        pass
                driver.switch_to_window(driver.window_handles[0])
            else:
                driver.switch_to_window(driver.window_handles[0])

            try:
                title = driver.find_element_by_xpath('/html/body/div/div[1]/div[2]/div[1]/div/div[1]/h3').text

                news_date = driver.find_element_by_xpath('/html/body/div/div[1]/div[2]/div[1]/div/div[1]/div[2]').text
                news_date = news_date.replace(' 게재', '')

                news_article = driver.find_element_by_class_name('article').text
            except:
                title = None
                news_date = None
                news_article = None
            
            article_list.append(news_article)
            date_list.append(news_date)
            title_list.append(title)

        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/ni_news/ni_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])
        driver.quit()

class DA_news:
    def __init__(self, company_name, maxpage, file_name):
        print("동아일보 크롤링")
        self.company_name= company_name
        self.maxpage = maxpage
        self.file_name = file_name
    
        self.da_news_url_crawler()
        self.da_article_crawler()


    def da_news_url_crawler(self):
        link_result =[]
        page = 0
        
        driver = News_Act.selenium_set(self)

        for page in tqdm(range(int(self.maxpage))): 
            
            url = 'https://www.donga.com/news/search?p='+str((15*page) + 1)+'&query=' + self.company_name + '&check_news=2&more=1&sorting=1&search_date=1&v1=&v2=&range=1'
            driver.get(url)

            links = driver.find_elements_by_xpath('/html/body/div[3]/div[4]/div[3]/div[3]/div[1]/div/div[2]/p[1]')
            
            for i in links:
                aTag = i.find_element_by_tag_name('a')
                hrefs = aTag.get_attribute('href')
                link_result.append(hrefs)
    
            page += 1
    
        driver.quit()
    
        result= {"링크" : link_result} 
        df_result = pd.DataFrame(result)

        df_result.to_csv('./data/news/links/da_news/'+self.file_name, mode='w', encoding='utf-8-sig')

    def da_article_crawler(self):

        driver = News_Act.selenium_set(self)


        title_list = []
        date_list = []
        article_list = []

        df = pd.read_csv('./data/news/links/da_news/'+self.file_name)

        for i in tqdm(range(len(df))):

            driver.get(df["링크"][i])
            
            if len(driver.window_handles) >= 2:
                for i in range(len(driver.window_handles)-1):
                    try:
                        driver.switch_to_window(driver.window_handles[i+1])
                        driver.close()
                    except:
                        pass
                driver.switch_to_window(driver.window_handles[0])
            else:
                driver.switch_to_window(driver.window_handles[0])

            try:
                title = driver.find_element_by_xpath('/html/body/div[5]/div[1]/div/div[1]/h2').text
                title_list.append(title)

                news_date = driver.find_element_by_xpath('/html/body/div[5]/div[1]/div/div[1]/p[2]/span[1]').text
                date_list.append(news_date)

                news_article = driver.find_element_by_xpath('/html/body/div[5]/div[1]/div/div[3]/div[1]/div/div[1]/div[1]').text
                article_list.append(news_article)
            except:
                news_article = None
                news_date = None
                title = None
            
            article_list.append(news_article)
            date_list.append(news_date)
            title_list.append(title)

        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/da_news/da_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])
        driver.quit()
        
class MI_news:
    def __init__(self, company_name, maxpage, file_name):
        print("매일일보 크롤링")
        self.company_name= company_name
        self.maxpage = maxpage
        self.file_name = file_name
    
        self.mi_news_url_crawler()
        self.mi_article_crawler()


    def mi_news_url_crawler(self):
        link_result =[]
        page = 0
        
        driver = News_Act.selenium_set(self)

        for page in tqdm(range(int(self.maxpage))): 
            
            url = 'https://www.m-i.kr/news/articleList.html?page='+str(page)+'&box_idxno=&sc_area=A&view_type=tm&sc_word=' + self.company_name
            driver.get(url)

            links = driver.find_elements_by_xpath('/html/body/div[1]/div/div[2]/div/section/div[3]/div[2]/div[2]/div/section/article/div[2]/section/div/div/div')
            
            for i in links:
                aTag = i.find_element_by_tag_name('a')
                hrefs = aTag.get_attribute('href')
                link_result.append(hrefs)
    
            page += 1
    
        driver.quit()
    
        result= {"링크" : link_result} 
        df_result = pd.DataFrame(result)

        df_result.to_csv('./data/news/links/mi_news/'+self.file_name, mode='w', encoding='utf-8-sig')

    def mi_article_crawler(self):

        driver = News_Act.selenium_set(self)


        title_list = []
        date_list = []
        article_list = []

        df = pd.read_csv('./data/news/links/mi_news/'+self.file_name)

        for i in tqdm(range(len(df))):

            driver.get(df["링크"][i])
            
            if len(driver.window_handles) >= 2:
                for i in range(len(driver.window_handles)-1):
                    try:
                        driver.switch_to_window(driver.window_handles[i+1])
                        driver.close()
                    except:
                        pass
                driver.switch_to_window(driver.window_handles[0])
            else:
                driver.switch_to_window(driver.window_handles[0])

            try:
                title = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/section/div[3]/header/div/div').text
                
                news_date = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/section/div[3]/header/section/div/ul/li[2]').text
                news_date = news_date.replace('승인 ', '')
                
                news_article_body = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/section/div[3]/div[3]/div/section/div/article[1]/div[2]')
                article = news_article_body.find_elements_by_tag_name('p')
                add = []
                for at in article:
                    add.append(at.text)

            except:
                add = None
                news_date = None
                title = None
            
            article_list.append(add)
            date_list.append(news_date)
            title_list.append(title)

        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/mi_news/mi_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])
        driver.quit()
        
class Seoul_news:
    def __init__(self, company_name, maxpage, file_name):
        print("서울신문 크롤링")
        self.company_name= company_name
        self.maxpage = maxpage
        self.file_name = file_name
    
        self.seoul_news_url_crawler()
        self.seoul_article_crawler()


    def seoul_news_url_crawler(self):
        link_result =[]
        page = 0
        
        driver = News_Act.selenium_set(self)

        for page in tqdm(range(int(self.maxpage))): 
            
            url = 'https://search.seoul.co.kr/index.php?scope=&sort=&cpCode=seoul&period=&sDate=&eDate=&keyword='+ self.company_name +'&iCategory=&pCategory=undefined&pageNum=' + str(page+1)
            driver.get(url)

            links = driver.find_elements_by_xpath('/html/body/div/div[3]/div[1]/div[3]/dl/dt')
            
            for i in links:
                aTag = i.find_element_by_tag_name('a')
                hrefs = aTag.get_attribute('href')
                link_result.append(hrefs)
    
            page += 1
    
        driver.quit()
    
        result= {"링크" : link_result} 
        df_result = pd.DataFrame(result)

        df_result.to_csv('./data/news/links/seoul_news/'+self.file_name, mode='w', encoding='utf-8-sig')

    def seoul_article_crawler(self):

        driver = News_Act.selenium_set(self)


        title_list = []
        date_list = []
        article_list = []

        df = pd.read_csv('./data/news/links/seoul_news/'+self.file_name)

        for i in tqdm(range(len(df))):

            driver.get(df["링크"][i])
            
            if len(driver.window_handles) >= 2:
                for i in range(len(driver.window_handles)-1):
                    try:
                        driver.switch_to_window(driver.window_handles[i+1])
                        driver.close()
                    except:
                        pass
                driver.switch_to_window(driver.window_handles[0])
            else:
                driver.switch_to_window(driver.window_handles[0])

            try:
                title = driver.find_element_by_xpath('/html/body/div[2]/div[6]/div[2]/div[2]/h1').text
            
                news_date = driver.find_element_by_xpath('/html/body/div[2]/div[6]/div[2]/div[2]/div/div[2]/span[1]/span').text
                
                news_article_body = driver.find_element_by_xpath('/html/body/div[2]/div[6]/div[3]/div[1]/div/div[1]').text
                
            except:
                news_article_body = None
                news_date = None
                title = None
            
            article_list.append(news_article_body)
            date_list.append(news_date)
            title_list.append(title)

        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/seoul_news/seoul_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])
        driver.quit()

class AT_news:
    def __init__(self, company_name, maxpage, file_name):
        print("아시아투데이 크롤링")
        self.company_name= company_name
        self.maxpage = maxpage
        self.file_name = file_name
    
        self.at_news_url_crawler()
        self.at_article_crawler()


    def at_news_url_crawler(self):
        link_result =[]
        page = 0
        
        driver = News_Act.selenium_set(self)

        for page in tqdm(range(int(self.maxpage))): 
            
            url = 'https://www.asiatoday.co.kr/kn_search.php?ob=a&page='+str(page+1)+'&period=a&period_sd=&period_ed=&scope=a&de_a=&de_b=&de_c=&sword=' + self.company_name
            driver.get(url)

            links = driver.find_elements_by_xpath('/html/body/div[1]/div[3]/div/div[2]/div[2]/dl/dd')
            
            for i in links:
                aTag = i.find_element_by_tag_name('a')
                hrefs = aTag.get_attribute('href')
                link_result.append(hrefs)
    
            page += 1
    
        driver.quit()
    
        result= {"링크" : link_result} 
        df_result = pd.DataFrame(result)

        df_result.to_csv('./data/news/links/at_news/'+self.file_name, mode='w', encoding='utf-8-sig')

    def at_article_crawler(self):

        driver = News_Act.selenium_set(self)

        title_list = []
        date_list = []
        article_list = []

        df = pd.read_csv('./data/news/links/at_news/'+self.file_name)

        for i in tqdm(range(len(df))):

            driver.get(df["링크"][i])
            
            if len(driver.window_handles) >= 2:
                for i in range(len(driver.window_handles)-1):
                    try:
                        driver.switch_to_window(driver.window_handles[i+1])
                        driver.close()
                    except:
                        pass
                driver.switch_to_window(driver.window_handles[0])
            else:
                driver.switch_to_window(driver.window_handles[0])

            try:
                title = driver.find_element_by_class_name('section_top_box').text

                news_date = driver.find_element_by_class_name('wr_day').text
                news_date = news_date.replace('기사승인 ', '')
                

                news_article_body = driver.find_element_by_class_name('news_bm').text
                
            except:
                title = None
                news_date = None
                news_article_body = None
            
            title_list.append(title)
            date_list.append(news_date)
            article_list.append(news_article_body)

        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/at_news/at_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])
        driver.quit()
        
class HG_news:
    def __init__(self, company_name, maxpage, file_name):
        print("한겨레 크롤링")
        self.company_name= company_name
        self.maxpage = maxpage
        self.file_name = file_name
    
        self.hg_news_url_crawler()
        self.hg_article_crawler()


    def hg_news_url_crawler(self):
        link_result =[]
        page = 0
        
        driver = News_Act.selenium_set(self)

        for page in tqdm(range(int(self.maxpage))): 
            
            url = 'https://search.hani.co.kr/Search?command=query&keyword='+self.company_name +'&media=news&submedia=&sort=d&period=all&datefrom=1988.01.01&dateto=2021.11.08&pageseq=' + str(page)
            driver.get(url)

            links = driver.find_elements_by_xpath('/html/body/div[3]/div/div[1]/div[3]/ul/li/dl/dt')
            
            for i in links:
                aTag = i.find_element_by_tag_name('a')
                hrefs = aTag.get_attribute('href')
                link_result.append(hrefs)
    
            page += 1
    
        driver.quit()
    
        result= {"링크" : link_result} 
        df_result = pd.DataFrame(result)

        df_result.to_csv('./data/news/links/hg_news/'+self.file_name, mode='w', encoding='utf-8-sig')

    def hg_article_crawler(self):

        driver = News_Act.selenium_set(self)


        title_list = []
        date_list = []
        article_list = []

        df = pd.read_csv('./data/news/links/hg_news/'+self.file_name)

        for i in tqdm(range(len(df))):

            driver.get(df["링크"][i])
            
            if len(driver.window_handles) >= 2:
                for i in range(len(driver.window_handles)-1):
                    try:
                        driver.switch_to_window(driver.window_handles[i+1])
                        driver.close()
                    except:
                        pass
                driver.switch_to_window(driver.window_handles[0])
            else:
                driver.switch_to_window(driver.window_handles[0])
                
            try:
                title = driver.find_element_by_xpath('/html/body/div[4]/div[2]/div[2]/h4/span').text

                news_date = driver.find_element_by_xpath('/html/body/div[4]/div[2]/div[2]/p[2]/span[1]').text
                news_date = news_date.replace('등록 :', '')

                news_article_body = driver.find_element_by_xpath('/html/body/div[4]/div[2]/div[3]/div[1]/div/div/div[2]/div/div[2]').text

            except:
                title = None
                news_date = None
                news_article_body = None
        
            title_list.append(title)
            date_list.append(news_date)
            article_list.append(news_article_body)
            
        driver.quit()
        
        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/hg_news/hg_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])

class News_DF_Processing:
    def __init__(self, file_name):
        print("NewsArticleDFProcessing - News_DF_Processing  Start")
        self.file_name = file_name
        
        self.open_news()
        self.merge_news()
        self.gen_news_data_df()
        
    def open_news(self):
        print("NewsArticleDFProcessing - News_DF_Processing - open_news  Start")
        try:
            print("----- opening kh_news Start -----")
            kh_news = pd.read_csv('./data/news/cr_article/kh_news/kh_'+self.file_name+'_news_article.csv', encoding ="utf-8")
            kh_n = kh_news.copy()
            kh_n = kh_n.dropna()
            kh_n = kh_n.reset_index(drop=False)
            self.kh_n = kh_n.drop(['index'], axis = 1)
            print("----- opening kh_news Done -----")
        except:
            print("----- kh_news = None -----")
            self.kh_n = None
        try:
            print("----- opening ni_news Start -----")
            ni_news = pd.read_csv('./data/news/cr_article/ni_news/ni_'+self.file_name+'_news_article.csv', encoding ="utf-8")
            ni_n = ni_news.copy()
            ni_n = ni_n.dropna()
            ni_n = ni_n.reset_index(drop=False)
            self.ni_n = ni_n.drop(['index'], axis = 1)        

            print("----- transforming date type Start -----")
            for i in tqdm(range(len(self.ni_n))):  
                a_1 = datetime.strptime(str(self.ni_n['date'][i]), format_ni)
                self.ni_n['date'][i] = str(datetime.strftime(a_1, format_l))
            print("----- transforming date type Done -----")
            print("----- opening ni_news Done -----")
        except:
            print("----- ni_news = None -----")
            self.ni_n = None
        try:
            print("----- opening da_news Start -----")
            da_news = pd.read_csv('./data/news/cr_article/da_news/da_'+self.file_name+'_news_article.csv', encoding ="utf-8")
            da_n = da_news.copy()
            da_n = da_n.dropna()
            da_n = da_n.reset_index(drop=False)
            self.da_n = da_n.drop(['index'], axis = 1)
            
            print("----- transforming date type Start -----")
            for i in tqdm(range(len(self.da_n))):
              a_1 = datetime.strptime(self.da_n['date'][i], format_ni)
              self.da_n['date'][i] = str(datetime.strftime(a_1, format_l))
            print("----- transforming date type Done -----")
            print("----- opening da_news Done -----")      
        except:
            print("----- da_news = None -----")
            self.da_n = None
        try:
            print("----- opening mi_news Start -----")
            mi_news = pd.read_csv('./data/news/cr_article/mi_news/mi_'+self.file_name+'_news_article.csv', encoding ="utf-8")
            mi_n = mi_news.copy()
            mi_n = mi_n.dropna()
            mi_n = mi_n.reset_index(drop=False)
            self.mi_n = mi_n.drop(['index'], axis = 1)   
            print("----- opening da_news Done -----")     
        except:
            print("----- da_news = None -----")
            self.mi_n = None
        try:
            print("----- opening seoul_news Start -----")
            seoul_news = pd.read_csv('./data/news/cr_article/seoul_news/seoul_'+self.file_name+'_news_article.csv', encoding ="utf-8")
            seoul_n = seoul_news.copy()
            seoul_n = seoul_n.dropna()
            seoul_n = seoul_n.reset_index(drop=False)
            self.seoul_n = seoul_n.drop(['index'], axis = 1)
            
            print("----- transforming date type Start -----")
            for i in tqdm(range(len(self.seoul_n))):
              a_1 = datetime.strptime(self.seoul_n['date'][i], format_seoul)
              self.seoul_n['date'][i] = str(datetime.strftime(a_1, format_l))  
            print("----- transforming date type Done -----")
            print("----- opening seoul_news Done -----")            
        except:
            print("----- seoul_news = None -----") 
            self.seoul_n = None
        try:
            print("----- opening at_news Start -----") 
            at_news = pd.read_csv('./data/news/cr_article/at_news/at_'+self.file_name+'_news_article.csv', encoding ="utf-8")
            at_n = at_news.copy()
            at_n = at_n.dropna()
            at_n = at_n.reset_index(drop=False)
            self.at_n = at_n.drop(['index'], axis = 1)        
            print("----- transforming date type Start -----")
            for i in tqdm(range(len(self.at_n))):
                a_1 = datetime.strptime(self.at_n['date'][i], format_at)
                self.at_n['date'][i] = str(datetime.strftime(a_1, format_l))
            print("----- transforming date type Done -----")
            print("----- opening at_news Done -----")      
        except:
            print("----- at_news = None -----")
            self.at_n = None
        try:
            print("----- opening hg_news Start -----")
            hg_news = pd.read_csv('./data/news/cr_article/hg_news/hg_'+self.file_name+'_news_article.csv', encoding ="utf-8")
            hg_n = hg_news.copy()
            hg_n = hg_n.dropna()
            hg_n = hg_n.reset_index(drop=False)
            self.hg_n = hg_n.drop(['index'], axis = 1) 
            print("----- transforming date type Start -----")
            for i in tqdm(range(len(self.hg_n))):
                a_1 = datetime.strptime(self.at_n['date'][i], format_seoul)
                self.at_n['date'][i] = str(datetime.strftime(a_1, format_l))
            print("----- transforming date type Done -----")
            print("----- opening hg_news Done -----")             
        except:
            print("----- hg_news = None -----")
            self.hg_n = None
            
        print("NewsArticleDFProcessing - News_DF_Processing - open_news  Done")
    
    def merge_news(self):
        print("NewsArticleDFProcessing - News_DF_Processing - merge_news  Start")
        print("----- concat dfs Start -----")
        result_news = pd.concat([self.ni_n, self.mi_n, self.kh_n, self.seoul_n,
                                 self.da_n, self.at_n, self.hg_n], ignore_index=True)
        
        print("----- concat dfs Done -----")
        result_news = result_news.dropna()
        result_news = result_news.reset_index(drop=False)
        result = result_news.drop(['index'], axis=1)

        print("----- saving csv Start -----")
        result.to_csv("./data/news/cr_article/result/"+self.file_name+"_news_article_result.csv", encoding="utf8", index=False)
        print("----- saving csv Done -----")
        print("NewsArticleDFProcessing - News_DF_Processing - merge_news  Done")
        
    #-----------회사 뉴스 데이터----------------------------------------------------------------------------------------------------------
    def gen_news_data_df(self):
        print("NewsArticleDFProcessing - News_DF_Processing - gen_news_data_df  Start")
        print("----- reading csv Start -----")
        # ---news csv파일 open --------------------------------------------
        data_df = pd.read_csv('./data/news/cr_article/result/'+self.file_name+"_news_article_result.csv", encoding="utf8")
        # ------------------------------------------------------------------
        print("----- reading csv Done -----")
        print("----- clearing strings Start -----")
        # ----------------정규식을 이용하여 기사 내용 중 불 필요한 부분 제거-----------------------------------------------------
        data_df['title'] = data_df['title'].apply(lambda x: re.sub(r'[^ a-zA-z ㅣㄱ-ㅣ가-힣]+', " ", str(x)))  
        data_df['article'] = data_df['article'].apply(lambda x: re.sub(r'[^ a-zA-z ㅣㄱ-ㅣ가-힣]+', " ", str(x)))
        data_df['article'] = data_df['article'].apply(lambda x: re.sub(r"오류를 우회하기 위한 함수 추가", " ", str(x) ))
        # ---------------------------------------------------------------------------------------------------------------------
        print("----- clearing strings Done -----")
        data_df_sorted = data_df.sort_values(by='date', axis = 0) # 'date'를 기준으로 오름차순 정렬하여 data_df_sorted로 저장
        print("----- saving csv Start -----")
        data_df_sorted.to_csv('./data/news/sorted_article/'+self.file_name+'_data_df_sorted.csv', index = False, encoding ="utf-8")
        print("----- saving csv Done -----")
        print("NewsArticleDFProcessing - News_DF_Processing - gen_news_data_df  Done")
        print("NewsArticleDFProcessing - News_DF_Processing  Done")
    # ---------------------------------------------------------------------------------------------------------------------------------------











 
 
 
 
 