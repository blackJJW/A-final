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

class news_act:
    def __init__(self, company_name, file_name, maxpage):
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
        news_df.to_csv('./data/news/cr_article/kh_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

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

        driver.implicitly_wait(180)


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

        driver.implicitly_wait(180) # 10초대기


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
        news_df.to_csv('./data/news/cr_article/ni_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

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

        driver.implicitly_wait(180) # 10초대기


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
        news_df.to_csv('./data/news/cr_article/da_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

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

        driver.implicitly_wait(180) # 10초대기


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
        news_df.to_csv('./data/news/cr_article/mi_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

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

        driver.implicitly_wait(180) # 10초대기


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
                title = driver.find_element_by_class_name('atit2').text
            
                news_date = driver.find_element_by_xpath('/html/body/div[2]/div[6]/div[2]/div[2]/div/div[2]/span[1]/span').text
                
                news_article_body = driver.find_element_by_class_name('S20_v_article').text
                
                if title is None:
                    title = driver.find_element_by_class_name('articleTitleDiv').text
                    news_date = driver.find_element_by_class_name('articleDay').text
                    news_article_body = driver.find_element_by_class_name('articleDiv').text
                    
                
            except:
                news_article_body = None
                news_date = None
                title = None
            
            article_list.append(news_article_body)
            date_list.append(news_date)
            title_list.append(title)

        news_df = pd.DataFrame({"title" : title_list, "date" : date_list, "article" : article_list})
        news_df.to_csv('./data/news/cr_article/seoul_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

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

        driver.implicitly_wait(180) # 10초대기


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
        news_df.to_csv('./data/news/cr_article/at_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

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

        driver.implicitly_wait(180) # 10초대기


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
        news_df.to_csv('./data/news/cr_article/hg_'+self.file_name+'_news_article.csv', index = False, encoding ="utf-8")

        print("첫 날짜 : "+news_df["date"][0])
        print("끝 날짜 : "+news_df["date"][len(news_df)-1])
        
        


news_act('오뚜기', '오뚜기.csv')




