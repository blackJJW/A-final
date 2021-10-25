#---------custom modules---------------
import cust_dates
import gen_df
import crawling
#--------------------------------------

# --------- 날짜, 시간 데이터 format -----------------
format = ' %Y.%m.%d %H:%M' # ex) ' 2021.10.20 17:20'
format_s = '%Y/%m/%d'      # ex) '2021/10/20'
format_t = '%Y-%m-%d'      # ex) '2021-10-20'
time_format ='%H:%M:%S'    # ex) '17:20:00'
# ----------------------------------------------------

#------------회사명 변수--------------
company_name = "GS건설"
#------------------------------------

#-----------기간 내 날짜수, 주말 리스트 생성 변수-----
start_date = "2020-10-14"
end_date = "2021-10-14"
#----------------------------------------------------

#-----------정규분포 확률 변수----------
prob = 0.75
#--------------------------------------

#-----------krx 휴장일 출력 기준 년도 변수-----------------
start_year = 2020
end_year = 2021
#--------------------------------------------------------


def main_menu():
    print("1. 뉴스 크롤링")
    print("2. 주식 크롤링")
    print("3. 데이터 프레임 생성")
    print("4. 모델 학습")
    print("5. TEST")
    print("6. 성능비교")
    print("7. 종료")

    menu = input("메뉴 선택 : ")

    return int(menu)

def menu_1():
    print("1. 뉴스 크롤링")
    print("2. 주식 크롤링")
    print("3. 메인메뉴")

    menu_1 = input("메뉴 선택 : ")

    return int(menu_1)


def run():
    while 1: # 메인메뉴 while
        menu = main_menu()

        if menu == 1:
            while 1:
                menu_a = menu_1()

                if menu_a == 1 :
                    print("1. 뉴스 링크 크롤링")
                    name = input("company name : ")
                    code = input("company code : ")
                    page = input("max page : ")
                    crawling.url_crawler(name, code, page)

                elif menu_a == 2:
                    print("2. 뉴스 기사 크롤링")
                    name = input("company name : ")

                    crawling.article_crawler(name)
                elif menu_a == 3:
                    break
                
        elif menu == 2:
            pass
        elif menu == 3:
            while 1:

                '''
                company_stock_data_sorted = gen_df.gen_stock_data_df(company_name)
                company_news_data_sorted = gen_df.gen_news_data_df(company_name)

                dates = cust_dates.date_range(start_date, end_date)

                data_total_df = gen_df.gen_total_df(company_stock_data_sorted, dates, prob)

                news_updown = gen_df.gen_news_updown(company_news_data_sorted, data_total_df)

                sentiDic = gen_df.gen_sentiDic(company_name, company_news_data_sorted, news_updown, data_total_df, start_year, end_year, start_date, end_date)
                '''
        elif menu == 4:
            pass
        elif menu == 5:
            pass
        elif menu == 6:
            pass
        elif menu == 7:
            break

run()




































