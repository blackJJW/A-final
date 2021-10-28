#---------custom modules---------------
import gen_df
import crawling
import cust_noun
#--------------------------------------
import pathlib
import os, glob
import os.path

#------------- 주식데이터 다운로드 path------------------------------------
current_path = str(pathlib.Path(__file__).parent.absolute())
download_path = current_path+'\data\stock'
#-------------------------------------------------------------------------

#-----------정규분포 확률 변수----------
prob = 0.75
#--------------------------------------

def display_dir_path(dir_name): # dir 내 파일 목록 출력
    file_dir = "./data/"+dir_name 
    list_files = os.listdir(file_dir)

    print('\n')
    print('*'*10+'데이터 파일 목록'+'*'*10)

    for i in list_files :
        print(i)
    print('*'*35)


def main_menu():
    print("1. 뉴스 크롤링")
    print("2. 주식 크롤링")
    print("3. 데이터 프레임 생성")
    print("4. 명사 추출")
    print("5. TEST")
    print("6. 성능비교")
    print("7. 종료")

    menu = input("메뉴 선택 : ")

    return int(menu)

def menu_1():
    print("1. 뉴스 링크 크롤링")
    print("2. 뉴스 기사 크롤링")
    print("3. 메인메뉴")

    menu_1 = input("메뉴 선택 : ")

    return int(menu_1)

def menu_2():
    print("1. 주식 데이터 다운로드")
    print("2. 메인메뉴")

    menu_2 = input("메뉴 선택 : ")

    return int(menu_2)

def menu_3():
    print("1. 데이터 프레임 생성")
    print("2. 메인메뉴")

    menu_3 = input("메뉴 선택 : ")

    return int(menu_3)

def menu_4():
    print("1. 명사 추출")
    print("2. 긍부정 지수 계산")
    print("3. 메인메뉴")

    menu_4 = input("메뉴 선택 : ")

    return int(menu_4)


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
                
        elif menu == 2: # 주식 데이터 다운로드
            while 1:
                menu_b = menu_2()

                if menu_b == 1:
                    print("종목 코드를 입력하시오. ex) 006300")
                    c_code = input("종목 코드 : ")
                    print("시작 날짜를 입력하시오. ex) 20201014")
                    date_1 = input("시작 날짜 : ")
                    print("끝 날짜를 입력하시오. ex) 20211014")
                    date_2 = input("끝 날짜 : ")

                    crawling.download_stock_data(c_code, date_1, date_2, download_path)
                
                elif menu_b == 2:
                    break

        elif menu == 3: # 데이터 프레임 생성
            while 1:
                menu_c = menu_3()
                if menu_c == 1:

                    display_dir_path("stock")

                    stock_file_name = input("주식데이터 파일명을 입력하시오.(확장자 명 포함 필수) : ")
                    company_data_df_sorted = gen_df.gen_stock_data_df(stock_file_name)
                    total_df = gen_df.gen_total_df(company_data_df_sorted, prob)

                    display_dir_path("news")
                    
                    news_file_name = input("뉴스데이터 파일명을 입력하시오.(확장자 명 포함 필수): ")
                    news_data_df = gen_df.gen_news_data_df(news_file_name)
                    
                    senti_name = input("완성된 데이터 프레임의 이름을 입력하시오. (확장자 명 미 포함) : ")

                    gen_df.gen_senti(senti_name, news_data_df, total_df)

                elif menu_c == 2:
                    break

        elif menu == 4:
            while 1:
                menu_d = menu_4()
                if menu_d == 1:

                    display_dir_path("dict")

                    senti_name = input("파일 명을 입력하시오.(확장자명 필수) : ")
                    cust_noun.gen_noun_df(senti_name)
                
                elif menu_d == 2:

                    display_dir_path("dict")
                    
                    senti_name = input("파일 명을 입력하시오.(확장자명 필수) : ")

                    display_dir_path("nouns")
                    
                    noun_df_name = input("파일 명을 입력하시오.(확장자명 필수) : ")

                    cust_noun.gen_nouns_freq(senti_name, noun_df_name)

                elif menu_d == 3:
                    break

        elif menu == 5:
            pass
        elif menu == 6:
            pass
        elif menu == 7:
            break

run()




































