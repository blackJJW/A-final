import pandas as pd

seoul_news = pd.read_csv('./data/news/cr_article/seoul_news/seoul_네이버.csv_news_article.csv', index = False,  encoding ="utf-8")

print(seoul_news)