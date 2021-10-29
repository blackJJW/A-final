from konlpy.tag import Kkma
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm
import json


with open('./data/nouns/GS건설20201026_senti.csv_nouns_freq.json', 'r') as f:
  data = json.load(f)
  
nouns_freq = pd.DataFrame.from_dict(data, orient='columns')

nouns_dic = nouns_freq.transpose()
nouns_dic.index.name= "noun"

nouns_dic = nouns_dic[nouns_dic.freq != 1]

nouns_dic_del_0 = nouns_dic[(nouns_dic['posRatio'] <= 0.05) & (nouns_dic['negRatio'] <= 0.05)].index
nouns_dic_del = nouns_dic.drop(nouns_dic_del_0)

c=[]

for i in range(len(nouns_dic_del)):
  if len(nouns_dic_del.index[i]) < 2:
    c.append(i)
    #freq	up	down	same	posRatio	negRatio
    nouns_dic_del.at[nouns_dic_del.index[i],"freq"] = None
    nouns_dic_del.at[nouns_dic_del.index[i],"up"] = None
    nouns_dic_del.at[nouns_dic_del.index[i],"down"] = None
    nouns_dic_del.at[nouns_dic_del.index[i],"same"] = None
    nouns_dic_del.at[nouns_dic_del.index[i],"posRatio"] = None
    nouns_dic_del.at[nouns_dic_del.index[i],"negRatio"] = None

nouns_dic_del = nouns_dic_del.dropna(axis=0)

nouns_dic_del.to_json('./data/nouns/Sk_train_noun_result.json', orient= 'index')

'''
kkma = Kkma()

a_article = pd.read_csv("./data/news/sorted_article/GS건설20211029_links.csv_news_article.csvdata_df_sorted.csv", encoding="utf8") 

with open("./data/nouns/GS건설20201026_senti.csv_nouns_freq.json", 'r') as f:
  nouns_freq = json.load(f)

y = 0

p_list = []
n_list = []

for i in tqdm(range(len(a_article))):
  noun_list = kkma.nouns(a_article["article"][i])    
  lst = set(noun_list)
  sumPos = 0
  sumNeg = 0
  for j in lst:
    if j in nouns_freq: 
      sumPos += nouns_freq[j]["posRatio"] 
      sumNeg += nouns_freq[j]["negRatio"]
  p_list.append(sumPos)
  n_list.append(sumNeg)


a_article["sumPos"] = p_list
a_article["sumNeg"] = n_list

print(a_article)

a_article.to_csv('./data/result/a_article_result.csv', index=True, encoding= 'cp949')
'''