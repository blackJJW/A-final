from konlpy.tag import Kkma
#from eunjeon import Mecab
import pandas as pd
from tqdm import tqdm
import csv
import json

kkma = Kkma()

def gen_noun_df(file_name):
    company_senti = pd.read_csv('./data/dict/'+file_name, encoding="cp949") 

    y = 0

    noun_df = pd.DataFrame(columns=["index", "nouns"])

    for y in tqdm(range(len(company_senti))):
      nouns_list = kkma.nouns(company_senti['article'][y])
      nouns_list = set(nouns_list)
      for n in nouns_list:
        data_insert = {"index":company_senti["index"][y], "nouns": n}
        noun_df = noun_df.append(data_insert, ignore_index=True)

    noun_df.to_csv('./data/nouns/noun_df/'+file_name+'_noun_df.csv', encoding="cp949")

def gen_nouns_freq(senti_file_name, noun_df_file_name):

    company_senti = pd.read_csv('./data/dict/'+senti_file_name, encoding="cp949")
    noun_df = pd.read_csv('./data/nouns/noun_df/'+noun_df_file_name, encoding="cp949")

    updown_dict = dict(zip(company_senti["index"], company_senti["up/down"]))

    nouns_freq = dict()

    for i in tqdm(range(len(noun_df))):
       nouns_freq.setdefault(noun_df["nouns"][i], { 'freq':0, 'up': 0, 'down':0, 'same':0})
       nouns_freq[noun_df["nouns"][i]]['freq'] += 1

       if updown_dict[noun_df["index"][i]] == 1:
         nouns_freq[noun_df["nouns"][i]]['up'] += 1

       elif updown_dict[noun_df["index"][i]] == 0:
         nouns_freq[noun_df["nouns"][i]]['same'] += 1

       elif updown_dict[noun_df["index"][i]] == -1:
         nouns_freq[noun_df["nouns"][i]]['down'] += 1

    for k in tqdm(nouns_freq.keys()):
       nouns_freq[k]['posRatio'] = nouns_freq[k]['up'] / nouns_freq[k]['freq']
       nouns_freq[k]['negRatio'] = nouns_freq[k]['down'] / nouns_freq[k]['freq']
       nouns_freq[k]['sameRatio'] = nouns_freq[k]['same'] / nouns_freq[k]['freq']
               
    nouns_freq = pd.DataFrame.from_dict(nouns_freq, orient='columns')

    nouns_dic = nouns_freq.transpose()
    nouns_dic.index.name= "noun"

    nouns_dic = nouns_dic[nouns_dic.freq != 1]

    nouns_dic_del_0 = nouns_dic[(nouns_dic['posRatio'] <= 0.0005) & (nouns_dic['negRatio'] <= 0.0005)].index

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

    nouns_dic_del = nouns_dic_del.dropna(axis=0) # 결측값이 있는 행 전체 삭제
    
    d = []
    e = []
    for i in range(len(nouns_dic_del)):
      d.append(max(nouns_dic_del['posRatio'][i], nouns_dic_del['negRatio'][i]) / (nouns_dic_del['negRatio'][i] + nouns_dic_del['posRatio'][i]))

    nouns_dic_del["prob"] = d
    
    for i in range(len(nouns_dic_del)):
      e.append(nouns_dic_del["freq"][i] * nouns_dic_del["prob"][i])
      
    nouns_dic_del["weight"] = e
    
    nouns_dic_del.to_json('./data/nouns/nouns_freq/'+senti_file_name+'_nouns_freq.json', orient= 'index')

     
def pos_neg_points(file_name, article, nouns_freq):
    a_article = pd.read_csv("./data/dict/"+article, encoding="cp949")

    with open("./data/nouns/nouns_freq/"+nouns_freq, 'r') as f:
        nouns_freq = json.load(f)

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
        
    a_article.to_csv('./data/result/'+file_name+'_result.csv', index=True, encoding= 'cp949')
