from konlpy.tag import Kkma
import pandas as pd
from tqdm import tqdm
import csv
import json

kkma = Kkma()

class Noun_Analysis_1:
    def __init__(self, senti_file_name):
        self.senti_file_name = senti_file_name
        print("AnalysisNoun - Noun_Analysis_1  Start")
        self.gen_noun_df()
        
    def gen_noun_df(self):
        print("AnalysisNoun - Noun_Analysis_1 - gen_noun_df  Start")
        print("----- reading company_senti start -----")
        company_senti = pd.read_csv('./data/dict/'+self.senti_file_name, encoding="cp949")
        print("----- reading company_senti Complete -----") 

        y = 0
        print("----- selecting columns Start -----")
        noun_df = pd.DataFrame(columns=["index", "nouns"])
        print("----- selecting columns Complete -----")

        print("----- extracting nouns Start -----")
        for y in tqdm(range(len(company_senti))):
          nouns_list = kkma.nouns(company_senti['article'][y])
          nouns_list = set(nouns_list)
          for n in nouns_list:
            data_insert = {"index":company_senti["index"][y], "nouns": n}
            noun_df = noun_df.append(data_insert, ignore_index=True)
        print("----- extracting nouns Complete -----")

        print("----- saving noun_df Start -----")
        noun_df.to_csv('./data/nouns/noun_df/'+self.senti_file_name+'_noun_df.csv', encoding="cp949")
        print("----- saving noun_df Complete -----")
        print("AnalysisNoun - Noun_Analysis_1 - gen_noun_d  Done")
        print("AnalysisNoun - Noun_Analysis_1  Done")

class Noun_Analysis_2:
    def __init__(self, senti_file_name, noun_df_file_name):
        print("AnalysisNoun - Noun_Analysis_2 Start")
        self.senti_file_name = senti_file_name
        self.noun_df_file_name = noun_df_file_name
        
        self.gen_nouns_freq()
        
    def gen_nouns_freq(self):
        print("AnalysisNoun - Noun_Analysis_2 - gen_nouns_freq  Start")
        print("----- reading csv Start -----")
        company_senti = pd.read_csv('./data/dict/'+self.senti_file_name, encoding="cp949")
        noun_df = pd.read_csv('./data/nouns/noun_df/'+self.noun_df_file_name, encoding="cp949")
        print("----- reading csv Complete -----")

        print("----- selecting columns Start -----")
        updown_dict = dict(zip(company_senti["index"], company_senti["up/down"]))
        print("----- selecting columns Done -----")
        nouns_freq = dict()

        print("----- calculating features Start -----")
        for i in tqdm(range(len(noun_df))):
           nouns_freq.setdefault(noun_df["nouns"][i], { 'freq':0, 'up': 0, 'down':0, 'same':0})
           nouns_freq[noun_df["nouns"][i]]['freq'] += 1

           if updown_dict[noun_df["index"][i]] == 1:
             nouns_freq[noun_df["nouns"][i]]['up'] += 1

           elif updown_dict[noun_df["index"][i]] == 0:
             nouns_freq[noun_df["nouns"][i]]['same'] += 1

           elif updown_dict[noun_df["index"][i]] == -1:
             nouns_freq[noun_df["nouns"][i]]['down'] += 1
        print("----- calculating features Done -----")

        print("----- creating ratios Start -----")
        for k in tqdm(nouns_freq.keys()):
           nouns_freq[k]['posRatio'] = nouns_freq[k]['up'] / nouns_freq[k]['freq']
           nouns_freq[k]['negRatio'] = nouns_freq[k]['down'] / nouns_freq[k]['freq']
           nouns_freq[k]['sameRatio'] = nouns_freq[k]['same'] / nouns_freq[k]['freq']
        print("----- creating ratios Done -----")

        print("----- setting freq / dic Start -----")
        nouns_freq = pd.DataFrame.from_dict(nouns_freq, orient='columns')

        nouns_dic = nouns_freq.transpose()
        nouns_dic.index.name= "noun"

        nouns_dic = nouns_dic[nouns_dic.freq != 1]

        nouns_dic_del_0 = nouns_dic[(nouns_dic['posRatio'] <= 0.0005) & (nouns_dic['negRatio'] <= 0.0005)].index

        nouns_dic_del = nouns_dic.drop(nouns_dic_del_0)
        print("----- setting freq / dic Done -----")

        c=[]
        print("----- setting None Start -----")
        for i in tqdm(range(len(nouns_dic_del))):
          if len(nouns_dic_del.index[i]) < 2:
            c.append(i)
            #freq	up	down	same	posRatio	negRatio
            nouns_dic_del.at[nouns_dic_del.index[i],"freq"] = None
            nouns_dic_del.at[nouns_dic_del.index[i],"up"] = None
            nouns_dic_del.at[nouns_dic_del.index[i],"down"] = None
            nouns_dic_del.at[nouns_dic_del.index[i],"same"] = None
            nouns_dic_del.at[nouns_dic_del.index[i],"posRatio"] = None
            nouns_dic_del.at[nouns_dic_del.index[i],"negRatio"] = None
        print("----- setting None Done -----")
        
        nouns_dic_del = nouns_dic_del.dropna(axis=0) # 결측값이 있는 행 전체 삭제

        print("----- creating 'prob' column Start -----")
        d = []
        e = []
        for i in tqdm(range(len(nouns_dic_del))):
          d.append(max(nouns_dic_del['posRatio'][i], nouns_dic_del['negRatio'][i]) / (nouns_dic_del['negRatio'][i] + nouns_dic_del['posRatio'][i]))

        nouns_dic_del["prob"] = d
        
        print("----- creating 'prob' column Done -----")

        print("----- creating 'weight' column Start -----")
        for i in tqdm(range(len(nouns_dic_del))):
          e.append(nouns_dic_del["freq"][i] * nouns_dic_del["prob"][i])

        nouns_dic_del["weight"] = e
        print("----- creating 'weight' column Done -----")

        print("----- saving json file Start -----")
        nouns_dic_del.to_json('./data/nouns/nouns_freq/'+self.senti_file_name+'_nouns_freq.json', orient= 'index')
        print("----- saving json file Done -----")
        print("AnalysisNoun - Noun_Analysis_2 - gen_nouns_freq  Done")
        print("AnalysisNoun - Noun_Analysis_2  Done")
        
class Pos_Neg_Points:
    def __init__(self, senti_file_name, nouns_freq_file_name):
        print("AnalysisNoun - Pos_Neg_Points  Start")
        self.senti_file_name = senti_file_name
        self.nouns_freq_name = nouns_freq_file_name
        
        self.pos_neg_points()
        
    def pos_neg_points(self):
        print("AnalysisNoun - Pos_Neg_Points - pos_neg_points  Start")
        
        print("----- reading files Start -----")
        a_article = pd.read_csv("./data/dict/"+self.senti_file_name, encoding="cp949")
        with open("./data/nouns/nouns_freq/"+self.nouns_freq_name, 'r') as f:
            nouns_freq = json.load(f)
        print("----- reading files Done -----")
        
        p_list = []
        n_list = []
        
        print("----- calculating points Start -----")
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
        print("----- calculating points Done -----")
        a_article["sumPos"] = p_list
        a_article["sumNeg"] = n_list

        print("----- saving csv Start -----")
        a_article.to_csv('./data/result/'+self.senti_file_name+'_result.csv', index=True, encoding= 'cp949')
        print("----- saving csv Done -----")
        print("AnalysisNoun - Pos_Neg_Points - pos_neg_points  Done")
        print("AnalysisNoun - Pos_Neg_Points  Done")
        