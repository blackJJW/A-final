from konlpy.tag import Kkma
import pandas as pd
from tqdm import tqdm
import csv

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

    noun_df.to_csv('./data/nouns/'+file_name+'_noun_df.csv', encoding="cp949")

def gen_nouns_freq(senti_file_name, noun_df_file_name):

    company_senti = pd.read_csv('./data/dict/'+senti_file_name, encoding="cp949")
    noun_df = pd.read_csv('./data/nouns/'+noun_df_file_name, encoding="cp949")

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

    with open('./data/nouns/'+senti_file_name+'_nouns_freq.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(nouns_freq.keys())
      writer.writerow(nouns_freq.values())