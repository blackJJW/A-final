from konlpy.tag import Kkma
import pandas as pd
from tqdm import tqdm

kkma = Kkma()

# senti, nouns_sample, 

updown_dict = (zip(senti["index"], senti["up/down"]))

# nouns_sample read csv

nouns_freq = dict()

for i in range(len(nouns_sample)):
   nouns_freq.setdefault(nouns_sample["nouns"][i], { 'freq':0, 'up': 0, 'down':0, 'same':0})
   nouns_freq[nouns_sample["nouns"][i]]['freq'] += 1

# noun_df read csv

nouns_freq = dict()

for i in range(len(noun_df)):
   nouns_freq.setdefault(noun_df["nouns"][i], { 'freq':0, 'up': 0, 'down':0, 'same':0})
   nouns_freq[noun_df["nouns"][i]]['freq'] += 1
   if b[noun_df["index"][i]] == 1:
     nouns_freq[noun_df["nouns"][i]]['up'] += 1
   elif b[noun_df["index"][i]] == 0:
     nouns_freq[noun_df["nouns"][i]]['same'] += 1
   elif b[noun_df["index"][i]] == -1:
     nouns_freq[noun_df["nouns"][i]]['down'] += 1

for k in nouns_freq.keys():
   nouns_freq[k]['posRatio'] = nouns_freq[k]['up'] / nouns_freq[k]['freq']
   nouns_freq[k]['negRatio'] = nouns_freq[k]['down'] / nouns_freq[k]['freq']
   