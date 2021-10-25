from konlpy.tag import Kkma
import pandas as pd
from tqdm import tqdm

kkma = Kkma()

GS_senti = pd.read_csv("./data/dict/GS건설_sentiDic_1.csv", encoding="cp949") 

y = 0

noun_df = pd.DataFrame(columns=["index", "nouns"])

for y in tqdm(range(len(GS_senti))):
  nouns_list = kkma.nouns(GS_senti['article'][y])
  nouns_list = set(nouns_list)
  for n in nouns_list:
    data_insert = {"index":GS_senti["index"][y], "nouns": n}
    noun_df = noun_df.append(data_insert, ignore_index=True)
print(y)

noun_df.to_csv("./data/dict/noun_df.csv", encoding="cp949")


