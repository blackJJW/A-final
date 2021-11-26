import pandas as pd
from tqdm import tqdm
from datetime import datetime

class Refine_Result:
    def __init__(self, result_file_name, stock_file_name):
        print("Gen_Stock_Pos_Neg - Refine_Result  Start")
        self.result_file_name = result_file_name
        self.stock_file_name = stock_file_name
        
        self.refine_result()
        
    def refine_result(self):
        print("Gen_Stock_Pos_Neg - Refine_Result - refine_result  Start")
        print("----- reading csv Start -----")
        company_result = pd.read_csv("./data/result/"+self.result_file_name, encoding="cp949")
        company_stock = pd.read_csv("./data/stock/"+self.stock_file_name, encoding="cp949")
        print("----- reading csv Done -----")
        company_stock = company_stock.sort_values(by = '일자', axis = 0)
        s_d = company_stock.reset_index(drop=False, inplace=False)

        date_list = list(company_result['dates'])

        d_list = []
        print("----- trainsforming date type  Start -----")
        for i in tqdm(range(len(date_list))):

          temp = datetime.strptime(date_list[i], '%Y.%m.%d %H:%M')
          temp_1 = datetime.strftime(temp, '%Y/%m/%d')
          d_list.append(temp_1)
        print("----- trainsforming date type  Done -----")
        print("----- setting columns Start -----")
        company_result_1 = company_result.copy()

        company_result_1['date'] = d_list

        company_result_2 = company_result_1.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
        company_result_2 = company_result_2[['index', 'title', 'article', 'dates', 'date', 'apply_date', 'ratio', 'up/down', 'sumPos', 'sumNeg']]
        c = sorted(list(set(list(company_result_2['date']))))
        print("----- setting columns Done -----")
        
        stock_list = list(s_d['일자'])

        print("----- trainsforming date type  Start -----")
        for i in tqdm(range(len(stock_list))):
          t = datetime.strptime(stock_list[i], '%Y/%m/%d')
          stock_list[i] = datetime.strftime(t, '%Y-%m-%d')
        print("----- trainsforming date type  Done -----")
        
        l = []
        pos = 0
        neg = 0
        
        print("----- filtering / calculating Start -----")
        for i in tqdm(range(len(c))):
          l_2 = []
          res_list = list(filter(lambda x: company_result_2['date'][x] == c[i], range(len(company_result_2))))

          for j in res_list:
            pos += company_result_2['sumPos'][j]
            neg += company_result_2['sumNeg'][j]

          l_2.append(c[i])
          l_2.append(pos/len(res_list))
          l_2.append(neg/len(res_list))
          pos = 0
          neg = 0
          l.append(l_2)
        print("----- filtering / calculating Done -----")
        
        l_df = pd.DataFrame(l, columns=['일자', 'sumPos', 'sumNeg'])

        print("----- merging df Start -----")
        pd_merge  = pd.merge(s_d, l_df, how='left')
        pd_merge_1 = pd_merge.dropna()
        print("----- merging df Done -----")
        s = pd_merge_1.reset_index()
        print("----- saving csv Start -----")
        s.to_csv('./data/stock_pos_neg/'+self.stock_file_name+'pos_neg_result.csv', encoding='cp949')
        print("----- saving csv Done -----")
        print("Gen_Stock_Pos_Neg - Refine_Result - refine_result  Done")
        print("Gen_Stock_Pos_Neg - Refine_Result  Done")