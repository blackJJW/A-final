from scipy.stats import norm
import numpy as np

def norm_dist(list, p): # 정규분포
    m = list.mean() #평균
    s = list.std() #표준편차

    rv = norm(loc = m, scale = s) # loc : 평균  scale : 표준편차

    return rv.ppf(p)