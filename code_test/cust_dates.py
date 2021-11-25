# ---------datetime import -------------
import datetime
from datetime import datetime
from datetime import timedelta
# --------------------------------------
import pandas as pd

# --------- 날짜, 시간 데이터 format -----------------
format = ' %Y.%m.%d %H:%M' # ex) ' 2021.10.20 17:20'
format_s = '%Y/%m/%d'      # ex) '2021/10/20'
format_t = '%Y-%m-%d'      # ex) '2021-10-20'
time_format ='%H:%M:%S'    # ex) '17:20:00'
# ----------------------------------------------------

# ----------날짜 기간 내 모든 날짜 리스트---------------------------------------------------------------
def date_range(start, end): 
    start = datetime.strptime(start, format_t)
    end = datetime.strptime(end, format_t)
    dates = [(start + timedelta(days = i)).strftime(format_t) for i in range((end - start).days+1)]
  
    return dates
# -----------------------------------------------------------------------------------------------------

# ------------날짜 data 형식 변환-----------------------------------
def type_date_transform(day, dateform):    
    origin_date = datetime.strptime(day, dateform)
    transform_date = datetime.strftime(origin_date, format_t)

    return transform_date
# -----------------------------------------------------------------

#-------------시간 data 형식 인식-----------------
def type_time_transform(time):
    return datetime.strptime(time, time_format)
#------------------------------------------------

#--------------날짜(일수) 더하기-------------------
def date_plus(day, num):
    return day + timedelta(days = num)
#--------------------------------------------------

#--------------주말 리스트 생성----------------------------------------------------------------------
def weekend(day1, day2):
    sat = pd.date_range(start = day1, end = day2, freq='W-SAT')
    sun = pd.date_range(start = day1, end = day2, freq='W-SUN')

    sat_days = []
    for time in sat:
        sat_days.extend(pd.date_range(time, freq = 'S', periods = 1).strftime(format_t).tolist())

    sun_days = []
    for time in sun:
        sun_days.extend(pd.date_range(time, freq = 'S', periods = 1).strftime(format_t).tolist())

    weekend_days = sat_days + sun_days
    weekend_days.sort()

    return weekend_days
#---------------------------------------------------------------------------------------------------------