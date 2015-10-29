from bs4 import BeautifulSoup
import urllib2 as ul
import pandas as pd

# Pull down yearly financial reporting data from the Kimono API
sym_file = open("https://github.com/wbconnelly/Bill_Data_Sci/blob/master/project%20draft/symbol_list.txt")
sym_list = sym_file.read()
sym_list = sym_list.split(',')
data = []

for i in range(0, len(sym_list)-1):
    results = urllib.urlopen(
    "http://sec.kimonolabs.com/companies/"+ sym_list[i]+"/forms/10-K/ANN/2015,2014,2013,2012,2011/Q4?apikey=yaimV7I2x4hOIMnVTpdOOmz4etlnIxdn")
    results = pd.read_json(results.read())
    data.append(results)

dataset = pd.concat(data)
#write data to csv
dataset.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/fin_data.csv")
