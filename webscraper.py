from bs4 import BeautifulSoup
import urllib2 as ul
import pandas as pd


def print_page(Base):
    
    target = Base
    page = ul.urlopen(target)
    bs = BeautifulSoup(page.read())
    try:
        industry =  bs.find(name = 'a', attrs = {'id':'sector'}).text
        return industry
    except: 
        return "None Found"
        
industry_list= []

for symb in sym_list:
    Base_URL = 'https://www.google.com/finance?q='+symb +'&ei=q8gVVqn1GoKjmAHX7oWwCg'
 
    #print (symb, print_page(Base_URL))
    industry_list.append([symb, print_page(Base_URL)])

pd.concat(industry_list)
company_sectors = pd.DataFrame(industry_list, columns = ['Symbol', 'Sector'])

#write the data to a csv
company_sectors.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/company_sectors.csv")