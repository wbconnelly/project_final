from bs4 import BeautifulSoup
import urllib2 as ul
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
plt.style.use('fivethirtyeight')


feature_cols = list(company_avg.loc[:, company_avg.dtypes == np.float64].columns)

col_list_delete = ['extraordinaryitems',
'deferredcharges',
'accountingchange',
'amended',
'audited',
'Unnamed: 0_y', 'preliminary',
'Unnamed: 0_x',
 'year',
 'quarter',
 'restated',
'company_cik', 
'usdconversionrate',
'periodlength',
'original', 'crosscalculated', 'discontinuedoperations', 
'changeininventories', 'inventoriesnet',' otherequity']

for col in col_list_delete:
    try:    
        feature_cols.remove(col)    
    except:
        pass
feature_cols.append('Sector')

#fill NaNs with the average of each column
sector_list = ['None Found',
 'Basic Materials',
 'Cyclical Consumer Goods & Services',
 'Technology',
 'Healthcare',
 'Non-Cyclical Consumer Goods & Services',
 'Financials',
 'Industrials',
 'Utilities',
 'Energy',
 'Telecommunications Services']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# impute all NaN values with the mean from it's respective sector




y_vals = pd.read_csv('https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/y_vals.csv')
x = pd.read_csv('https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/x.csv')


x = scaler.fit_transform(x[feature_cols])
x = pd.DataFrame(x, columns = feature_cols)

from sklearn.tree import DecisionTreeClassifier
#treeclf = DecisionTreeClassifier(max_depth=10, random_state=929)

model_list = {}

for sector in sector_list:
    try:
        x = x[feature_cols]
        y= y_vals[sector]
        treeclf = DecisionTreeClassifier(max_depth=10, random_state=1)   
        model_list[sector] = treeclf.fit(x, y)
        x['predicted'] = model_list[sector].predict(x)
        print sector, '---',metrics.accuracy_score(y, x.predicted) #, len(feature_lists[sector])
        confusion = metrics.confusion_matrix(y, x.predicted)
        TP = confusion[1][1]
        TN = confusion[0][0]
        FP = confusion[0][1]
        FN = confusion[1][0]
        print 'True Positives:', TP
        print 'True Negatives:', TN
        print 'False Positives:', FP
        print 'False Negatives:', FN
    except:
        pass

#----------------------- test the models ---------------------#
x_test = pd.read_csv('https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/x_test.csv')
y_test = pd.read_csv('https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/y_test.csv')
#x_test_unscaled = pd.read_csv('https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/x_test_unscaled.csv')

for sector in sector_list:
    try:
        xt = x_test[feature_cols]
        yt = y_test[sector]
        x_pred = model_list[sector].predict(xt)
        print sector, '---',metrics.accuracy_score(yt, x_pred)
        confusion = metrics.confusion_matrix(yt, x_pred)
        TP = confusion[1][1]
        TN = confusion[0][0]
        FP = confusion[0][1]
        FN = confusion[1][0]
        print 'True Positives:', TP
        print 'True Negatives:', TN
        print 'False Positives:', FP
        print 'False Negatives:', FN
    except:
        pass