
# Read in the data. Fin_data is the basic financial reporting data 
# from kimono Labs, company_sectors is the list of symbols with the 
# corresponding industry classification


fin_data = pd.read_csv("https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/fin_data.csv")
company_sectors = pd.read_csv("https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/company_sectors.csv")

# rename the symbol column so the name match the company_symbol column in fin_data
company_sectors.rename(columns={'Symbol':'company_symbol'}, inplace = True)

# join the two to get a classification for each company
company_data = pd.merge(fin_data, company_sectors, on = 'company_symbol')


# compute the average values for each company across the years in the sample

company_avg = company_data.groupby('company_symbol').mean()

# reattch the sectors since they were dropped when getting averages
company_avg['company_symbol'] = company_avg.index
company_avg = pd.merge(company_avg, company_sectors, on = 'company_symbol')

# find the number of companies in each sector
company_avg.Sector.value_counts()

# find average number of missing values
null_sum = pd.DataFrame(company_avg.isnull().sum()).reset_index()
null_sum.rename(columns = {0:'null_count', 'index':'col_title'}, inplace = True)
company_avg.isnull().sum().mean()

# find columns with few missing values for each sector grouping
null_list = {}

# add each to  data frame to see which columns for each sector are good candidates as predictors
for sector in company_data.Sector.unique():
    null_df = pd.DataFrame(company_data[company_data.Sector == sector].isnull().sum()).reset_index()
    null_df.rename(columns = {0:'null_count', 'index':'col_title'}, inplace = True)
    null_list[sector] = null_df



# ----------Use logistic regression to predict the industry classifiaction based on well populated columns -------------#

# eliminate unwaned features from the list of feture columns

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
'changeininventories', 'inventoriesnet',' otherequity', 'equityearnings']


for col in col_list_delete:
    try:    
        feature_cols.remove(col)    
    except:
        pass
feature_cols.append('Sector')

#fill NaNs with the average of each column
sector_list = list(company_avg.Sector.unique())

# impute all NaN values with the mean from it's respective sector with a loop that subsets the data
# by sector reattaches the Sector column stores the data frame in a list ant then after the loop
# ends concatenate the frames back together

feature_dfs = []
for sector in sector_list:  
    x = company_avg[company_avg.Sector == sector][feature_cols]
    industry = pd.DataFrame(x.Sector, columns = ['Sector'])
    x.drop('Sector', axis = 1, inplace= True)
    x = x.fillna(x.mean())
    #x = (x - x.mean())/ np.std(x)
    #x = scaler.fit_transform(x)    
    x = x.join(industry)
    feature_dfs.append(x)
    
x = pd.concat(feature_dfs)
x.reset_index(inplace = True)

# impute any remaining nulls with the Global mean for the column.  This inly affected the 
# Telecommunictions Sector for the feature "researchdevelopmentexpense"
x = x.fillna(x.mean())

# Create tables in the var_list dictionary that hold the variance of each column for each sector

var_list = {}
for sector in sector_list:
    var_tbl = pd.DataFrame(x[feature_cols][x.Sector == sector].var().reset_index())
    var_tbl.rename(columns = {0:'variance', 'index':'col_title'}, inplace = True)
    var_tbl.sort('variance', ascending = False, inplace = True)
    var_tbl['total_var']= var_tbl['variance'].cumsum(skipna = True)/var_tbl.variance.sum()
    var_list[sector] = var_tbl

# Get dummy values for the Sectors and drip the 'Sector' column
y_vals = pd.get_dummies(x.Sector)

 # drop the categorical "Sector" column

x.drop('Sector', axis = 1, inplace= True)
feature_cols.remove('Sector')

# Create a for-loop that records logistic Regression performance at successively higher variance levels#

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
scaler = StandardScaler()
logreg = LogisticRegression(C=1)

# define a funciton that will increment in fractional values
def frange(init, end, step):
    steps = []    
    while init < end:
        steps.append(init)
        init += step
    return steps

x = scaler.fit_transform(x[feature_cols])
x = pd.DataFrame(x, columns = feature_cols)

# create empty dictionaries to store the data
false_pred_dict = {}
num_feat_dict = {}
var_limit_dict = {}
accuracy_dict = {}

# Begin the loop
for sector in sector_list:
    false_pred = []
    num_feat = []
    var_limit = []
    accuracy_list = []
    for var in frange(0.5,1,.005):
        
        x2 = x[var_list[sector]['col_title'][var_list[sector].total_var <= var]]
        y = y_vals[sector]
        logreg.fit(x2, y)
        x_pred = logreg.predict(x2)
        accuracy = metrics.accuracy_score(y, x_pred)
        confusion = metrics.confusion_matrix(y, x_pred)
        TP = confusion[1][1]
        TN = confusion[0][0]
        FP = confusion[0][1]
        FN = confusion[1][0]
        total_false = FP + FN
        false_pred.append(total_false)
        var_limit.append(var)
        num_feat.append(len(x2.columns))
        accuracy_list.append(accuracy)
        
        false_pred_dict[sector] = false_pred
        num_feat_dict[sector] = num_feat
        var_limit_dict[sector] = var_limit
        accuracy_dict[sector] = accuracy_list

# Observe the effect of accounting for successively higher levels of variance in the data in the graphs below
'Cyclical Consumer Goods & Services',
 'Telecommunications Services',
 'Industrials',
 'Energy',
 'None Found',
 'Utilities',
 'Financials',
 'Healthcare',
 'Basic Materials',
 'Technology',
 'Non-Cyclical Consumer Goods & Services'

sector = 'Technology'
plt.scatter(x = num_feat_dict[sector], y = var_limit_dict[sector])
plt.scatter(x = num_feat_dict[sector], y = false_pred_dict[sector])
plt.scatter(x = var_limit_dict[sector], y = false_pred_dict[sector])
plt.scatter(x = var_limit_dict[sector], y = accuracy_dict[sector])
plt.scatter(x = num_feat_dict[sector], y = accuracy_dict[sector])



# ------ Use a loop that creates models for a preset level of total variance ------ #

x = scaler.fit_transform(x[feature_cols])
x = pd.DataFrame(x, columns = feature_cols)

model_list = {}
feature_lists = {}
logreg = LogisticRegression(C=1e9)

for sector in sector_list:
    features = var_list[sector]['col_title'][var_list[sector].total_var <=.99999999999999999]
    x2 = x[features]   
    feature_lists[sector] = features
    feature_lists[sector] = feature_cols
    y = y_vals[sector]
    
    model_list[sector] = logreg.fit(x2, y)
    x2['predicted'] = model_list[sector].predict(x2)
    print sector, '--- Accuracy',metrics.accuracy_score(y, x2.predicted), 'Number of features ', len(feature_lists[sector])
    confusion = metrics.confusion_matrix(y, x2.predicted)
    TP = confusion[1][1]
    TN = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    print 'True Positives:', TP
    print 'True Negatives:', TN
    print 'False Positives:', FP
    print 'False Negatives:', FN


#----------------------- test the models ---------------------#
x_test = pd.read_csv('https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/x_test.csv')
y_test = pd.read_csv('https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/y_test.csv')

# x_test is already centered and scaled so there is no need to repeat that step

pred_list = []
for sector in sector_list:
    try:
        x_cols = feature_lists[sector]
        y = y_test[sector]

# Manually ensemble all the results in this inner loop

        for i in range(1,100):
            y_pred = model_list[sector].predict(x_test[x_cols])
            pred_list.append(np.array(y_pred))
            y_pred = np.round(np.mean(pred_list, axis = 0)).astype(int)
        print sector, '---',metrics.accuracy_score(y, y_pred), "number of features ", len(x_cols)
        confusion = metrics.confusion_matrix(y, y_pred)
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


