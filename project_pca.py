import matplotlib.pyplot as plt
from sklearn import metrics
plt.style.use('fivethirtyeight')

import pandas as pd
import numpy as np
from pylab import plot,subplot,axis,stem,show,figure

#---------------------------------------#


fin_data = pd.read_csv("https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/fin_data.csv")
company_sectors = pd.read_csv("https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/company_sectors.csv")
company_sectors.rename(columns={'Symbol':'company_symbol'}, inplace = True)

# Company data is the unaveraged dataset
company_data = pd.merge(fin_data, company_sectors, on = 'company_symbol')

#company_data.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/final_company_data.csv")


#get average values for each company across the years in the sample
company_data.groupby('company_symbol').mean().shape
company_avg = company_data.groupby('company_symbol').mean()

company_avg['company_symbol'] = company_avg.index

# reattch the symbols since they were dropped when getting averages
company_avg = pd.merge(company_avg, company_sectors, on = 'company_symbol')

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
'changeininventories', 'inventoriesnet',' otherequity',
]

for col in col_list_delete:
    try:    
        feature_cols.remove(col)    
    except:
        pass
feature_cols.append('Sector')


#fill NaNs with the average of each column
sector_list = list(company_avg.Sector.unique())

# impute all NaN values with the mean from it's respective sector

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
x = x.fillna(x.mean())

 
x.drop('Sector', axis = 1, inplace= True)
feature_cols.remove('Sector')

#------------End dataset creation--------------------------#


#M1 = np.apply_along_axis(divide, 0 , M)
# Note: I did not develop this function myself. I obtained it from a coworker

def princomp(A):
 """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables. 

 Returns :  
  coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
  score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
  latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
 """
 # computing eigenvalues and eigenvectors of covariance matrix
 M = (A-np.mean(A))  #center the data
 M = M / np.std(A)   #scale the data
  
 U, s, V= np.linalg.svd(M, full_matrices=False, compute_uv='True')
  
 coeff = V.T # the right singular vectors are the factor loadings
 latent = s / np.sqrt(51-1) # the singular values / sqrt(n-1) are the eigenvalues
 score = U * s # projection of the data in the new space; or principal components(scores)
 return coeff, score, latent
 
 
# run the PCA function on x[feature cols]
 
coeff, score, latent = princomp(x[feature_cols])
 
y_mt = pd.DataFrame(score)

for col in y_mt.columns:
    x[col] = y_mt[col]

# Create the correlation matrix between the original variables and the components and examine
# it to eliminate features of low variance.

x.corr().to_csv("C:/Users/wiconnelly/Documents/pythonex/proj/corr_mt.csv")



