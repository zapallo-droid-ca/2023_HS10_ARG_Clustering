### --- CLUSTERING

##-- Environment Setting
import pandas as pd ; from pandas.tseries.offsets import MonthEnd
import numpy as np
import matplotlib as plt
import seaborn as sns

import os
import pyodbc

import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from statsmodels.tsa.seasonal import seasonal_decompose

# General Settings
csvAttr_imp = {'sep': ';' , 'encoding': 'UTF-8'} #csv settings - export


#- Work Directory
wd = 'C:/Users/jrab9/OneDrive/08.Github/2023.HS10-ARG.Clustering/01.Data/'

os.chdir(wd)

##-- DATA
df_base = pd.read_csv(wd + '/dfM.csv.gz', sep = csvAttr_imp['sep'], encoding = csvAttr_imp['encoding'])


##-- TRANSFORMATION
## TIME SERIES DECOMPOSITION
partnerCodes = df_base['partnerCodeISO'].unique()
cereals = df_base['desc_l2'].unique()

dfClustered_trend = pd.DataFrame()
dfClustered_seasonal = pd.DataFrame()
dfClustered_resid = pd.DataFrame()


for cereal in cereals:    
    workPaper = df_base[df_base['desc_l2'] == cereal].reset_index(drop = True).copy()    
    
    for partner in partnerCodes:
        df_temp = workPaper[workPaper['partnerCodeISO'] == partner]
    
        ##INDEX
        df_temp.set_index(pd.to_datetime(df_temp['date']), inplace = True)
        df_temp = df_temp.asfreq('m')
        
        sdModel = 'additive'
        
        ## -- EXPORTS
        sdDF = seasonal_decompose(df_temp['t_exp'], model = sdModel)
        
        clusteredData = pd.DataFrame({'trend': sdDF.trend,
                                      'seasonal': sdDF.seasonal,
                                      'residual': sdDF.resid
                                     })
        
        df_temp_trend = clusteredData.trend.T
        df_temp_trend['partnerCodeISO'] = partner
        df_temp_trend['flow'] = 'export'
        df_temp_trend['component'] = 'trend'
        
        df_temp_seasonal = clusteredData.seasonal.T
        df_temp_seasonal['partnerCodeISO'] = partner
        df_temp_seasonal['flow'] = 'export'
        df_temp_seasonal['component'] = 'seasonal'
        
        df_temp_resid = clusteredData.residual.T
        df_temp_resid['partnerCodeISO'] = partner
        df_temp_resid['flow'] = 'export'
        df_temp_resid['component'] = 'residual'    
        
        df_temp_trend['cereal'] = cereal  
        df_temp_seasonal['cereal'] = cereal 
        df_temp_resid['cereal'] = cereal  
        
        dfClustered_trend = dfClustered_trend.append(df_temp_trend)
        dfClustered_seasonal = dfClustered_seasonal.append(df_temp_seasonal)
        dfClustered_resid = dfClustered_resid.append(df_temp_resid)
        
        del(df_temp_trend,df_temp_seasonal,df_temp_resid,sdDF,clusteredData)   
        print(f'exports of {cereal} done for partner {partner}')
        
        ## -- IMPORTS    
        sdDF = seasonal_decompose(df_temp['t_imp'], model = sdModel)
        
        clusteredData = pd.DataFrame({'trend': sdDF.trend,
                                      'seasonal': sdDF.seasonal,
                                      'residual': sdDF.resid
                                     })
        
        df_temp_trend = clusteredData.trend.T
        df_temp_trend['partnerCodeISO'] = partner
        df_temp_trend['flow'] = 'import'
        df_temp_trend['component'] = 'trend'
        
        df_temp_seasonal = clusteredData.seasonal.T
        df_temp_seasonal['partnerCodeISO'] = partner
        df_temp_seasonal['flow'] = 'import'
        df_temp_seasonal['component'] = 'seasonal'
        
        df_temp_resid = clusteredData.residual.T
        df_temp_resid['partnerCodeISO'] = partner
        df_temp_resid['flow'] = 'import'
        df_temp_resid['component'] = 'residual'    
        
        df_temp_trend['cereal'] = cereal  
        df_temp_seasonal['cereal'] = cereal 
        df_temp_resid['cereal'] = cereal  
        
        dfClustered_trend = dfClustered_trend.append(df_temp_trend)
        dfClustered_seasonal = dfClustered_seasonal.append(df_temp_seasonal)
        dfClustered_resid = dfClustered_resid.append(df_temp_resid)
        
        del(df_temp_trend,df_temp_seasonal,df_temp_resid,sdDF,clusteredData)        
        print(f'imports of {cereal} done for partner {partner}')


##-- OPTIMAL CLUSTER NUMBERS
## WCSS (Within-Cluster Sum of Squares)
maxClusterNumbers = 10
randomStateValue = 2023
flowType = 'export'

numberOfClustersByCereal = pd.DataFrame()

for cereal in cereals:    
    print(f'processing {cereal}')
    
    df_temp = dfClustered_trend[(dfClustered_trend['cereal'] == cereal) & (dfClustered_trend['flow']  == flowType)].reset_index(drop = True).copy()
    X = df_temp.drop(columns = ['partnerCodeISO', 'flow','component', 'cereal']).fillna(0)        
    
    for clusters in range(2,maxClusterNumbers + 1):
        kmeans = KMeans(n_clusters = clusters, random_state = randomStateValue).fit(X)
        labels = kmeans.labels_
        
        score_s = silhouette_score(X, labels)
        score_c = calinski_harabasz_score(X, labels)
        
        numberOfClustersByCereal = numberOfClustersByCereal.append({'cereal': cereal,
                                                                    'clusters': clusters,
                                                                    'silhouetteScore': score_s,
                                                                    'calinskiHarabaszScore': score_c,
                                                                    'flowType':flowType
                                                                    }, ignore_index = True)        
        
        
        del(kmeans,labels,score_s,score_c)
    
    del(X,df_temp)
    print(f'cereal {cereal} evaluated')
        
    
optimalNumberOfClusters = pd.DataFrame()
for cereal in cereals:
    df_temp = numberOfClustersByCereal[numberOfClustersByCereal['cereal'] == cereal].reset_index(drop = True).copy()
    
    df_temp['maxValueOf_S'] = df_temp['silhouetteScore'] == df_temp['silhouetteScore'].max()
    df_temp['maxValueOf_C'] = df_temp['calinskiHarabaszScore'] == df_temp['calinskiHarabaszScore'].max()
    
    optimalNumberOfClusters = optimalNumberOfClusters.append(df_temp)

optimalNumberOfClusters = optimalNumberOfClusters[(optimalNumberOfClusters['maxValueOf_S']) | (optimalNumberOfClusters['maxValueOf_C'])].reset_index(drop = True)

del(numberOfClustersByCereal,df_temp)
