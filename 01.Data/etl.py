### --- EXPLORATORY DATA ANALYSIS

##-- Environment Setting
import pandas as pd ; from pandas.tseries.offsets import MonthEnd
import numpy as np
import matplotlib as plt
import seaborn as sns

import os
import pyodbc

import warnings
warnings.filterwarnings('ignore')


#General Settings
csvAttr_exp = {'index': False, 'sep': ';' , 'encoding': 'UTF-8'} #csv settings - export


#- Work Directory
wd = 'C:/Users/jrab9/OneDrive/08.Github/2023.HS10-ARG.Clustering/01.Data/'

os.chdir(wd)


#- Data-Warehouse Credentials
serverConnectString = open(wd + '/00.Resources/serverConnectionSTG.txt').read()


##-- DATA
##Queries
dfM_q = open(wd + '/00.Resources/dfM_q.sql').read()

#- DB Connection
conn = pyodbc.connect(serverConnectString)

cursor = conn.cursor()

#- Datasets
dfM = pd.read_sql(dfM_q, conn)
dfM_copy = dfM.copy()
print(f'Monthly Data DF has been readed with {dfM.shape} shape')

conn.close()


##-- TRANSFORM
#- Transforming DF - MONTHLY VALUES
idxs = dfM.columns.drop(['flowCode','flowDesc','netWeight','PrimaryValue'])
cols = 'flowDesc'
vals = ['netWeight','PrimaryValue']

dfM = dfM.pivot(index = idxs, columns = cols, values = vals).reset_index()

dfM.columns = dfM.columns.droplevel(1); 

colsNames = np.append(dfM.iloc[:,:-4].columns,['netWeight_exp','netWeight_imp','primaryValue_exp','primaryValue_imp'])

dfM.columns = colsNames


#- Feature Engineering
dfM = dfM.fillna(0)

#Trade balance calculation and sign correction
dfM['primaryValue_imp'] = dfM['primaryValue_imp']  * (-1)
dfM['tradeBalance'] = dfM['primaryValue_exp'] + dfM['primaryValue_imp']

dfM['netWeight_exp'] = dfM['netWeight_exp'] / 1000
dfM['netWeight_imp'] = dfM['netWeight_imp'] / 1000

dfM['t_total'] = dfM['netWeight_exp'] + dfM['netWeight_imp']

#Values by tonne and differences in the values between trade flows
dfM['valueByTonne_exp'] = np.where(dfM['netWeight_exp'].isna(), np.nan, dfM['primaryValue_exp'] / dfM['netWeight_exp'])
dfM['valueByTonne_imp'] = np.where(dfM['netWeight_imp'].isna(), np.nan, dfM['primaryValue_imp'] / dfM['netWeight_imp'])

dfM['logisticValue'] = dfM['valueByTonne_exp'] + dfM['valueByTonne_imp']

dfM = dfM.fillna(0)


#QA - Every transaction should have weight and primaryValue
dfM['primaryValue_imp'] = np.where((dfM['netWeight_imp'] == 0) & (dfM['primaryValue_imp'] != -0), -0, dfM['primaryValue_imp'])
dfM['netWeight_imp'] = np.where((dfM['primaryValue_imp'] == 0) & (dfM['netWeight_imp'] != -0), -0, dfM['netWeight_imp'])

dfM['primaryValue_exp'] = np.where((dfM['netWeight_exp'] == 0) & (dfM['primaryValue_exp'] != -0), -0, dfM['primaryValue_exp'])
dfM['netWeight_exp'] = np.where((dfM['primaryValue_exp'] == 0) & (dfM['netWeight_exp'] != -0), -0, dfM['netWeight_exp'])

dfM.rename(columns = {'netWeight_exp':'t_exp','netWeight_imp':'t_imp'}, inplace = True)


#Making descriptions shorter
dfM['desc_l2'].unique()

descL2_namesDict = {'Wheat and meslin':'Wheat', 
                    'Maize (corn)':'Maize', 
                    'Grain sorghum':'Sorghum',
                    'Buckwheat, millet and canary seeds; other cereals':'Others'}

dfM['desc_l2'] = dfM['desc_l2'].replace(descL2_namesDict)

dfM = dfM[(dfM['t_exp'] != 0) | (dfM['t_imp'] != 0)].reset_index(drop = True)



dfM['date'] = pd.to_datetime(dfM['calendarCode'],format = '%Y%m') + MonthEnd(0)
dfM.set_index(dfM['date'], inplace = True)

indexDatesRange = pd.date_range(dfM.index.min(), dfM.index.max(), freq = 'M')

#Completing dates without transactions using zeros
partnerCodes = dfM.partnerCode.unique()
itemCodes = dfM.un_code_l2.unique()

reporterCode_temp = '32'
reporterCodeISO_temp = 'ARG'
reporterDesc_temp = 'Argentina'
reporterRegionDesc_temp = 'South America'

colNamesIndex = ['reporterCode', 'reporterCodeISO', 'reporterDesc',
                 'reporterRegionDesc', 'partnerCode', 'partnerCodeISO', 'partnerDesc',
                 'partnerRegionDesc', 'un_code_l2', 'desc_l2']

df = dfM.copy()

counter = 0
partners = len(partnerCodes)

for partner in partnerCodes:
    df_aux_c = dfM[dfM['partnerCode'] == partner][['partnerCode', 'partnerCodeISO', 'partnerDesc','partnerRegionDesc']].drop_duplicates()
    
    partnerCode_temp = df_aux_c.partnerCode.values[0]
    partnerCodeISO_temp = df_aux_c.partnerCodeISO.values[0]
    partnerDesc_temp = df_aux_c.partnerDesc.values[0]
    partnerRegionDesc_temp = df_aux_c.partnerRegionDesc.values[0]
    
    counter += 1
    
    for item in itemCodes:           
        df_aux_i = dfM[dfM['un_code_l2'] == item][['un_code_l2', 'desc_l2']].drop_duplicates()
        df_test = dfM[(dfM['partnerCode'] == partner) & (dfM['un_code_l2'] == item)]     
        
        un_code_l2_temp = df_aux_i.un_code_l2.values[0]
        desc_l2_temp = df_aux_i.desc_l2.values[0]
        
        tempIndex = df_test.index

        tempValues = df_test[colNamesIndex].drop_duplicates().reset_index(drop = True)
        
        datesToImput = indexDatesRange[np.isin(indexDatesRange,tempIndex) == False]
        
        for date in datesToImput:      
            
            calendarCode_temp = date.strftime('%Y%m')       

            df = df.append({'calendarCode':calendarCode_temp, 
                            'reporterCode':reporterCode_temp, 
                            'reporterCodeISO':reporterCodeISO_temp, 
                            'reporterDesc':reporterDesc_temp,
                            'reporterRegionDesc':reporterRegionDesc_temp, 
                            'partnerCode':partnerCode_temp, 
                            'partnerCodeISO':partnerCodeISO_temp, 
                            'partnerDesc':partnerDesc_temp,
                            'partnerRegionDesc':partnerRegionDesc_temp, 
                            'un_code_l2':un_code_l2_temp, 
                            'desc_l2':desc_l2_temp, 
                            't_exp':0, 
                            't_imp':0,
                            'primaryValue_exp':0, 
                            'primaryValue_imp':0, 
                            'tradeBalance':0, 
                            't_total':0,
                            'valueByTonne_exp':0, 
                            'valueByTonne_imp':0, 
                            'logisticValue':0,
                            'date': date}, ignore_index = True)       
            
        del(df_aux_i,df_test,un_code_l2_temp,desc_l2_temp,tempIndex,tempValues,datesToImput)
    print(f'partner {partnerDesc_temp} finished, {np.round(counter / partners,2) * 100} %')  
    del(df_aux_c,partnerCode_temp,partnerCodeISO_temp,partnerDesc_temp,partnerRegionDesc_temp)

print(df.shape)

#Export
df.to_csv(wd + '/dfM.csv.gz', index = csvAttr_exp['index'], sep = csvAttr_exp['sep'], encoding = csvAttr_exp['encoding'])
