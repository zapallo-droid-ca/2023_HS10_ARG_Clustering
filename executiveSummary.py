### --- CLUSTERING
##-- Environment Setting
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os

import warnings
warnings.filterwarnings('ignore')



##-- GENERAL SETTINGS
csvAttr_imp = {'sep': ';' , 'encoding': 'UTF-8'} #csv settings - export

cereal = 'Maize'
maxClusterNumbers = 6
randomStateValue = 2023
flowType = 'export'

flowColumn = 't_exp' if flowType.lower() == 'export' else 't_imp'


##-- WORK DIRECTORY
wd = 'C:/Users/jrab9/OneDrive/08.Github/2023.HS10-ARG.Clustering/01.Data/'

os.chdir(wd)

##-- DATA
df_base = pd.read_csv(wd + '/df_baseWithClusters.csv.gz', sep = csvAttr_imp['sep'], encoding = csvAttr_imp['encoding'])
df_base['date'] = pd.to_datetime(df_base['date'], format = '%Y-%m-%d')
df_base['year'] = df_base['date'].dt.year

colsRenameDict = {'desc_l2':'Cereal'}

df_base.rename(columns = colsRenameDict, inplace = True)


df_metrics = pd.read_csv(wd + '/df_metrics.csv.gz', sep = csvAttr_imp['sep'], encoding = csvAttr_imp['encoding']).merge(df_base[['partnerCodeISO','partnerDesc','partnerRegionDesc']].drop_duplicates(), on = 'partnerCodeISO', how = 'left')

df_centersExp01 = pd.read_csv(wd + '/clustersExp01_centers.csv.gz', sep = csvAttr_imp['sep'], encoding = csvAttr_imp['encoding'])
df_centersExp02 = pd.read_csv(wd + '/clustersExp02_centers.csv.gz', sep = csvAttr_imp['sep'], encoding = csvAttr_imp['encoding'])
df_centersExp03 = pd.read_csv(wd + '/clustersExp03_centers.csv.gz', sep = csvAttr_imp['sep'], encoding = csvAttr_imp['encoding'])


##-- TABLE RESOURCES
auxTab1A = df_metrics.t_exp.sum()
auxTab1B = df_metrics.t_exp_tot.sum()

tab1 = df_metrics.sort_values('t_exp', ascending = False).reset_index(drop = True)
tab1['share'] = np.round((tab1['t_exp'] / auxTab1A) * 100,1)
tab1['share_totalCereals'] = np.round((tab1['t_exp_tot'] / auxTab1B) * 100,1)
tab1 = tab1[['partnerCodeISO','partnerDesc','partnerRegionDesc','t_exp','t_exp%Cereal','share','trxExp','share_totalCereals']].head(10)

tab1.rename(columns = {'partnerDesc':'PARTNER','partnerRegionDesc':'REGION','t_exp':'MAIZE EXPORTS (TONNES)',
                       't_exp%Cereal':'% MAIZE IN CEREAL EXPORTS','trxExp':'EXPORTS FREQUENCY (MONTHS) IN A YEAR',
                       'share':'PARTNER % SHARE IN MAIZE EXPORTS','share_totalCereals':'PARTNER % SHARE IN CEREALS EXPORTS'}, inplace = True)


tab2 = tab1.merge(df_base[['partnerCodeISO','clusterByMetrics','clusterByTWDs','clusterByTrendComponent','clusterBySeasonalComponent', 'clusterByResidualComponent']].drop_duplicates(), on = 'partnerCodeISO',how = 'left')


##-- METRIC RESOURCES
metric1 = tab1['PARTNER % SHARE IN MAIZE EXPORTS'].sum()
metric2 = tab1['PARTNER % SHARE IN CEREALS EXPORTS'].sum()

##-- FIG RESOURCES

fig1 = px.bar(df_base.groupby(['date','Cereal']).agg(Value = ('t_exp','sum')).reset_index().sort_values('date',ascending = True), x = 'date', y ='Value', color = 'Cereal', title = 'Argentinian Cereals Exports')


partnersOfInterest = tab1.partnerCodeISO.unique()

df_tempAux = df_base[(df_base['Cereal'] == 'Maize') & (df_base.partnerCodeISO.isin(partnersOfInterest))].sort_values('date', ascending = True).reset_index(drop = True).copy()
df_tempAux = df_tempAux[['partnerDesc','date','t_exp','partnerCodeISO']].rename(columns = {'partnerDesc':'Time Serie','date':'Date','t_exp':'Value'})

df_tempAuxScalated = pd.DataFrame()
#Escalamos, centroiodes escalados por Z-Score
for partner in df_tempAux.partnerCodeISO.unique():
    df_temp = df_tempAux[df_tempAux['partnerCodeISO'] == partner]        
    
    #-SCALING
    scaler = StandardScaler()    
    df_temp['Value'] = scaler.fit_transform(df_temp['Value'].values.reshape(-1,1))
    
    df_tempAuxScalated = df_tempAuxScalated.append(df_temp)

del(df_temp)
df_tempAuxScalated.drop(columns = 'partnerCodeISO', inplace = True)
df_tempAuxScalated


def cerealsExpTimeSeries():
    return fig1

def centroidTrendTS(partners = False):
    df_tempFig2 = df_centersExp03[df_centersExp03['exp'] == '3_clustersTrend'].drop(columns = ['cereal','exp']).sort_values('variable', ascending = True).reset_index(drop = True)
    df_tempFig2['cluster'] = 'Cluster ' + df_tempFig2['cluster'].astype(str)
    df_tempFig2.columns = ['Time Serie','Date','Value']
    
    if partners == True:
        df_tempFig2 = df_tempFig2.append(df_tempAuxScalated)
        titleText = 'Series and Trend Centroid by Cluster'
    titleText = 'Trend Centroid by Cluster'

    fig2 = px.line(df_tempFig2, x = 'Date', y ='Value', color = 'Time Serie', title = titleText, width=1200, height=600)
    return fig2


def centroidSeasonalTS(partners = False):
    df_tempFig3 = df_centersExp03[df_centersExp03['exp'] == '3_clustersSeasonal'].drop(columns = ['cereal','exp']).sort_values('variable', ascending = True).reset_index(drop = True)
    df_tempFig3['cluster'] = 'Cluster ' + df_tempFig3['cluster'].astype(str)
    df_tempFig3.columns = ['Time Serie','Date','Value']
    
    if partners == True:
        df_tempFig3 = df_tempFig3.append(df_tempAuxScalated)
        titleText = 'Series and Seasonal Component Centroid by Cluster'
    titleText = 'Seasonal Component Centroid by Cluster'
    fig3 = px.line(df_tempFig3, x = 'Date', y ='Value', color = 'Time Serie', title = titleText, width=1200, height=600)
    return fig3


def centroidResidualTS(partners = False):
    df_tempFig4 = df_centersExp03[df_centersExp03['exp'] == '3_clustersResidual'].drop(columns = ['cereal','exp']).sort_values('variable', ascending = True).reset_index(drop = True)
    df_tempFig4['cluster'] = 'Cluster ' + df_tempFig4['cluster'].astype(str)
    df_tempFig4.columns = ['Time Serie','Date','Value']
    
    if partners == True:
        df_tempFig4 = df_tempFig4.append(df_tempAuxScalated)
        titleText = 'Series and Residual Component Centroid by Cluster'
    titleText = 'Residual Component Centroid by Cluster'
    fig4 = px.line(df_tempFig4, x = 'Date', y ='Value', color = 'Time Serie', title = titleText, width=1200, height=600)
    
    return fig4


def tableTopPartners():
    return tab1.drop(columns = 'partnerCodeISO')

def tableTopPartnersWithClusterIdMetrics():
    return tab2.drop(columns = 'partnerCodeISO')

def shareSummaryTop10():      
    df_temp = pd.DataFrame().append({'% SHARE MAIZE':metric1,'% SHARE CEREALS':metric2}, ignore_index = True)      
    return df_temp

def percentageOfChangeCereals():
    df_temp = df_base[df_base.year.isin([2022,2021])].groupby(['year','Cereal']).agg(Value = ('t_exp','sum')).reset_index()
    df_temp = df_temp[df_temp['year'] == 2022].merge(df_temp[df_temp['year'] == 2021], on = 'Cereal', how = 'left', suffixes=('_2022','_2021')).drop(columns = ['year_2022','year_2021'])
    df_temp['% of Change 2022-2021'] = np.round((df_temp['Value_2022'] - df_temp['Value_2021']) * 100 / df_temp['Value_2021'],1) 
    df_temp.fillna(0, inplace = True)
    
    fig_temp = px.bar(df_temp, x = '% of Change 2022-2021' , y = 'Cereal', orientation = 'h', title = '% of change in Exports of Cereals for 2022')
    
    return fig_temp
        
def metricsClusterCentroids():
    catLabels = df_centersExp01.columns.drop(['cluster','cereal'])
    clusterIds = df_centersExp01.cluster.unique()
    
    df_temp = df_centersExp01.drop(columns = 'cereal').copy()
    
    fig = go.Figure()
    
    for iclusterId in clusterIds:        
        fig.add_trace(go.Scatterpolar(
                                      r= df_temp[df_temp['cluster'] == iclusterId].values[0],
                                      theta=catLabels,
                                      fill='toself',
                                      name=f'Cluster {iclusterId}'
                                ))
        
    fig.update_layout( polar=dict(
                        radialaxis=dict(
                          visible=True,
                          range=[0, 6]
                        )),
                      showlegend=True,
                      title = 'Centroids by Cluster',
                      width=800,
                      height=600 
                    )
    return fig
    
