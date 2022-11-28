# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:08:31 2022

@author: user
"""
# %%

import glob
import os
import pandas as pd
files=glob.glob('C:/Users/user/Desktop/강화학습_sample_trading/강화학습용_종목/train_test_set/*.csv')

path = "C:/Users/user/Desktop/강화학습_sample_trading/강화학습용_종목/train_test_set"

file_list =os.listdir(path)

len(file_list)

df=[]

for file in file_list:
    path = "C:/Users/user/Desktop/강화학습_sample_trading/강화학습용_종목/train_test_set"
    data=pd.read_csv(path+"/"+file)
    data=data.drop(['Unnamed: 0'],axis=1)
    df.append(data)


stock_name=pd.DataFrame({'stock_name':file_list})


for i in range(len(df)):
    stock_name['stock_name'][i] =stock_name['stock_name'][i].strip(".csv")
    
    
    
#model train/test set 생성  
stock_price=[]


############train/test 분리 ###
for i in range(0,72):
    train=None
    train=df[i]['date'].str.contains('2018|2019|2020')
    stock_price.append(df[i][train])    
    

for i in range(len(stock_price)):
    stock_price[i] = stock_price[i][['date','close']]


stock_data = pd.DataFrame(columns=stock_name['stock_name'])


for i in range(len(stock_price)):
    stock_data[stock_data.columns[i]] = stock_price[i]['close']
    
    
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler().fit(stock_data)
train = scaler.transform(stock_data)

# %%

###modeling
# Native libraries
import os
import math
# Essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Algorithms
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(train))))

##하이퍼파라미터
s_time = pd.Timestamp.now()
print('시작시간:',s_time,'\n')

#원하는 파라미터 조합 리스트화
map_n= [n for n in range(2,6)]
para_sigma= [np.round(sigma*0.1,2) for sigma in range(1,10)]
para_learning_rate= [np.round(learning_rate*0.1,2) for learning_rate in range(1,10)]

#결과 값을 담을 리스트 res 생성
res = []
#모든 조합에 대해 모델 생성 및 qe,te값 계산
for n in map_n:
    for sigma in para_sigma:
        for lr in para_learning_rate:
            
            try:
                #랜덤으로 초기값을 설정하는 경우
                estimator = MiniSom(n,n,72,sigma =sigma, learning_rate = lr, topology='hexagonal',random_seed=0)
                estimator.random_weights_init(train)
                estimator.train(train,1000,random_order=True)
                qe = estimator.quantization_error(train)
                #te = estimator.topographic_error(data.values)
                winner_coordinates = np.array([estimator.winner(x) for x in train]).T
                cluster_index = np.ravel_multi_index(winner_coordinates,(n,n))
                
                res.append([str(n)+'x'+str(n),sigma,lr,'random_init',qe,len(np.unique(cluster_index))])

                #pca로 초기값을 설정하는 경우
                estimator = MiniSom(n,n,72,sigma =sigma, learning_rate = lr,topology='hexagonal', random_seed=0)
                estimator.pca_weights_init(train)
                estimator.train(train,1000,random_order=True)
                qe = estimator.quantization_error(train)
                #te = estimator.topographic_error(data.values)
                winner_coordinates = np.array([estimator.winner(x) for x in train]).T
                cluster_index = np.ravel_multi_index(winner_coordinates,(n,n))
                
                res.append([str(n)+'x'+str(n),sigma,lr,'pca_init',qe,len(np.unique(cluster_index))])
                
            except ValueError as e:
                print(e)
            
#결과 데이터프레임 생성 및 sorting 
df_res = pd.DataFrame(res,columns=['map_size','sigma','learning_rate','init_method','qe','n_cluster']) 
df_res.shape
df_res.sort_values(by=['qe'],ascending=True,inplace=True,ignore_index=True)
df_res.head(10)



som_x = 5
som_y = 5
# I didn't see its significance but to make the map square,
# I calculated square root of map size which is 
# the square root of the number of series
# for the row and column counts of som

som = MiniSom(5,5,72,sigma=0.4,learning_rate=0.7,topology='hexagonal',neighborhood_function='gaussian',activation_distance='euclidean', random_seed=0)

som.random_weights_init(train)
som.train(train, 50000)


def plot_som_series_averaged_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x,som_y,figsize=(30,25))
    for x in range(5):
        for y in range(5):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5) 
                axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}")

    # plt.show()
    plt.savefig("stock_market_clustering.png",dpi=300)
    
win_map = som.win_map(train)
# Returns the mapping of the winner nodes and inputs

plot_som_series_averaged_center(som_x, som_y, win_map)



cluster_c = []
cluster_n = []
for x in range(som_x):
    for y in range(som_y):
        cluster = (x,y)
        if cluster in win_map.keys():
            cluster_c.append(len(win_map[cluster]))
        else:
            cluster_c.append(0)
        cluster_number = x*som_y+y+1
        cluster_n.append(f"Cluster {cluster_number}")

plt.figure(figsize=(25,5))
plt.title("Cluster Distribution for SOM")
plt.bar(cluster_n,cluster_c)
plt.show()


cluster_map = []
for idx in range(len(stock_name)):
    winner_node = som.winner(train[idx])
    cluster_map.append((stock_name['stock_name'][idx],f"Cluster {winner_node[0]*som_y+winner_node[1]+1}"))

som_result = pd.DataFrame(cluster_map,columns=["Series","Cluster"]).sort_values(by="Cluster")

som_result['Cluster'].value_counts()


# %%

# DTW k-means


cluster_count = math.ceil(math.sqrt(len(train))) 
# A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN

#cluster 개수 : 

km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")

labels = km.fit_predict(train)


plot_count = math.ceil(math.sqrt(cluster_count))

fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(train[i],c="gray",alpha=0.4)
                cluster.append(train[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0
        
plt.show()

fancy_names_for_labels = [f"Cluster {label}" for label in labels]

DTW_result = pd.DataFrame(zip(stock_name['stock_name'],fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster")


DTW_result['Cluster'].value_counts()

# %%
from tslearn.clustering import KShape
ks = KShape(n_clusters=28, max_iter=100, n_init=10,verbose=1,random_state=2019)

ks.fit(train)

ksahpe_labels = ks.predict(train)

plot_count = math.ceil(math.sqrt(28))

fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0

for label in set(ksahpe_labels):
    cluster = []
    for i in range(len(ksahpe_labels)):
            if(ksahpe_labels[i]==label):
                axs[row_i, column_j].plot(train[i],c="gray",alpha=0.4)
                cluster.append(train[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0
        
plt.show()

fancy_names_for_labels = [f"Cluster {label}" for label in labels]

kshape_result = pd.DataFrame(zip(stock_name['stock_name'],fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster")

kshape_result['Cluster'].value_counts()


#%%

#가장 많이 나오는 시계열 패턴으로 주식 종목들을 군집화

som_result = som_result.reset_index(drop=True)
DTW_result =  DTW_result.reset_index(drop=True)
kshape_result=  kshape_result.reset_index(drop=True)


# som_result['Cluster'].unique()[0]


# stock_data = pd.DataFrame(train,columns=stock_name['stock_name'])

# stock_data
# #Minisom 
# stock_data.columns = stock_name['stock_name']



#minisom cluster
som_cluster =[]

for i in range(len(som_result['Cluster'].value_counts())) :
    som_cluster.append(som_result[som_result['Cluster']==som_result['Cluster'].unique()[i]])
    som_cluster[i] = som_cluster[i].reset_index(drop=True)

        
#DTW cluster
dtw_cluster =[]

for i in range(len(DTW_result['Cluster'].value_counts())) :
    dtw_cluster.append(DTW_result[DTW_result['Cluster']==DTW_result['Cluster'].unique()[i]])
    dtw_cluster[i] = dtw_cluster[i].reset_index(drop=True)

        
#kshape cluster
kshape_cluster =[]

for i in range(len(kshape_result['Cluster'].value_counts())) :
    kshape_cluster.append(kshape_result[kshape_result['Cluster']==kshape_result['Cluster'].unique()[i]])
    kshape_cluster[i] = kshape_cluster[i].reset_index(drop=True)



####주식 
som_stock_price =[]

for i in range(len(som_cluster)):
    som_stock_price.append(stock_data[som_cluster[i]['Series']])
    som_stock_price[i] = som_stock_price[i].reset_index(drop=True)




dtw_stock_price =[]

for i in range(len(dtw_cluster)):
    dtw_stock_price.append(stock_data[dtw_cluster[i]['Series']])
    dtw_stock_price[i] = dtw_stock_price[i].reset_index(drop=True)


kshape_stock_price =[]

for i in range(len(kshape_cluster)):
    kshape_stock_price.append(stock_data[kshape_cluster[i]['Series']])
    kshape_stock_price[i] = kshape_stock_price[i].reset_index(drop=True)


#%%
    
##minisom 
# # trading signal plot
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# jupyter notebook 내 그래프를 바로 그리기 위한 설정
%matplotlib inline

# unicode minus를 사용하지 않기 위한 설정 (minus 깨짐현상 방지)
plt.rcParams['axes.unicode_minus'] = False


plt.rcParams['font.family'] = 'NanumGothic'

    
for i in range(len(som_stock_price)):
    # fig = plt.figure(figsize = (20,10))
    # plt.rc('font', size=30)
    # plt.xticks(rotation=45,size=25)
    # som_stock_price[i].plot(figsize=(30,30))#.autoscale(axis='y',tight=True)
    fig = plt.figure(figsize = (40,40))
    plt.rc('font', size=30)
    plt.plot(som_stock_price[i], lw=2.,label = som_stock_price[i].columns)
    plt.xticks(rotation=45,size=25)
    # plt.title('RL_test_trading_signal_{}'.format(stock_name['stock_name'][i]))
    plt.legend(fontsize=30,loc='lower left')
    plt.show()
    

##dtw_kmeans
    
for i in range(len(dtw_stock_price)):
    # fig = plt.figure(figsize = (20,10))
    # plt.rc('font', size=30)
    # plt.xticks(rotation=45,size=25)
    # som_stock_price[i].plot(figsize=(30,30))#.autoscale(axis='y',tight=True)
    fig = plt.figure(figsize = (40,40))
    plt.rc('font', size=30)
    plt.plot(dtw_stock_price[i], lw=2.,label = dtw_stock_price[i].columns)
    plt.xticks(rotation=45,size=25)
    # plt.title('RL_test_trading_signal_{}'.format(stock_name['stock_name'][i]))
    plt.legend(fontsize=30,loc='lower left')
    plt.show()
    
       
#kshape    
for i in range(len(kshape_stock_price)):
    # fig = plt.figure(figsize = (20,10))
    # plt.rc('font', size=30)
    # plt.xticks(rotation=45,size=25)
    # som_stock_price[i].plot(figsize=(30,30))#.autoscale(axis='y',tight=True)
    fig = plt.figure(figsize = (40,40))
    plt.rc('font', size=30)
    plt.plot(kshape_stock_price[i], lw=2.,label = kshape_stock_price[i].columns)
    plt.xticks(rotation=45,size=25)
    # plt.title('RL_test_trading_signal_{}'.format(stock_name['stock_name'][i]))
    plt.legend(fontsize=30,loc='lower left')
    plt.show()
#%%
    
##pairs trading
from statsmodels.tsa.stattools import coint    

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    all_pairs = []
    pairs = []

    # result
    stock1 = []
    stock2 = []
    pvalue_list = []
    check_95 = []
    check_98 = []

    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue


            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
                check_95.append('Y')
            else:
                check_95.append('N')

            if pvalue < 0.02:
                check_98.append('Y')
            else:
                check_98.append('N')

            # result
            stock1.append(keys[i])
            stock2.append(keys[j])
            pvalue_list.append(pvalue)


    pair_pvalue = pd.DataFrame()
    pair_pvalue['s1'] = stock1
    pair_pvalue['s2'] = stock2
    pair_pvalue['pvalue'] = pvalue_list
    pair_pvalue['check_95'] = check_95
    pair_pvalue['check_98'] = check_98

    pair_pvalue.sort_values('pvalue', ascending=True, inplace=True) # ascending=True 오름차순

    return score_matrix, pvalue_matrix, pair_pvalue, pairs



instrumentIds = list(som_stock_price[0].columns.values)

scores, pvalues, pair_pvalue, pairs = find_cointegrated_pairs(som_stock_price[0])
import seaborn
m = [0,0.2,0.4,0.6,0.8,1]
seaborn.heatmap(pvalues, xticklabels=instrumentIds,
                yticklabels=instrumentIds, cmap='RdYlGn_r',
                mask = (pvalues >= 0.95))
plt.show()

# 유의한 pair 출력
print(pairs)

pair_pvalue


# 가장 유의성이 높은 2개 종목을 추출한다.
s1_nm = '008770_호텔신라'
s2_nm = '029780_삼성카드'
S1 = som_stock_price[0][s1_nm]
S2 = som_stock_price[0][s2_nm]


ratios = S1 / S2
cut = int(len(ratios)*0.7)
train = ratios[:cut]
test = ratios[cut:]

S1_train = S1.iloc[:cut]
S2_train = S2.iloc[:cut]
S1_test = S1.iloc[cut:]
S2_test = S2.iloc[cut:]


ratios_mavg5 = train.rolling(window=5,
                               center=False).mean()
ratios_mavg60 = train.rolling(window=60,
                               center=False).mean()
std_60 = train.rolling(window=60,
                        center=False).std()
zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60

plt.figure(figsize=(15,7))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)
plt.legend(['Ratio','5d Ratio MA', '60d Ratio MA'])
plt.ylabel('Ratio')
plt.show()


plt.figure(figsize=(15,7))
zscore_60_5.plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()


# Plot the ratios and buy and sell signals from z score
plt.figure(figsize=(15,7))
train[60:].plot()
buy = train.copy()
sell = train.copy()
buy[zscore_60_5>-1] = 0
sell[zscore_60_5<1] = 0
buy[60:].plot(color='g', linestyle='None', marker='^')
sell[60:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,ratios.min(),ratios.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()


# Plot the prices and buy and sell signals from z score
plt.figure(figsize=(18,9))

S1_log = S1_train[60:].map(lambda x : np.log(x))
S2_log = S2_train[60:].map(lambda x : np.log(x))

S1_log[60:].plot(color='b')
S2_log[60:].plot(color='c')
buyR = 0*S1_log.copy()
sellR = 0*S1_log.copy()
# When buying the ratio, buy S1 and sell S2
buyR[buy!=0] = S1_log[buy!=0]
sellR[buy!=0] = S2_log[buy!=0]
# When selling the ratio, sell S1 and buy S2
buyR[sell!=0] = S2_log[sell!=0]
sellR[sell!=0] = S1_log[sell!=0]
buyR[60:].plot(color='g', linestyle='None', marker='^')
sellR[60:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,min(S1_log.min(),S2_log.min())-1,max(S1_log.max(),S2_log.max())+1))
plt.legend([s1_nm, s2_nm, 'Buy Signal', 'Sell Signal'])
plt.show()
