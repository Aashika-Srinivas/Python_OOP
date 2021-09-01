#!/usr/bin/env python
# coding: utf-8

# In[22]:


class outlier():
        
    def removeOutlier(data, threshold):
        #importing library
        import pandas as pd  #bibioteca responsável para o tratamento e limpeza dos dados
        import numpy as np #biblioteca utilizada para o tratamento eficiente de dados numéricos
        import datetime  #biblioteca utilizada para trabalhar com datas
        from matplotlib import pyplot as plt  #plotar os gráficos
        import seaborn as sns #plot de gráficos
        import io, os, sys, types
        
        #Original Data size
        data_col = data.shape[1]
        data_row = data.shape[0]
        
        #Z-Score calculus --------------------------------------------
        # signed number of standard deviations by which the value of an observation or data point is above the mean value
        # Z-Score tells how far a point is from the mean of dataset in terms of standard deviation

        cols = list(data.columns)

        for col in cols:
            col_zscore = col + '_zscore'
            data[col_zscore] = (data[col] - data[col].mean())/data[col].std(ddof=0)
            
        #Threashold for z-score is an empirical value. Might be changed according the dataset
        t= threshold; 

        # Selecting half of the dataset where Z columns are located (down half)
        dataSizeHalf_col = int(data.shape[1]/2) 
        dataSize_col = int(data.shape[1]) 

        #Removing Outliers where z < threashold and adding to dataNew
        dataNew = data[(data[data.columns[dataSizeHalf_col:dataSize_col]] < t).all(axis=1)]

        #Removing Z Colums to final matrix (Upper half)
        data_final = dataNew.iloc[:, 0:dataSizeHalf_col]
        
        #Amount of outlier rows eliminated
        originalData = int(data.shape[0])
        finalData = int(data_final.shape[0])
        rowsEliminated =  originalData - finalData

        print("Amount of Eliminated Rows: ", rowsEliminated)
        print("")


        return data_final    


# In[27]:


import pandas as pd
original_data = pd.read_excel('Real estate valuation data set.xlsx')
print(' ############ ORIGINAL DATA ############' )
print('')
print(original_data)
print('')
print('')


print(' ############ DATA WITHOUT OUTLIERS - Z-SCORE ############' )
print('')
new_data=outlier.removeOutlier(original_data, 2) #STD Deviation with Z-score. the second input is threshold
print(new_data)


# In[ ]:




