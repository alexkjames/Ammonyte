# -*- coding: utf-8 -*-
"""
N. Ahmad, S. Derrible, T. Eason, and H. Cabezas, 2016, “Using Fisher information to track stability in multivariate systems”,
Royal Society Open Science, 3:160582, DOI: 10.1098/rsos.160582

Edits have been made by Alexander James to streamline the code for usage in this package
"""

import pandas as pd 
import math
import numpy as np
import pyleoclim as pyleo

__all__ = [
    'fisher_information',
    'smooth_series'
]

def fisher_information(eig_data,w_size,w_incre):
    Data_num=[]
    Time=[]
    
    for row in eig_data:
        Time.append(row[0])
        temp=[]
        for i in range(1,len(row)):
            if row[i]=='':
                temp.append(0)
            else:
                temp.append(float(row[i]))
        Data_num.append(temp)
        
    sost_data = SOST(eig_data,w_size)
    sost = sost_data.values[0]
    
    FI_final=[]
    k_init=[]
    for i in range(0,len(Data_num),w_incre):
        
        Data_win=Data_num[i:i+w_size]
        win_number=i
        
        if len(Data_win)==w_size:
            Bin=[]
            for m in range(len( Data_win)):
                Bin_temp=[]
                
                for n in range(len( Data_win)):
                    if m==n:
                        Bin_temp.append('I')
                    else:
                        Bin_temp_1=[]
                    
                        for k in range(len(Data_win[n])):
                            if (abs(Data_win[m][k]-Data_win[n][k]))<=sost[k]:
                                Bin_temp_1.append(1)
                            else:
                                Bin_temp_1.append(0)
                                
                        Bin_temp.append(sum(Bin_temp_1))
                        
                Bin.append(Bin_temp)
            
            FI=[]
            for tl in range(1,101):
                tl1=len(sost)*float(tl)/100
                Bin_1=[]
                Bin_2=[]
                
                for j in range(len(Bin)):
                    if j not in Bin_2:
                       
                        Bin_1_temp=[j]
                        for i in range(len(Bin[j])):
                            if Bin[j][i]!='I' and Bin[j][i]>=tl1 and i not in Bin_2:
                                Bin_1_temp.append(i)
                                
                        Bin_1.append(Bin_1_temp)
                        Bin_2.extend(Bin_1_temp)

                prob=[0]
                for i in Bin_1:
                    prob.append(float(len(i))/len(Bin_2))
                    
                prob.append(0)
                
                prob_q=[]
                for i in prob:
                    prob_q.append(math.sqrt(i))
                    
                FI_temp=0
                for i in range(len(prob_q)-1):
                    FI_temp+=(prob_q[i]-prob_q[i+1])**2
                FI_temp=4*FI_temp    
                
                FI.append(FI_temp)
                
            for i in range(len(FI)):
                if FI[i]!=8.0:
                    k_init.append(FI.index(FI[i]))
                    break
                
            FI_final.append(FI)
            
    if len(k_init)==0:
        k_init.append(0)
        
    time_axis = []
    
    for i in range(0,len(FI_final)):
        FI_final[i].append(float(sum(FI_final[i][min(k_init):len(FI_final[i])]))/len(FI_final[i][min(k_init):len(FI_final[i])]))
        time_axis.append(Time[(i*w_incre+w_size)-1])
        
    FI_final = pd.DataFrame(FI_final)
    FI_series=pyleo.Series(time=time_axis,value=FI_final.iloc[:,-1],label='Fisher Information')
    
    return FI_series
        
def SOST(eig_data,s_for_sd):
    Data_num=[]
    
    for row in eig_data:
        temp=[]
        for i in range(1,len(row)):
            if row[i]=='':
                temp.append(0)
            else:
                temp.append(float(row[i]))
        Data_num.append(temp)
        
    df=pd.DataFrame(Data_num)
    
    sos=[]
    for j in range(len(df.columns)):
        sos_temp=[]
        for i in df.index:
            A=list(df[j][i:i+s_for_sd])
            A_1=[float(i) for i in A if i!=0 ]
        
            if len(A_1)==s_for_sd:
                sos_temp.append(np.std(A_1,ddof=1))
                
        if len(sos_temp)==0:
            sos.append(0)
        else:
            
            sos.append(min(sos_temp)*2)
        
    df_sos=pd.DataFrame(sos)
    df_sos=df_sos.transpose()
    
    return df_sos
    
def smooth_series(series,block_size):
    '''Function to smooth series by averaging chunks
    
    Parameters
    ----------
    
    series : pyleoclim.Series object
        Series to be smoothed
        
    block_size : int
        Size of each block, assumes an evenly spaced series'''
    
    values = series.value
    
    smoothed_values=[]
   
    for i in range(block_size,len(values)+block_size,block_size):
        for j in range(i-block_size,i):
            smoothed_values.append(float(sum(values[i-block_size:i]))/len(values[i-block_size:i]))
            
    smoothed_values=smoothed_values[0:len(values)]
    
    series.value = smoothed_values
    
    return series