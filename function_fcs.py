import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *
import FlowCal as FC
import seaborn as sns
from scipy.stats import ttest_ind, t

from scipy.optimize import curve_fit,fsolve, minimize
from scipy.stats import gamma
from itertools import chain

import rpy2.robjects as robjects
from rpy2.robjects import FloatVector


ch1_='FSC-H'
ch2_='SSC-H'

events_num=100000
gatefrac=0.9
alpha_=0.02

#------------------------
def fcs_dataframe_(files,print_=True):
    fcslist_=[]
    FSC_SD=[]
    FSC_median=[]
    SSC_median=[]
    fcsdata=pd.DataFrame(dict())

    for f in files:      
        if str(f)!='nan':
            file_=f

            fcs_file=read_fcs(file_,ch1_,ch2_,events_num,gatefrac,width_gate=False)
            median_FSCH=np.median(fcs_file[:,'FSC-H'])
            vol_=float(fcs_file.text['$VOL'])/1000
            if print_: print(median_FSCH, vol_)
            fcslist_.append(fcs_file)
            FSC_SD.append(np.std(fcs_file[:,'FSC-H']))
            FSC_median.append(np.median(fcs_file[:,'FSC-H']))
            SSC_median.append(np.median(fcs_file[:,'SSC-H']))


    fcsdata['fcs']=fcslist_
    fcsdata['FSC_median']=FSC_median
    fcsdata['SSC_median']=SSC_median
    fcsdata['FSC_SD']=FSC_SD

    return fcsdata


def variants_(data, data_ref, p_value=0.01):

    data_H=data[:,'BL1-H']
    ref_H=data_ref[:,'BL1-H']
    data_A=data[:,'BL1-A']
    ref_A=data_ref[:,'BL1-A']
    
    df=len(ref_H)-1
    refH_log=np.log(ref_H+0.001)

    critical_t_statistic_H = t.ppf(1 - p_value, df)
    threshold_H=exp(critical_t_statistic_H*(np.std(refH_log))+np.mean(refH_log))
    variants_indx_H=data_H>threshold_H

    #----------------------
    data_A=data[:,'BL1-A']
    ref_A=data_ref[:,'BL1-A']
    
    df=len(ref_A)-1
    refA_log=np.log1p(ref_A[ref_A>0])

    critical_t_statistic_A = t.ppf(1 - p_value, df)
    threshold_A=exp(critical_t_statistic_A*(np.std(refA_log))+np.mean(refA_log))
    variants_indx_A=data_A>threshold_A

    variants_indx=variants_indx_H * variants_indx_A
    
    return variants_indx


def mut_dataset(data,data_ref,p_value=0.01):
    mut_ratio=np.zeros((len(data)))
    for j in range(0,len(data)):
        tmp=data[j]
        tmp_ref=data_ref[j]
        indx=variants_(tmp,tmp_ref,p_value)
        mut=tmp[indx]
        r=len(mut)/len(tmp)
        mut_ratio[j]=r
        
    return mut_ratio

def read_fcs(file,ch1_,ch2_, events_num, gatefrac=0.9, width_gate=False):
    tmp=FC.io.FCSData(file)
    tmp_gated=FC.gate.density2d(tmp,channels=[ch1_, ch2_],gate_fraction=gatefrac)
    if width_gate:
        tmp_gated[:,'YL2-W']=(tmp_gated[:,'SSC-A']/tmp_gated[:,'SSC-H'])
        tmp_gated[:,'BL1-W']=(tmp_gated[:,'FSC-A']/tmp_gated[:,'FSC-H'])
        tmp_gated_2=FC.gate.high_low(tmp_gated,channels=['YL2-W', 'BL1-W'],high=[1.5,1.5],low=None)
        tmp_gated=tmp_gated_2
    
    return tmp_gated

#------------------------------

def fig_boxplot_N(data_set,labellist,title,ax,median_color='orange'):

    ax.set_xscale('log')
    ax.set_axisbelow(True)

    ax.boxplot(data_set, widths=np.array(labellist)/5.0, medianprops=dict(color=median_color, linewidth=3), \
                showfliers=False, flierprops=dict(marker='o', markersize=2),positions=labellist)
    
    for i in range(0,len(data_set)):
        data_=data_set[i]
        ax.scatter(np.ones(len(data_))*labellist[i],data_,c='royalblue',s=5, alpha=1)

    ax.tick_params(which="both",axis="both",direction="in",labelsize=9)
    ax.set_ylabel('mutant ratio (%)',fontsize=10)
    ax.set_xlabel('plasmid copy number',fontsize=10)
    
    # take only positive tick values
    yticks_=ax.get_yticks()[ax.get_yticks()>=0]
    yticks_str=[f'{x:.0f}' for x in yticks_[:1]]+[f'{x:.2f}' for x in yticks_[1:]]
    ax.set_yticks(yticks_,yticks_str)
    
    return ax
    

#---------------------------------------
def fig_mutprob(data_mutprob, data_sdmutprob, labels_, colors_, ax):    
    ax.set_axisbelow(True)
    ax.plot(labels_, data_mutprob, color='darkgrey', lw=0.75)
    for i in range(0,len(labels_)):
        ax.errorbar(labels_[i],data_mutprob[i], yerr=data_sdmutprob[i],fmt='o', markersize=6, capsize=2,c=colors_(i/len(labels_)),lw=1.5, linestyle='None')
    ax.set_ylabel('phenotypic\nmutation rate ($10^{-4}$)', fontsize=10)
    ax.tick_params(direction='in',which='both',axis='both',labelsize=9)
    ax.set_yticks([0,0.5e-4, 1e-4, 1.5e-4, 2.0e-4], ['0', '0.5', '1.0','1.5','2.0'])
    ax.set_ylim([0, 2.1e-4])
    ax.set_xscale('log')
    ax.set_xlabel('plasmid copy number',fontsize=10)
    
    return ax


#----------------------------
def write_mutant_file(mut_data, mutant_file):
    
    with open(mutant_file, 'w') as file:
        for i in range(0,len(mut_data)):
            sequence_=mut_data[i]
            for j in range(0,len(sequence_)):
                seq_=sequence_[j]
                if j==len(sequence_)-1:
                    s=''.join(str(seq_)+'\n')
                else:
                    s=''.join(str(seq_)+',')
                file.write(s)
    file.close()

    return

def write_events_file(data_, data_file):
    
    with open(data_file, 'w') as file:
        for i in range(0,len(data_)):
            sequence_=data_[i]
            for j in range(0,len(sequence_)):
                seq_=float(sequence_[j])
                if j==len(sequence_)-1:
                    s=''.join(str(seq_)+'\n')
                else:
                    s=''.join(str(seq_)+',')
                file.write(s)

def read_mut_file(mut_file, events_file, method):
    
    robjects.r('library(flan)')
    robjects.r("file_path<-'"+mut_file+"'")
    robjects.r("file_fn<-'"+events_file+"'")
    
    r_code="""
    data<-read.table(file_path, fill=TRUE,sep=',')
    fn_data<-read.table(file_fn, fill=TRUE,sep=',')
    """
    robjects.r(r_code)

    if method =='GF' or method =='ML':
        flan_1=robjects.r('data_<-unlist(data[1,]); fn_<-unlist(fn_data[1,]); mutestim(mc=data_,fn=fn_,model="H", method="'+method+'", plateff=1,fitness=1)')
        flan_2=robjects.r('data_<-unlist(data[2,]); fn_<-unlist(fn_data[2,]); mutestim(mc=data_,fn=fn_,model="H", method="'+method+'", plateff=1,fitness=1)')
        flan_3=robjects.r('data_<-unlist(data[3,]); fn_<-unlist(fn_data[3,]); mutestim(mc=data_,fn=fn_,model="H", method="'+method+'", plateff=1,fitness=1)')
        flan_4=robjects.r('data_<-unlist(data[4,]); fn_<-unlist(fn_data[4,]); mutestim(mc=data_,fn=fn_,model="H", method="'+method+'", plateff=1,fitness=1)')
        flan_5=robjects.r('data_<-unlist(data[5,]); fn_<-unlist(fn_data[5,]); mutestim(mc=data_,fn=fn_,model="H", method="'+method+'", plateff=1,fitness=1)')

        data_mutprob=np.array([flan_1[0][0], flan_2[0][0], flan_3[0][0], flan_4[0][0], flan_5[0][0]])
        data_sdmutprob=np.array([flan_1[1][0], flan_2[1][0], flan_3[1][0], flan_4[1][0], flan_5[1][0]])
    else:
        print('incorrect method, use either ML or GF')
        return


    return data_mutprob, data_sdmutprob
    
#============================================================================================
# r=5, placIO single copy plasmid 0.1 (0.5 transpcript/s, Elowitz, low-copy plasmid) x 20 proteins/transcript (Elowitz)
# "average translation efficiency, 20 proteins per transcript"
# placIq: 15, 3x of placIq
# Pupsp2_tot, 60
# placO on p15A, approx 15 copies, tot 50 

# all promoter_activity is for the promoter drives GFP
# lacI binding DNA n = 2.93 ± 0.42, Du et al. 2019 
def const_(promoter_rate, plasmid_mut, N_p):
    if np.max(plasmid_mut)>N_p:
        print('plasmid_mut error')
        return 
        
    promoter_activity_=promoter_rate*plasmid_mut
    return promoter_activity_


# predefined pupsp2_rate_total=100

def IFFL_(Np, pupsp2_rate):
    mutPCN=np.arange(0, Np+1).astype(int)
    promoter_arr=np.zeros(len(mutPCN))

    pupsp2_rate_total=pupsp2_rate
    gfp_product=pupsp2_rate_total/Np*mutPCN

    return gfp_product
    
# no mutation, promoter_activity=promoter_rate*N_p/(1+(N_p/0.5)**2.0)
#-------------------------------------------------------------------------------
def promotermut_varyingCN(promoter_rate, placIq_rate, Np, Kd, Kd_mut, n=2):
    mutPCN=np.arange(0, Np+1).astype(int)
    wt_PCN=Np-mutPCN

    lacIwt_conc=Np*placIq_rate
    promoter_activity_=promoter_rate*wt_PCN/(1+ (lacIwt_conc/Kd)**n)+\
                        promoter_rate*mutPCN/(1+lacIwt_conc/Kd_mut)

    return promoter_activity_, lacIwt_conc


#==============================

def lacINF_promoter_simple(x,Np,mut_PCN,Kd, Kd_mut, r, n):
    wtPCN=Np-mut_PCN
    # assumption LacI per cell is homogenous
    
    degrad=1
    # x is total lacI in cell, assuming most lacI are free
    eq_=r*wtPCN*30/(1+(x/Kd)**n)+r*mut_PCN*30/(1+(x/Kd_mut))-degrad*x
    return eq_


def lacINF_promoter(y,Np, decoy_, mut_PCN,Kd, Kd_mut, r, theta):
    wtPCN=Np-mut_PCN
    degrad=1

    x0=y/(y+Kd)*decoy_
    production=r*wtPCN*10/(1+(1+theta*x0)*(y/Kd))+r*mut_PCN*10/(1+(1+theta*x0)*y/Kd_mut)

    eq_=production-degrad*(y+x0)
    return eq_


def promotermut_NF(promoter_rate, Np, decoy_, Kd, Kd_mut, theta):
    mutPCN=np.arange(0, Np+1).astype(int)
    wtPCN=Np-mutPCN
    promoter_arr=np.zeros(len(mutPCN))
    lacI_arr=np.zeros(len(mutPCN))
    lacI_tot=np.zeros(len(mutPCN))

    P_initial_guess=Kd
    
    for i in range(0,len(mutPCN)): 
        lacI_free=fsolve(lacINF_promoter, P_initial_guess,args=(Np, decoy_, mutPCN[i], Kd, Kd_mut, promoter_rate, theta))[0]
        x0=lacI_free/(lacI_free+Kd)*decoy_

        promoter_arr[i]=promoter_rate*wtPCN[i]/(1+(1+theta*x0)*lacI_free/Kd)+\
                        promoter_rate*mutPCN[i]/(1+(1+theta*x0)*(lacI_free)/Kd_mut)

        
        lacI_arr[i]=(1+x0)*(lacI_free)/Kd_mut
        
    return promoter_arr, lacI_arr



#-------------------------
def lacI_NF(y,Np,decoy_, mut_PCN,Kd, r, theta):
    wtPCN=Np-mut_PCN    
    degrad=1
    x0=y/(y+Kd)*decoy_

    production=r*wtPCN*10/(1+(1+theta*x0)*(y/Kd))

    eq_=production-degrad*(y+x0)#+x1)
     
    return eq_


#-----------------------------
def repressormut_NF_(promoter_rate, Np,decoy_, Kd, theta):
    mutPCN=np.arange(0, Np+1).astype(int)
    promoter_arr=np.zeros(len(mutPCN))
    lacI_arr=np.zeros(len(mutPCN))
    P_initial_guess=Kd
    
    # for n=2, the following code is good
    for i in range(0,len(mutPCN)):
        lacI_free=fsolve(lacI_NF, P_initial_guess,args=(Np, decoy_, mutPCN[i], Kd, promoter_rate, theta))[0]
        x0=lacI_free/(lacI_free+Kd)*decoy_

        promoter_arr[i]=promoter_rate*Np/(1+(1+theta*x0)*lacI_free/Kd)
        
        lacI_arr[i]=lacI_free
    

    return promoter_arr, lacI_arr
    

#====================================
# repressor mutation
def repressormut_varyingCN(promoter_rate, placIq_rate, Np,Kd):
    mutPCN=np.arange(0, Np+1).astype(int)
    plasmid_wt=Np-mutPCN

    # lacI expression under placIq
    lacIwt_conc=plasmid_wt*placIq_rate
    promoter_activity_=promoter_rate*Np/(1+(lacIwt_conc/Kd))   

    return promoter_activity_, lacIwt_conc


#----------------------------------
def repressor_fixedCN(promoter_rate, Np, Kd,pupsp2_rate):
    
    mutPCN=np.arange(0, Np+1).astype(int)
    plasmid_wt=Np-mutPCN
    
    pupsp2_rate_total=pupsp2_rate

    # lacI expression under IFFL regulation, total expression times the fraction of wt ones
    lacIwt_conc=pupsp2_rate_total*plasmid_wt/Np

    # on a p15A plasmid, assuming 10 copies
    promoter_activity_=promoter_rate/(1+(lacIwt_conc/Kd)) 

    return promoter_activity_, lacIwt_conc
