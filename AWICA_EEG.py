#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:47:07 2021

@author: Claudio LoBraico

The following script reads an edf file from the TUH EEG Corpus, then performs 
automatic artifact mitigation based on a given start and end time.  

Note that nedc_pystream is given on TUH EEG Corpus Database.

The algorithm used is automatic wavelet-independent component analysis (AWICA)
popularized by Mammone et al (2007). Entropy (randomness) and kurtosis (peakiness)
are used to determine whether a given component is artifactual.


"""
import pandas as pd
#import pyedflib
#from dit import other
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from pywt import wavedec
from pywt import waverec
import nedc_pystream as nedc


# function to window the data
# (Taken from Dr. Mogul's Lab,written originally by Ashley Cahoon, edited by Claudio LoBraico)
def slicing_window(a,w_size,overlap):
    #new_arr = np.zeros(shape=(1,len(a)//w_size))
    new_arr = []
    # if the length of the data is less than the selected window length, return the original array 
    if(len(a)< w_size):
        return a
    
    # i increases by the overlap size, bounded by the total length of the array
    for i in range(0,len(a),overlap): # range(start,stop,step); (0, len(data), step size to increase by with the rows); 5 sec overlap = 12500
      
        # only appends the data if there is a full window size remaining
        if len(a[i:i+w_size])==w_size:
            new_arr.append(a[i:i+w_size])
#            np.array(new_arr.append(a[i:i+n]))
        
        else:
            break
    return new_arr

# find number of bins for a 2D array (wIMF in this case)
# Originally written by Claudio LoBraico for Dr. Mogul's Lab
def find_bins(data):
    bins =  np.zeros([len(data)]).astype(int)
    for i in range(len(data)):
        N = len(data)
        bin_width = 2* stats.iqr(data[i]) * N**(-1/3)
        num_bins = (data[i].max() - data[i].min()) / bin_width
        bins[i] = np.ceil(num_bins)
    return bins


# to compute shannon entropy
def shan_entropy(w_data,bins):
    c_data = np.histogram(w_data,bins)[0]      #counts for w_data for given bin size
    non_zero_c = c_data[c_data != 0]
    h_data = -sum(non_zero_c * np.log2(non_zero_c))
    return h_data

# compute renyi's entropy for each window (alpha = 2)
def renyi_entropy(w_data,bins):
    alpha = 2
    # find prob, not counts 
    c_data = np.histogram(w_data,bins)[0]      #counts for w_data for given bin size
    non_zero_c = c_data[c_data != 0]
    #non_zero_p = non_zero_c/
    p_2 = np.linalg.norm(non_zero_c,ord = alpha)
    h_data = (1/(1-alpha)) * np.log2((p_2))
    return h_data
 

#------------------------------------------------------------------------------
# 
# DATA PARSING AND INITIALIZATION
#
#------------------------------------------------------------------------------

   
# LOADING DATA ----------------------------------------------------------------

edf_file =  '00001402_s004_t000.edf'
fsamp, sig, labels = nedc.nedc_load_edf(edf_file)


eeg_dict = dict()
for i in range(len(sig)-4):
    if (labels[i] != 'EEGEKG1-REF')  & (labels[i] != 'EMG-REF') :
        eeg_dict[labels[i]] = sig[i]    


eeg_df = pd.DataFrame.from_dict(eeg_dict)


# SPLICING DATA TO FOCUS ON REPORTED ARTIFACT ---------------------------------

# Reported eye movement artifact between 75 and 83s.

montage = input('Enter desired electrode montage: ')
print(montage)

start_t = int(input('Enter start time: '))
print(start_t)
end_t = int(input('Enter end time: '))
print(end_t)

art_c0 = int(input('Enter artifactual channel 1: '))
print(art_c0)
art_c1 = int(input('Enter artifactual channel 2: '))
print(art_c1)

#start_t = 54
#end_t = 62
#art_c0 = 0
#art_c1 = 1
start = start_t * fsamp[0]   # sample rate should be same for each channel
end = end_t * fsamp[0]     



ch0 = sig[0][start:end]     #EEGFP1-REF
ch1 = sig[1][start:end]     #EEGFP2-REF
ch2 = sig[2][start:end]     #EEGF3-REF
ch3 = sig[3][start:end]     #EEGF4-REF
ch4 = sig[4][start:end]     #EEGC3-REF
ch5 = sig[5][start:end]     #EEGC4-REF
ch6 = sig[6][start:end]     #EEGP3-REF
ch7 = sig[7][start:end]     #EEGP4-REF
ch8 = sig[8][start:end]     #EEGO1-REF
ch9 = sig[9][start:end]     #EEGO2-REF
ch10 = sig[10][start:end]   #EEGF7-REF
ch11 = sig[11][start:end]   #EEGF8-REF
ch12 = sig[12][start:end]   #EEGT3-REF
ch13 = sig[13][start:end]   #EEGT4-REF
ch14 = sig[14][start:end]   #EEGT5-REF
ch15 = sig[15][start:end]   #EEGT6-REF
ch16 = sig[16][start:end]   #EEGA1-REF
ch17 = sig[17][start:end]   #EEGA2-REF
ch18 = sig[18][start:end]   #EEGFZ-REF
ch19 = sig[19][start:end]   #EEGCZ-REF
ch20 = sig[20][start:end]   #EEGPZ-REF
ch21 = sig[21][start:end]   #EEGROC-REF
ch22 = sig[22][start:end]   #EEGLOC-REF
ch23 = sig[23][start:end]   #EEGEKG1-REF
ch24 = sig[24][start:end]   #EMG-REF
ch25 = sig[25][start:end]   #EEG26-REF
ch26 = sig[26][start:end]   #EEG27-REF
ch27 = sig[27][start:end]   #EEG28-REF'
ch28 = sig[28][start:end]   #EEG29-REF
ch29 = sig[29][start:end]   #EEG30-REF
ch30 = sig[30][start:end]   #EEGT1-REF
ch31 = sig[31][start:end]   #EEGT2-REF
ch32 = sig[32][start:end]   #PHOTIC-REF
ch33 = sig[33][start:end]   #IBI
ch34 = sig[34][start:end]   #BURSTS
ch35 = sig[35][start:end]   #SUPR


# Selecting montage (i.e. channels)

if montage == 'Mammone':
    eeg = np.array([ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9])
    labels = np.array([labels[2],labels[3],labels[4],labels[5],labels[6],labels[7],labels[8],labels[9]])

elif montage == 'Z3':
    eeg = np.array([ch0,ch10,ch12])
    labels = np.array([labels[0],labels[10],labels[12]])

else:
    eeg = np.array([ch0,ch1,ch4,ch5,ch10,ch11,ch14,ch15])
    labels = np.array([labels[0],labels[1],labels[4],labels[5],labels[10],labels[11],labels[14],labels[15]])


# Making pd dataframe (not currently used)
eeg_df = eeg_df.iloc[start:end]

# Plot for visualizing artifact
for i in range(len(eeg)):
    plt.plot(eeg[i], label=labels[i])

plt.xlabel('Time [samples]', fontsize=14, labelpad=10)
plt.ylabel('Voltage [\u03BCV]', fontsize=14)
plt.title('Resting state EEG ({} channels)'.format(len(eeg)), fontsize=14)
#plt.savefig('eeg_all.png')
plt.show()


# visualizing channels of interest
fig, axs = plt.subplots(2,1, figsize=(15, 7), sharex=True, sharey=True)
axs = axs.ravel()
plt.margins(x=0.001)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs[0].plot(eeg[art_c0], label=labels[art_c0], color='rosybrown')
axs[0].legend(loc="upper right", fontsize=12)
axs[1].plot(eeg[art_c1], label=labels[art_c1], color='silver')
axs[1].legend(loc="upper right", fontsize=12)
plt.xlabel('Time [samples]', fontsize=14, labelpad=15)
plt.ylabel('Voltage [\u03BCV]', fontsize=14, labelpad=15)
plt.title('Pre Artifact Mitigation')
#plt.savefig('fp1_fp2.png')
plt.show()

#------------------------------------------------------------------------------
# 
# ARTIFACT MITIGATION
#
#------------------------------------------------------------------------------


# STEP 1: WAVELET DECOMPOSITION ------------------------------------------------


# Level = log2(fs/N) - 1 , where N is the number wavelet selected
# In this case, level = 6 because we have a 250 Hz sampling rate with N = 4

level = 6
ch_coeffs = []

# for each channel, find wavelet coefficients
for i in range(len(eeg)):
    ch_coeffs.append(wavedec(eeg[i],'db4',level=level))
 



# STEP 2: CRITICAL WC SELECTION -----------------------------------------------

# 2.A Kurtosis 
# Find kurtosis of each coeff array within each channel, then normalize to zero mean unit var 

kurt = []

#   Finding kurtosis for each WC within each channel

# for each channel
for i in range(len(ch_coeffs)):
    ch_kurt = []
    
    # for each wavelet coeff array
    for j in range(len(ch_coeffs[i])):
        coeff_kurt = stats.kurtosis(ch_coeffs[i][j])
        ch_kurt.append(coeff_kurt)
    
    #   Normalizing 
    ch_kurt = stats.zscore(ch_kurt)   
    kurt.append(ch_kurt)




# 2.B Renyi's Entropy
# Calculate renyi's entropy of order 2, then normalize to zero mean, unit var

#   Must first bin the data
bins = []
for i in range(len(ch_coeffs)):
    bins.append(find_bins(ch_coeffs[i]))


#   Finding entropy within each WC within each channel
entropy = []
for i in range(len(ch_coeffs)):
    ch_entropy = []
    for j in range(len(ch_coeffs[i])):
        coeffs_entropy = (renyi_entropy(ch_coeffs[i][j], bins[i][j]))
        ch_entropy.append(coeffs_entropy)
    
    # Normalizing
    ch_entropy = stats.zscore(ch_entropy)
    entropy.append(ch_entropy)
    

# 2.C Selecting Critical WCs 


#   Question to review: why are we using abs of entropy and kurtosis? (neg entropy...)
entropy_thresh = 1.3
kurt_thresh = 1.3

crit_coeffs_j = set()
for i in range(len(ch_coeffs)):
    for j in range(len(ch_coeffs[i])):
        if (abs(entropy[i][j]) > entropy_thresh) | (abs(kurt[i][j]) > kurt_thresh):
            crit_coeffs_j.add(j)




# STEP 3: WAVELET INDEPENDENT COMPONENT (WIC) EXTRACTION-----------------------------------------------
# STEP 4: ARTIFACTUAL WIC SELECTION -----------------------------------------------
# STEP 5: ARTIFACTUAL WIC DELETION -----------------------------------------------


# ICA on wavelet coefficients, THEN window, then determine if artifactual
# for each channel

# for each critical WC
for j in crit_coeffs_j:
    
    '''
    # for each channel
    for i in range(len(ch_coeffs)):
        plt.plot(np.arange(len(ch_coeffs[i][j])),ch_coeffs[i][j])
        
    plt.xlabel('Time [samples]', fontsize=14, labelpad=10)
    plt.ylabel('Voltage [\u03BCV]', fontsize=14)
    plt.title('WC {} pre-cleaning({} channels)'.format(j,len(eeg)), fontsize=14)
    #plt.savefig('eeg_all.png')
    plt.show()
    '''
    #for each channel within a given critical WC
    x_data = ch_coeffs[0][j]
    for i in range(1,len(ch_coeffs)):
        crit_IC_i = []
        x_data = np.vstack((x_data,ch_coeffs[i][j]))
    
    x_data = x_data.T
    ica = FastICA(n_components = len(eeg) , random_state = 0,tol = .01,max_iter=1000)
    ICs = ica.fit_transform(x_data)
        
    ICs = ICs.T                 # transposing to make into a row vector
    ICs_bins = find_bins(ICs)
        
    # window each independent component into trials, figure out if critical or not (using entropy/kurtosis)
    for i in range(len(ICs)):
        num_samples = len(ICs[i])
        fs = num_samples / (end_t - start_t)
        w_size = int(np.ceil(.5 * fs))    # .5s second windows
        overlap = w_size                  # no overlap
        
        comp_trials = slicing_window(ICs[i],w_size,overlap)
        trials_bins = find_bins(comp_trials)
        
        kurt_trials = []
        entropy_trials = []
        #for each window
        for k in range(len(comp_trials)):
            kurt_trials.append(stats.kurtosis(comp_trials[k]))
            entropy_trials.append(renyi_entropy(comp_trials[k], trials_bins[k]))
        
        kurt_trials = stats.zscore(kurt_trials)
        entropy_trials = stats.zscore(entropy_trials)
        
        crit_trials = 0
        #for each window
        for k in range(len(kurt_trials)):
            if (abs(kurt_trials[k]) > kurt_thresh) or (abs(entropy_trials[k]) > entropy_thresh):
                crit_trials += 1
        
        percent_crit = crit_trials / len(comp_trials)
    
        if percent_crit > .2:
            # keep track of index for deletion !
            crit_IC_i.append(i)
            
        
    cleaned_ICs = ICs.copy()
    for i in crit_IC_i:
        cleaned_ICs[i] = 0
    
    cleaned_ICs = cleaned_ICs.T
    cleaned_x_data = ica.inverse_transform(cleaned_ICs)
    cleaned_x_data = cleaned_x_data.T
    
    
    # restoring WCs
    for i in range(len(ch_coeffs)):
        ch_coeffs[i][j] = cleaned_x_data[i]
    '''    
    plt.plot(np.arange(len(ch_coeffs[i][j])),ch_coeffs[i][j])
    
    plt.xlabel('Time [samples]', fontsize=14, labelpad=10)
    plt.ylabel('Voltage [\u03BCV]', fontsize=14)
    plt.title('Channels following artifact mitigation for WC {}({} channels)'.format(j,len(eeg)), fontsize=14)
    #plt.savefig('eeg_all.png')
    plt.show()
    ''' 


    
# restoring channels
restored_eeg = eeg.copy()
for i in range(len(eeg)):
    restored_eeg[i] = waverec(ch_coeffs[i],'db4')        
    # plot
    plt.plot(np.arange(len(restored_eeg[i])),restored_eeg[i]) 
    #eeg_df.iloc[start:end].plot(figsize=(15,5), legend=False)
    
plt.xlabel('Time [samples]', fontsize=14, labelpad=10)
plt.ylabel('Voltage [\u03BCV]', fontsize=14)
plt.title('EEG post-artifact mitigation ({} channels)'.format(len(eeg)), fontsize=14)
#plt.savefig('eeg_all.png')
plt.show()


# visualizing channels post artifact mitigation
fig, axs = plt.subplots(2,1, figsize=(15, 7), sharex=True, sharey=True)
axs = axs.ravel()
plt.margins(x=0.001)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs[0].plot(eeg[art_c0], label=(labels[art_c0]+'_pre'), color='rosybrown')
axs[0].plot(restored_eeg[art_c0], label=(labels[art_c0] + '_post'), color='maroon')
axs[0].legend(loc="upper right", fontsize=12)
axs[1].plot(eeg[art_c1], label=(labels[art_c1]+'_pre'), color='silver')
axs[1].plot(restored_eeg[art_c1], label=(labels[art_c1]+'_post'), color='dimgray')
axs[1].legend(loc="upper right", fontsize=12)
plt.xlabel('Time [samples]', fontsize=14, labelpad=15)
plt.ylabel('Voltage [\u03BCV]', fontsize=14, labelpad=15)
plt.savefig('post_AWICA_{}_{}.png'.format(start_t,end_t))
plt.show()

