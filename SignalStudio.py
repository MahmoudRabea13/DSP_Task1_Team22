from asyncore import write
from functools import cache
import random 
from itertools import count
from turtle import onclick
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import math

if 'frequencies' not in st.session_state:
    st.session_state.frequencies = []
if 'amplitudes' not in st.session_state:
    st.session_state.amplitudes = []


# --- FUNCTIONS ---

# Plotting Function For Signal Mixer
def addSignalplotting(freq, amp):
    signalFig = plt.figure(figsize = (10,3))
    time = np.arange(start = -5, stop = 5, step = 0.01)
    signal = 0
    iterator = 0
    while(iterator < len(freq)):
        signal += amp[iterator] * np.sin(2 * np.pi * freq[iterator] * time) 
        iterator+=1    
    if np.all(signal == 0):
        st.markdown('##### Please Enter Frequency and Amplitude.')
    else:
        st.markdown('##### Signal Generator')
        plt.plot(time, signal)
    if generate_noise:
        plt.clf()
        power=signal**2
        snr_db = SNR_level*60 # add SNR of 20 dB.
        signal_average_power = np.mean(power) # Calculate signal power
        signal_averagepower_db= 10 * np.log10(signal_average_power) #convert signal power to dB
        noise_db = signal_averagepower_db - snr_db # Calculate noise
        noise_watts = 10 ** (noise_db / 10) #convert noise from db to watts
        # Generate an sample of white noise
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(signal))
        noise_signal = signal + noise # Noise added to the original signal
        plt.clf()
        plt.plot(time,noise_signal)
    plt.xlabel('Time')
    plt.ylabel('Sum(sine functions)')
    st.pyplot(signalFig)


# Sinc Function For Interpolation Equation
def sinc(x):
    try:
        return math.sin(math.pi*x) / (math.pi*x)
    except ZeroDivisionError:
        return 1.0



# Signal Addition For Interpolation Equation 
@st.cache
def signal(iterator = 0,signal = 0, time=0):
    freq= st.session_state.frequencies
    amp= st.session_state.amplitudes
    while(iterator < len(freq)):
        signal += amp[iterator] * np.sin(2 * np.pi * freq[iterator] * time) 
        iterator+=1
    return signal 


# Plotting Function For Signal Reconstruction
def reconstructPlotting(freq,freq_comp):  
    st.markdown('##### Reconstructed Signal')
    ReconstFig = plt.figure(figsize=(10,3))
    fmax= max(freq)
    time= np.arange(start = -5, stop = 5, step = 0.01)
    signal_as_dictionary= dict()
    sig_nal= signal(0,0,time)
    for t, s in zip(time,sig_nal):
        signal_as_dictionary[t]= s

    F_sample= freq_comp*fmax
    T_sample = 1/F_sample
    no_samples= int(20*F_sample)
    x_constructed= dict()
    sampled_signal_amplitude= dict()
    for n in np.arange(-no_samples, no_samples, T_sample):
            sampled_signal_amplitude[n*T_sample]= signal(0,0,n*T_sample )

    for t in signal_as_dictionary.keys():
        x_constructed_summation= 0.0
        for n in np.arange(-no_samples, no_samples, T_sample):
            x_constructed_summation= sampled_signal_amplitude.get(n*T_sample)*sinc(F_sample*(t-(n*T_sample))) + x_constructed_summation
        x_constructed[t] = x_constructed_summation

    plt.plot(x_constructed.keys(), x_constructed.values())
    plt.xlabel('Time')
    plt.ylabel('Reconstructed Signal')
    plt.gca().axes.get_yaxis().set_visible(False)
    st.pyplot(ReconstFig)

# Browse Interpolation 
def reconstruct(sample_R):  
    st.markdown('##### Reconstructed Signal')
    ReconstFig = plt.figure(figsize=(8,4.5))
    time= np.arange(-2.5, 2.5 ,0.01)
    F_sample= sample_R
    T_sample = 1/F_sample
    no_samples= int(5*F_sample)
    x_constructed= dict()
    sampled_sin= dict()
    for n in np.arange(-no_samples, no_samples, T_sample):
        sampled_sin[n*T_sample]= math.sin(2*math.pi*2*n*T_sample)

    for t in time:
        x_constructed_summation= 0.0
        for n in  np.arange(-no_samples, no_samples, T_sample):
            x_constructed_summation= (sampled_sin.get(n*T_sample)*sinc(F_sample*(t-(n*T_sample)))) + x_constructed_summation     
        x_constructed[t]= x_constructed_summation

    plt.plot(x_constructed.keys(),x_constructed.values(),'b-')
    plt.xlabel('Time')
    plt.ylabel('Reconstructed Signal')
    st.plotly_chart(ReconstFig)


# --- WEBSITE ---

st.set_page_config(page_title="Signal Studio",page_icon=":chart_with_upwards_trend:",layout='wide')
st.sidebar.header('Tools')
browse = st.sidebar.checkbox('Browse')
generate = st.sidebar.checkbox('Generate a Singal')

col1, col2 = st.columns([3, 1])

# Browsing 
if browse and not generate:
    header_name=["signal","time"]
    uploaded_file = st.file_uploader('choose a signal',type='csv')
    if uploaded_file is not None:
        # st.markdown('---')
        st.markdown('##### Signal')
        fig = plt.figure(figsize=(8,4.5))
        df = pd.read_csv(uploaded_file,names=  header_name)
        sampleRate = st.sidebar.slider('Sample Frequency',2,6,step = 2)
        SNR_level=st.sidebar.number_input('Add SNR value',0.0,1.0,1.0,step=0.01)
        #noise for browsing #
        power=df["signal"]**2
        signalpower_db= 10 * np.log10(power)
        snr_db = SNR_level*(60) # add SNR of 20 dB.
        signal_average_power = np.mean(power) # Calculate signal power
        signal_averagepower_db= 10 * np.log10(signal_average_power) #convert signal power to dB
        noise_db = signal_averagepower_db - snr_db # Calculate noise
        noise_watts = 10 ** (noise_db / 10) #convert noise from db to watts
        # Generate a sample of white noise
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(df["signal"]))
        signal_as_list= []
        noise_signal = df["signal"] + noise
        for s in np.sin(noise_signal):
            signal_as_list.append(s)
        #end of noise for browsing
        if sampleRate != 0 :
            T = 1/sampleRate 
            n= np.arange(0,5/T,step=0.8) #num of samples , step ????????????/
            nT=n*T   #x_axis for the sampling
            y_sampled= np.sin(2*np.pi*2*nT)
            reconstruct_box=st.sidebar.checkbox('Reconstruct')
            if reconstruct_box :
                plt.plot(nT,y_sampled,'ro')
                plt.plot(np.linspace(0,5,df['time'].count()),np.sin(noise_signal),'b-')
                plt.xlabel('Time')
                plt.ylabel('Signal')
                st.plotly_chart(fig)
                reconstruct(sampleRate)
            else:
                plt.plot(nT,y_sampled,'ro')
                plt.plot(np.linspace(0,5,df['time'].count()),np.sin(noise_signal),'b-')
                plt.xlabel('Time')
                plt.ylabel('Signal')
                st.plotly_chart(fig)


# Signal Mixer

with col1:
    if generate and not browse:
        if 'frequency' not in st.session_state:
           st.session_state.frequency = 1
        st.session_state.frequency = st.sidebar.number_input('Frequency', 1, 10, step = 1)

        if 'amplitude' not in st.session_state:
           st.session_state.amplitude = 1
        st.session_state.amplitude = st.sidebar.number_input('Amplitude', 1, 10, step = 1)

        add = st.sidebar.button('Add', onclick)
        if add:
           st.session_state.frequencies.append(st.session_state.frequency)
           st.session_state.amplitudes.append(st.session_state.amplitude)        

        if st.session_state.frequencies != []:
            removeByfrequency = st.sidebar.checkbox('Remove by frequency', key = 1)
            removeByamplitude = st.sidebar.checkbox('Remove by amplitude', key = 2)
            if removeByfrequency and not removeByamplitude:
                removedFrequency = st.sidebar.number_input('Frequency to be removed', 1, 10, step = 1)
                i = 0
                while i < len(st.session_state.frequencies):
                   if removedFrequency == st.session_state.frequencies[i]:
                       removedAmplitude = st.session_state.amplitudes[i]
                       break
                   i += 1
                if st.sidebar.button('ok'):
                   if removedFrequency in st.session_state.frequencies:
                      st.session_state.frequencies.remove(removedFrequency)
                      st.session_state.amplitudes.remove(removedAmplitude)
                   else:
                      st.sidebar.warning('Please enter an exiseted component.')
            elif removeByamplitude and not removeByfrequency:
                removedAmplitude = st.sidebar.number_input('Amplitude to be removed', 1, 10, step = 1)
                i = 0
                while i < len(st.session_state.amplitudes):
                   if removedAmplitude == st.session_state.amplitudes[i]:
                      removedFrequency = st.session_state.frequencies[i]
                      break
                   i += 1
                if st.sidebar.button('ok'):
                   if removedAmplitude in st.session_state.amplitudes:
                      st.session_state.frequencies.remove(removedFrequency)
                      st.session_state.amplitudes.remove(removedAmplitude)
                   else:
                      st.sidebar.warning('Please enter an exiseted component.')
            elif removeByamplitude and removeByfrequency:
                removedFrequency = st.sidebar.number_input('Frequency to be removed', 1, 10, step = 1)
                removedAmplitude = st.sidebar.number_input('Amplitude to be removed', 1, 10, step = 1)
                if st.sidebar.button('ok'):
                   if removedFrequency in st.session_state.frequencies and removedAmplitude in st.session_state.amplitudes:
                      st.session_state.frequencies.remove(removedFrequency)
                      st.session_state.amplitudes.remove(removedAmplitude)
                   else:
                      st.sidebar.warning('Please enter an exiseted component.')
              

            generate_noise = st.sidebar.checkbox('Noise')
            if generate_noise:
                SNR_level = st.sidebar.number_input('SNR value', 0.0, 1.0,1.0, step=0.01)

            addSignalplotting(st.session_state.frequencies, st.session_state.amplitudes)
            

            reconstruct = st.sidebar.checkbox('Reconstruct')           
            if reconstruct:
                freq_comp = st.sidebar.slider('Freq', 1, 3, step = 1, value = 2)
                reconstructPlotting(st.session_state.frequencies, freq_comp)

            df1 = pd.DataFrame(st.session_state.frequencies, columns=['Frequency'])
            df2=  pd.DataFrame(st.session_state.amplitudes, columns=['Amplitude'])
            df = pd.DataFrame.to_csv(pd.concat([df1,df2],axis=1)) 
            save = st.sidebar.download_button(data=df,label='Save',file_name='signal.csv')

            with col2:
               if len(st.session_state.frequencies) != 0: 
                 dfTable = pd.DataFrame(pd.concat([df1,df2], axis=1)) 
                 st.table(dfTable)

        else:
            st.markdown('##### Please Enter Frequency and Amplitude')

        
if not generate:
    st.session_state.frequencies = []
    st.session_state.amplitudes = []
    

if not generate and not browse:
    st.markdown('# Welcome To Signal Studio :wave:')
if generate and browse:
    st.markdown('# Kindly choose only one mode :pray:.')

# -- End Of Signal Mixer --
