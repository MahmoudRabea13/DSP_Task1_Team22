from asyncore import write
from functools import cache
from operator import truediv
import random 
from itertools import count
from turtle import onclick, width
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import math
from scipy.signal import find_peaks
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.signal import find_peaks
import re

# Initializing two lists for frequencies and amplitudes
if 'frequencies' not in st.session_state:
    st.session_state.frequencies = []
if 'amplitudes' not in st.session_state:
    st.session_state.amplitudes = []


# --- Functions ---

# Plotting Function For Signal Generator
def addSignalplotting(freqList, ampList):
    # time = np.arange(start = 0, stop = 5, step = 0.02)
    time = np.linspace(0, 5, 1000)
    signal = 0  
    iterator = 0     
    while(iterator < len(freqList)):
        signal += ampList[iterator] * np.sin(2 * np.pi * freqList[iterator] * time) 
        iterator += 1    
    power = signal ** 2
    snr_db = SNR_level * 0.6 # add SNR of 20 dB.
    signal_average_power = np.mean(power) # Calculate signal power
    signal_averagepower_db = 10 * np.log10(signal_average_power) #convert signal power to dB
    noise_db = signal_averagepower_db - snr_db # Calculate noise
    noise_watts = 10 ** (noise_db / 10) #convert noise from db to watts
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(signal))
    noise_signal = signal + noise # Noise added to the original signal
    return time, noise_signal


# Sinc Function For Interpolation Equation
def sinc(x):
    try:
        return math.sin(math.pi * x) / (math.pi * x)
    except ZeroDivisionError:
        return 1.0


# Signal Addition For Interpolation Equation 
@st.cache
def signal(time):
    freq = st.session_state.frequencies
    amp = st.session_state.amplitudes
    signal = 0
    iterator = 0
    while(iterator < len(freq)):
        signal += amp[iterator] * np.sin(2 * np.pi * freq[iterator] * time) 
        iterator += 1
    return signal 


# Plotting Function For Signal Reconstruction
def reconstructPlotting(freqList, freq_factor, sRate):  
    fmax= max(freqList)
    time = np.linspace(0, 5, 1100)

    if freq_comp:
        F_sample = freq_factor * fmax
    if sRate:
        F_sample = sRate
    T_sample = 1/F_sample
    no_samples= int(5*F_sample)
    signal_constructed = dict()
    sampled_signal = dict()
    factor = 1

    for n in np.arange(0, no_samples, T_sample):
        sampled_signal[n * T_sample] = signal(n * T_sample)

    for t in time:
        signal_constructed_summation = 0.0
        for n in np.arange(0, no_samples, T_sample):
            signal_constructed_summation += sampled_signal.get(n * T_sample) * sinc(F_sample * (t - (n * T_sample)))
        if freq_factor != 1 and sRate == 0:
            factor= signal(t) / signal_constructed_summation
        elif sRate >= 2 * fmax:
            factor= signal(t) / signal_constructed_summation
        signal_constructed[t] = factor * signal_constructed_summation
    
    
    timeList = list(signal_constructed.keys())
    recSignallist = list(signal_constructed.values())
    Sampled_time= list(sampled_signal.keys())
    sampledSignal = list(sampled_signal.values())
    return timeList, recSignallist, Sampled_time, sampledSignal


# Reconstruction function for uploaded file
def reconstruct(sample_Rate):  
    time = np.arange(-2.5, 2.5 ,0.01)
    F_sample = sample_Rate
    T_sample = 1/F_sample
    no_samples = int(5*F_sample)
    signal_constructed = dict()
    sampled_sin = dict()
    factor = 1
    if F_sample == 2:
        factor= (10 ** (16))/6
    elif F_sample == 4:
        factor = 0.5
    else:
        factor = 1/6
    for n in np.arange(-no_samples/2, no_samples/2, T_sample):
        sampled_sin[n * T_sample] = math.sin(2 * math.pi * 2 * n * T_sample)

    for t in time:
        signal_constructed_summation = 0.0
        for n in  np.arange(-no_samples/2, no_samples/2, T_sample):
            signal_constructed_summation = (sampled_sin.get(n*T_sample)*sinc(F_sample*(t-(n*T_sample)))) + signal_constructed_summation     
        signal_constructed[t] = signal_constructed_summation * factor

    timeList = list(np.arange(0, 5 ,0.01))
    signalList = list(signal_constructed.values())
    return timeList, signalList



# --- WEBSITE ---

st.set_page_config(page_title="Signal Studio",page_icon=":chart_with_upwards_trend:",layout='wide')
st.markdown("""
        <style>
                html {
                    font-size: 0.835rem;
                }
                .css-18e3th9 {
                    padding-top: 2rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
                .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
                .css-k1vhr4 {
                        backface-visibility: hidden;
                        display: flex;
                        flex-direction: column;
                        width: 100%;
                        overflow: hidden;
                        -webkit-box-align: center;
                        align-items: center;
                }
                .css-hxt7ib {
                        padding-top: 2rem;
                        padding-left: 1rem;
                        padding-right: 1rem;
                }
                .css-18e3th9 {
                        padding-left: 5rem;
                        padding-right: 1.2rem;
                }
        </style>
        """, unsafe_allow_html=True)



col1, col2 = st.columns([10, 2.5])

# Browsing section
uploaded_file = st.sidebar.file_uploader('choose a signal',type='csv')

if uploaded_file is not None:
    fmax = 2 
    SNR_level = st.sidebar.slider('Add SNR value', 0, 100, 100, step=10)
    sampling = st.sidebar.radio('sampling method', ['sample rate','Maximum Frequency'])
    if sampling == 'sample rate':
        sampleRate = st.sidebar.slider('Sample Rate', 1, 10, step = 1)
        freq = 0
    if sampling == 'Maximum Frequency':
        freq = st.sidebar.slider('Frequency', fmax, 3*fmax,step = fmax)
        sampleRate = 0
    reconstruct_box = st.sidebar.checkbox('Reconstruct', key = 1)
    st.session_state.frequencies = []
    st.session_state.amplitudes = []
    header_name = ["signal", "time"]
    df = pd.read_csv(uploaded_file,names =  header_name)
    #noise for browsing #
    power = df["signal"]**2
    signalpower_db = 10 * np.log10(power)
    snr_db = SNR_level*(0.6) # add SNR of 20 dB.
    signal_average_power = np.mean(power) # Calculate signal power
    signal_averagepower_db = 10 * np.log10(signal_average_power) #convert signal power to dB
    noise_db = signal_averagepower_db - snr_db # Calculate noise
    noise_watts = 10 ** (noise_db / 10) #convert noise from db to watts
    # Generate a sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(df["signal"]))
    signal_as_list = []
    noise_signal = df["signal"] + noise
    for s in np.sin(noise_signal):
        signal_as_list.append(s)
    #end of noise for browsing
    with col1:
        if sampleRate != 0 :
            plottingFreq = sampleRate 
        if freq != 0:
            plottingFreq = freq
        T = 1/plottingFreq
        n = np.arange(0,5/T,step=0.8) #num of samples
        nT = n*T   #x_axis for the sampling
        y_sampled = np.sin(2*np.pi*2*nT)
        if reconstruct_box :
            layout = go.Layout(autosize=False, width=850, height=500,
                               margin=go.layout.Margin(l=50, r=80, b=100, t=140, pad = 4))
            timeList, signalList = reconstruct(plottingFreq)
            fig = go.Figure(data=[go.Scatter(x = nT, y = y_sampled, mode = 'markers', marker_color= 'blue')], layout= layout)
            fig.add_trace(go.Scatter(x = np.linspace(0,5,df['time'].count()), y =np.sin(noise_signal), name = 'Signal'))
            fig.add_trace(go.Scatter(x = timeList, y = signalList, name = 'Rec. Signal'))
            fig.update_layout(xaxis={'title':'Time (sec)'},yaxis={'title':'Sinusoidal Functions (volt)'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            layout = go.Layout(autosize=False, width=850, height=500,
                               margin=go.layout.Margin(l=50, r=80, b=100, t=140, pad = 4))
            fig = go.Figure(data=[go.Scatter(x = nT, y = y_sampled, mode = 'markers', marker_color= 'blue')], layout= layout)
            fig.add_trace(go.Scatter(x = np.linspace(0,5,df['time'].count()), y =np.sin(noise_signal), name = 'Signal'))
            fig.update_layout(xaxis={'title':'Time (sec)'},yaxis={'title':'Sinusoidal Functions (volt)'})
            st.plotly_chart(fig, use_container_width=True)


# Generation section
if uploaded_file is None:
    with col2:
        if 'frequency' not in st.session_state:
            st.session_state.frequency = 1
        st.session_state.frequency = st.number_input('Frequency', 1, 10, step = 1)

        if 'amplitude' not in st.session_state:
            st.session_state.amplitude = 1
        st.session_state.amplitude = st.number_input('Amplitude', 1, 10, step = 1)

        add = st.button('Add', onclick)
        if add:
           st.session_state.frequencies.append(st.session_state.frequency)
           st.session_state.amplitudes.append(st.session_state.amplitude)

        if 'firstRun' not in st.session_state:
           st.session_state.firstRun = 0

        if st.session_state.frequencies == [] and st.session_state.firstRun == 0:
           st.session_state.frequencies.append(1)
           st.session_state.amplitudes.append(1)

        signalRemoved = []
        for i, a in enumerate(zip(st.session_state.frequencies, st.session_state.amplitudes)):
            signalRemoved.append(f'Frequency={a[0]}, Amplitude={a[1]}')

        if st.session_state.frequencies != []:
            removedSignal = st.selectbox('Remove', signalRemoved)
            removeList = [int(s) for s in re.findall(r'\b\d+\b', removedSignal)]
            removedFrequency = removeList[0]
            removedAmplitude = removeList[1]

            if st.button('ok'):
              st.session_state.frequencies.remove(removedFrequency)
              st.session_state.amplitudes.remove(removedAmplitude)
              st.session_state.firstRun = 1
              st.experimental_rerun()
            

        SNR_level = st.slider('SNR value', 0, 100, 100, step=10)

        reconstructBox = st.checkbox('Reconstruct', key = 2)

    with col1:
        if st.session_state.frequencies != []:
            time, addSignal = addSignalplotting(st.session_state.frequencies, st.session_state.amplitudes)
            layout = go.Layout(autosize=False, width=850, height=500,
            margin=go.layout.Margin(l=50, r=80, b=100, t=140, pad = 4))
            fig = go.Figure(layout= layout)
            fig.add_trace(go.Scatter(x = time, y = addSignal, name = 'Signal'))
            fig.update_layout(xaxis={'title':'Time (sec)'},yaxis={'title':'Sinusoidal Functions (volt)'})    

    with col2:
        if reconstructBox:    
            radioButton = st.radio('sampling method', ['Maximum Frequency', 'sample rate'])
            if radioButton == 'Maximum Frequency':
               freq_comp = st.slider('Frequency Factor', 1, 3, step = 1)
               rateSample = 0
            if radioButton == 'sample rate':
               rateSample = st.slider('sample rate', 1, 15, step = 1)
               freq_comp = 0
        
    with col1:
        if reconstructBox:
            recTime, recSignal, Sampled_time, samplePoints = reconstructPlotting(st.session_state.frequencies, freq_comp, rateSample)
            fig.add_trace(go.Scatter(x = recTime, y = recSignal, name = 'Reconstructed Signal'))
            fig.update_layout(xaxis={'title':'Time (sec)'},yaxis={'title':'Sinusoidal Functions (volt)'})
            fig.add_trace(go.Scatter(x = Sampled_time, y = samplePoints, name = 'Samples', mode = 'markers', marker_color= 'blue' ))

        if st.session_state.frequencies != []:
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

