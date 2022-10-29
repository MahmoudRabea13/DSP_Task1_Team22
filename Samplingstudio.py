from asyncore import write
from functools import cache
from operator import truediv
import random 
from itertools import count
from turtle import onclick
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



if 'frequencies' not in st.session_state:
    st.session_state.frequencies = []
if 'amplitudes' not in st.session_state:
    st.session_state.amplitudes = []


# Plotting Function For Signal Mixer
def addSignalplotting(freq, amp):
    time = np.arange(start = 0, stop = 5, step = 0.01)
    signal = 0
    iterator = 0
    while(iterator < len(freq)):
        signal += amp[iterator] * np.sin(2 * np.pi * freq[iterator] * time) 
        iterator+=1    
    power=signal**2
    snr_db = SNR_level*0.6 # add SNR of 20 dB.
    signal_average_power = np.mean(power) # Calculate signal power
    signal_averagepower_db= 10 * np.log10(signal_average_power) #convert signal power to dB
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
    ReconstFig = plt.figure(figsize=(10,3))
    fmax= max(freq)
    time= np.arange(start = 0, stop = 5, step = 0.01)

    F_sample= freq_comp*fmax
    T_sample = 1/F_sample
    no_samples= int(10*F_sample)
    x_constructed= dict()
    sampled_signal_amplitude= dict()
    factor=1

    for n in np.arange(0,1,T_sample):
        sampled_signal_amplitude[n*T_sample]= signal(0,0,n*T_sample )

    for t in time:
        x_constructed_summation= 0.0
        for n in np.arange(0,1,T_sample):
            x_constructed_summation += sampled_signal_amplitude.get(n*T_sample)*sinc(F_sample*(t-(n*T_sample)))
        if freq_comp!=1:
            factor= signal(0,0,t)/x_constructed_summation
        x_constructed[t] = factor*x_constructed_summation
    
    
    timeList = list(x_constructed.keys())
    recSignallist = list(x_constructed.values())
    return timeList, recSignallist

# Browse Interpolation 
def reconstruct(sample_R):  
    time= np.arange(-2.5, 2.5 ,0.01)
    F_sample= sample_R
    T_sample = 1/F_sample
    no_samples= int(5*F_sample)
    x_constructed= dict()
    sampled_sin= dict()
    factor=1
    if F_sample==2:
        factor= (10**(16))/6
    elif F_sample==4:
        factor=0.5
    else:
        factor= 1/6
    for n in np.arange(-no_samples/2, no_samples/2, T_sample):
        sampled_sin[n*T_sample]= math.sin(2*math.pi*2*n*T_sample)

    for t in time:
        x_constructed_summation= 0.0
        for n in  np.arange(-no_samples/2, no_samples/2, T_sample):
            x_constructed_summation= (sampled_sin.get(n*T_sample)*sinc(F_sample*(t-(n*T_sample)))) + x_constructed_summation     
        x_constructed[t]= x_constructed_summation*factor

    timeList = list(np.arange(0, 5 ,0.01))
    signalList = list(x_constructed.values())
    return timeList, signalList



# --- WEBSITE ---

st.set_page_config(page_title="Signal Studio",page_icon=":chart_with_upwards_trend:",layout='wide')
st.markdown("""
        <style>
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
        </style>
        """, unsafe_allow_html=True)



col1, col2 = st.columns([10, 3])
group_labels = ['Signal', 'Reconstructed Signal']

uploaded_file = st.sidebar.file_uploader('choose a signal',type='csv')



if uploaded_file is not None:
    sampleRate = st.sidebar.slider('Sample Frequency',2,6,step = 2)
    SNR_level=st.sidebar.slider('Add SNR value',0,100, 100, step=10)
    reconstruct_box=st.sidebar.checkbox('Reconstruct', key = 1)
    st.session_state.frequencies = []
    st.session_state.amplitudes = []
    header_name=["signal","time"]
    fig = plt.figure(figsize=(10,3))
    df = pd.read_csv(uploaded_file,names=  header_name)
    #noise for browsing #
    power=df["signal"]**2
    signalpower_db= 10 * np.log10(power)
    snr_db = SNR_level*(0.6) # add SNR of 20 dB.
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
    with col1:
      if sampleRate != 0 :
        T = 1/sampleRate 
        n= np.arange(0,5/T,step=0.8) #num of samples
        nT=n*T   #x_axis for the sampling
        y_sampled= np.sin(2*np.pi*2*nT)
        if reconstruct_box :
            layout = go.Layout(autosize=False, width=750, height=500,
                               margin=go.layout.Margin(l=50, r=80, b=100, t=140, pad = 4))
            timeList, signalList = reconstruct(sampleRate)
            fig = go.Figure(layout= layout)
            fig.add_trace(go.Scatter(x = timeList, y = signalList, name = 'Rec. Signal'))
            fig.update_layout(xaxis={'title':'Time (sec)'},yaxis={'title':'Sinusoidal Functions (volt)'})
            st.plotly_chart(fig)
        else:
            layout = go.Layout(autosize=False, width=750, height=500,
                               margin=go.layout.Margin(l=50, r=80, b=100, t=140, pad = 4))
            fig = go.Figure(data=[go.Scatter(x = nT, y = y_sampled, mode = 'markers', marker_color= 'blue')], layout= layout)
            fig.add_trace(go.Scatter(x = np.linspace(0,5,df['time'].count()), y =np.sin(noise_signal), name = 'Signal'))
            fig.update_layout(xaxis={'title':'Time (sec)'},yaxis={'title':'Sinusoidal Functions (volt)'})
            st.plotly_chart(fig)

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

        st.markdown('###### Remove')        

        removedFrequency = st.number_input('Frequency to be removed', 1, 10, step = 1)
        if st.button('ok'):
                if removedFrequency in st.session_state.frequencies:
                    st.session_state.frequencies.remove(removedFrequency)
                else:
                    with col2:
                       st.warning('Please enter an exiseted component.')
            

        SNR_level = st.slider('SNR value', 0, 100, 100, step=10)

        reconstructBox = st.checkbox('Reconstruct', key = 2)

    with col1:
        if st.session_state.frequencies != []:
            time, addSignal = addSignalplotting(st.session_state.frequencies, st.session_state.amplitudes)
            layout = go.Layout(autosize=False, width=750, height=500,
            margin=go.layout.Margin(l=50, r=80, b=100, t=140, pad = 4))
            fig = go.Figure(layout= layout)
            fig.add_trace(go.Scatter(x = time, y = addSignal, name = 'Signal'))
            fig.update_layout(xaxis={'title':'Time (sec)'},yaxis={'title':'Sinusoidal Functions (volt)'})            
    with col2:
        freq_comp = st.slider('Frequency Factor', 1, 3, step = 1, value = 2)
        
    with col1:
        if reconstructBox:
            recTime, recSignal = reconstructPlotting(st.session_state.frequencies, freq_comp)
            fig.add_trace(go.Scatter(x = recTime, y = recSignal, name = 'Reconstructed Signal'))
            fig.update_layout(xaxis={'title':'Time (sec)'},yaxis={'title':'Sinusoidal Functions (volt)'})

        if st.session_state.frequencies != []:
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig)
