#!/usr/bin/env python
# coding: utf-8

# Importing required modules
import os
import sys

import time
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Se recomienda mirar los parámetros noverlap y nperseg de la función signal.csd
# El parámetro fs (fs_welch para esta función) de np.csd al menos para este caso
# llega a la frec 
def Coherency1(data, fs_welch=20, points_interval=0, nps = 250, Tnps = 100, nover = 230 , Tnover = 90):
    '''From a matrix of time series calculates a matrix of coherenecy between each series.
    For two signals (time series) x(t) and y(t) the coherency is defined by Pxy/(PxxPyy)^{1/2}
    where Pxy, Pxx, and Pyy are the power spectral density. The Welch's method with Hanh funtions 
    for the windows are used for this.

    For time-dependet coherency use the parameter point_interval.

    Parameters
    ----------
    data : Matrix with the time series
           Rows -> signals
           Columns -> time

    fs_welch: Sampling (domain) frecuency of power spectral density. Default 20.

    points_interval: A divisor of columns number. If 0 the function only returns 
    stationary coherency. Default 0.

    Returns
    -------
    f: Real array (n,) of frecuency domain where n its the number
    of windows generate by Hanh method and the values are between 0 and fs/2.

    coherency: Complex array (nRows, nRows, n). The element coherency[i,j,k] is the of coherency
    between the i signal and j signal evaluated in the frecuency Tf[k]. 

    Tf: Only if points_interval != 0. Real array (n,) of frecuency domain where n its the number
    of windows generate by Hanh method and the values are between 0 and fs/2.

    Tcoherency: Only if points_interval != 0. Complex array (nRows, nRows, n, n_interval) where 
    n_interval = nColumns/points_interval. The element Tcoherency[i,j,k,z] is the of coherency
    between the i signal and j signal evaluated in the frecuency Tf[k] for z interval. 
    '''

    

    N = data.shape[0]
    
    #Stationary coherency
    f = signal.csd(data[0, :], data[0, :], fs=fs_welch, noverlap = nover, nperseg = nps)[0]
    Nf = f.shape[0]
    
    coherency = np.zeros((N, N, Nf), complex)
    
    for x in np.arange(N):
        for y in np.arange(N):
            if x >= y:
                f1, Pxy = signal.csd(data[x, :], data[y, :], fs=fs_welch, noverlap = nover, nperseg = nps)
                Pxx = np.abs(signal.csd(data[x, :], data[x, :], fs=fs_welch, noverlap = nover, nperseg = nps)[1]) # Each auto power spectra have imaginary part equal to 0
                Pyy = np.abs(signal.csd(data[y, :], data[y, :], fs=fs_welch, noverlap = nover, nperseg = nps)[1])
                
                coherency[x,y,:] = Pxy/(Pxx*Pyy)**(1/2)
                coherency[y,x,:] = Pxy/(Pxx*Pyy)**(1/2)
                
                # I'm not shure if for same fs and kind of windows the array f is the same
                if (f1!=f).any():
                    return True

    Returns = (f, coherency)
    
    # Time dependet coherency
    if points_interval:
        S = points_interval
        
        if data.shape[1]%S != 0:
            print("Error: Enter a valid (divisor of number of columns data) points_interval value.")
            return False

        Nt = int(data.shape[1]/S) #Numbers of interval
        Tf = signal.csd(data[0, :S], data[0, :S], fs=fs_welch ,nperseg = Tnps, noverlap = Tnover)[0]  # Array of frecuencies (The power spectra depends of frecuency)
        TNf = Tf.shape[0] 

        Tcoherency = np.zeros((N, N, TNf, Nt), complex)
        
        for i in range(Nt):
            data_i = data[:, i*S:(i + 1)*S]
            for x in np.arange(N):
                for y in np.arange(N):
                    if x >= y:
                        Tf1, Pxy = signal.csd(data_i[x, :], data_i[y, :], fs=fs_welch, nperseg = Tnps, noverlap = Tnover) 
                        Pxx = np.abs(signal.csd(data_i[x, :], data_i[x, :], fs=fs_welch, nperseg = Tnps, noverlap = Tnover)[1]) # Each auto power spectra have imaginary part equal to 0
                        Pyy = np.abs(signal.csd(data_i[y, :], data_i[y, :], fs=fs_welch, nperseg = Tnps, noverlap = Tnover)[1]) 
                        
                        Tcoherency[x,y,:,i] = Pxy/(Pxx*Pyy)**(1/2)
                        Tcoherency[y,x,:,i] = Pxy/(Pxx*Pyy)**(1/2)
                        
                        # I'm not shure if for same fs and kind of windows the array f is the same
                        if (Tf1!=Tf).any():
                            return True

        Returns += (Tf,Tcoherency)
    
    return Returns 

def Coherency(data, fs_welch=20, nps = 250, nover = 230):
    '''From a matrix of time series calculates a matrix of coherenecy between each series.
    For two signals (time series) x(t) and y(t) the coherency is defined by Pxy/(PxxPyy)^{1/2}
    where Pxy, Pxx, and Pyy are the power spectral density. The Welch's method with Hanh funtions 
    for the windows are used for this.

    For time-dependet coherency use the parameter point_interval.

    Parameters
    ----------
    data : Matrix with the time series
           Rows -> signals
           Columns -> time

    fs_welch: Sampling (domain) frecuency of power spectral density. Default 20.

    points_interval: A divisor of columns number. If 0 the function only returns 
    stationary coherency. Default 0.

    Returns
    -------
    f: Real array (n,) of frecuency domain where n its the number
    of windows generate by Hanh method and the values are between 0 and fs/2.

    coherency: Complex array (nRows, nRows, n). The element coherency[i,j,k] is the of coherency
    between the i signal and j signal evaluated in the frecuency Tf[k]. 

    Tf: Only if points_interval != 0. Real array (n,) of frecuency domain where n its the number
    of windows generate by Hanh method and the values are between 0 and fs/2.

    Tcoherency: Only if points_interval != 0. Complex array (nRows, nRows, n, n_interval) where 
    n_interval = nColumns/points_interval. The element Tcoherency[i,j,k,z] is the of coherency
    between the i signal and j signal evaluated in the frecuency Tf[k] for z interval. 
    '''

    

    N = data.shape[0]
    
    #Stationary coherency
    f = signal.csd(data[0, :], data[0, :], fs=fs_welch, noverlap = nover, nperseg = nps)[0]
    Nf = f.shape[0]
    
    coherency = np.zeros((N, N, Nf), complex)
    
    for x in np.arange(N):
        for y in np.arange(N):
            if x >= y:
                f1, Pxy = signal.csd(data[x, :], data[y, :], fs=fs_welch, noverlap = nover, nperseg = nps)
                Pxx = np.abs(signal.csd(data[x, :], data[x, :], fs=fs_welch, noverlap = nover, nperseg = nps)[1]) # Each auto power spectra have imaginary part equal to 0
                Pyy = np.abs(signal.csd(data[y, :], data[y, :], fs=fs_welch, noverlap = nover, nperseg = nps)[1])
                
                coherency[x,y,:] = Pxy/(Pxx*Pyy)**(1/2)
                coherency[y,x,:] = Pxy/(Pxx*Pyy)**(1/2)
                
                # I'm not shure if for same fs and kind of windows the array f is the same
                if (f1!=f).any():
                    return True
 
    return f, coherency 

def TCoherency(data, fs_welch=20, points_interval=300, nps = 250, Tnps = 100, nover = 230 , Tnover = 90, stat = True, Toverlap = 270):
    '''From a matrix of time series calculates a matrix of coherenecy between each series.
    For two signals (time series) x(t) and y(t) the coherency is defined by Pxy/(PxxPyy)^{1/2}
    where Pxy, Pxx, and Pyy are the power spectral density. The Welch's method with Hanh funtions 
    for the windows are used for this.

    For time-dependet coherency use the parameter point_interval.

    Parameters
    ----------
    data : Matrix with the time series
           Rows -> signals
           Columns -> time

    fs_welch: Sampling (domain) frecuency of power spectral density. Default 20.

    points_interval: A divisor of columns number. If 0 the function only returns 
    stationary coherency. Default 0.

    Returns
    -------
    f: Real array (n,) of frecuency domain where n its the number
    of windows generate by Hanh method and the values are between 0 and fs/2.

    coherency: Complex array (nRows, nRows, n). The element coherency[i,j,k] is the of coherency
    between the i signal and j signal evaluated in the frecuency Tf[k]. 

    Tf: Only if points_interval != 0. Real array (n,) of frecuency domain where n its the number
    of windows generate by Hanh method and the values are between 0 and fs/2.

    Tcoherency: Only if points_interval != 0. Complex array (nRows, nRows, n, n_interval) where 
    n_interval = nColumns/points_interval. The element Tcoherency[i,j,k,z] is the of coherency
    between the i signal and j signal evaluated in the frecuency Tf[k] for z interval. 
    '''

    Return = tuple()

    N = data.shape[0]
    TT = data.shape[1]
    # Stationary Coherency
    if stat:
        Return += Coherency(data, fs_welch=fs_welch, nps = nps, nover = nover) 
    # Time dependet Coherency
    if not points_interval:
        return True

    S = points_interval

    Tf = Coherency(data[:, :S], fs_welch=fs_welch, nps = Tnps, nover = Tnover)[0]  # Array of frequencies (The power spectra depends of frequency)
    TNf = Tf.shape[0] 
        
    #Time windowing
    aux = S - Toverlap
    Nt = (TT  - S) // aux #Number of intervals

    Tcoherency = np.zeros((N, N, TNf, Nt), complex) 
    for i in range(Nt-1):
        data_i = data[:, aux*i:aux*i + S]
        Tf1,  Tcoherencyi = Coherency(data_i, fs_welch=fs_welch, nps = Tnps, nover = Tnover)
        Tcoherency[:,:,:,i] = Tcoherencyi
        # I'm not shure if for same fs and kind of windows the array f is the same
        if (Tf1!=Tf).any():
            return True
    data_i = data[:, TT - S:]
    Tcoherency[:,:,:,Nt-1] = Coherency(data_i, fs_welch=fs_welch, nps = Tnps, nover = Tnover)
    Return +=  (Tf, Tcoherency)

    return Return  