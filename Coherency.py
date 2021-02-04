#!/usr/bin/env python
# coding: utf-8

### Importing required modules
import os
import sys

import time
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# Add custom library

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print("Directory changed to\n{}".format(dname))
sys.path.append('modules/')

import Functions as fnc

### List of animals to process
IDs = ['ID1597', 'ID1659']
#, 'SERT1678', 'SERT1908', 'SERT1984', 'SERT1985', 'SERT2014', 'SERT1668', 'SERT1665', 'SERT2018', 'SERT2024', 'SERT2013']

### Butterworth filter
fs = 1000.0
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band') 
    return b, a

## DUDA: En la documentación btype no tiene la opción band tiene las opciones {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

def Cohy_epoch(data, save = True, points_interval = 300, fs = 20, nps = 250, Tnps = 250, nover = 230 , Tnover = 90, stat = True, Toverlap = 270):
    ''' Devuelve una tupla (f, Cohy_eps, Tf, TCohy_eps). f y ft son arreglos unidimensionales.
        Cohy_eps y TCohy_eps son arreglos multidimensionales con (epoch, channel, channel, f.shape) y
        (epoch, channel, channel, Tf.shape, interval) respectivamente
    '''    

    #(epoch,(f, coherency, Tf, Tcoherency)) 
    ds = data.shape
    Cohy_eps = np.zeros((ds[2]), object)
    TCohy_eps = np.zeros((ds[2]), object)

    comp = fnc.TCoherency(data[:,:,0], points_interval = points_interval, fs_welch = fs, nps = nps, Tnps = Tnps, nover = nover , Tnover = Tnover, stat = stat)[0]
    
    if points_interval:
        comp1 = fnc.TCoherency(data[:,:,0], points_interval = points_interval, fs_welch = fs, nps = nps, Tnps = Tnps, nover = nover , Tnover = Tnover, stat = stat)[2]
    
    
    for epoch in range(ds[2]):
        COHY = fnc.TCoherency(data[:,:,epoch], points_interval = points_interval, fs_welch = fs, nps = nps, Tnps = Tnps, nover = nover , Tnover = Tnover, stat = stat)
    
        #Stationary Coherency
        f = COHY[0]
        Cohy_eps[epoch] = COHY[1]
        
        # I'm not shure if for same fs and kind of windows the array f is the same
        if (comp != f).any():
            print("Problem with frecuency stationary domain")
            return True     
        #Time depending Coherency
        if points_interval:
            Tf = COHY[2]
            TCohy_eps[epoch] = COHY[3]

            # I'm not shure if for same fs and kind of windows the array f is the same
            if (comp1 != Tf).any():
                print("Problem with frecuency time-dependent domain")
                return True
    
    Cohy_eps = np.stack(Cohy_eps)
    TCohy_eps = np.stack(TCohy_eps)
    print(f'Coherency calculated in {time.time() - clock :.2f} seconds')

    if save:
        print("Saving stationary coherency...")
        fnc.savesplitdata(Cohy_eps, f'DATA/Cohy/{ID}_{band}_Cohy')
        np.save(f'DATA/Cohy/{ID}_{band}_freq', f)
        if points_interval:
            print("Saving time-dependent coherency")
            fnc.savesplitdata(TCohy_eps, f'DATA/Cohy/{ID}_{band}_TCohy')
            np.save(f'DATA/Cohy/{ID}_{band}_Tfreq', Tf)
    
    print(f"Stationary Coherency shape: {Cohy_eps.shape}")
    print(f"Frequency shape: {comp.shape} Frequency max: {comp[-1]}")
        
    if points_interval:
        print(f"Time-dependent Coherency shape: {TCohy_eps.shape}")
        print(f"Frequency domain shape {comp1.shape} Frequency max: {comp1[-1]}")
    
    return comp, Cohy_eps, comp1, TCohy_eps

### Filter parameters. You can add more frequency bands (lowcuts and highcuts in Hz)
filter_parameters = {'theta': {'N': 2, 'lowcut': 4,  'highcut': 20}} 

### Creating a filter for each frequency band
butterworths = {key: butter_bandpass(val["lowcut"], val["highcut"], fs, order = val["N"]) for key, val in filter_parameters.items()}

### Main loop
CohysID = dict()
for ID in IDs: 
    Cohys = dict()
    epochs_dir = 'DATA/epochs/'
    print('\nLoading mouse {}...'.format(ID))
    
    ### Loading data
    data = np.load(epochs_dir + ID + '_epochs.npy')
    print(data.shape)

    # Filters the 32 channels matrix along the time_points axis 
    print('Filtering...')    
    for band in filter_parameters.keys():
        filtered = signal.filtfilt(b=butterworths[band][0], a=butterworths[band][1],
                                   x=data, axis=1)

        print(filtered.shape)

        print('Calculating iCoh for {} band...'.format(band))
        clock = time.time()

        n_epochs = filtered.shape[2]
        time_points = filtered.shape[1]
        num_points = 2000 #int(input(f"Enter the size of time interval in ms.\nWrite a number of this set (100,200,300,400,500).\n"))
        inter_num = int(time_points/num_points) 
        
        # Coherency by epoch
        Cohys[band] = Cohy_epoch(filtered, points_interval = 300, fs = 40, nps = 250, Tnps = 250, nover = 230 , Tnover = 230)
        
        #print(f"functions {fnc.Coherency(filtered[:,:,0], points_interval = 2000)[3].shape}")
    CohysID[ID] = Cohys
print('Done!')

# COMENTARIO DEL OUTPUT:
#
# El tiempo que arroja sobrepasa el minuto y el algoritmo solo se demora segundos
 
