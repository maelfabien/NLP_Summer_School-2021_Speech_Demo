#!/usr/bin/env python

# @Authors: Johan Rohdin
# @Email: rohdin@fit.vutbr.cz

# Kaldi style energy based VAD based on feature extraction code in the VBx recipe.
# Based on "predict.py" by Lukas Burget, Federico Landini, Jan Profant in the
# BUT-Phonexia VBx recipe and voice-activity-detection.cc/h by Vassil Panayotov,
# Matthew Maciejewski, Daniel Povey in Kaldi.

import argparse, glob, os.path, shutil
from src.VBx import features
import soundfile as sf
import numpy as np
import re
import os
from scipy.io import wavfile

def run_vad(input_dir:str, output_dir:str):
    """
    Run VAD on an input directory, and generates output directory
    """

    dither_type = "N" 
    dither_value = 1
    vad_energy_threshold = 5.5
    vad_energy_mean_scale = 0.5
    vad_frames_context = 2
    vad_proportion_threshold = 0.12

    f_out = vad(input_dir, output_dir, dither_type, dither_value, vad_energy_threshold, vad_energy_mean_scale, vad_frames_context, vad_proportion_threshold)
    return f_out

def vad(filename, out_dir, dither_type, dither_value, vad_energy_threshold, vad_energy_mean_scale, vad_frames_context, vad_proportion_threshold):
    """
    Kaldi-style energy-based voice activity detection
    """

    f = filename

    np.random.seed(3)  
    assert(vad_energy_mean_scale >= 0.0)
    
    file_name = f.split("/")[-1][:-4]
    signal, samplerate = sf.read(f)

    # NOTE: Because we don't want any other features than energy, we set NUMCHANS=0.
    #       This means fbank_mx.shape=(nfft/2+1,0) = (129,0) in this case. I.e. it is
    #       an empty array.

    if samplerate == 8000:
        noverlap = 120   # 10ms * 8 sample/ms shift => 200 - 80 = 120 overlap
        winlen   = 200   # 25ms * 8 sample/ms
        window   = features.povey_window(winlen)
        fbank_mx = features.mel_fbank_mx(
            winlen, samplerate, NUMCHANS=0, LOFREQ=20.0, HIFREQ=3700, htk_bug=False)
    elif samplerate == 16000:
        noverlap = 240
        winlen   = 400
        window   = features.povey_window(winlen)
        fbank_mx = features.mel_fbank_mx(
            winlen, samplerate, NUMCHANS=0, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
    else:
        raise ValueError(f'Only 8kHz and 16kHz are supported. Got {samplerate} instead.')
    
    # Make the signal signed integer values although still as type float
    signal = (signal*2**15)

    # Apply dither
    if ( dither_type == "U" ):
        signal = features.add_dither(signal.astype(int), int(dither_value) )      
    elif ( dither_type == "N" ):
        signal += np.random.randn(signal.shape[0])*dither_value
    else:
        print("WARNING dither is not used")

    # Mirror noverlap//2 initial and final samples (as Kaldi does)
    signal = np.r_[signal[noverlap // 2 - 1::-1],
                signal, signal[-1:-winlen // 2 - 1:-1]]

    
    log_energy = features.fbank_htk(signal, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True, _E="first", ENORMALISE=False)
    log_energy = np.squeeze( log_energy )
    
    energy_threshold  = vad_energy_threshold + vad_energy_mean_scale * np.mean(log_energy)
    
    # This does the Kaldi style VAD smoothing.
    vad = np.zeros_like( log_energy )
    for t in range(len(log_energy)):
        num_count = den_count = 0
        context = vad_frames_context
        for t2 in range( t - context, t + context+1):
            if (t2 >= 0 and t2 < len(log_energy)):
                den_count +=1
                if (log_energy[t2] > energy_threshold):
                    num_count +=1
            
            if (num_count >= den_count * vad_proportion_threshold):
                vad[t] = 1.0
            else:
                vad[t] = 0.0

    # Convert to HTK MLF format (with seconds as units) Print the VAD output
    f_out = file_name + ".lab"

    #f_out = out_dir + "/" + re.sub('\.wav$', '',f).split("/")[-1] + "/" + re.sub('\.wav$', '',f).split("/")[-1] + ".lab"
    with open(f_out, "w") as f:
        
        prev_sym = -1;
        n=0;
        speech_start =0;
        
        for sym in vad:
            if (sym == 1):
                if (prev_sym != 1):
                    speech_start = n/100 #*100000
                    prev_sym = 1;    

            elif(sym == 0):
                if (prev_sym == 1):
                    speech_end = n/100  #*100000
                    f.write(str(speech_start) + " " + str(speech_end) + " sp\n")
                    prev_sym = 0;    
                
            else:
                print("ERROR")
                
            n +=1
        
        # If last one was speech we need to print it.
        if (prev_sym == 1):
            speech_end = n/100 #*100000
            f.write(str(speech_start) + " " + str(speech_end) + " sp\n")
            prev_sym = 0
    
    signal, samplerate = sf.read(filename)

    full_signal = []

    f = open(f_out, "r")
    for line in f:

        times = line.split(" ")
        start = int(float(times[0])*samplerate)
        end = int(float(times[1])*samplerate)

        data_split = np.array(signal)[start:end]
        full_signal.extend(data_split)
    
    full_signal = np.array(full_signal)

    wavfile.write(out_dir, samplerate, full_signal)
