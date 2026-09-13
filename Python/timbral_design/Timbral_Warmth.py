from __future__ import division
import numpy as np
import soundfile as sf
import scipy.fftpack as fft
from scipy.signal import spectrogram
import scipy.stats
from sklearn import linear_model
from . import timbral_util
from pprint import pprint


def warm_region_cal_dft(audio_samples, fs):
    """
      Function for calculating various warmth parameters.

    :param audio_samples:   numpy.array, an array of the audio samples, reques only one dimension.
    :param fs:              int, the sample ratr of the audio file.

    :return:                four outputs: mean warmth region, weighted-average warmth region, mean high frequency level,
                            weighted-average high frequency level.
    """
    # Window the audio
    windowed_samples = timbral_util.window_audio(audio_samples)

    # Define a function for the roughness stimuli, emphasising the 20 - 40 region (of the bark scale)
    min_bark_band = 10
    max_bark_band = 40
    mean_bark_band = (min_bark_band + max_bark_band) / 2.0
    array = np.arange(min_bark_band, max_bark_band)
    x = timbral_util.normal_dist(array, theta=0.01, mean=mean_bark_band)
    x -= np.min(x)
    x /= np.max(x)

    wr_array = np.zeros(240)
    wr_array[min_bark_band:max_bark_band] = x

    # Define a second array emphasising the 20 - 40 region (of the bark scale)
    min_bark_band = 80
    max_bark_band = 240
    mean_bark_band = (min_bark_band + max_bark_band) / 2.0
    array = np.arange(min_bark_band, max_bark_band)
    x = timbral_util.normal_dist(array, theta=0.01, mean=mean_bark_band)
    x -= np.min(x)
    x /= np.max(x)

    hf_array = np.zeros(240)
    hf_array[min_bark_band:max_bark_band] = x

    windowed_loud_spec = []
    windowed_rms = []

    wr_vals = []
    hf_vals = []

    for i in range(windowed_samples.shape[0]):
        samples = windowed_samples[i, :]
        N_entire, N_single = timbral_util.specific_loudness(samples, Pref=100.0, fs=fs, Mod=0)

        # Append the loudness spec
        windowed_loud_spec.append(N_single)
        windowed_rms.append(np.sqrt(np.mean(samples * samples)))

        wr_vals.append(np.sum(wr_array * N_single))
        hf_vals.append(np.sum(hf_array * N_single))

    mean_wr = np.mean(wr_vals)
    mean_hf = np.mean(hf_vals)
    weighted_wr = np.average(wr_vals, weights=windowed_rms)
    weighted_hf = np.average(hf_vals, weights=windowed_rms)

    return mean_wr, weighted_wr, mean_hf, weighted_hf


def timbral_warmth(fname, segments_data=None, phase_correction=False, clip_output=False,
                   max_WR=12000, fs=0, verbose=True):
    """
     This function estimates the perceptual Warmth of an audio file.
     A warm sound is one that promotes a sensation analogous to that caused by a physical increase in temperature.

     This model of timbral_warmth contains self loudness normalising methods and can accept arrays as an input
     instead of a string filename.

     Version 0.4

     Required parameter
    :param fname:                   string, Audio filename to be analysed, including full file path and extension.

    Optional parameters
  
    :param phase_correction:        bool, if the inter-channel phase should be estimated when performing a mono sum.
                                    Defaults to False.
    :param max_FFT_frame_size:      int, Frame size for calculating spectrogram, default to 8192.
    :param max_WR:                  float, maximun allowable warmth region frequency, defaults to 12000.

    :return:                        Estimated warmth of audio file , segments data

    Copyright 2018 Andy Pearce, Institute of Sound Recording, University of Surrey, UK.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    """

    if verbose:
        print("~~~ function timbral_warmth ~~~\n")

    '''
      Read input
    '''
    if segments_data == None:
        first_call = True
        segments_data = [] # [ [onset1 , onset2 ...] , [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx1[], peak_value1[], peak_x1[] ] ] ,   [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx2[], peak_value2[], peak_x2[] ] ]  , ... ]
    else:
        first_call = False

    audio_samples, fs = timbral_util.file_read(fname, phase_correction=phase_correction)

    # Get the weighted high frequency content
    mean_wr, _, _, weighted_hf = warm_region_cal_dft(audio_samples, fs)

    if verbose:
        print(f"\tMean warmth region: {mean_wr}")
        print(f"\tWeighted-average high frequency level: {weighted_hf}")

    if first_call:
        # Calculate the onsets
        envelope = timbral_util.sample_and_hold_envelope_calculation(audio_samples, fs, decay_time=0.1)
        envelope_time = np.arange(len(envelope)) / float(fs)

        # Calculate the onsets
        nperseg = 4096
        onsets = timbral_util.calculate_onsets(audio_samples, envelope, fs, nperseg=nperseg)

        if verbose:
            print(f"\tOnsets array: {onsets}")

        # If onsets don't exist, set it to time zero
        if not onsets:
            onsets = [0]
        # Set to start of file in the case where there is only one onset
        if len(onsets) == 1:
            onsets = [0]

        segments_data.append(onsets) # [ [onset1 , onset2 ...] , [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx1[], peak_value1[], peak_x1[] ] ] ,   [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx2[], peak_value2[], peak_x2[] ] ]  , ... ]
    else:
        onsets = segments_data[0]


    # Calculate metrics for each onset
    for idx, onset in enumerate(onsets):
        
        if first_call:
            segments_data.append([]) # [ [onset1 , onset2 ...] , [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx1[], peak_value1[], peak_x1[] ] ] ,   [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx2[], peak_value2[], peak_x2[] ] ]  , ... ]

        if verbose:
            print(f"\tonset n : {idx}")

        if onset == onsets[-1]:
            # This is the last onset
            segment = audio_samples[onset:]
        else:
            segment = audio_samples[onset:onsets[idx + 1]]

        #segment_rms = np.sqrt(np.mean(segment * segment))
        #all_rms.append(segment_rms)

        if first_call:

            # Get FFT of segment         
            segment_length = len(segment)
            if segment_length < 1024:
                freqs, time, mags = spectrogram(segment, fs, nperseg=segment_length, nfft=1024)
            
            else:
                freqs, time, mags = spectrogram(segment, fs, nperseg=1024, nfft=1024)
            if verbose:
                pass  
                #pprint(f"FFT. FREQ: {freqs} , LEN: {len(freqs)}")
                #pprint(f"FFT. TIME: {time} , LEN: {len(time)}")
                #pprint(f"FFT. SPEC: {mags} , N frames: {len(mags)}, LEN per FRAME: {len(mags[0])}")

            # flatten the audio to 1 dimension.  Catches some strange errors that cause crashes
            if mags.shape[1] > 1:
                mags = np.sum(mags, axis=1)
                mags = mags.flatten()

            # normalise for this onset
            mags = np.array(list(mags)).flatten()
            this_shape = mags.shape
            mags /= max(abs(mags))

            segments_data[-1].append([]) # [ [onset1 , onset2 ...] , [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx1[], peak_value1[], peak_x1[] ] ] ,   [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx2[], peak_value2[], peak_x2[] ] ]  , ... ]
            segments_data[-1][0].append(mags) 
            segments_data[-1][0].append(freqs)

        else:
            mags = segments_data[idx + 1][0][0]
            freqs = segments_data[idx + 1][0][1]

            # Normalise for this onset
            #positive_magnitudes = np.array(list(positive_magnitudes)).flatten()
            #positive_magnitudes /= max(abs(positive_magnitudes))


        '''
          Estimate of fundamental frequency (first call only)
        '''
        #print(f"\n\n MAGS: {mags}")
        #print(f"\n\n FREQS: {freqs}")
        #print(f"\n\n fs: {fs}")
        if first_call:
            # Peak picking algorithm
            peak_idx, peak_value, peak_x = timbral_util.detect_peaks(mags, freq=freqs, cthr=0.001, fs=fs)
            # Find lowest peak
            fundamental = np.min(peak_x)
            fundamental_idx = np.min(peak_idx)

            segments_data[-1].append([peak_idx, peak_value, peak_x]) # [ [onset1 , onset2 ...] , [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx1[], peak_value1[], peak_x1[] ] ] ,   [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx2[], peak_value2[], peak_x2[] ] ]  , ... ]

            if verbose:
                pprint(f"\tPeaks x: {peak_x}")
                print(f"\tFundamental : {fundamental} (idx: {fundamental_idx})")
        else:
            fundamental = segments_data[idx + 1][1]
            fundamental_idx = segments_data[idx + 1][2]

        '''
         Warmth region calculation (each call)
        '''
        # Estimate the Warmth region
        WR_upper_f_limit = fundamental * 3.5
        if WR_upper_f_limit > max_WR:
            WR_upper_f_limit = 12000
        tpower = np.sum(mags)
        #WR_upper_f_limit_idx = int(np.where(freqs > WR_upper_f_limit)[0][0])
        WR_upper_f_limit_idx_array = np.where(freqs > WR_upper_f_limit)[0]
        if len(WR_upper_f_limit_idx_array) > 0:
            WR_upper_f_limit_idx = int(WR_upper_f_limit_idx_array[0])
        else:
            WR_upper_f_limit_idx = len(freqs) - 1  # Set to the last index if no values found


        if fundamental < 260:
            # Find frequency bin closest to 260Hz
            top_level_idx = int(np.where(freqs > 260)[0][0])
            # Sum energy up to this bin
            low_energy = np.sum(mags[fundamental_idx:top_level_idx])
            # Sum all energy
            tpower = np.sum(mags)
            # Take ratio
            ratio = low_energy / float(tpower)
        else:
            # Make exception where fundamental is greater than
            ratio = 0       
        if verbose:
            print(f"\tRatio: {ratio}")

        #all_ratio.append(ratio)

        '''
         Spectral centroid of the segment (each call)
        '''
        # Spectral centroid
        centroid = np.sum(freqs * mags) / float(np.sum(mags))
        if verbose:
            print(f"\tspectral centroid: {centroid}")
        #all_SC.append(SC)

        '''
         HF decay (each call)
         - linear regression of the values above the warmth region
        '''
        above_WR_spec = np.log10(mags[WR_upper_f_limit_idx:])
        #above_WR_freq = np.log10(freqs[WR_upper_f_limit_idx:])
        above_WR_freq = np.log10(np.clip(freqs[WR_upper_f_limit_idx:], a_min=1e-10, a_max=None))

        np.ones_like(above_WR_freq)
        metrics = np.array([above_WR_freq, np.ones_like(above_WR_freq)])

        # Create a linear regression model
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(metrics.transpose(), above_WR_spec)
        decay_score = model.score(metrics.transpose(), above_WR_spec)
        if verbose:
            print(f"\tHF decay: {decay_score}")
        #all_decay_score.append(decay_score)


        metrics = np.ones(6)
        metrics[0]=(np.log10(centroid))
        metrics[1]=(weighted_hf)
        metrics[2]=(mean_wr)
        metrics[3]=(decay_score)
        metrics[4]=(ratio)
        #weighted_mean_ratio = np.average(ratio, weights=segment_rms)

        if verbose:
            pprint(f"metrics: {metrics}")
    '''
        Apply regression model
    '''
   
    coefficients = np.array([-4.464258317026696, # higher centroid value => less warmth
                                -0.08819320850778556, # higher average high frequency level => less warmth
                                0.29156539973575546, # higher Mean warmth region => more warmth
                                17.274733561081554, # higher HF decay => more warmth
                                8.403340066029507, # higher (low end energy : whole energy) ratio => more warmth
                                45.21212125085579]) # offset

    warmth = np.sum(metrics * coefficients) 

    if verbose:
        pprint(f"metrics * coefficients = {metrics * coefficients}")
        print(f"\tWarmth: {warmth}")
        print("~~~ end of function timbral_warmth ~~~\n")
    # Clip output between 0 and 100
    if clip_output:
        warmth = timbral_util.output_clip(warmth)


    return warmth , segments_data

# -1.635458 , 40.99827192, 5.92857095, 14.70369, 0 , 45.21212125085579