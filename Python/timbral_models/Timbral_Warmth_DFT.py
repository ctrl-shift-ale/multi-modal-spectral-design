from __future__ import division
import numpy as np
import soundfile as sf
import scipy.fftpack as fft
import scipy.stats
from sklearn import linear_model
from . import timbral_util
from pprint import pprint

FFTSIZE = 8192

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


def timbral_warmth_dft(fname, dev_output=False, phase_correction=False, clip_output=False, max_FFT_frame_size=8192,
                   max_WR=12000, fs=0, verbose=True):
    """
     This function estimates the perceptual Warmth of an audio file.

     This model of timbral_warmth contains self loudness normalising methods and can accept arrays as an input
     instead of a string filename.

     Version 0.4

     Required parameter
    :param fname:                   string, Audio filename to be analysed, including full file path and extension.

    Optional parameters
    :param dev_output:              bool, when False return the warmth, when True return all extracted features in a
                                    list.
    :param phase_correction:        bool, if the inter-channel phase should be estimated when performing a mono sum.
                                    Defaults to False.
    :param max_FFT_frame_size:      int, Frame size for calculating spectrogram, default to 8192.
    :param max_WR:                  float, maximun allowable warmth region frequency, defaults to 12000.

    :return:                        Estimated warmth of audio file.

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
        print("Function Timbral_Warmth DFT")

    '''
      Read input
    '''
    audio_samples, fs = timbral_util.file_read(fname, fs, phase_correction=phase_correction)

    # Get the weighted high frequency content
    mean_wr, _, _, weighted_hf = warm_region_cal_dft(audio_samples, fs)

    # Calculate the onsets
    envelope = timbral_util.sample_and_hold_envelope_calculation(audio_samples, fs, decay_time=0.1)
    envelope_time = np.arange(len(envelope)) / float(fs)

    # Calculate the onsets
    nperseg = 4096
    original_onsets = timbral_util.calculate_onsets(audio_samples, envelope, fs, nperseg=nperseg)

    if verbose:
        pprint(f"Original onsets: {original_onsets}")

    # If onsets don't exist, set it to time zero
    if not original_onsets:
        original_onsets = [0]
    # Set to start of file in the case where there is only one onset
    if len(original_onsets) == 1:
        original_onsets = [0]

    '''
      Initialise lists for storing features
    '''
    # Set defaults for holding
    all_rms = []
    all_ratio = []
    all_SC = []
    all_WR_Ratio = []
    all_decay_score = []

    # Calculate metrics for each onset
    for idx, onset in enumerate(original_onsets):
        
        if verbose:
            pprint(f"onset n : {idx}")

        if onset == original_onsets[-1]:
            # This is the last onset
            segment = audio_samples[onset:]
        else:
            segment = audio_samples[onset:original_onsets[idx + 1]]

        segment_rms = np.sqrt(np.mean(segment * segment))
        all_rms.append(segment_rms)

        # Get FFT of signal
        #segment = segment[:FFTSIZE]  # Trim to the first FFTSIZE samples
        #segment = np.asarray(segment, dtype=np.float32)
        #padded_segment = np.pad(segment, (0, max(0, FFTSIZE - len(segment))), mode='constant')
        fft_magnitudes = np.abs(fft.fft(segment))
        #fft_magnitudes = np.abs(fft.fft(segment))
        fft_freqs = fft.fftfreq(len(segment), d=1.0 / fs)

        # Keep only positive frequencies (first half of the spectrum)
        positive_freqs = fft_freqs[:len(fft_freqs) // 2]
        positive_magnitudes = fft_magnitudes[:len(fft_freqs) // 2]

        #if verbose:
        #    pprint(f"DFT. FREQ: {positive_freqs} , LEN: {len(positive_freqs)}")
        #    pprint(f"DFT. MAGS: {positive_magnitudes} , LEN: {len(positive_magnitudes)}")

        # Normalise for this onset
        positive_magnitudes = np.array(list(positive_magnitudes)).flatten()
        positive_magnitudes /= max(abs(positive_magnitudes))

        '''
          Estimate of fundamental frequency
        '''
        # Peak picking algorithm
        peak_idx, peak_value, peak_x = timbral_util.detect_peaks(positive_magnitudes, freq=positive_freqs, fs=fs)
        # Find lowest peak
        fundamental = np.min(peak_x)
        fundamental_idx = np.min(peak_idx)

        '''
         Warmth region calculation
        '''
        # Estimate the Warmth region
        WR_upper_f_limit = fundamental * 3.5
        if WR_upper_f_limit > max_WR:
            WR_upper_f_limit = 12000
        tpower = np.sum(positive_magnitudes)
        #WR_upper_f_limit_idx = int(np.where(positive_freqs > WR_upper_f_limit)[0][0])
        WR_upper_f_limit_idx_array = np.where(positive_freqs > WR_upper_f_limit)[0]
        if len(WR_upper_f_limit_idx_array) > 0:
            WR_upper_f_limit_idx = int(WR_upper_f_limit_idx_array[0])
        else:
            WR_upper_f_limit_idx = len(positive_freqs) - 1  # Set to the last index if no values found


        if fundamental < 260:
            # Find frequency bin closest to 260Hz
            top_level_idx = int(np.where(positive_freqs > 260)[0][0])
            # Sum energy up to this bin
            low_energy = np.sum(positive_magnitudes[fundamental_idx:top_level_idx])
            # Sum all energy
            tpower = np.sum(positive_magnitudes)
            # Take ratio
            ratio = low_energy / float(tpower)
        else:
            # Make exception where fundamental is greater than
            ratio = 0

        all_ratio.append(ratio)

        '''
         Spectral centroid of the segment
        '''
        # Spectral centroid
        SC = np.sum(positive_freqs * positive_magnitudes) / float(np.sum(positive_magnitudes))
        if verbose:
            pprint(f"spectral centroid (dft): {SC}")
        all_SC.append(SC)

        '''
         HF decay
         - linear regression of the values above the warmth region
        '''
        above_WR_spec = np.log10(positive_magnitudes[WR_upper_f_limit_idx:])
        #above_WR_freq = np.log10(positive_freqs[WR_upper_f_limit_idx:])
        above_WR_freq = np.log10(np.clip(positive_freqs[WR_upper_f_limit_idx:], a_min=1e-10, a_max=None))

        np.ones_like(above_WR_freq)
        metrics = np.array([above_WR_freq, np.ones_like(above_WR_freq)])

        # Create a linear regression model
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(metrics.transpose(), above_WR_spec)
        decay_score = model.score(metrics.transpose(), above_WR_spec)
        all_decay_score.append(decay_score)

    '''
     Get mean values
    '''
    mean_SC = np.log10(np.mean(all_SC))
    mean_decay_score = np.mean(all_decay_score)
    weighted_mean_ratio = np.average(all_ratio, weights=all_rms)

    if dev_output:
        return mean_SC, weighted_hf, mean_wr, mean_decay_score, weighted_mean_ratio
    else:
        '''
         Apply regression model
        '''
        all_metrics = np.ones(6)
        all_metrics[0] = mean_SC
        all_metrics[1] = weighted_hf
        all_metrics[2] = mean_wr
        all_metrics[3] = mean_decay_score
        all_metrics[4] = weighted_mean_ratio

        coefficients = np.array([-4.464258317026696,
                                 -0.08819320850778556,
                                 0.29156539973575546,
                                 17.274733561081554,
                                 8.403340066029507,
                                 45.21212125085579])

        warmth = np.sum(all_metrics * coefficients) 

        # Clip output between 0 and 100
        if clip_output:
            warmth = timbral_util.output_clip(warmth)

        return warmth