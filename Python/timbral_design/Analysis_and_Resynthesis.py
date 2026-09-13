import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fft
import librosa
#import librosa.display
from pprint import pprint

def analyse_audiofile(fname, fft_size=8192, hop_length=256, verbose=False):
    # Read audio file
    sample_rate, audio_data = wav.read(fname)
    dtype = audio_data.dtype  # bit depth

    if verbose:
        print(f"Analysing {fname}...")
        print(f"sr: {sample_rate}, bit depth: {dtype}")
    
    # Handle stereo by converting to mono if necessary
    if len(audio_data.shape) > 1:
        if verbose:
             print(f"Converting to mono")
        audio_data = np.mean(audio_data, axis=1)  # Convert to mono
    
    # Calculate rms
    original_rms = np.sqrt(np.mean(audio_data.astype(float)**2))

    # Perform Short-Time Fourier Transform (STFT)
    mags = librosa.stft(audio_data.astype(float), n_fft=fft_size, hop_length=hop_length)
    freqs = np.fft.rfftfreq(fft_size, d=1/sample_rate)

    return [ sample_rate, dtype, original_rms, fft_size, hop_length, mags, freqs ]

def edit_spectrum(fft_data, modifications):
    #fft_data = [
    #   sample_rate,
    #   dtype,
    #   original_rms,
    #   fft_size,
    #   hop_length,
    #   mags_arr,
    #   freq_arr
    #   ]
    #modification = [ [time_start1, time_end1, freq_range_min1, freq_range_max1, factor1] , [time_start2, time_end2, freq_range_min2, freq_range_max2, factor2] , ...]
    # Apply frequency-based multipliers
    sample_rate = fft_data[0]
    hop_length = fft_data[4]
    mags = fft_data[5]
    freqs = fft_data[6]
    
    for modification in modifications:
        #print(f"modification: {modification}")
        start_time = modification[0]
        end_time = modification[1]
        min_freq = modification[2]
        max_freq = modification[3] 
        factor = modification[4]
        # Identify bins within the time and frequency range
        bins = np.where((freqs >= min_freq) & (freqs <= max_freq))
        frame_idx_min = 0 if start_time == 0 else int(start_time / (hop_length / sample_rate))
        frame_idx_max = mags.shape[1] if end_time == -1 else int(end_time / (hop_length / sample_rate))


        # Apply multiplier to magnitude
        #mags[bins, frame_idx_min:frame_idx_max] *= factor
        mags[np.ix_(bins[0], np.arange(frame_idx_min, frame_idx_max))] *= factor
        
        # Repalce original mags data with modified mags data
        fft_data[5] = mags
        
    return fft_data

def resynthesize_audiofile(file_name, fft_data):
    #fft_data = [
    #   sample_rate,
    #   dtype,
    #   original_rms,
    #   fft_size,
    #   hop_length,
    #   mags_arr,
    #   freq_arr
    #   ]
    sample_rate = fft_data[0]
    dtype = fft_data[1]
    original_rms = fft_data[2]
    hop_length = fft_data[4]
    mags = fft_data[5]

    # Perform inverse STFT to get the resynthesized signal
    resynthesized_data = librosa.istft(mags, hop_length=hop_length)

    # Calculate the RMS energy of the modified signal
    modified_rms = np.sqrt(np.mean(resynthesized_data**2))

    # Scale the modified signal to match the expected loudness
    # The scaling factor is the ratio of the original RMS to the modified RMS
    scaling_factor = original_rms / modified_rms
    resynthesized_data = resynthesized_data * scaling_factor

    # Rescale based on bit depth without normalization
    if dtype == np.int32:
        resynthesized_data = np.clip(resynthesized_data, -2**31, 2**31 - 1).astype(np.int32)
    elif dtype == np.int16:
        resynthesized_data = np.clip(resynthesized_data, -32768, 32767).astype(np.int16)
    else:
        resynthesized_data = resynthesized_data.astype(np.float32)  # Keep float if needed

    # Save the resynthesized audio file
    wav.write(file_name, sample_rate, resynthesized_data)
    print(f"Resynthesized file saved as {file_name}")


