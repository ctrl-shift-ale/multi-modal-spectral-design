from timbral_design import *
from timbral_models import timbral_warmth as timbral_warmth_original
from pprint import pprint
import random

"""
    1. ANALYZE AUDIOFILE IN HI-RES (FOR FINAL RESYNTHESIS WITH EDITED SPECTRUM)
    2. RUN TIMBRAL_WARMTH ONCE -> RETURNS WARMTH AND FFTDATA
    3. CREATE FOR LOOP WITH N ITERATIONS
    4. INSIDE THE LOOP:
            A. RUN EDIT_SPECTRUM WITH A RANDOM MODIFICATION ON A NARROW BAND OF SPECTRUM
            B. RESYNTHESIZE THE AUDIO WITH EDITED SPECTRUM IN LORES
"""

path = "C:/Projects/SpectralDesign/samples/"
file_in = "Bassoon_C3_MF.wav"
target_warmth_mod_range = [10, 15] 
n_iterations = 1
#resynthesised_filename_hires = (file_in.replace(".wav", "_edited.wav"))
#resynthesised_filename_lores = (file_in.replace(".wav", "_lores.wav"))
#analysis_hires = analyse_audiofile(path+file_in)
#analysis_lores = analyse_audiofile(path+file_in,fft_size=1024, hop_length=512)

# GET WARMTH AND SPECTRAL DATA FROM THE AUDIOFILE
warmth, spec_data = timbral_warmth(path+file_in)
ONSET_IDX = 0
ONSET_DATA = 1
frame_idx = 1
FFTDATA = 0
FFTDATA_MAGS = 0
FFTDATA_FREQS = 1
PEAKSDATA = 1
PEAKIDX = 0
PEAKVAL = 1

#spec_data = [ [onset1 , onset2 ...] , [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx1[], peak_value1[], peak_x1[] ] ] ,   [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx2[], peak_value2[], peak_x2[] ] ]  , ... ]
print(f"warmth: {warmth}")
#print(f"spec_data: {spec_data}\n")
strategy = {
    'low_end': {
        'tonal': 0,
        'noise': 0
        } ,
    'mids': {
        'tonal': 0,
        'noise': 0
        } ,
    'highs': {
        'tonal': 0,
        'noise': 0
        } ,
    'high_end': {
        'tonal': 0,
        'noise': 0
        } 
}
"""
strategies for more warmth:
    -lower centroid
    - lower average high frequency level
    - higher mean warmth region
    - higher HF decay
    - higher (low end energy : whole energy) ratio
    
coefficients = np.array([-4.464258317026696, # higher centroid value => less warmth
                                -0.08819320850778556, # higher average high frequency level => less warmth
                                0.29156539973575546, # higher Mean warmth region => more warmth
                                17.274733561081554, # higher HF decay => more warmth
                                8.403340066029507, # higher (low end energy : whole energy) ratio => more warmth
                                45.21212125085579]) # offset
"""

#FIND HARMONICS 
harmonics = [] #idx, magnitude
for onset_idx in range(len(spec_data[ONSET_IDX])): #spec_data[ONSET_DATA][frame_idx][PEAKS]
    harmonics.append([])
    ONSET_IDX = 0
    ONSET_IDX_OFFSET = 1
    FFTDATA = 0
    FFTDATA_MAGS = 0
    FFTDATA_FREQS = 1
    PEAKSDATA = 1
    PEAKIDCS = 0
    PEAKVALS = 1
    PEAKFREQS = 2

#spec_data = [ [onset1 , onset2 ...] , [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx1[], peak_value1[], peak_x1[] ] ] ,   [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx2[], peak_value2[], peak_x2[] ] ]  , ... ]


    peak_freqs = spec_data[ONSET_IDX_OFFSET + onset_idx][PEAKSDATA][PEAKFREQS]
    peak_mags = spec_data[ONSET_IDX_OFFSET + onset_idx][PEAKSDATA][PEAKVALS]
    peak_indeces = spec_data[ONSET_IDX_OFFSET + onset_idx][PEAKSDATA][PEAKIDCS]
    print(f"peak indeces: {peak_indeces}\n")
    print(f"peak freqs: {peak_freqs}\n")
    print(f"peak mags: {peak_mags}\n")
    
    fundamental_idx = np.min(peak_indeces)
    idx = np.where(peak_indeces == fundamental_idx)[0][0]
    print(f"numpy idx: {idx}\n")
    fundamental_freq =  peak_freqs[idx] #[ peak_idx1[], peak_value1[], peak_x1[] ]
    fundamental_mag = peak_mags[idx] #[ peak_idx1[], peak_value1[], peak_x1[] ]
    print(f"fundamental_freq: {fundamental_freq}\n")
    
    #harmonics[onset_idx].append([fundamental_idx,fundamental_mag])
    for peak_idx in range(len(peak_indeces)):
        if peak_idx % fundamental_idx <= 1 or peak_idx % fundamental_idx >= fundamental_idx - 1: #this is the supposed index of the harmonic
            mags_harm_window = []
            if peak_idx > 0:
                mags_harm_window.append(peak_mags[peak_idx - 1])
            mags_harm_window.append(peak_mags[peak_idx])
            if peak_idx < len(peak_indeces) - 1:
                mags_harm_window.append(peak_mags[peak_idx + 1])

            harm_mag = np.max(mags_harm_window)
            harm_idx = peak_indeces[peak_idx] + mags_harm_window.index(harm_mag) - 1
            harmonics[onset_idx].append([harm_idx,harm_mag, peak_freqs[idx]])
    print(f"harmonics: {harmonics}\n")
#    while warmth not in range(target_warmth_mod_range[0],target_warmth_mod_range[1]+1):


#original_warmth = timbral_warmth_original(path+file_in)
#print(f"original warmth: {original_warmth}")
#spec_data: # # [ [onset1 , onset2 ...] , [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx1[], peak_value1[], peak_x1[] ] ] ,   [ [fft_mags_frame1[],fft_mags_frame2[],...],freqs[] ], [ peak_idx2[], peak_value2[], peak_x2[] ] ]  , ... ]

"""for _ in range(n_iterations):
    
modified_data_hires = edit_spectrum(analysis_hires, modification)
modified_data_lores = edit_spectrum(analysis_lores, modification)

resynthesize_audiofile(path+resynthesised_filename_hires, modified_data_hires)
resynthesize_audiofile(path+resynthesised_filename_lores, modified_data_lores)"""