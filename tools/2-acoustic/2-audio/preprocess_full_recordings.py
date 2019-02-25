#!/usr/bin/env python3

import numpy as np
import librosa
import glob
import os

# import matplotlib.pyplot as plt
# import librosa.display

def get_data_dir():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_directory, '..', '..', 'data', '2-acoustic', 'original')

def get_data_path(filename):
    return os.path.join(get_data_dir(), filename)

def get_audio_files():
    data_dir = glob.escape(os.path.join(get_data_dir(), ''))
    return sorted(glob.glob(data_dir + '*.flac'))

def log(status, duration, filename_base):
    if duration is None:
        duration_fmt = '      '
    elif duration >= 1000.0:
        duration_fmt = '% 5.0fs' % duration
    else:
        duration_fmt = '% 5.1fs' % duration
    print('\r\x1B[K[%s] [   ] %s [%s]' % (duration_fmt, filename_base, status), end='', flush=True)

def test_peaks(rms, p):
    noise_floor = np.percentile(rms, p)
    diff        = np.diff(np.pad(1 * (rms > noise_floor), (1,0), mode='constant'))
    starts_1,   = np.where(diff > 0)
    print('%2.2f%%: %d' % (p, len(starts_1)))

def process_recording(filename):
    duration      = None
    filename_base = os.path.basename(filename)
    log('loading', duration, filename_base)

    y, sr = librosa.load(filename, sr=None, mono=True)
    duration = y.size / sr
    log('creating spectrogram', duration, filename_base)

    stft = librosa.stft(y, n_fft=8192, hop_length=512)
    S    = np.abs(stft)**2

    f = librosa.fft_frequencies(sr=sr, n_fft=8192)
    f_min_index = np.argmax(f > 4000) - 1
    S[0:f_min_index,:] = 0

    rms,    = librosa.feature.rmse(S=S)
    print()
    print()
    test_peaks(rms, 90)
    test_peaks(rms, 95)
    test_peaks(rms, 98)
    test_peaks(rms, 99)
    test_peaks(rms, 99.5)
    test_peaks(rms, 99.8)
    test_peaks(rms, 99.9)
    test_peaks(rms, 99.95)
    test_peaks(rms, 99.98)
    test_peaks(rms, 99.99)
    print()
    print()
    rms_low = np.percentile(rms, 95)

    print()
    print()
    print(np.percentile(rms,   0))
    print(np.percentile(rms,  10))
    print(np.percentile(rms,  20))
    print(np.percentile(rms,  30))
    print(np.percentile(rms,  40))
    print(np.percentile(rms,  50))
    print(np.percentile(rms,  60))
    print(np.percentile(rms,  70))
    print(np.percentile(rms,  80))
    print(np.percentile(rms,  90))
    print(np.percentile(rms,  95))
    print(np.percentile(rms,  98))
    print(np.percentile(rms, 100))
    print()
    print(rms_low)
    exit()

    log('creating figure', duration, filename_base)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
    plt.figure(figsize=(7, 7))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, fmax=22050, cmap='gray_r', y_axis='log', x_axis='time')
    plt.title('Power spectrogram')

    log('saving figure', duration, filename_base)
    plt.savefig('spectrogram.pdf', format='pdf', bbox_inches='tight', metadata={'CreationDate':None})
    plt.close()

    log('done', duration, filename_base)

for filename in get_audio_files():
    process_recording(filename)
    print()
