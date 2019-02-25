#!/usr/bin/env python3

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import pandas as pd
import os

def get_data_dir():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_directory, '..', 'data', '2-acoustic')

def get_data_path(filename):
    return os.path.join(get_data_dir(), filename)

def get_audio_files():
    return librosa.util.find_files(get_data_dir(), recurse=False)

def process_file(mfcc_vectors, filename):
    filename_base = os.path.basename(filename)
    filename_root, filename_ext = os.path.splitext(filename_base)
    audio_words = filename_root.split('_')
    category_id = int(audio_words.pop(0))

    print('[     ] [  ] %s [processing]' % filename_base, end='', flush=True)
    y, sr = librosa.load(filename, sr=None, mono=True)
    duration = y.size / sr
    print('\r[%02.1fs] [  ] %s [processing]' % (duration, filename_base), end='', flush=True)

    stft  = librosa.stft(y, n_fft=8192, hop_length=512)
    S     = np.abs(stft)**2

    f = librosa.fft_frequencies(sr=sr, n_fft=8192)
    f_min_index = np.argmax(f > 165) - 1
    f_max_index = np.argmax(f > 3400)
    S[0:f_min_index,:] = 0
    S[f_max_index:,:] = 0

    rms,          = librosa.feature.rmse(S=S)
    rms_low       = np.percentile(rms, 60)
    rms_gdiff     = np.diff(np.pad(1 * (rms > rms_low), (1,0), mode='constant'))
    rms_1_starts, = np.where(rms_gdiff > 0)
    rms_1_ends,   = np.where(rms_gdiff < 0)
    rms_1_groups  = rms_1_ends - rms_1_starts
    rms_0_starts  = np.concatenate([[0], rms_1_ends])
    rms_0_ends    = np.concatenate([rms_1_starts, [rms.size]])
    rms_0_groups  = rms_0_ends - rms_0_starts
    rms_groups    = np.empty((rms_0_groups.size + rms_1_groups.size,), dtype=rms_0_groups.dtype)
    rms_groups[0::2] = rms_0_groups
    rms_groups[1::2] = rms_1_groups
    rms_groups_ms = np.round(rms_groups * 512 / sr * 1000).astype(int)

    rms_groups_y = np.concatenate([np.full(size, size * 512 / sr) for size in rms_groups])
    rms_groups_p = np.concatenate([np.full(g.size, np.linalg.norm(g)) for g in np.split(rms, np.cumsum(rms_groups[:-1]))])

    rms_group_types = np.tile([0,1], rms_groups.size)[0:rms_groups.size]
    rms_groups = list(zip(np.cumsum(rms_groups) - rms_groups, rms_groups, rms_group_types))

    cut_samples = []
    rms_cuts = np.full(rms.size, False)
    for offset, size, value in rms_groups:
        if value == 0 and size > 25:
            if offset > 0:
                rms_cuts[offset + 1] = True
                cut_samples.append((offset + 1) * 512)
            if offset + size < rms.size:
                rms_cuts[offset + size - 1] = True
                cut_samples.append((offset + size - 1) * 512)

    cuts = np.split(y, cut_samples)
    for i in range(0, len(cuts), 2):
        cuts[i] = np.zeros(int(0.1 * sr), dtype=y.dtype)
    for i in range(1, len(cuts), 2):
        cut_duration = cuts[i].size / sr
        cut = librosa.util.normalize(
            librosa.util.fix_length(
                librosa.effects.time_stretch(cuts[i], cut_duration / 0.4),
                int(0.4 * sr))) * 0.5
        for j, power in enumerate(1 - (1 - np.linspace(0, 1, int(0.05 * sr)))**3):
            cut[j] *= power
            cut[cut.size - 1 - j] *= power

        cuts[i] = cut

        window = int(np.exp2(np.ceil(np.log2(cut.size / 2.5))))
        stride = int(np.floor((cut.size - window) / 2))
        S_cut  = librosa.feature.melspectrogram(y=cut, sr=sr, n_fft=window, hop_length=stride)
        mfccs  = librosa.feature.mfcc(S=librosa.power_to_db(S_cut), n_mfcc=7)

        variant_id = int((i + 1) / 2)
        mfcc_vectors.append((
            '%03d_%02d' % (category_id, variant_id),
            category_id,
            variant_id,
            1,      # speaker_id
            'F',    # speaker_sex
            'low',  # variant_type
            audio_words[(variant_id - 1) % len(audio_words)],  # word
            ' '.join(map(str, list(np.ravel(mfccs)))),         # vector
        ))

    print()
    return

    y_cut = np.concatenate(cuts)
    librosa.output.write_wav('cut_%s.flac' % filename_root, y_cut, sr=sr)
    cuts_count = len(range(1, len(cuts), 2))
    print('\r[%02.1fs] [%02d] %s [processing]' % (duration, cuts_count, filename_base), end='', flush=True)

    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
    plt.ioff()
    plt.figure(figsize=(7, 7))
    ax1 = plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, cmap='gray_r', y_axis='log', x_axis='time')
    plt.title('Power spectrogram')

    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(times, 2 + 0.97 * rms_cuts, alpha=0.8, label='RMS high (forward)')
    plt.plot(times, 2 + 0.97 * rms / rms.max(), alpha=0.8, label='RMS (forward)')
    plt.plot(times, 1 + 0.97 * rms_groups_p / rms_groups_p.max(), alpha=0.8, label='RMS width')
    plt.plot(times, 0 + 0.97 * rms_groups_y, alpha=0.8, label='RMS power')
    # plt.legend(frameon=True, framealpha=0.75)
    plt.ylabel('Normalized strength')
    plt.yticks([])
    plt.axis('tight')

    fig_filename = 'fig_%s.pdf' % filename_root
    print('\r[%02.1fs] [%02d] %s [saving figure: %s]' % (duration, cuts_count, filename_base, fig_filename), end='', flush=True)
    plt.savefig(fig_filename, format='pdf', bbox_inches='tight', metadata={'CreationDate':None})
    plt.close('all')
    print('\r[%02.1fs] [%02d] %s [done] [p=%3dms] [v=%3dms] %s' % (duration, cuts_count, filename_base, np.median(rms_groups_ms[0::2]), np.median(rms_groups_ms[1::2]), ' '*40))


mfcc_vectors = []
for filename in get_audio_files():
    process_file(mfcc_vectors, filename)

mfcc_vectors_data = pd.DataFrame.from_records(data=mfcc_vectors, index='id', columns=[
    'id',
    'category_id',
    'variant_id',
    'speaker_id',
    'speaker_sex',
    'variant_type',
    'word',
    'vector',
])
mfcc_vectors_data.to_csv('mfcc_vectors.csv')
print()
print(mfcc_vectors_data)
