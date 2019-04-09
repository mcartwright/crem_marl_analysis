import glob
import os
import numpy as np
import pandas as pd
import tqdm
import h5py

from sklearn.manifold import TSNE

import scipy.signal
import scipy.fftpack as fft

from librosa import cqt, magphase, note_to_hz, stft, resample, to_mono, load
from librosa import amplitude_to_db, get_duration, time_to_frames, power_to_db
from librosa.util import fix_length
from librosa.feature import melspectrogram, rmse, tempogram
from librosa.decompose import hpss
from librosa.onset import onset_strength

from librosa.filters import get_window
from librosa.core.audio import resample
from librosa.core import tempo_frequencies
from librosa import util


def tempogram_ratio(tgram, sr=22050, hop_length=512, start_bpm=120.0, std_bpm=1):
    
    # each row of tempogram is a lag of hop_length / sr
    times = 60 * (sr / 512.) / np.arange(0, len(tgram))
    
    # estimate per-frame tempo
    window = np.exp(-0.5 * ((np.log2(times / start_bpm)) / std_bpm)**2)
    window[0] = 0
    
    win_tg = tgram * window[:, np.newaxis]
    
    local_tempo = win_tg.argmax(axis=0)
        
    ref_strength = tgram[local_tempo, range(len(local_tempo))]
    
    # construct an interpolator on the y-axis
    f = scipy.interpolate.RectBivariateSpline(np.arange(tgram.shape[0]),
                                              np.arange(tgram.shape[1]),
                                              tgram,
                                              kx=2,
                                              ky=1)
    out = []

    for factor in [1./3 * 2./32, 4./32, 6./32,  # 32nds
                   1./3 * 2./16, 4./16, 6./16,  # 16ths
                   1./3 * 2./8,  4./8,  6./8,   # 8ths
                   1./3 * 2./4,  4./4,  6./4,   # 4ths
                   1./3 * 2./2,  4./2,  6./2,   # 2nds
                   1./3 * 2./1,  4./1,  6./1]:  # 1sts
        
        out.append(f.ev(local_tempo / factor, np.arange(len(local_tempo))))
    
    # Only normalize if there's enough energy
    ref_strength[ref_strength < 1e-2] = 1
    
    out = np.asarray(out) / ref_strength[np.newaxis, :]
    
    return np.maximum(out, 0.0) # correct for interpolation errors


def tempo(tg, sr=22050, onset_envelope=None, hop_length=512, start_bpm=120,
          std_bpm=1.0, ac_size=8.0, max_tempo=320.0, aggregate=np.mean):
    """Estimate the tempo (beats per minute) 

    (from librosa.beat.tempo)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
        sampling rate of the time series

    onset_envelope    : np.ndarray [shape=(n,)]
        pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length of the time series

    start_bpm : float [scalar]
        initial guess of the BPM

    std_bpm : float > 0 [scalar]
        standard deviation of tempo distribution

    ac_size : float > 0 [scalar]
        length (in seconds) of the auto-correlation window

    max_tempo : float > 0 [scalar, optional]
        If provided, only estimate tempo below this threshold

    aggregate : callable [optional]
        Aggregation function for estimating global tempo.
        If `None`, then tempo is estimated independently for each frame.

    Returns
    -------
    tempo : np.ndarray [scalar]
        estimated tempo (beats per minute)
    """

    if start_bpm <= 0:
        raise ParameterError('start_bpm must be strictly positive')

    # Eventually, we want this to work for time-varying tempo
    if aggregate is not None:
        tg = aggregate(tg, axis=1, keepdims=True)

    # Get the BPM values for each bin, skipping the 0-lag bin
    bpms = tempo_frequencies(tg.shape[0], hop_length=hop_length, sr=sr)

    # Weight the autocorrelation by a log-normal distribution
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm)**2)

    # Kill everything above the max tempo
    if max_tempo is not None:
        max_idx = np.argmax(bpms < max_tempo)
        prior[:max_idx] = 0

    # Really, instead of multiplying by the prior, we should set up a
    # probabilistic model for tempo and add log-probabilities.
    # This would give us a chance to recover from null signals and
    # rely on the prior.
    # it would also make time aggregation much more natural

    # Get the maximum, weighted by the prior
    best_period = np.argmax(tg * prior[:, np.newaxis], axis=0)

    tempi = bpms[best_period]
    # Wherever the best tempo is index 0, return start_bpm
    tempi[best_period == 0] = start_bpm
    return tempi


def get_onset_patterns_tsne(feat_files, perplexity=30.0):
    onset_patterns = []
    idx = 0
    idxs = []
    for feat_file in tqdm.tqdm(feat_files):
        try: 
            features = np.load(feat_file)
            onset_patterns.append(features['onset_patterns'])
            idxs.append(idx)
        except Exception as e:
            print('Skipping {}. {}'.format(feat_file, e))
        idx += 1
    onset_patterns = np.array(onset_patterns)

    onset_patterns_tsne = TSNE(n_components=2).fit_transform(onset_patterns.reshape([onset_patterns.shape[0],-1]))
    
    return onset_patterns_tsne, np.array(idxs)


def get_avg_power_spec_tsne(feat_files, perplexity=30.0):
    avg_power_specs = []
    idx = 0
    idxs = []
    for feat_file in tqdm.tqdm(feat_files):
        try:
            features = np.load(feat_file)
            avg_power_specs.append(np.mean(features['linspec_mag'] ** 2, axis=1))
            idxs.append(idx)
        except Exception as e:
            print('Skipping {}. {}'.format(feat_file, e))
        idx += 1
    avg_power_specs = np.array(avg_power_specs)

    avg_power_specs_tsne = TSNE(n_components=2).fit_transform(avg_power_specs)
    
    return avg_power_specs_tsne, np.array(idxs)


def get_mean_percussive_ratio_dbs(feat_files, margin=3.0):
    mean_percussive_ratios_db = []
    idx = 0
    idxs = []
    for feat_file in tqdm.tqdm(feat_files):
        try:
            features = np.load(feat_file)
            idxs.append(idx)
            idx += 1
        except Exception as e:
            print('Skipping {}. {}'.format(feat_file, e))
            idx += 1
            continue
        D = features['linspec_mag'] * np.exp(1.j * features['linspec_mag'])
        
        H, P = hpss(D, margin=margin)
        Pm, Pp = magphase(P)
        S, phase = magphase(D)

        P_rms = rmse(S=Pm)
        S_rms = rmse(S=S)
        percussive_ratio = P_rms / S_rms
        mean_percussive_ratio_db = amplitude_to_db(np.array([np.mean(percussive_ratio)]))[0]
        
        mean_percussive_ratios_db.append(mean_percussive_ratio_db)
    mean_percussive_ratios_db = np.array(mean_percussive_ratios_db)
    
    return mean_percussive_ratios_db, idxs


def get_audio_filenames(feat_files):
    return [os.path.splitext(os.path.basename(ff))[0] + '.wav' for ff in feat_files]


def get_onset_patterns(feats):
    return feats['onset_patterns']


def get_tempogram_features(feats):
    # filtered tempogram
    tempogram = feats['tempogram']
    tgf = util.normalize(np.maximum(0.0, tempogram - scipy.ndimage.median_filter(tempogram, size=(9, 1))))
    tr = util.normalize(tempogram_ratio(tgf))
    return np.median(tgf, axis=1), np.median(tr, axis=1), tempo(tgf, aggregate=np.median)


def get_mean_power_spec(feats, perplexity=30.0):
    return np.mean(feats['linspec_mag'] ** 2, axis=1)


def get_median_percussive_ratio_db(feats):
    return np.median(feats['percussive_ratio'])


def get_audio_filename(feat_file):
    return os.path.splitext(os.path.basename(feat_file))[0] + '.wav'


def main(feat_dir):
    feat_files = glob.glob(os.path.join(feat_dir, '*.npz'))

    with h5py.File(os.path.join(feat_dir, 'feats.h5')) as h5:
        d = h5.create_dataset('features',
                              (len(feat_files),),
                              dtype=[('filename', 'S50'),
                                     ('onset_patterns', 'f4', (60,60)),
                                     ('median_filt_tempogram', 'f4', (384,)),
                                     ('median_tempogram_ratio', 'f4', (18,)),
                                     ('estimated_tempo', 'f4'),
                                     ('mean_power_spec', 'f4', (1025,)),
                                     ('median_percussive_ratio_db', 'f4')],
                              compression='lzf')

        for k,feat_file in tqdm.tqdm(enumerate(feat_files)):
            feats = np.load(feat_file)
            filename = get_audio_filename(feat_file)
            onset_patterns = get_onset_patterns(feats)
            median_filt_tempogram, median_tempogram_ratio, est_tempo = get_tempogram_features(feats)
            mean_power_spec = get_mean_power_spec(feats)
            median_percussive_ratio_db = get_median_percussive_ratio_db(feats)
            d[k] = (filename, onset_patterns, median_filt_tempogram, median_tempogram_ratio, 
                    est_tempo, mean_power_spec, median_percussive_ratio_db)

if __name__ == '__main__':
    main('/scratch/mc6591/crem_features')
