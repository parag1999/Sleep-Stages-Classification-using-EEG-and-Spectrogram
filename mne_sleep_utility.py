import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet.age import fetch_data

from lspopt import spectrogram_lspopt
from scipy.signal import spectrogram


def lspopt_spec(X, sf):
    fig = plt.figure()
    fig.set_size_inches(7, 3.5, forward=True)
    f, t, Sxx = spectrogram_lspopt(X, sf, c_parameter=20.0)
    plt.title("lspopt spec")
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def matplot_spec(X, sf):
    fig = plt.figure()
    fig.set_size_inches(7, 3.5, forward=True)
    plt.title("matplotlib spec")
    plt.specgram(X, Fs=sf, cmap="plasma")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def scipy_spec(X, sf):
    fig = plt.figure()
    fig.set_size_inches(7, 3.5, forward=True)
    f, t, Sxx = spectrogram(X, sf)
    plt.title("scipy spec")
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def get_sleep_data():
    ALICE, BOB = 0, 1
    [alice_files, bob_files] = fetch_data(subjects=[ALICE, BOB], recording=[1])
    
    raw_train = mne.io.read_raw_edf(alice_files[0])
    
    annot_train = mne.read_annotations(alice_files[1])
    
    raw_train.set_annotations(annot_train, emit_warning=False)
    
    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}
    
    events_train, _ = mne.events_from_annotations(
        raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)
    
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}
    
    tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included
    
    epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    
    X = (epochs_train.get_data(picks="eeg"))[:,0,:]
    Y = epochs_train.events[:, 2]
    
    return X, Y


if __name__ == "__main__":
    X, Y = get_sleep_data()
    lspopt_spec(X[0], 100)
    matplot_spec(X[0], 100)
    scipy_spec(X[0], 100)



    