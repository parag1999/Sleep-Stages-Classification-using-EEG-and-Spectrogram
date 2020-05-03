import matplotlib.pyplot as plt
import numpy as np
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


def get_sleep_data(no_of_people):
    no_of_people %= 20
    people = [i for i in range(no_of_people)]
    sf = 100
    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}
    
    tmax = 30. - 1. / sf
    
    files = fetch_data(subjects=people, recording=[1])
    
    count = 0
    for file in files:
        raw = mne.io.read_raw_edf(file[0])
        annot = mne.read_annotations(file[1])
        raw.set_annotations(annot, emit_warning=False)
        events, _ = mne.events_from_annotations(
            raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)
        epochs = mne.Epochs(raw=raw, events=events,
                                  event_id=event_id, tmin=0., tmax=tmax, baseline=None)
        
        if count == 0:
            X = (epochs.get_data(picks="eeg"))[:,0,:]
            Y = epochs.events[:, 2]
            count += 1
        else:
            x = (epochs.get_data(picks="eeg"))[:,0,:]
            y = epochs.events[:, 2]
            X = np.concatenate((X, x), axis=0)
            Y = np.concatenate((Y, y), axis=0)
        
    #X = (epochs_test.get_data(picks="eeg"))[:,0,:]
    #Y = epochs_test.events[:, 2]
    
    return X, Y


if __name__ == "__main__":
    X, Y = get_sleep_data(no_of_people = 4)
    lspopt_spec(X[0], 100)
    matplot_spec(X[0], 100)
    scipy_spec(X[0], 100)



    