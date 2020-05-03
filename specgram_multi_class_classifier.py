import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet.age import fetch_data

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def get_spectrogram(data, sf):
    rate = sf
    nfft= sf
    noverlap= sf-1
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75,0.5]
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    plt.close(fig)
    return imarray


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def normalize_gray(array):
    return (array - array.min())/(array.max() - array.min())
  

def preprocess():
    ALICE, BOB = 0, 1
    [alice_files, bob_files] = fetch_data(subjects=[ALICE, BOB], recording=[1])
    
    mapping = {'EOG horizontal': 'eog',
               'Resp oro-nasal': 'misc',
               'EMG submental': 'misc',
               'Temp rectal': 'misc',
               'Event marker': 'misc'}
    
    raw_train = mne.io.read_raw_edf(alice_files[0])
    annot_train = mne.read_annotations(alice_files[1])
    
    raw_train.set_annotations(annot_train, emit_warning=False)
    raw_train.set_channel_types(mapping)
    
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
    
    X_raw = (epochs_train.get_data(picks="eeg"))[:,0,:]
    X = []
    for x in X_raw:
        spectrogram = get_spectrogram(x, sf=100)
        #graygram      = rgb2gray(spectrogram)
        normgram      = normalize_gray(spectrogram)
        X.append(normgram)
        
    X = np.array(X, dtype="float32")
    Y = epochs_train.events[:, 2]
    Y -= 1
    Y = keras.utils.to_categorical(Y, 5)
    
    return X, Y

if __name__ == "__main__":
    
    X, Y = preprocess()
    imheight, imwidth = (36, 54)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    input_shape = (imheight, imwidth, 3)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test), callbacks=[ModelCheckpoint('model_categorical_3_channel.h5',save_best_only=True)])

    
