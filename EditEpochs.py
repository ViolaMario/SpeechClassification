import os
import numpy as np
from tqdm import tqdm
import LoadData as loadData
from collections import namedtuple
import mne
from eeglabio import epochs
os.system('cls')


DATASET_FOLDER_PATH = 'F:/Intership/Large_Spanish_EEG-main/data/npz'
trial_info = namedtuple('trial_structure', ['subject', 'task', 'stim', 'epoch', 'label'])
eeg_load_trial = namedtuple('eeg_preload_trial', ['data', 'label'])
trials = []  # Trials will be stored here
stims = list(range(1, 31))
stim2id = {stim_value: index+1 for index, stim_value in enumerate(stims)}

fs = 250
t = 3.9
channels = 64
window = int(fs*t)   # 3.9 seconds

ch_names = [f'EEG {i + 1}' for i in range(channels)]
ch_types = ['eeg'] * channels
info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

for epo in os.listdir(DATASET_FOLDER_PATH):
    if 'Fs250_0.5-80hz_20sessions_perception_0' in epo:
        filename = os.path.join(DATASET_FOLDER_PATH, epo)
        print(filename)
        EventProduction = loadData.load_eeg(filename, event_key='perception')

        for key in tqdm(EventProduction.keys()):
            subject, task, stim = key.split('_')
            num_epoch, _, samples = EventProduction[key].shape  # num_epoch, channels, samples

            for epoch_index in range(num_epoch):
                trials.append(
                    trial_info(
                        subject=subject,
                        task=task,
                        stim=stim,
                        epoch=epoch_index,
                        label=stim2id[int(stim)],
                    )
                )

        X = np.zeros((len(trials), channels, window))
        Y = np.zeros((len(trials)), dtype=int)

        for idx, single_trial in tqdm(enumerate(trials), total=len(trials)):
            subject = single_trial.subject
            task = single_trial.task
            stim = single_trial.stim
            epoch = single_trial.epoch
            text = f"{subject}_{task}_{stim}"
            X[idx, :, :] = EventProduction[text][epoch][:, :window]
            Y[idx] = single_trial.label

        np.save(f"{epo[:-4]}_label.npy", Y)

        n_epochs = len(Y)

        events = np.column_stack(
            [np.arange(0, int(t * fs * len(Y)), int(t * fs)), np.zeros(n_epochs, dtype=int), Y])
        epochs = mne.EpochsArray(X, info, events, tmin=0, event_id=stims)

        final_file = f"{epo[:-4]}_epo.fif"
        epochs.save(fname=final_file, overwrite=True, verbose=None)
        np.save(f"{epo[:-4]}_label.npy", Y)





