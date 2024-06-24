import os
import numpy as np
from tqdm import tqdm
import LoadData as loadData
from collections import namedtuple
import mne
from eeglabio import epochs
os.system('cls')



###============================ Load data ============================###
# filename1 = f"F:\\Intership\\Large_Spanish_EEG-main\\data\\npz\\average_1-40hz_icaLabel95confidence_eyes_40sessions.npz"
# filename2 = f"F:\\Intership\\Large_Spanish_EEG-main\\data\\npz\\average_1-40hz_icaLabel95confidence_eyes_20sessions.npz"
#
# dataset1 = loadData.load_eeg(filename1, event_key='perception')
# dataset2 = loadData.load_eeg(filename2, event_key='perception')
#
# EventProduction = {**dataset1, **dataset2}
filename = f"F:\\Intership\\Large_Spanish_EEG-main\\data\\npz\\Fs250_0.5-80hz_20sessions_perception_production_01.npz"
EventProduction = loadData.load_eeg(filename, event_key='perception')
###============================ Manage data ============================###
trial_info = namedtuple('trial_structure', ['subject', 'task', 'stim', 'epoch', 'label'])
eeg_load_trial = namedtuple('eeg_preload_trial', ['data', 'label'])

trials = []  # Trials will be stored here
stims = list(range(1, 31))
stim2id = {stim_value: index+1 for index, stim_value in enumerate(stims)}

for key in tqdm(EventProduction.keys()):
    subject, task, stim = key.split('_')
    num_epoch, channels, samples = EventProduction[key].shape   # num_epoch, channels, samples

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



fs = 250
t = 3.9
window = int(fs*t)   # 3.9 seconds
X = np.zeros((len(trials), channels, window))
Y = np.zeros((len(trials)), dtype=int)

event_ids = {}

for idx, single_trial in tqdm(enumerate(trials), total=len(trials)):
    subject = single_trial.subject
    task = single_trial.task
    stim = single_trial.stim
    epoch = single_trial.epoch
    text = f"{subject}_{task}_{stim}"
    event = f"{subject}_{task}_{stim}_{epoch}"
    X[idx, :, :] = EventProduction[text][epoch][:, :window]
    Y[idx] = single_trial.label
    event_ids[event] = single_trial.label


sfreq = 250  # 采样率
n_epochs, n_channels, n_samples = X.shape
ch_names = [f'EEG {i + 1}' for i in range(n_channels)]
ch_types = ['eeg'] * n_channels
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
# events = np.column_stack([np.arange(n_epochs), np.zeros(n_epochs, dtype=int), Y])
events = np.column_stack([np.arange(0, int(t*sfreq*len(Y)), int(t*sfreq)), np.zeros(n_epochs, dtype=int), Y])
epochs = mne.EpochsArray(X, info, events, tmin=0, event_id=event_ids)

annotations = mne.Annotations(onset=np.arange(0, int(3.896*len(Y)), 3.896),
                              duration=[3.896]*len(Y),
                              description=list(event_ids.keys()),
                              orig_time=None)

epochs.set_annotations(annotations)

epochs.save(fname='Fs128_1-40hz_20sessions_perception_01_epo2.fif', overwrite=True, verbose=None)

# sfreq = 250  # 采样率
# n_epochs, n_channels, n_samples = X.shape
# ch_names = [f'EEG {i + 1}' for i in range(n_channels)]
# tmax = 3.896
# # Create the data for the AnnotationsList
# descriptions = list(event_ids.keys())
# onsets = np.arange(0, int(tmax*len(Y)), tmax)  # Starting time in seconds
# durations = [tmax]*len(Y)    # Duration in seconds
# # Create the AnnotationsList
# annotations_list = [descriptions, onsets, durations]
# events = np.column_stack([np.arange(0, int(tmax*sfreq*len(Y)), int(tmax*sfreq)), np.zeros(n_epochs, dtype=int), Y])
#
# epochs.export_set(fname='Fs128_1-40hz_20sessions_perception_01_epo.set', data=X,
#                   sfreq=sfreq, events=events, tmin=0.0, tmax=tmax,
#                   ch_names=ch_names, annotations=annotations_list)


