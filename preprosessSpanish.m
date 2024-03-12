% EEGLAB Propessing file generated on the 07-Mar-2024
% ------------------Steps-------------------------
% 1. Load EEG datasets in .edf format
% 2. Remove non-EEG channels, including:'HEO','VEO','EKG','EMG','Trigger'
% 3. Load channel locations and edit channel name
% 4. Bandpass filter [0.5 80]
% 5. Re-reference by average
% 6. Save EEG datasets in .set format
% ------------------------------------------------

clear;clc;close all;
eeglab;

%% 1. Load EEG datasets in .edf format

DATASET_FOLDER_PATH = 'E:\.BME\semester3\Thesis1\Large_Spanish_EEG-main\ds004279-download';
DATASET_SAVE_PATH = 'F:\\Intership\\eeglab_current\\MyProject\\preprocessing';
locFile = 'F:\\Intership\\eeglab_current\\plugins\\dipfit5.2\\standard_BEM\\elec\\standard_1005.elc';

% List contents of DATASET_FOLDER_PATH
subject_name = {dir(fullfile(DATASET_FOLDER_PATH, 'sub-*')).name};

% Loop through each subject directory
for subject_idx = 1:numel(subject_name)
    % Extract subject name
    subject = subject_name{subject_idx};

    % Check if the subject name starts with 'sub-'
    if startsWith(subject, 'sub-')
        edf_file_path = fullfile(subject, 'ses-01', 'eeg', ...
                sprintf('%s_ses-01_task-sentences_eeg.edf', subject));

        edf_full_path = fullfile(DATASET_FOLDER_PATH, edf_file_path);
        EEG = pop_biosig(edf_full_path);
        EEG.setname = subject;
        %pop_eegplot( EEG, 1, 1, 1);

%%% 2. Remove non-EEG channels, including:'HEO','VEO','EKG','EMG','Trigger'

        EEG = pop_select( EEG, 'rmchannel',{'HEO','VEO','EKG','EMG','Trigger'});

%%% 3. Load channel locations and edit channel name

        EEG = pop_chanedit(EEG, 'lookup',locFile, ...
            'changefield',{1,'labels','Fp1'}, 'changefield',{2,'labels','Fpz'}, ...
            'changefield',{3,'labels','Fp2'}, 'changefield',{10,'labels','Fz'}, ...
            'changefield',{19,'labels','FCz'}, 'changefield',{28,'labels','Cz'}, ...
            'changefield',{38,'labels','CPz'}, 'changefield',{48,'labels','Pz'}, ...
            'changefield',{56,'labels','POz'},'changefield',{62,'labels','Oz'}, ...
            'changefield',{64,'labels','I2'}, 'changefield',{60,'labels','I1'}, ...
            'lookup',locFile, ...
            'changefield',{60,'labels','CB1'},'changefield',{64,'labels','CB2'},'rplurchanloc',1);
        
        %figure; topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);

%%% 4. Bandpass filter [0.5 80]
       
        EEG = pop_firws(EEG, 'fcutoff', [0.5 80], 'ftype', 'bandpass', 'wtype', 'kaiser', 'warg', 5.65326, 'forder', 4530, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);

%%% 5. Re-reference by average

        EEG = pop_reref( EEG, []);

%%% 6. Save EEG datasets in .set and .edf format       
        
        pop_saveset(EEG, 'filename', sprintf('%s.set',subject), 'filepath', DATASET_SAVE_PATH);
        
        edf_filename = sprintf('%s.edf',subject);
        pop_writeeeg(EEG, sprintf('%s.edf',subject), 'TYPE', 'EDF', 'dataformat', 'double', 'overwrite', 'on');
        movefile(edf_filename, fullfile(DATASET_SAVE_PATH, edf_filename));

    end
end

