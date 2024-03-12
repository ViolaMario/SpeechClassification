function [EEG] = pre_processing(subject,session,file_path,locFile)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%%  Load EEG datasets in .edf format
edf_file_path = sprintf('%s/%s/eeg/%s_%s_task-sentences_eeg.edf', subject, session, subject, session);
edf_full_path = fullfile(file_path, edf_file_path);
EEG = pop_biosig(edf_full_path);
EEG.setname = subject;

%% Load events in .tsv file
events_file_path = sprintf('%s/%s/eeg/%s_%s_task-sentences_events.tsv', subject, session, subject, session);
events_full_path = fullfile(file_path,events_file_path);
events = readtable(events_full_path, 'FileType', 'text', 'Delimiter', '\t');
events = table2struct(events);

%% Convert events to EEGLAB event structure
for i = 1:length(events)
    EEG.event(i).type = events(i).trial_type;
    EEG.event(i).latency = events(i).onset * EEG.srate; % Convert onset to samples
    EEG.event(i).duration = events(i).duration * EEG.srate; % Convert duration to samples
end

%% Remove non-EEG channels, including:'HEO','VEO','EKG','EMG','Trigger'
EEG = pop_select( EEG, 'rmchannel',{'HEO','VEO','EKG','EMG','Trigger'});

%% Load channel locations and edit channel name
EEG = pop_chanedit(EEG, 'lookup',locFile, ...
    'changefield',{1,'labels','Fp1'}, 'changefield',{2,'labels','Fpz'}, ...
    'changefield',{3,'labels','Fp2'}, 'changefield',{10,'labels','Fz'}, ...
    'changefield',{19,'labels','FCz'}, 'changefield',{28,'labels','Cz'}, ...
    'changefield',{38,'labels','CPz'}, 'changefield',{48,'labels','Pz'}, ...
    'changefield',{56,'labels','POz'},'changefield',{62,'labels','Oz'}, ...
    'changefield',{64,'labels','I2'}, 'changefield',{60,'labels','I1'}, ...
    'lookup',locFile, ...
    'changefield',{60,'labels','CB1'},'changefield',{64,'labels','CB2'},'rplurchanloc',1);

clc;
disp("Load Channel down");

%% Bandpass filter [0.5 80]
EEG = pop_firws(EEG, 'fcutoff', [0.5 80], 'ftype', 'bandpass', 'wtype', 'kaiser', 'warg', 5.65326, 'forder', 4530, 'minphase', 0, 'usefftfilt', 0, 'plotfresp', 0, 'causal', 0);

disp("Bandpass filter down");

%% Re-reference by average
EEG = pop_reref( EEG, []);

end