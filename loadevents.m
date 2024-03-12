
eeglab;
subject = "sub-003";
session = "ses-01";
file_path = 'E:\.BME\semester3\Thesis1\Large_Spanish_EEG-main\ds004279-download';
locFile = 'F:\\Intership\\eeglab_current\\plugins\\dipfit5.2\\standard_BEM\\elec\\standard_1005.elc';

% EEG = pre_processing(subject,session,file_path,locFile);

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

%% Create epochs
% Define event types
event_types = {'rest', 'perception', 'preparation', 'production'};

% Iterate through each event
for i = 1:length(events)
    event_type = events(i).trial_type;
    % Check if the event type matches any of the defined event types
    for j = 1:length(event_types)
        if startsWith(event_type, event_types{j})
            % Extract the suffix number from the event type
            suffix = str2double(event_type(length(event_types{j}) + 2:end));
            % Create an epoch for this event type and suffix
            EEG = pop_epoch(EEG, {event_type}, [0, events(i).duration], 'eventindices', i, 'epochinfo', 'yes', 'precompute', 'all', 'verbose', 'off');
            % Rename the event type to remove the suffix for consistency
            EEG.event(end).type = event_types{j};
            % Break the inner loop once the event type is found
            break;
        end
    end
end
%%
% Define the event types to include
event_types= {'rest', 'perception', 'preparation', 'production'};

% Iterate through each event
for i = 1:length(events)
    event_type = events(i).trial_type;
    
    % Check if the event type belongs to the specified types
    if any(strcmp(event_type, event_types))
        EEG2 = pop_epoch(EEG, {event_type}, [0, events(i).duration], 'eventindices', i, 'epochinfo', 'yes', 'precompute', 'all', 'verbose', 'off');
    end
end




