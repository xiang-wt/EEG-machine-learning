for i = 1:90
    filename = {'data.bdf','evt.bdf'};
    pathname =[names(:,:,i),'\']; 
    EEG = pop_importNeuracle(filename, pathname);
    EEG = eeg_checkset( EEG );
    EEG = pop_select( EEG, 'rmchannel',{'A1','A2','Oz','CB1','CB2','FC1','FC2','FC3','FC4','FC5','FC6','FT7','FT8'}); % channel selection
    EEG = eeg_checkset( EEG );
    EEG = pop_chanedit(EEG, 'lookup','D:\\MATLAB\\R2023b\\eeglab\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc','load',{'D:\\MATLAB\\R2023b\\eeglab\\19.ced'},'filetype','autodetect'); % channel location
    EEG = eeg_checkset( EEG );
    EEG = pop_resample( EEG, 256); % resample
    EEG = eeg_checkset( EEG );
    EEG = pop_reref( EEG, []); % re-reference
    EEG = eeg_checkset( EEG );
    EEG = pop_basicfilter( EEG,  1:19 , 'Boundary', 'boundary', 'Cutoff', [ 0.5 70], 'Design', 'butter', 'Filter', 'bandpass', 'Order',  2 );
    EEG = pop_basicfilter( EEG,  1:19 , 'Cutoff',  50, 'Design', 'notch', 'Filter', 'PMnotch', 'Order',  180 ); % filter
    EEG = eeg_checkset( EEG );
    stru = EEG.chanlocs;
    EEG = pop_rejchan(EEG, 'elec',[1:19] ,'threshold',5,'norm','on','measure','kurt'); % automatic interpolation to remove bad channels
    EEG = pop_interp(EEG, [stru], 'spherical');
    EEG = eeg_checkset( EEG );
    EEG = pop_autobssemg( EEG, [6], [6], 'bsscca', {'eigratio', [1000000]}, 'emg_psd', {'ratio', [10],'fs', [256],'femg', [15],'estimator',spectrum.welch({'Hamming'}, 128),'range', [0  9]}); % automatically remove EMG artifacts
    EEG = eeg_checkset( EEG );
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
    EEG = eeg_checkset( EEG );
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on'); % two ICA
    EEG = eeg_checkset( EEG );
    n = table2array(biyan(i,:));
    middleIndex = round(n/2);
    if n <= 300
        a=n;
        EEG = pop_epoch( EEG, {  '闭眼'  }, [0 a]); % extract the middle 5min data. if less than 5min, extract all.
        EEG = eeg_checkset( EEG );
    else
        startIndex = middleIndex - round(300 / 2);
        endIndex = startIndex + 300 - 1;
        a=[startIndex endIndex];
        EEG = pop_epoch( EEG, {  '闭眼'  }, a); 
        EEG = eeg_checkset( EEG );
    end
    EEG = eeg_regepochs(EEG, 'recurrence', 2, 'limits', [0 2], 'rmbase', NaN); % divide 2s segment
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',[num2str(i),'.set'],'filepath','TD');
    EEG = eeg_checkset( EEG );
end
% manual artifact removal (e.g., eye movements, muscle artifacts, etc.)
% save the cleaned dataset again,'_final.set'
% count and record the number of retained epochs after cleaning