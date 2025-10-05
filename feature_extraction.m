for a =1:90
    EEG = pop_loadset('filename', [num2str(a),'_final.set'], 'filepath', 'TD');
    [chan, len, epoch] = size(EEG.data);
    mkdir('TD',['TD',num2str(a)]);
    % set frequency analysis parameters
    freq_range = [0.5 70]; % frequency range

    % extract time domain features
    time_features = zeros(chan, 14, epoch);
    leng = 16;
    for j = 1: 1:epoch
        for i = 1:chan
            x = EEG.data(i, :, j);
            time_features(i,1,j) = mean(x); % mean
            time_features(i,2,j) = std(x); % standard deviation
            time_features(i,3,j) = mean(abs(diff(x)))/std(x); % normalized first-order difference
            time_features(i,4,j) = sum(x.^2); % energy
            time_features(i,5,j) = length(findpeaks(x,MinPeakProminence=3)); % minimums and maximums (MMAX)
            time_features(i,6,j) = kurtosis(x); % kurtosis
            time_features(i,7,j) = skewness(x); % skewness
            time_features(i,8,j) = Katz_FD(x); % Katz fractal dimension
            time_features(i,9,j) = Higuchi_FD(x,20); % Higuchi fractal dimension
            n = length(x)/leng;
            means = zeros(1, n);
            for ii = 1:n
                segment = x((ii-1)*leng + 1 : ii*leng);
                means(i) = mean(segment);
            end
            time_features(i,10,j) = std(means); % Non-Stationary Index (NSI) 
            time_features(i,11,j) = zerocrossrate(x); % zero-crossing rate
            [time_features(i,12,j),time_features(i,13,j),time_features(i,14,j)]= hjorth(x); % Hjorth parameter (activity, mobility, complexity)
        end
    end
    time_features = mean(time_features,3);
    save(['TD',num2str(a),'\time_features_TD',num2str(a),'.mat'],'time_features') 

    % extract frequency domain features
    delta_band=[0.5 4];
    theta_band=[4 8];
    alpha_band=[8 12];
    beta_band=[13 30];
    gamma_band=[31 70];
    band_name=["delta" "theta" "alpha" "beta" "gamma"];
    band=[[delta_band];[theta_band];[alpha_band];[beta_band];[gamma_band]];
    fs=256;
    frames = 512;
    m=2;
    r=0.2;
    for k=1:length(band)
        freq_features = zeros(chan,7, epoch);
        for j = 1:1:epoch
            x = EEG.data(:,:,j);
            wave=bandpass(x,band(k,:),fs);
            data=transpose(detrend(wave));
            [pxx, f] = pwelch(data, hamming(fs), 0.5*fs,frames, fs);
            freq_features(:,1,j) = mean(pxx,1); % mean PSD
            freq_features(:,2,j) = std(pxx,0,1); % standard deviation of PSD
            freq_features(:,3,j) = meanfreq(pxx,f); % mean frequency (MNF)
            freq_features(:,4,j) = bandpower(pxx,f,'psd'); % mean frequency band power
            prob_density = pxx./sum(pxx);
            freq_features(:,5,j) = -sum(prob_density.*log2(prob_density)); % power spectrum entropy (PSE)
            de_feature=zeros(size(x,1),1);
            for i=1:size(x,1)
                de_feature(i)=diff_entropy(x(i,:),m,r);
            end
            freq_features(:,6,j) = de_feature; % differential entropy (DE)
            for i = 1:chan
                freq_features(i,7,j) = pentropy(pxx(:,i),f,512,"Instantaneous",false); % spectral entropy (SE)  
            end
        end
        freq_features=mean(freq_features,3);
        filename=strcat('TD',num2str(a),'\freq_features_',num2str(band_name(k)),'_TD',num2str(a),'.mat');
        save(filename,'freq_features');
    end

    % extract time-frequency domain features
    tf_features = zeros(chan,5, epoch);
    for j = 1:1:epoch
        for i = 1:chan
            x = EEG.data(i, :, j);
            % Continuous Wavelet Transform (CWT)
            [cwtmatr,freq] = cwt(x, EEG.srate); % cwtmatr represents the wavelet coefficients at the corresponding scale, and freq is the scale-to-frequency conversion of CWT
            energy_mean = mean(abs(cwtmatr).^2, 2);
            abs_mean=mean(abs(cwtmatr),2); 
            abs_std=std(abs(cwtmatr),[],2); 
            tf_features(i, 1, j) = mean(energy_mean); % energy
            tf_features(i, 2, j) = mean(abs_mean); % absolute mean
            tf_features(i, 3, j) = mean(abs_std); % standard deviation
            cul_energy=0;
            for k = 1:size(freq,1)
                energy = sum(abs(cwtmatr(k,:)).^2);
                cul_energy=energy+cul_energy;
                ree(k) = cul_energy/energy;
            end
            tf_features(i, 4, j) = mean(ree); % recursive energy efficiency
            waveletCoeffs = abs(cwtmatr); % calculate the absolute value of the wavelet transform coefficients
            numbin = size(freq,1);
            for k = 1:numbin
                [counts, ~] = hist(waveletCoeffs(k,:), numbin); % compute histograms for estimating probability distributions
                probabilities = counts / numel(x); % calculate probability distributions
                probabilities = nonzeros(probabilities); % remove items with probability 0 to avoid errors in logarithmic operations
                entropy(k) = -sum(probabilities.* log2(probabilities));
            end
            tf_features(i, 5, j) = mean(entropy); % wavelet time frequency entropy
        end
    end
    tf_features=mean(tf_features,3);
    save(['TD',num2str(a),'\tf_features_TD',num2str(a),'.mat'],'tf_features');
end

%hjorth and diff_entropy are implemented in separate function files.