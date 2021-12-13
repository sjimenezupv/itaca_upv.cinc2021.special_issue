%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% get_features
%%% Extract features from ECG signals of every lead
%%%
%%% Inputs:
%%%  1. ECG data from available leads (data)
%%%  2. Header files including the number of leads (header_data)
%%%  3. The available leads index (in data/header file)
%%%
%%% Outputs:
%%%  features - Array of signal features
%%%
%%% Author:  Santiago Jiménez-Serrano [sanjiser@upv.es]
%%% Version: 1.0
%%% Date:    2020-03-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function features = get_features(data, header_data, leads_idx)


    % read number of leads, sample frequency and adc_gain from the header.
    [recording, Total_time, num_leads, ...
     Fs, adc_gain, age, sex, Baseline] ...
        = extract_data_from_header(header_data);
    
    num_leads = length(leads_idx);
    
    % Initialize features array
    num_features_x_lead=81;
    num_features       = 2 + (num_features_x_lead*num_leads);    
    
    % Initial set of features
    features = [age, sex];
    
    
    try
        
        % Get the features for eache lead
        for i = [leads_idx]            
            
            %if i == 1
            if i > 0
                
                %subplot(2, 1, 1)
                %plot(data(i,:));
                
                % Apply adc_gain and remove baseline
                LeadWGain = (data(i,:)-Baseline(i))./adc_gain(i);

                %subplot(2, 1, 2)
                %plot(LeadWGain);
                %pause
                
                try
                    % CinC sjimenez features
                    fsjs = GetChallengeFeatures_CinC2021_v03(LeadWGain, Fs);                    
                catch
                    fsjs = nan(1, num_features_x_lead);
                end
                
                % Append features
                features = [features, fsjs];
            end

        end

    catch ex
                
        % Uncomment to test
        % plot(LeadWGain)        
        % rethrow(ex)
        % Some error happened ==> Set the length of a common feature array

        features = nan(1, num_features);
    end

    
end




