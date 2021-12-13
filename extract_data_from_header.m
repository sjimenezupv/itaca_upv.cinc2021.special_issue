function [recording,Total_time,num_leads,Fs,adc_gain,age_data,sex_data,Baseline]=extract_data_from_header(header_data);

% Extract header information:
 
tmp_hea = strsplit(header_data{1},' ');
recording = tmp_hea{1};             % recording ID
num_leads = str2num(tmp_hea{2});    % number of leads
Fs = str2num(tmp_hea{3});           % sampling frequency  
Total_time = str2num(tmp_hea{4})/Fs;% total time (seconds) of the ECG
adc_gain = zeros(1,num_leads);      % initialize adc_gains or floating digits of the ECG leads 
Baseline = zeros(1,num_leads);      % initialize reported baseline of ECG leads

for ii=1:num_leads
    tmp_hea = strsplit(header_data{ii+1},' ');
    tmp_gain=strsplit(tmp_hea{3},'/');
    adc_gain(ii)=str2num(tmp_gain{1});   % (ADC units) a floating-point number of the ECG leads
    Baseline(ii)=str2num(tmp_hea{5});    % reported baseline of ECG leads
end

% Extract demographic information
for tline = 1:length(header_data)
    if startsWith(header_data{tline},'#Age')
        tmp = strsplit(header_data{tline},': ');
        age_data = str2num(tmp{2});  % age of the subject
        % encode gender of the subject:
        % 0 for female
        % 1 for male
        % NaN for other
    elseif startsWith(header_data{tline},'#Sex')
        tmp = strsplit(header_data{tline},': ');
        if strcmp(tmp{2},'female') || strcmp(tmp{2},'f')...
                || strcmp(tmp{2},'F') || strcmp(tmp{2},'Female') || strcmp(tmp{2},'FEMALE')
            sex_data = 0;
        elseif strcmp(tmp{2},'male') || strcmp(tmp{2},'m')...
                || strcmp(tmp{2},'M') ||strcmp(tmp{2},'Male') || strcmp(tmp{2},'MALE')
            sex_data = 1;
        else
            sex_data = nan;
        end
    end
end


end
