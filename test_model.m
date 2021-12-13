function test_model(model_directory, input_directory, output_directory)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose:
% Test model and obtain test outputs
% *** Do not edit this script.
% Inputs:
% 1. model_directory: the directory containing the models
% 2. input_directory: the directory containing test data and header files
% 3. output_directory: a directory to save the test output in
%
% Outputs:
% output csv file: the recording name, class and output scores and labels
%
% Author: Nadi Sadr, PhD, <nadi.sadr@dbmi.emory.edu>
% Version 1.0
% Date 9-Dec-2020
% Version 2.0 25-Jan-2021
% Version 2.0 26-April-2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Define lead sets (e.g 12, 6, 4, 3 and 2 lead ECG sets)
    twelve_leads = [{'I'}, {'II'}, {'III'}, {'aVR'}, {'aVL'}, {'aVF'}, {'V1'}, {'V2'}, {'V3'}, {'V4'}, {'V5'}, {'V6'}];
    six_leads    = [{'I'}, {'II'}, {'III'}, {'aVR'}, {'aVL'}, {'aVF'}];
    four_leads   = [{'I'}, {'II'}, {'III'}, {'V2'}];
    three_leads  = [{'I'}, {'II'}, {'V2'}];
    two_leads    = [{'I'}, {'II'}];
    one_leads    = [{'I'}];
    lead_sets = {twelve_leads, six_leads, four_leads, three_leads, two_leads, one_leads};

    % Find files
    input_files = {};
    for f = dir(input_directory)'
        if exist(fullfile(input_directory, f.name), 'file') == 2 && f.name(1) ~= '.' && all(f.name(end - 2 : end) == 'mat')
            input_files{end + 1} = f.name;
        end
    end

    % Check output directory exists
    if ~exist(output_directory, 'dir')
        mkdir(output_directory)
    end

    % Create the cell containing the models
    model=cell(1,length(lead_sets));
          
        
    
    %% Predicting the outputs
    % Iterate over files.
    disp('Predicting ECG leads labels...')
    num_files = length(input_files);
    %num_files = 100 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% cambiaaa
    
    for i = 1 : num_files

        if mod(i, 1000) == 0
            disp(['    ', num2str(i), '/', num2str(num_files), '...'])
        end

                % Load test data.
        file_tmp            = strsplit(input_files{i},'.');
        tmp_input_file      = fullfile(input_directory, file_tmp{1});
        [data, header_data] = load_challenge_data(tmp_input_file);

        %% Check the available ECG leads
        tmp_hea   = strsplit(header_data{1},' ');
        num_leads = str2num(tmp_hea{2});
        [leads, leads_idx] = get_leads(header_data, num_leads);
        for kk = 1:length(lead_sets)
            if length(lead_sets{kk})==length(leads)
                if ((strcmp(lead_sets{kk}, leads)) == 1) % if the leads are from the defined leads sets
                    %% Load model.
                    % This function is **required**.
                    % if the model has not been loaded for another data in the dataset, load the model
                    if isempty(model{kk})==1
                        disp(['Loading ',num2str(length(lead_sets{kk})),'-leads ECG model...'])
                        model{kk} = load_ECG_leads_model(model_directory,length(lead_sets{kk}));
                    end

                    [current_score,current_label,classes] = team_testing_code(data, header_data, model{kk});

                    %% Save model outputs.
                    save_challenge_predictions(output_directory, file_tmp{1}, current_score, current_label, classes);
                else
                    disp('The leads of the input data do not match the defined leads...')
                end
            end
        end
    end

    disp('Done.')
end




%% Load test data
function [data, tlines] = load_challenge_data(filename)

    % Opening header file
    fid=fopen([filename '.hea']);
    if (fid<=0)
        disp(['error in opening file ' filename]);
    end

    tline = fgetl(fid);
    tlines = cell(0,1);
    while ischar(tline)
        tlines{end+1,1} = tline;
        tline = fgetl(fid);
    end
    fclose(fid);

    f=load([filename '.mat']);
    try
        data = f.val;
    catch ex
        rethrow(ex);
    end

end

%% save predictions
function save_challenge_predictions(output_directory,recording, scores, labels,classes)

    output_file = ([output_directory filesep recording '.csv']);

    Total_classes = strjoin(classes,','); %insert commaas
    %write header to file
    fid = fopen(output_file,'w');
    fprintf(fid,'#%s\n',recording);
    fprintf(fid,'%s\n',Total_classes);
    fclose(fid);

    %write data to end of file
    dlmwrite(output_file,labels,'delimiter',',','-append','precision',4);
    dlmwrite(output_file,scores,'delimiter',',','-append','precision',4);

end

%% Load your trained ECG models
% This function is **required**
% Do **not** change the arguments of this function.
function model = load_ECG_leads_model(model_directory, num_leads)
    out_file=[num2str(num_leads),'_lead_ecg_model.mat'];
    filename=fullfile(model_directory,out_file);
    A=load(filename);
    model=A;
end
