%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% team_training_code
%%% Train ECG leads and obtain classifier models
%%% for 12, 6, 4, 3, 2 and 1-lead ECG sets
%%%
%%% Inputs:
%%%  input_directory
%%%  output_directory
%%%
%%% Outputs:
%%%  model - trained model
%%%
%%% Author:  Santiago Jiménez-Serrano [sanjiser@upv.es]
%%% Version: 1.0
%%% Date:    2021-12-07
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function  model = team_training_code(input_directory, output_directory)

	%% Define lead sets (e.g 12, 6, 4, 3 and 2 lead ECG sets)
	twelve_leads = [{'I'}, {'II'}, {'III'}, {'aVR'}, {'aVL'}, {'aVF'}, {'V1'}, {'V2'}, {'V3'}, {'V4'}, {'V5'}, {'V6'}];
	six_leads    = [{'I'}, {'II'}, {'III'}, {'aVR'}, {'aVL'}, {'aVF'}];
	four_leads   = [{'I'}, {'II'}, {'III'}, {'V2'}];
	three_leads  = [{'I'}, {'II'}, {'V2'}];
	two_leads    = [{'I'}, {'II'}];
    one_leads    = [{'I'}];
	lead_sets    = {twelve_leads, six_leads, four_leads, three_leads, two_leads, one_leads};
    


    %% Get the file names
    [input_files, num_files] = getInputFiles(input_directory);	
    fprintf('Number of files in input directory = %d \n', num_files);
    
    
	%% Filter data????
    %num_files = 7143
	%input_files = input_files(1:num_files);
	% End Filter data

    
    %% Extract classes from dataset    
    [classes, num_classes] = getClasses_2();
        
    
    %% Set the number of features (for each lead and the total ones)
    nfeatures_x_lead = 81;
    num_features     = 2 + (nfeatures_x_lead*12); % First ones will be age & sex
    
    
    %% Initialize the dataset (X=features, y=labels)
    labels     = zeros(num_files, num_classes);
    features   = zeros(num_files, num_features);    
    debug_step = int32(1000);
    
    if 1 == 1

        %% Load data recordings and header files        
        tic        
        fprintf('\nReading data and Getting Features...\n')    
        parfor i = 1 : num_files

            % Debug
            if mod(i, debug_step) == 0
                fprintf('Loading & Featuring    %5d/%5d ...\n', i, num_files);
            end    

            % Load data & Extract features
            [features(i,:), labels(i, :)] = read_features(input_directory, input_files{i}, classes, num_features);

        end

        % Filter the features and labels that does not belongs to any class
        filter = sum(labels, 2) == 0; % sum in the dimension of columns
        features(filter, :) = []; % Remove the samples
        labels(filter,   :) = [];
        fprintf('[INFO]: Filtered %d samples with no class!\n', sum(filter));

        % Now, filter the classes that have no enough samples to be trained
        filter = sum(labels) < 150;
        labels(:, filter) = [];
        fprintf('[INFO]: Filtered %d classes with no samples (<%d)!\n', sum(filter), 150);


        %% Outliers filtering
        [features, muOld, sgOld, muNew, sgNew, totFiltered] = FilterOutliers(features); % Esto debe considerarse en el modelo final


        %% NaNs filtering
        [features, medianas] = FilterNaN(features);


        %% Plot the Boxplots
        %%plotBx(dataset, class);


        %% Apply z-Score
        [features, mu, sigma] = ApplyZScore(features);


        %% Save after filtering        
        save('./features_v35_f3_time.mat', 'features', 'labels', 'mu', 'sigma', 'medianas');
        disp('Save Features -> DONE');
        
        fprintf('[TOC] Seconds needed for feature extracion...\n');
        toc;
        fprintf('\n')
    
    else
        disp('Loading Features...');        
        S = load('./features_v35_f3_time.mat');
        features = S.features;
        labels   = S.labels;
        mu       = S.mu;
        sigma    = S.sigma;
        medianas = S.medianas;
        disp('Loading Features -> DONE');
    end
    

    %% Train the models
        
    warning('off');
    
    % For each lead combination...
    for i=length(lead_sets):-1:1


        % Train the corresponding ECG model
        num_leads = length(lead_sets{i});
        fprintf('Training %d-lead ECG model...\n', num_leads);
        tic
        
        % Error -> this instruction return a sorted vector in leads_idx
        %%[leads, leads_idx] = get_leads(header_data, num_leads);
                
        % Real leads idx
        if num_leads == 12
            leads_idx = 1:12;
        elseif num_leads == 6
            leads_idx = 1:6;
        elseif num_leads == 4
            leads_idx = [1, 2, 3, 8]; % I II III V2
        elseif num_leads == 3
            leads_idx = [1, 2, 8]; % I II V2
        elseif num_leads == 2
            leads_idx = [1, 2]; % I II
        elseif num_leads == 1
            leads_idx = 1; % I
        end
        
        leads_idx
        
        % Get the feature indexes
        Features_leads_idx = get_features_idx(leads_idx, nfeatures_x_lead);
        
        % Features for those leads
        Features_leads = features(:, Features_leads_idx);

        % Train the model
        nets = train_whole_models(Features_leads, labels, classes);
        save_ECGleads_model(num_leads, nets, mu(Features_leads_idx), sigma(Features_leads_idx), medianas(Features_leads_idx), output_directory, classes);        
        
        % Get the number of seconds for testing the model_i
        fprintf('[TOC] model training in seconds for %d leads\n', num_leads);
        toc
        fprintf('\n');
    end

    model = nets;

    warning('on');
    
end

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get Input files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [input_files, nfiles] = getInputFiles(input_directory)

    fprintf('\n Getting training file names ...\n');
    
    list = dir(input_directory)';
    nfiles = 0;
    
    % 1 - Read the number of valid files
    for f = [ list ]
        if exist(fullfile(input_directory, f.name), 'file') == 2 && ...
           f.name(1) ~= '.' && ...
           all(f.name(end - 2 : end) == 'mat')
            nfiles = nfiles + 1;
        end
    end
    
    % Initialize cell of input file paths
    input_files = cell(nfiles, 1);
    
    % Current file index
    filei = 1;
    
    for f = [ list ]
        if exist(fullfile(input_directory, f.name), 'file') == 2 && ...
           f.name(1) ~= '.' && ...
           all(f.name(end - 2 : end) == 'mat')
            input_files{filei} = f.name;
            filei = filei + 1;
        end
    end

end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature Extraction from one given sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [features, labels] = read_features(input_directory, input_file, classes, num_features)
% Load data recordings and header files
    
    % Load data
    [data, header_data] = read_sample(input_directory, input_file);
    
    
    %% Extract labels
    % Vector that will containt a 1 if the sample belongs to such class
    labels = zeros(1, length(classes));
    
    for j = 1 : length(header_data)
        
        if startsWith(header_data{j},'#Dx')
            
            tmp = strsplit(header_data{j},': ');
            % Extract more than one label if avialable
            Dx_classes = strsplit(tmp{2},',');
            
            for k=1:length(Dx_classes)
                
                % Check the classes that share score
                if strcmp(Dx_classes{k}, '164909002') == 1
                    Dx_classes{k} = '733534002';
                elseif strcmp(Dx_classes{k}, '59118001') == 1
                    Dx_classes{k} = '713427006';
                elseif strcmp(Dx_classes{k}, '63593006') == 1
                    Dx_classes{k} = '284470004';
                elseif strcmp(Dx_classes{k}, '17338001') == 1
                    Dx_classes{k} = '427172004';
                end
                
                idx=find(strcmp(classes, Dx_classes{k}));
                labels(idx)=1;
            end
            
            break
        end
    end
    
    % The sample does not belong to any class that we want to classify
    if sum(labels) == 0
        features = zeros(1, num_features);
        return;
    end
    
    
    %% Extract features

    % Check the number of available ECG leads
    tmp_hea   = strsplit(header_data{1},' ');
    num_leads = str2num(tmp_hea{2});
    [~, leads_idx] = get_leads(header_data, num_leads);

    % Extract features
    features = get_features(data, header_data, leads_idx);

end


function [data, header_data] = read_sample(input_directory, input_file)
% Load data recordings and header files
    
    % Load data
    file_tmp            = strsplit(input_file,'.');
    tmp_input_file      = fullfile(input_directory, file_tmp{1});
    [data, header_data] = load_challenge_data(tmp_input_file);
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Features Index - Utils
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function features_idx = get_features_idx(leads_idx, num_features_x_lead)
    
    
    num_features = 2 + (num_features_x_lead*12);
    features_idx = zeros(1, num_features);
    
    % 2 first features always are the age and sex
    features_idx(1) = 1;
    features_idx(2) = 1;
    
    for i = [leads_idx]

        % Get the start and end feature index
        start_idx = 3 + ((i-1)*num_features_x_lead);
        end_idx   = start_idx + num_features_x_lead - 1;

        % Append features
        features_idx(start_idx:end_idx) = 1;
    end    
    
    % Mask to index array
    features_idx = find(features_idx);
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model Saving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function save_ECGleads_model(num_leads, model, mu, sigma, medianas, output_directory, classes) %save_ECG_model
    % Save results
	modelname = [num2str(num_leads),'_lead_ecg_model.mat'];
    filename  = fullfile(output_directory, modelname);
    save(filename, 'model', 'mu', 'sigma', 'medianas', 'classes', '-v7.3');    
    disp(['Save Model ',  filename,' -> Done']);
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dataset/Features Saving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function save_ECGleads_features(features, output_directory)
    % Save features
    filename=fullfile(output_directory, 'features.mat');
    save(filename, 'features');
    disp('Save Features -> DONE')
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File Utilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [classes, num_classes] = get_classes(input_directory, files)
    % find unique number of classes

    fprintf('\n Getting unique class identifiers ...\n');

    classes={};
    
    num_files = length(files);
    k=1;
    
    % For each file...
    for i = 1:num_files
        
        g = strrep(files{i},'.mat','.hea');
        input_file = fullfile(input_directory, g);
        fid=fopen(input_file);
        tline  = fgetl(fid);
        tlines = cell(0,1);

        while ischar(tline)
            tlines{end+1,1} = tline;
            tline = fgetl(fid);
            if startsWith(tline,'#Dx')
                tmp   = strsplit(tline,': ');
                tmp_c = strsplit(tmp{2},',');
                for j=1:length(tmp_c)
                    idx2 = find(strcmp(classes,tmp_c{j}));
                    if isempty(idx2)
                        classes{k}=tmp_c{j};
                        k=k+1;
                    end
                end
                break
            end
        end

        fclose(fid);
    end
    
    classes     = sort(classes);
    num_classes = length(classes);
    
end


function [classes, num_classes] = getClasses_2()
    classes = cell(26, 1);
    classes{ 1} = '164889003';
    classes{ 2} = '164890007';
    classes{ 3} =   '6374002';
    classes{ 4} = '426627000';
    classes{ 5} = '733534002'; % same than '164909002'
    classes{ 6} = '713427006'; % same than '59118001';
    classes{ 7} = '270492004';
    classes{ 8} = '713426002';
    classes{ 9} =  '39732003';
    classes{10} = '445118002';    
    classes{11} = '251146004';
    classes{12} = '698252002';
    classes{13} = '426783006'; %% Sinus Rhythm
    classes{14} = '284470004'; % same than '63593006'
    classes{15} =  '10370003';
    classes{16} = '365413008';
    classes{17} = '427172004'; % same than '17338001'
    classes{18} = '164947007';
    classes{19} = '111975006';
    classes{20} = '164917005';
    classes{21} =  '47665007';    
    classes{22} = '427393009';
    classes{23} = '426177001';
    classes{24} = '427084000';    
    classes{25} = '164934002';
    classes{26} =  '59931005';    
    classes     = sort(classes);
    num_classes = length(classes);
end


function [data, tlines] = load_challenge_data(filename)

    % Opening header file
    fid=fopen([filename '.hea']);

    if (fid<=0)
        disp(['error in opening file ' filename]);
    end

    tline  = fgetl(fid);
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


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dataset preprocessing/filtering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [dataset, muOld, sgOld, muNew, sgNew, totFiltered] = ...
    FilterOutliers(dataset)

    nsg = 3;                  % Number of std to consider outlier
    mu  = nanmean(dataset);   % mean
    sg  = nanstd(dataset);    % std
    ic1 = mu - nsg .* sg;     % Confidence interval, lower limit
    ic2 = mu + nsg .* sg;     % Confidence interval, upper limit
    totFiltered = 0;          % Counter of filtered samples
    [nr, nf] = size(dataset); % Number of columns/features

    % For each column/feature
    for i = 1 : nf

        % Get the data vector
        v = dataset(:, i);

        % If it is lower than mu, we must to check the lower limit
        signo = v>=mu(i);
        signo(signo==0) = -1; 

        % Filter indexes with the data to be normalized/filtered
        f = find(v < ic1(i) | v > ic2(i));
        
        % Check if exist some outlier
        if length(f) > 0            
            
            % Debug
            fprintf('Replacing Outliers in col [%d] => %d Outliers in %d rows \n' , ...
                i, length(f), nr);
            
            % Set the outliers to its corresponding limit (lower or upper)
            v(f)= mu(i) + (signo(f)) .* nsg * sg(i);

            % Replace the data in its own column
            dataset(:, i) = v;

            % Count the number os filtered samples
            totFiltered = totFiltered + length(f);            

        end
    end
    
    % Save the old mu and sigma vectors
    muOld = mu;
    sgOld = sg;
    
    % Save the new mu and sigma vectors
    muNew = nanmean(dataset);
    sgNew = nanstd(dataset);

end

function [dataset, medianas] = FilterNaN(dataset)

    [nrows, ncols] = size(dataset);
    medianas = nanmedian(dataset);    
    
    for i = 1 : ncols
        
        % Get the column values
        x = dataset(:, i);
        
        % Get the indexes where NaN values exist
        nan_idx = isnan(x);
        
        % Check if exist some NaN value
        if sum(nan_idx) > 0            
            
            % Debug
            fprintf('Replacing NaNs in col [%d] => %d NaNs in %d rows (replaced by median: %f) \n' , ...
                i, sum(nan_idx), nrows, medianas(i));
            
            % Set the median values in the NaN
            dataset(nan_idx, i) = medianas(i);
        end
    end
    
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Z-Scoring
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dataset, mu, sigma] = ApplyZScore(dataset)

    [dataset, mu, sigma] = zscore(dataset);

end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Networks training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [g] = getBestThresholds_nb(nb, features, y, class_name)

    % Create the models response vector
    [nsamples, nf] = size(features);
    yhat           = zeros(1, nsamples);

    fprintf('Classifiyng for class %s ... #samples = %d, #features = %d [naive_bayes]', class_name, nsamples, nf);
    parfor i = 1 : nsamples
        yhat(i) = predict(nb, features(i, :));
    end

    % Get the best threshold
    [g] = getNBg(y, yhat);    
    fprintf(' [g] = [%.4f] \n', g);    
end

function [g] = getBestThresholds_cn(ld, features, y, class_name)

    % Create the models response vector
    [nsamples, nf] = size(features);
    yhat           = zeros(1, nsamples);

    fprintf('Classifiyng for class %s ... #samples = %d, #features = %d [ld|cnet]', class_name, nsamples, nf);
    parfor i = 1 : nsamples
        yhat(i) = ld.predict(features(i, :));
    end

    % Get the best threshold
    [g] = getNBg(y, yhat);    
    fprintf(' [g] = [%.4f] \n', g);    
end


function [th, best_g] = getBestThresholds(net, features, y, class_name)

    % Create the models response vector
    [nsamples, nf] = size(features);    
    yhat = zeros(1, nsamples);

    fprintf('Classifiyng for class %s ... #samples = %d, #features = %d ', class_name, nsamples, nf);
    parfor i = 1 : nsamples
        yhat(i) = net(features(i, :)', 'useGPU', 'no');
    end

    % Get the best threshold
    [th, best_g] = getBestTh(y, yhat);
    fprintf(' best [th, g] = [%.4f, %.4f] \n', th, best_g);    
    
end


function [net] = setBestThresholds(net, features, y, class_name)
   
    % Set the value into the struct
    [th, g] = getBestThresholds(net, features, y, class_name);
    
    net.userdata.th = th;
    net.userdata.g  = g;    
end

function [train_x, train_y, test_x, test_y] = get_train_test(features, y)

    % Get the positive and negative samples indexes
    ynegidx = find(y==0);
    yposidx = find(y==1);
    
    % Random permutation of indexes
    ynegidx = ynegidx(randperm(length(ynegidx)));
    yposidx = yposidx(randperm(length(yposidx)));
    
    % Get 75% of data
    cutoff_neg = int32(length(ynegidx)*0.75);
    cutoff_pos = int32(length(yposidx)*0.75);
    
    % 75% data for training and testing
    tridx = [ynegidx(1:cutoff_neg);     yposidx(1:cutoff_pos)    ];
    teidx = [ynegidx(cutoff_neg+1:end); yposidx(cutoff_pos+1:end)];
    
    % Random permutation of indexes - again
    tridx = tridx(randperm(length(tridx)));
    teidx = teidx(randperm(length(teidx)));
    
    % Training set
    train_x = features(tridx, :);
    train_y = y(tridx);
    
    % Test set
    test_x = features(teidx, :);
    test_y = y(teidx);
    
end


function [net] = train_nn_step01(features, y, class_name, layer_size)

    fprintf('Layers: %d: \n', layer_size);

    ynegidx = find(y==0);
    yposidx = find(y==1);
    
    cutoff_neg = int32(length(ynegidx)*0.75);
    cutoff_pos = int32(length(yposidx)*0.75);
    
    tridx = [ynegidx(1:cutoff_neg);     yposidx(1:cutoff_pos)];
    teidx = [ynegidx(cutoff_neg+1:end); yposidx(cutoff_pos+1:end)];
    
    % Random permutation
    tridx = tridx(randperm(length(tridx)));
    teidx = teidx(randperm(length(teidx)));
    
    % Create the net
    net = feedforwardnet(layer_size, 'trainscg'); % Trainscg is the trainFcn    
    
    % Configure the net
    net = configure(net, features(tridx, :)', y(tridx)');    
    
    % Avoid gui    
    net.trainParam.showWindow = 0;
    
    % Negative inputs must be -1
    y(y==0) = -1;

    % Check point - At least 100 rows to train
    if length(y(tridx)) < 100
        net.userdata.th = 0;
        net.userdata.g  = 0;
        return;
    end    
    
    % Train the net
    net = train(net, features(tridx, :)', y(tridx)', ...
        'useGPU', 'yes', ...
        'showResources', 'no');
    
    % Get the best Threshold
    net = setBestThresholds(net, features(teidx, :), y(teidx), class_name);
    
end

function [net] = train_nn_step01_SIMPLE(features, y, class_name, layer_size, test_x, test_y)

    net = feedforwardnet(layer_size, 'trainscg'); % Trainscg is the trainFcn    
    net = configure(net, features', y');
    net.trainParam.showWindow = 0;
    
    % Negative inputs must be -1
    y(y==0) = -1;

    % Check point - At least 100 rows to train
    if length(y) < 100
        net.userdata.th = 0;
        net.userdata.g  = 0;
        return;
    end    
    
    % Train the net
    net = train(net, features', y', 'useGPU', 'yes', 'showResources', 'no');
    
    % Get the best Threshold
    net = setBestThresholds(net, test_x, test_y, class_name);
    
end



function [new_features, features_indexes] = train_feature_selection(features, y, class_name)
    
    % First two columns always set
    features_indexes = [1, 2];
    
    [~, n] = size(features);
    
    nf1 = 0; % #features filtered at stage 1
    nf2 = 0; % #features filtered at stage 2
    
    for i = 3 : n
        %features_indexes = [features_indexes, i];
        [h, p] = ttest2(features(y==1, i), features(y==0, i)); %, 'Alpha', 0.01);
        if h == 1
            features_indexes = [features_indexes, i];
            %fprintf('Feature selected for class %s: Fi=%d, p_value=%f \n', class_name, i, p);
        else
            nf1 = nf1 + 1;
        end        
    end
    
    % Correlation coefficient threshold
    corr_coeff_th = 0.95;
    
    for i = 3 : length(features_indexes)
        for j = i + 1 : length(features_indexes)
            c = corrcoef([features(:, features_indexes(i)), features(:, features_indexes(j))]);
            if abs(c(1, 2)) >= corr_coeff_th
                features_indexes(j) = -1;
                nf2 = nf2 +1;
                %fprintf('Feature selection removed corrcoef for class %s: Fi=%d, Fj=%d \n', class_name, features_indexes(i), features_indexes(j));
            end
        end
        
        features_indexes(features_indexes == -1) = [];
    end
    
    
    fprintf('\nFS [%s]: #features=%d #nfiltered_1=%d #nfiltered_2=%d #n_total_filtered=%d\n\n', class_name, length(features_indexes), nf1, nf2, nf1+nf2);
    
    new_features = features(:, features_indexes);
    
end


function [model] = train_nbayes(features, y, class_name)

    % 0 - Perform the feature selection
    [features, features_indexes] = train_feature_selection(features, y, class_name);    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    % Set the model properties
    model.features_idx = features_indexes;
    model.type         = 'nb';    
    
    % Train a Naive Bayes model
    try
        nb = fitcnb(features, y);
        nb = compact(nb);    
        [nb_g] = getBestThresholds_nb(nb, features, y, class_name);        
        model.model = nb;
        model.th    = 0.5;
        model.g     = nb_g;
        
    catch ME        
        fprintf('Error training Naive Bayes...\n');
        %%%getReport(ME)
        
        model.th   = 0.0;
        model.g    = 0.0;
        model.type = 'error';
    end
    
end


function [model] = train_nn(features, y, class_name)

    % 0 - Perform the feature selection
    [features, features_indexes] = train_feature_selection(features, y, class_name);    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    nlayers = [18, 32];
    
    
    % Train the neural networks and mantaing the one with best G value
    for i = 1 : length(nlayers)        
        
        % Train the neural network
        net_aux = train_nn_step01(features, y, class_name, nlayers(i));
        
        % Save first network or best, comparing g values
        if i==1 || net_aux.userdata.g > model.g
            model.model = net_aux;
            model.th    = net_aux.userdata.th;
            model.g     = net_aux.userdata.g;
        end        
    end

    % Set the model properties
    model.features_idx = features_indexes;
    model.type         = 'net';    
    
    
    % Test also a Naive Bayes model
    try
        nb = fitcnb(features, y);
        nb = compact(nb);
        [nb_g] = getBestThresholds_nb(nb, features, y, class_name);
        if nb_g > model.g
            model.model        = nb;
            model.th           = 0.5;
            model.g            = nb_g;
            model.type         = 'nb';
        end
    catch ME
        fprintf('Error training Naive Bayes...\n');
        %%%getReport(ME)
    end
    
    
    
% % %     % Test also a linear classification cnet network
% % %     try
% % %         
% % %         cn = fitcnet(features, y, ...
% % %             'LayerSizes', 25, ...
% % %             'Activations', 'relu', ...
% % %             'Lambda', 0, ...
% % %             'IterationLimit', 100, ...
% % %             'Standardize', false, ...
% % %             'ClassNames', [0; 1]);
% % %         
% % %         cn = compact(cn);
% % % 
% % %         [cn_g] = getBestThresholds_cn(cn, features, y, class_name);
% % %         if cn_g > model.g
% % %             model.model        = cn;
% % %             model.th           = 0.5;
% % %             model.g            = cn_g;
% % %             model.type         = 'cn';
% % %         end
% % %     catch ME
% % %         fprintf('Error training cnet...\n');
% % %         getReport(ME)
% % %     end
    
end



function [model_km, datasets, y_km, CENTERS] = train_cluster_kmeans(features, y, class_name, nclusters)

    % 1 - Perform the kmeans
    [idx, CENTERS, sumd] = kmeans(features, nclusters, 'MaxIter', 200, 'Display', 'final', 'Replicates', 10);
    
    % Get the dataset for each cluster
    datasets = cell(1, nclusters);
    y_km     = cell(1, nclusters);
    
    for i = 1 : nclusters
        datasets{i} = features(idx==i, :);
        y_km{i}     = y(idx==i);
    end    
    
    % Initialize the meta-model
    model_km.best_models = cell(1, nclusters);
    model_km.type        = 'kmeans';
    model_km.nclusters   = nclusters;
    model_km.kmcenters   = CENTERS;
    model_km.ntotal      = sum(idx==[1:nclusters]);
    model_km.npositives  = sum(idx==[1:nclusters] & y==1);
    model_km.nnegatives  = sum(idx==[1:nclusters] & y==0);
    model_km.sumd        = sum(sumd);
    model_km.class_name  = class_name;
    
end


function [model_km] = train_nn_kmeans(features, y, class_name)

    % 0 - Perform the feature selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [features, features_indexes] = train_feature_selection(features, y, class_name);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % 1 - Clusterify %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Set the number of clusters
    nclusters = 3;
    
    % Get the clusters
    [model_km, datasets, y_km, CENTERS] = train_cluster_kmeans(features, y, class_name, nclusters);    
    
    % Set the features indexes for the kmeans model
    model_km.features_idx = features_indexes;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % #Layers to be tested
    nlayers = [18, 32];
    
    % For each cluster...
    for km = 1 : nclusters
        
        % Debug
        fprintf('\nTraining with km data: %d #samples=%d - #samples_pos=%d \n', km, model_km.ntotal(km), model_km.npositives(km));
        
        % Get the corresponding dataset for this cluster
        features = datasets{km};
        y        = y_km{km};
        
        % Perform a new feature selection for this cluster!
        [features, features_indexes] = train_feature_selection(features, y, class_name);
        
        % 75% samples for training, 25% samples for test
        %[train_x, train_y, test_x, test_y] = get_train_test(features, y);
        % Better to train and test with all the dataset 
        %  => (maybe small subsets)
        train_x = features;
        train_y = y;
        test_x  = features;
        test_y  = y;
        
        % Preconditions
        if model_km.npositives(km) < 40 || length(y) < 40
            fprintf('WARNING: Model with few positive samples.... jump it');
            model.th            = 0.0;
            model.g             = 0.0;
            model.type          = 'error';
            model.kmeans_center = CENTERS(km, :);
            model_km.best_models{km} = model;
            continue;
        end        
        

        % Train the neural networks
        for i = 1 : length(nlayers)

            fprintf('Layers: %d: \n', nlayers(i));
            net_aux = train_nn_step01_SIMPLE(train_x, train_y, class_name, nlayers(i), test_x, test_y);

            % Compare g values
            if i==1 || net_aux.userdata.g > model.g
                model.model = net_aux;
                model.th    = net_aux.userdata.th;
                model.g     = net_aux.userdata.g;
                model.type  = 'net';
                model.units = nlayers(i);
            end        
        end


        % Test also a Naive Bayes model
        try
            nb = fitcnb(train_x, train_y);
            nb = compact(nb);    
            [nb_g] = getBestThresholds_nb(nb, test_x, test_y, class_name);
            if nb_g > model.g
                model.model        = nb;
                model.th           = 0.5;
                model.g            = nb_g;
                model.type         = 'nb';
            end
        catch ME
            fprintf('Error training Naive Bayes...\n');
            %%%getReport(ME)            
        end
        
        % Set the model properties for the best model
        model.features_idx  = features_indexes;
        model.kmeans_center = CENTERS(km, :);
        
        % Save the best model for this kmeans-cluster
        model_km.best_models{km} = model;
    
    end
    
    % Get the mean g value for the whole model
    total_samples = 0;
    for k = 1 : nclusters
        total_samples = total_samples + model_km.ntotal(k);
    end
    
    mean_g = 0;
    for k = 1 : nclusters
        mean_g = mean_g + (model_km.best_models{k}.g * (model_km.ntotal(k) / total_samples));
    end
    model_km.mean_g = mean_g;
    model_km.g      = mean_g;
    
end



function [models] = train_whole_models(features, labels, classes)

    % Get the number of classes
    [~, nclasses] = size(labels);
    
    % One model for each class
    models = cell(1, nclasses);
    
    % 1 - Train one model for each class
    for i = 1 : nclasses
        fprintf('\n\nTraining model %2d/%2d for class: %20s \n', i, nclasses, classes{i});
        
        % Get the response for this model
        y = labels(:, i);
        
        % Class 21 - '47665007' (right axis deviation) 
        if strcmp(classes{i}, '47665007') == 1
            
            % => use always nbayes
            models{i} = train_nbayes(features, y, classes{i});
        
        % All the other classes (with no specific rules)
        else
        
            % Train a model with 
            nn_model     = train_nn(features, y, classes{i});
            kmeans_model = train_nn_kmeans(features, y, classes{i});

            g_nn    = nn_model.g;
            g_kmean = kmeans_model.mean_g;

            fprintf('G[nn, kmeans] = [%f, %f] \n', g_nn, g_kmean);

            if g_kmean > g_nn
                models{i} = kmeans_model;
                fprintf('Best model for this class is a kmeans model \n');
            else
                models{i} = nn_model;
                fprintf('Best model for this class is a simple %s model \n', nn_model.type);
            end
        end
        
    end
end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Threshold selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [g] = getNBg(y, y_hat)

    %y_hat = y_hat';
    n = length(y);
    
        
    % Count the true positive, true negatives...
    vp = 0;
    fp = 0;
    vn = 0;
    fn = 0;
    for i = 1 : n

        if y(i) == 1
            if y_hat(i) == 1
                vp = vp + 1;
            else
                fn = fn + 1;
            end

        else % y == 0

            if y_hat(i) == 0 
                vn = vn + 1;
            else
                fp = fp + 1;
            end

        end
    end

    % Get sensitivity and specificity
    sen = vp / (vp+fn);
    esp = vn / (vn+fp);

    % We want to improve the ratio among sensitivity and specificity
    g = sqrt(sen * esp);

end

function [best_th, best_g] = getBestTh(y, y_hat)

    %y_hat = y_hat';
    n = length(y);
    
    best_th = -1.0;
    best_g  =  0.0;
    
    min_th = min(y_hat);
    max_th = max(y_hat);
    step   = 0.005;
    
    for th = min_th : step : max_th
        
        % Count the true positive, true negatives...
        vp = 0;
        fp = 0;
        vn = 0;
        fn = 0;
        for i = 1 : n
            
            if y(i) == 1
                if y_hat(i) >= th
                    vp = vp + 1;
                else
                    fn = fn + 1;
                end
                    
            else % y == 0
                
                if y_hat(i) < th 
                    vn = vn + 1;
                else
                    fp = fp + 1;
                end
                
            end
        end
        
        % Get sensitivity and specificity
        sen = vp / (vp+fn);
        esp = vn / (vn+fp);
        
        % We want to improve the ratio among sensitivity and specificity
        g = sqrt(sen * esp);
        
        %fprintf('%.3f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n', th, vp, fp, vn, fn, sen, esp, g);
        
        % Select the best g score
        if g > best_g
            best_g  =  g;
            best_th = th;
        end        
    end
    
    %fprintf('BestTh %.4f Best_g %.4f \n', best_th, best_g);

end

