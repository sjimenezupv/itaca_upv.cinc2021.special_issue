%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% team_testing_code
%%% Apply classifier model to test set
%%%
%%% Inputs: ...
%%%
%%% Outputs: ...
%%%
%%% Author:  Santiago Jiménez-Serrano [sanjiser@upv.es]
%%% Version: 1.0
%%% Date:    2020-03-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [score, labels, classes] = team_testing_code(data, header_data, loaded_model)

    model    = loaded_model.model;
    classes  = loaded_model.classes;
    mu       = loaded_model.mu;
    sigma    = loaded_model.sigma;
    medianas = loaded_model.medianas;

    num_classes = length(classes);
    labels      = zeros(1, num_classes);
    score       = zeros(1, num_classes);

    %% Extract features from test data
    tmp_hea   = strsplit(header_data{1},' ');
    num_leads = str2num(tmp_hea{2});
    [leads, leads_idx] = get_leads(header_data,num_leads);
    features = get_features(data, header_data, leads_idx);
    
    %% Fill the NaN values with the median
    nan_idx = isnan(features);
    features(nan_idx) = medianas(nan_idx);
    
    %% Appy the z-score
    features=bsxfun(@minus,   features, mu);
    features=bsxfun(@rdivide, features, sigma);

    %% Use your classifier here to obtain a label and score for each class.

    % Get the score and label for each model
    parfor i = 1 : length(model)        
        if  strcmp(model{i}.type, 'kmeans') == 1
            [score(i), labels(i)] = eva_model_kmeans(model{i}, features);
        else
            [score(i), labels(i)] = eva_model(model{i}, features);
        end
    end

    
end

function [score, label] = eva_model(model, features)

    
    score = 0.0;
    label = 0;

    if strcmp(model.type, 'error') == 1        
        score = 0.0;
        label = 0;
        
    elseif model.g > 0.0 && strcmp(model.type, 'net') == 1
        
        score = model.model(features(:, model.features_idx)', 'useGPU', 'no');
        if score >= model.th
            label = 1;
        end

        if score < -1
            score = -1;
        elseif score > 1
            score = 1;
        end
        score = (score+1)/2;


    elseif model.g > 0.0 && strcmp(model.type, 'nb') == 1        
        
        [label, posterior] = predict(model.model, features(:, model.features_idx));
        score = posterior(label+1);
        
    elseif model.g > 0.0 && strcmp(model.type, 'cn') == 1
        [label, sc] = model.model.predict(features(:, model.features_idx));
        score = sc(label+1);
        
        
    end
    
end

function [score, label] = eva_model_kmeans(model_km, features)

    nclusters = model_km.nclusters;

    % Dists to the centers of the kmeans
    dists = ones(1, nclusters) .* 9999999999;
    
    some_model = false;
    score      = 0;
    label      = 0;
    
    % 3 kmeans - Search the model that best fits to the corresponding cluster
    for i = 1 : nclusters
        if strcmp(model_km.best_models{i}.type, 'error') ~= 1            
            dists(i)  = pdist2(model_km.best_models{i}.kmeans_center, features(:, model_km.features_idx));            
            some_model = true;
        end
    end
    
    if some_model == true
        
        % Select the model with minimun distance to the centroid
        [~, best_k] = min(dists);
    
        % Get the score and label
        % Warning! The inner model has its own features_idx based in the
        % kmeans training cluster!!!
        [score, label] = eva_model(model_km.best_models{best_k}, features(:, model_km.features_idx));
    end

end


