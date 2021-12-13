function train_model(input_directory, output_directory)

% *** Do not edit this script.
% Train models for three different ECG leads sets

if ~exist(output_directory, 'dir')
    mkdir(output_directory)
end

disp('Running training code...')
team_training_code(input_directory,output_directory); %team_training_code>train ECG leads classifier

disp('Done.')
end
