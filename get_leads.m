function [sorted_leads, sorted_leads_idx]= get_leads(header_data, num_leads)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose:
% Extract the available ECG leads
% 1. 12 leads are available or
% 2. Reduced leads: for any lead sets defined in lead_sets, e.g:
%    - If 6-leads are available: I, II, III, aVR, aVL & aVF or
%    - If 4-leads are avialable: I, II, III & V2 or
%    - If 3-leads are avialable: I, II & V2 or
%    - If 2-leads are avialable: I & II, etc.

% Inputs:
% 1. Header files and the number of leads
% 2. num_leads: number of leads
%
% Outputs:
% leads: available leads names
% leads_idx: The index of available leads
%
% Author: Nadi Sadr, PhD, <nadi.sadr@dbmi.emory.edu>
% Version 1.0
% Date 9-Dec-2020
% Version 2.0 26-Jan-2021
% Version 2.1 26-April-2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define lead sets (e.g 12, 6, 4, 3 and 2 lead ECG sets)
twelve_leads = [{'I'}, {'II'}, {'III'}, {'aVR'}, {'aVL'}, {'aVF'}, {'V1'}, {'V2'}, {'V3'}, {'V4'}, {'V5'}, {'V6'}];
six_leads    = [{'I'}, {'II'}, {'III'}, {'aVR'}, {'aVL'}, {'aVF'}];
four_leads   = [{'I'}, {'II'}, {'III'}, {'V2'}];
three_leads  = [{'I'}, {'II'}, {'V2'}];
two_leads    = [{'I'}, {'II'}];
one_leads    = [{'I'}];
lead_sets = {twelve_leads, six_leads, four_leads, three_leads, two_leads, one_leads};

leads = {};
jj =1;
tmp_hea = strsplit(header_data{1},' ');
Max_leads = str2num(tmp_hea{2});    % number of leads

% Extract 12 leads
if num_leads==12
    for ii=1:Max_leads
        tmp_hea   = strsplit(header_data{ii+1},' ');
        leads{ii} = tmp_hea{9};
        leads_idx{ii} = ii;
    end
    %Sorting the leads names as defined above on lead_sets
    [~, sorted_leads_idx] = ismember(twelve_leads,leads);
    sorted_leads = leads(sorted_leads_idx);

% Extract 6 leads
elseif num_leads==6
    for ii=1:Max_leads
        tmp_hea   = strsplit(header_data{ii+1},' ');
        Lead_name{ii} = tmp_hea{9};
        switch Lead_name{ii}
            case 'I'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'II'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'III'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'aVR'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'aVL'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'aVF'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
        end
        if length(leads)==num_leads
            break
        end
    end
    %Sorting the leads names as defined above on lead_sets
    [~, sorted_leads_idx] = ismember(six_leads,leads);
    sorted_leads = leads(sorted_leads_idx);

% Extract 4 leads
elseif num_leads==4
    for ii=1:Max_leads
        tmp_hea   = strsplit(header_data{ii+1},' ');
        Lead_name{ii} = tmp_hea{9};
        switch Lead_name{ii}
            case 'I'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'II'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'III'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'V2'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
        end
        if length(leads)==num_leads
            break
        end
    end
    %Sorting the leads names as defined above on lead_sets
    [~, sorted_leads_idx] = ismember(four_leads,leads);
    sorted_leads = leads(sorted_leads_idx);

% Extract 3 leads
elseif num_leads==3
    for ii=1:Max_leads
        tmp_hea   = strsplit(header_data{ii+1},' ');
        Lead_name{ii} = tmp_hea{9};
        switch Lead_name{ii}
            case 'I'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'II'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'V2'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
        end
        if length(leads)==num_leads
            break
        end
    end
    %Sorting the leads names as defined above on lead_sets
    [~, sorted_leads_idx] = ismember(three_leads,leads);
    sorted_leads = leads(sorted_leads_idx);

% Extract 2 leads
elseif num_leads==2
    for ii=1:Max_leads
        tmp_hea   = strsplit(header_data{ii+1},' ');
        Lead_name{ii} = tmp_hea{9};
        switch Lead_name{ii}
            case 'I'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
            case 'II'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
        end
        if length(leads)==num_leads
            break
        end
    end
    %Sorting the leads names as defined above on lead_sets
    [~, sorted_leads_idx] = ismember(two_leads,leads);
    sorted_leads = leads(sorted_leads_idx);
    
% Extract 1 leads
elseif num_leads==1
    for ii=1:Max_leads
        tmp_hea   = strsplit(header_data{ii+1},' ');
        Lead_name{ii} = tmp_hea{9};
        switch Lead_name{ii}
            case 'I'
                leads_idx{jj} = ii;
                leads{jj}  = Lead_name{ii};
                jj = jj+1;
        end
        if length(leads)==num_leads
            break
        end
    end
    %Sorting the leads names as defined above on lead_sets
    [~, sorted_leads_idx] = ismember(one_leads, leads);
    sorted_leads = leads(sorted_leads_idx);

end

end
