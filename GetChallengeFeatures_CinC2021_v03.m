%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GetChallengeFeatures_CinC2021_v01
%%% Extract ECG Features from the specified input signal
%%%
%%% Inputs:
%%% Y  - The input signal
%%% Fs - Frequency sampling
%%%
%%% Outputs:
%%% features - Array of signal features
%%%
%%% Author:  Santiago Jiménez-Serrano [sanjiser@upv.es]
%%% Version: 3.0
%%% Date:    2021-11-27
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [features] = GetChallengeFeatures_CinC2021_v03(Y, Fs)
%GETCHALLENGEFEATURES_CINC2021_V03 Extract ECG Features from the specified input signal

    % 1st stage - Signal Filtering
    [Y, Fs] = SignalFiltering_CinC2021_v03(Y', Fs);

    % Get qrs index 
    [ ~, qrs_indices, qrs_values ] = getQRS( Y, Fs );
    QRSms = (qrs_indices / Fs)*1000; % QRS waves occurrences in ms
    
    %%% T indexes
    msROffset = 100;
    msWindow  = 300;
    [ t_indices, t_values, diff_qt_ms ] = getTWave_mini( Y, Fs, qrs_indices, msROffset, msWindow );
    
    %%% QRS Pattern
    msOffset  = 100;    
    [ patron_r, per_descartes_r, rmse_r, amp_r ] = getQRS_Pattern_mini( Y, Fs, qrs_indices, msOffset );
    
    %%% T Pattern
    msOffset  = 100;    
    [ patron_t, per_descartes_t, rmse_t, amp_t ] = getQRS_Pattern_mini( Y, Fs, t_indices, msOffset);
    
 
    
    % 
    fvolt_r = getMuSg(qrs_values);   % r voltages       stats
    fvolt_t = getMuSg(t_values);     % t voltages       stats
    fqt     = getMuSg(diff_qt_ms);   % qt interval (ms) stats
    
    
    fqt_volt    = [0, 0, 0, 0, 0, 0, 0];
    fqt_volt(1) = amp_t ./ amp_r;
    fqt_volt(2) = sign(fvolt_t(1)); %[0, 2] (+1)
    fqt_volt(3) = sign(fvolt_r(1)); %[0, 2] (+1)
    fqt_volt(4) = max(abs(diff(patron_r)));
    fqt_volt(5) = max(abs(diff(patron_t)));
    fqt_volt(6) = max(abs(diff(patron_r, 2)));
    fqt_volt(7) = max(abs(diff(patron_t, 2)));

    pattern_features = [per_descartes_r, rmse_r, per_descartes_t, rmse_t ];
    
    [fspect] = getSpectralF(Y, Fs);
    
    
    rr   = diff(QRSms);          % RR in ms
    rr   = Filter_Outliers(rr);  % Best results filtering outliers
    rrd1 = diff(rr);
    rrd2 = diff(rrd1);    
    


     % Todas estas son las óptimas
     f0    = getMuSgKuSk(rr);             % All measures give us relevant information
     f1    = getMuSgKu(rrd1);             % Skewness do not give relevant information
     f2    = getSgKu(rrd2);               % Skewness do not give relevant information
     
     
     fcx   = getRRComplexFeatures(rr);                        %  2 features [49-50]
     frr   = [f0, f1, f2];                                    %  9 features [51-59]
     pf    = getPoincareFeatures(rr);                         %  4 features [60-63]
     lf    = getLorenzFeatures(rr);                           %  8 features [64-71]
     
     
     [pcaf] = getFeaturesPCA(rrd1);
     
     % QtC - Qt Corrected 
     % http://ve.scielo.org/scielo.php?script=sci_arttext&pid=S0367-47622008000300006
     % 1. Bazett:     QTc = QT / (RR)^0,5
     % 2. Fridericia: QTc = QT / (RR)0,33
     % 3. Framingham: QTc = QT + 0.154(1−RR)
     qtC1 = fqt(1) / sqrt(f0(1));           % Bazzet 
     qtC2 = fqt(1) / (f0(1) * 0.33333);     % Fridericia
     qtC3 = fqt(1) + (0.154 * (1 - f0(1))); % Framingham
     qtC  = [qtC1, qtC2, qtC3];
     
     rs_der = getRS_DerivativeFeatures(patron_r);
     t_der  = getRS_DerivativeFeatures(patron_t);
     
 
     
     % Final features vector
     features  = [fspect, fvolt_r, fvolt_t, fqt, fqt_volt, pattern_features, fcx, frr, pf, lf, pcaf, qtC, rs_der, t_der];
     

end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic Stats Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [s] = getMu(v)
    if isempty(v), s = 0;
    else,          s = nanmean(v);
    end
end

function [s] = getMuSg(v)
    if isempty(v), s = [0, 0];
    else,          s = [nanmean(v), nanstd(v)];
    end
end

function [s] = getSgKu(v)
    if isempty(v), s = [0, 0];
    else,          s = [nanstd(v), kurtosis(v)];
    end
end

function [s] = getMuSgKu(v)
    if isempty(v), s = [0, 0, 0];
    else,          s = [nanmean(v), nanstd(v), kurtosis(v)];
    end
end

function [s] = getMuSgKuSk(v)
   
    if isempty(v)
        s = [NaN, NaN, NaN, NaN];
    else        
        s = [nanmean(v), nanstd(v), kurtosis(v), skewness(v)];
    end
end

function [ x, filter, nout ] = Filter_Outliers( x )

    % Umbrales para outliers
    m = median(x);
    s = std(x);
    
    th_up = m + (3.0 * s);
    th_dn = m - (3.0 * s);
    
    filter    = x<th_dn | x>th_up;
    nout      = sum(filter);
    x(filter) = [];

end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RR Features functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [results] = getRRComplexFeatures(rr)
    
    % Antes Features 1
    entRR=shannonEntropy(rr);
    
    % Antes Feature 2 (lzcrr)
    aux = rr>median(rr);
    
    if length(aux) < 3
        lzcrr = entRR;
    else
        lzcrr=calc_lz_complexity(aux, 'exhaustive', 1);    
    end 

    results = [entRR, lzcrr];    

end


function [ values ] = getPoincareFeatures( rr ) %% rr must be in ms

    if isempty(rr)
        values = ones(1, 4);
        return;
    end

    rrd1 = diff(rr);
    
    % RMSSD
    RMSSD=sqrt(mean(rrd1.^2));
    
    % PNN50
    n    = length(rrd1);
    NN25 = sum(rrd1>25);
    NN50 = sum(rrd1>50); % > 50 ms
    NN75 = sum(rrd1>75);
    pNN25=NN25/n;
    pNN50=NN50/n;
    pNN75=NN75/n;    

    values = [RMSSD, pNN25, pNN50, pNN75];
    
end

function [results] = getLorenzFeatures(rr)
    
    if isempty(rr) || length(rr) < 3
        results = zeros(1, 8);
        return;
    end

    n     = length(rr);
    x     = rr(1:end-1);
    y     = rr(2:end);
    theta = atand(x ./ y);
    li    = sqrt(x.^2 + y.^2); % Sum of distances to the origin (sdo)
    L     = mean(li);
    VAI   = sum(abs(theta-45))./(n-1);
    VLI   = sqrt(sum((li-L).^2))./(n-1);
    
    rrd1 = diff(rr);
    x    = rrd1(1:end-1);
    y    = rrd1(2:end);
    
    % Sum of distances to the origin
    sdo = sqrt(x.^2 + y.^2); 
   
    % Sum of distnaces for consecutive points
    sdp = sqrt((x(1:end-1)-x(2:end)).^2 + (y(1:end-1)-y(2:end)).^2);
    
    % Difference between distances of tree at three points - w/window=1
    dife = sqrt((sdp(1:end-1) - sdp(2:end)).^2);    
    
    results = [VAI, VLI, getMuSg(sdo), getMuSg(sdp), getMuSg(dife)];
end


% Poincaré features
function [f] = getFeaturesPCA(rrd1)

   
    dmax = -Inf;
    dmin = +Inf;
    dsum = 0;
    n    = length(rrd1)-1;
    dd = [];
    
    for i = 1: n
        for j = i+1 : n
             if i ~= j
                p1 =[rrd1(i), rrd1(i+1)];
                p2 =[rrd1(j), rrd1(j+1)];

                dx = p1-p2;
                d = sqrt(sum((dx.*dx)));

                if(d > dmax), dmax = d; end
                if(d < dmin), dmin = d; end
                dsum = dsum + d;
                dd = [dd, d];
             end
        end
    end
    
    f = NaN(1, 7);
    
    if n > 1
    
        f(1) = dmax;
        f(2) = dmin;
        f(3) = abs(dmin-dmax);
        f(4) = mean(dd);
        f(5) = std(dd);
        f(6) = kurtosis(dd);
        f(7) = skewness(dd);    
    end
    
end




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spectral Features functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [r] = getSpectralF(Y, Fs)

    %[Pxx, F, T, FDom, PercentAreaFDom] = AnalisisEspectralPWelch(Y, Fs, true);
    
    [~, ~, FDom, PercentAreaFDom, Accum] = AnalisisEspectralPWelch_mini(Y, Fs);
%     FDom
%     PercentAreaFDom
%     Accum
%     size(Accum)
%     plot(Accum);
%     
%     pause;

    r = [FDom, PercentAreaFDom, Accum];

end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% R-S features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [r] = getRS_DerivativeFeatures(qrsPattern)

    
    try    
        [mx, imax] = max(qrsPattern);
        [mn, imin] = min(qrsPattern);

        % Avoid more than one minimun or maximun  
        mx   = mx(1);
        mn   = mn(1);
        imax = imax(1);
        imin = imin(1);

        % Define the pattern segments    
        if imax < imin
            s1 = qrsPattern(1:imax-1);
            s2 = qrsPattern(imax:imin);
            s3 = qrsPattern(imin+1:end);
        else
            s1 = qrsPattern(1:imin-1);
            s2 = qrsPattern(imin:imax);
            s3 = qrsPattern(imax+1:end);
        end   


        % Return the maximun and minimun derivative for each segment of signal
        d1 = diff(s1);
        d2 = diff(s2);
        d3 = diff(s3);

        % Maximun derivative
        mx1 = max(d1);
        mx2 = max(d2);
        mx3 = max(d3);

        % Minimun derivative
        mn1 = max(d1);
        mn2 = max(d2);
        mn3 = max(d3);

        % Avoid more than one maximun/minimun
        mx1 = mx1(1);
        mx2 = mx2(1);
        mx3 = mx3(1);
        mn1 = mn1(1);
        mn2 = mn2(1);
        mn3 = mn3(1);
        
        % R or S Dominant
        if abs(mx) > abs(mn)
            rs_dominant = 1;
        else
            rs_dominant = -1;
        end
        
        % sign+1 => [0, 2]
        
        % Features 
        %  r is dominant?
        %  maximun derivative values for (1st, 2nd, 3rd) stages 
        %  minimun derivative values for (1st, 2nd, 3rd) stages

        r = [rs_dominant, mx1, mx2, mx3, mn1, mn2, mn3];
    
    catch ME
        r = [0, 0, 0, 0, 0, 0, 0];
    end

end
