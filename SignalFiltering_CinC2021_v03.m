%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SignalFiltering_CinC2021_v01
%%% Performs the signal filtering for a given ecg signal
%%%
%%% Inputs:
%%% Y  - The input signal
%%% Fs - Frequency sampling
%%%
%%% Outputs:
%%% Y  - The output filtered signal
%%% Fs - Frequency output sampling (always 500Hz)
%%%
%%% Author:  Santiago Jiménez-Serrano [sanjiser@upv.es]
%%% Version: 2.0
%%% Date:    2021-11-27
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Y, Fs] = SignalFiltering_CinC2021_v03(Y, Fs)
%SIGNALFILTERING_CINC2021_V03 Performs the signal filtering for a given ecg signal


    % Debug?
    %subplot(2, 1, 1);
    % plot(Y);

    % Resampling to 500Hz ??
    if Fs ~= 500        
        Y = Signal_Resampling( Y, Fs, 500 );
        Fs = 500;
    end
    
    % Maximun, 30 seconds of signal to be analyzed/filtered
    max_samples = int32(30 * Fs);
    [nsamples, ~] = size(Y);
    if nsamples > max_samples
        Y(max_samples:end, :) = [];
    end

    
    % Filter in the 50Hz + [0.5, 40]Hz band
    Fc1 = 0.5;
    Fc2 = 40.0;
    Y = FiltraNotch(Y, Fs, 50); % Remove 50Hz freq with notch
    Y = FiltraPBanda(Y, Fs, Fc1, Fc2);
    %Y = FiltraPB(Y, Fs, Fc2);

    
    % Remove First and Last signal second
    iFs = int32(Fs);
    Y(1:iFs, :)       = [];
    Y(end-iFs:end, :) = [];
    
    
    % Artifacts filtering
    Y = Filter_Artifacts( Y, Fs );
    
    
    % Maximun, 15 seconds of signal
    max_samples = int32(15 * Fs);
    [nsamples, ~] = size(Y);
    if nsamples > max_samples
        Y(max_samples:end, :) = [];
    end
    
        
    % See:
    %  E:\matlab_workspace\sanjiser_toolbox\scripts\MCA\Filtra_CASEIB.m
    
    % Debug?
    %subplot(2, 1, 2);
    %plot(Y);
    %pause;
    
end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Artifacts signal filtering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ ecg_ok ] = Filter_Artifacts( ecg, Fs )
%FILTRAARTEFACTOS Filtra los artefactos que puede tener una senyal ECG.
%   De momento simplemente filtra los outliers que tiene la senyal.
%   Esta función se llama desde el script Filtra_LineaBase. Para un uso
%   óptimo, mejor llamar a la función Filtra_LineaBase.

% INPUTS
%         ecg: Senyal a filtrar sus artefactos. Vector o Matriz de senyales
%          Fs: Frecuencia de muestreo
    
    ecg_ok = ecg;
    
    % Filtramos outliers segundo a segundo, tanto en mínimos como en máximos
    N_MUESTRAS       = length(ecg);
    winStep          = int32(Fs/2);
    iIni             = int32(1);
    iFin             = winStep;
    outliers_indexes = [];
    
    % Por si la senyal es menor que la ventana
    if iFin > N_MUESTRAS
        iFin = N_MUESTRAS;
    end
    
    mins = [];
    maxs = [];
    
    % Obtenemos los máximos y los mínimos según una ventana de tiempos
    while iFin <= N_MUESTRAS
        
        if iFin + winStep > N_MUESTRAS
            iFin = N_MUESTRAS;
        end
        
        senyal = ecg(iIni: iFin);
        
        mins = [mins; min(senyal)];
        maxs = [maxs; max(senyal)];
        
        iIni = iIni + winStep;
        iFin = iFin + winStep;
    end
    
    % Obtenemos los umbrales
    median_max  = median(maxs);
    median_min  = median(mins);
    std_max     = std(maxs);    
    std_min     = std(mins);
    
    % Necesitamos saber si el R es negativo o positivo
    if abs(median_max) > abs(median_min)
        coef_up = 4;
        coef_dn = 5;
    else
        coef_up = 5;
        coef_dn = 4;
    end
    
    
    % 1 - Detectamos los outliers
    umbral_up = median_max + (std_max * coef_up);
    umbral_dn = median_min - (std_min * coef_dn);
    
    ind_up = find(ecg > umbral_up);
    ind_dn = find(ecg < umbral_dn);
    outliers_indexes = sort([ind_up; ind_dn]);
    
    
    
    % 2 - Ponemos a zero los tramos donde los outliers están juntos    
    nOutliers = length(outliers_indexes);
    
    % Ponemos a cero los outliers
    if nOutliers >= 1
        ecg_ok(outliers_indexes) = 0;    
    end
    
    for i = 1 : nOutliers
        index = outliers_indexes(i);
        value = ecg(index);
        
        signo = sign(value);
        j = index-1;
        while j >= 1 && sign(ecg(j)) == signo
            ecg_ok(j) = 0;
            j = j - 1;
        end        
        
        j = index+1;
        while j <= N_MUESTRAS && sign(ecg(j)) == signo
            ecg_ok(j) = 0;
            j = j + 1;
        end
    end
    
    % Ponemos a cero los tramos de senyal que tengan outliers muy cercanos
    if nOutliers > 1
        
        min_win = int32(Fs/2);
        for i = 1 : nOutliers-1
            iIni = outliers_indexes(i);
            iFin = outliers_indexes(i+1);
            
            % Si están muy cerca, ese tramo lo dejamos a cero
            if iFin-iIni <= min_win
                ecg_ok(iIni:iFin) = 0;
            end
        end
        
    end
    
end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filtering Signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ y ] = FiltraPA( x, Fs, Fc )
%FILTRAPA Aplica a la entrada un filtro Paso Alto
%
% INPUTS
%   x: Vector o Matriz de entrada de tamaño (SAMPLES x CHANNELS)
%  Fs: Frecuencia de Muestreo
%  Fc: Frecuencia de Corte
%
% OUTPUTS
%   y: Vector o Matriz resultado de filtrar con los parámetros de entrada

    [a, b] = creaFiltroPA(Fs, Fc);    
    y  = AplicaFiltFilt(x, a, b);

end

function [ y ] = FiltraPB( x, Fs, Fc )
%FILTRAPB Aplica a la entrada un filtro Paso Bajo
%
% INPUTS
%   x: Vector o Matriz de entrada de tamaño (SAMPLES x CHANNELS)
%  Fs: Frecuencia de Muestreo
%  Fc: Frecuencia de Corte
%
% OUTPUTS
%   y: Vector o Matriz resultado de filtrar con los parámetros de entrada

    [a, b] = creaFiltroPB(Fs, Fc);    
    y  = AplicaFiltFilt(x, a, b);

end

function [ y ] = FiltraPBanda( x, Fs, Fc1, Fc2 )
%FILTRAPBANDA Aplica a la entrada un filtro Pasa Banda
%
% INPUTS
%    x: Vector o Matriz de entrada de tamaño (SAMPLES x CHANNELS)
%   Fs: Frecuencia de Muestreo
%  Fc1: Frecuencia de Corte 1
%  Fc2: Frecuencia de Corte 2
%
% OUTPUTS
%   y: Vector o Matriz resultado de filtrar con los parámetros de entrada


    [a, b] = creaFiltroPBanda(Fs, Fc1, Fc2);    
    y  = AplicaFiltFilt(x, a, b);

end

function [ y ] = FiltraNotch( x, Fs, Fc )
%FILTRANOTCH Aplica a la entrada un filtro de tipo Notch
%
% INPUTS
%   x: Vector o Matriz de entrada de tamaño (SAMPLES x CHANNELS)
%  Fs: Frecuencia de Muestreo
%  Fc: Frecuencia de Corte
%
% OUTPUTS
%   y: Vector o Matriz resultado de filtrar con los parámetros de entrada

    [a, b] = creaFiltroNotch(Fs, Fc);    
    y  = AplicaFiltFilt(x, a, b);

end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filtering Coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a, b] = creaFiltroPA(Fs, Fc)

%     rp = 3;
%     rs = 60;
    
    rp = 0.5;
    rs = 15;

    Fn = Fs/2;
    
    % Wp > Ws
    Wp = (Fc+5)/Fn;
    Ws = Fc/Fn;  % Para que atenue los 15 db dejamos un margen de 5 Hz
    
    [n, Wn] = buttord(Wp, Ws, rp, rs);
    [b, a]  = butter(n, Wn, 'high');

end

function [a, b] = creaFiltroPB(Fs, Fc)

    Rp = 0.5; % Db de atenuación en la banda que pasa
    Rs = 15;  % Db de atenuación en la stopband

    Fn = Fs/2;
    
    % Wp < Ws
    Wp = Fc/Fn;
    Ws = (Fc+5)/Fn;  % Para que atenue los 15 db dejamos un margen de 5 Hz
    
    [n, Wn] = buttord(Wp, Ws, Rp, Rs);
    [b, a] = butter(n, Wn);%, 'low');

end

function [a, b] = creaFiltroPBanda(Fs, Fc1, Fc2)


    % Design a bandpass filter with a passband from 60 to 200 Hz with
    % at  most  3 dB of passband ripple and 
    % at least 40 dB attenuation in the stopbands. 
    % Specify a sampling rate of 1 kHz. 
    % Have the stopbands be 50 Hz wide on both sides of the passband. 
    % Find the filter order and cutoff frequencies.
    
     
%     Wp = [Fc1 Fc2]/Fn;
%     Ws = [1 50]/Fn;
%     Rp = 3;
%     Rs = 60;
%     [n, Wn] = buttord(Wp,Ws,Rp,Rs);
%     
%     % Calcula el filtro
%     [b, a] = butter(n, Wn, 'bandpass')

    
    rp = 3;
    rs = 60;
    
    if Fc2-Fc1 < 5
        rs = 40;
    end

    Fn    = Fs/2;
    w1    = Fc1/Fn;
    w2    = Fc2/Fn;
     n    = buttord(w1, w2, rp, rs);
    wn    = [w1 w2];
    [b,a] = butter(n, wn, 'bandpass');

end

function [a, b] = creaFiltroNotch(Fs, Fc)

    Q  = 35;
    Fn = Fs/2;
    
    W0 = Fc/Fn;
    %bw = W0/Q; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    bw = 0.1;
    
    
    [b, a] = iirnotch(W0, bw);
    
end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filtering Applying
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ y ] = AplicaFiltFilt( x, a, b )
%APLICAFILTFILT Aplica a la entrada el filtro especificado en los coeficientes
%
% INPUTS
%    x: Vector o Matriz de entrada de tamaño (SAMPLES x CHANNELS)
%    a: Coeficientes a del filtro
%    b: Coeficientes b del filtro
%
% OUTPUTS
%   y: Vector o Matriz resultado de filtrar con los parámetros de entrada


    [ x, ~, nchannels ] = AssertMatrixSize( x );
    y = zeros(size(x));
    
    for i = 1 : nchannels
        y(:, i) = filtfilt(b, a, x(:, i));
    end

end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Signal resampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ ynew, tmnew ] = Signal_Resampling( y, oldFs, newFs )
%RESAMPLEASENYAL Resamplea una senyal a la Fs indicada
%
% INPUTS
%     y: Vector o Matriz de senyales a remuestrear
% oldFs: Frecuencia de Muestreo de la senyal o senyales
% newFs: Nueva frecuencia de muestreo a la que se quiere remuestrear
%
% OUTPUTS
%  ynew: Matriz o Vector columna con las senyales remuestreadas
%  tnew: Vector de tiempos con la nueva frecuencia de muestreo

    
    [y, ~, nchannels] = AssertMatrixSize(y);

    if oldFs ~= newFs
        for i = 1 : nchannels
            
            if i == 1 
                % Detectamos el número de muestras para el primer canal
                x = Signal_Resampling_Aux( y(:, i), oldFs, newFs);
                nsamples = length(x);
                
                % Inicializamos la matriz de salida acorde a ese número de
                % muestras
                ynew = zeros(nsamples, nchannels);
                ynew(:, 1) = x;
            else            
                ynew(:, i) = Signal_Resampling_Aux( y(:, i), oldFs, newFs);
            end
        end
    else
        ynew = y;
    end
    
    
    if nargout > 1
        [m, n] = size(ynew);
        l      = max([m, n]);
        tmnew  = 1:l / newFs;
    end
end

function [ xnew ] = Signal_Resampling_Aux( x, oldFs, newFs)

    [P, Q]  = rat(newFs/oldFs);
    xnew    = resample(x, P, Q);
    
end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Utilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ M, nsamples, nvars ] = AssertMatrixSize( M )
%ASSERTMATRIXSIZE Asegura que el tamanyo de la matriz de datos sea
% de tipo Filas=Muestras, Columnas=Variables o Canales de las senyales
%
% INPUTS
%    M: Matriz o Vector de datos
%
% OUTPUTS
%          M: Matriz o Vector de datos en formato 
%               Filas=Muestras, Columnas=Variables o Canales de las senyales
%   nsamples: Número de muestras (FILAS)
%      nvars: Número de variables o canales de las senyales (COLUMNAS)

    [nsamples, nvars] = size(M);
    if nsamples < nvars
        M = M';
        [nsamples, nvars] = size(M);
    end

end

