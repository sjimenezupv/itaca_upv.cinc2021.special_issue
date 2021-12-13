function [ t_mask, t_indices, t_values, diff_qt_ms ] = getTWave( ecg, Fs, msROffset, msWindow, plot_flag, qrs )
%GETTWAVE Devuelve una máscara indicando dónde se encuentran las ondas P,
%así como sus índices y valores.
%   INPUTS
%         ecg: Senyal ECG
%          Fs: Frecuencia de muestreo
%   msROffset: Opcional. Offset respecto al pico R donde se posiciona la ventana para buscar la onda T, en ms.
%    msWindow: Opcional. Tamanyo de la ventana para buscar la onda T, en ms.
%   plot_flag: Opcional. (Default=false) Flag indicando si plotear los  resultados
%         qrs: Opcional. Máscara de ondas R, si queremos tener una como referencia. Si no se indica, se vuelve a calcular con getQRS.

    if nargin < 3; msROffset = 100;    end;
    if nargin < 4; msWindow  = 300;    end;
    if nargin < 5; plot_flag = false;  end;
    
    [n, m] = size(ecg);
    if m > n
        ecg = ecg';
        fprintf('xxx \n')
    end
    
    if nargin < 6
        [qrs, r_indices] = getQRS(ecg, Fs, plot_flag, false);
    else
        r_indices = find(qrs==1);
    end
    
    t_mask = zeros(size(qrs));
    
    % Iniciamos las variables locales
    N            = length(r_indices);           % Número de ondas R
    NECG         = length(ecg);                 % Longitud de la senyal ECG
    offsetR      = round(Fs * (msROffset/1000)); % Número de muestras a cada lado de la onda R    
    offsetWin    = round(Fs * (msWindow/1000)); % Número de muestras a cada lado de la onda R
    
    for i = 1 : N
        
        r_index = r_indices(i);
        index1  = r_index + offsetR;
        index2  =  index1 + offsetWin - 1;
        
        if index1 > 0 && index2 <= NECG
            
            %senyal = ecg(index1 : index2);
            
            [ma, mai] = max(ecg(index1 : index2));
            [mi, mii] = min(ecg(index1 : index2));
            
        
            if(abs(ma) > abs(mi))
                t_mask(r_index + offsetR + mai - 1) = 1;
            else
                t_mask(r_index + offsetR + mii - 1) = 1;
            end
        end
    end
    
    
    t_indices = find(t_mask==1);
    t_values  = ecg(t_indices);
    
    
    %%  QT interval in ms
    r_indices  = r_indices(1:length(t_indices));
    diff_qt_ms = (t_indices - r_indices) ./ Fs;

    %% Plot ??
    if plot_flag == true
        plotQRS(ecg, qrs, Fs);
        
        max_ecg = max(1, max(ecg));
        subplot(3, 1, 1);
        hold on
        T = getTimeVector(Fs, length(ecg));
        stem(T(t_mask==1), t_mask(t_mask==1) .* max_ecg, 'g-');
        hold off;
    end
end

