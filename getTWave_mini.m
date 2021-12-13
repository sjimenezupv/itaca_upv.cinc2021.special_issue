function [ t_indices, t_values, diff_qt_ms ] = getTWave_mini( ecg, Fs, r_indices, msROffset, msWindow )
%GETTWAVE Devuelve una máscara indicando dónde se encuentran las ondas P,
%así como sus índices y valores.
%   INPUTS
%         ecg: Senyal ECG
%          Fs: Frecuencia de muestreo
%   msROffset: Opcional. Offset respecto al pico R donde se posiciona la ventana para buscar la onda T, en ms.
%    msWindow: Opcional. Tamanyo de la ventana para buscar la onda T, en ms.
%   plot_flag: Opcional. (Default=false) Flag indicando si plotear los  resultados
%         qrs: Opcional. Máscara de ondas R, si queremos tener una como referencia. Si no se indica, se vuelve a calcular con getQRS.

    if nargin < 4
        msROffset = 100;
    end
    
    if nargin < 5
        msWindow  = 300;    
    end
    
    [n, m] = size(ecg);
    if m > n
        ecg = ecg';        
    end
    
    t_indices = zeros(size(r_indices));
    
    % Iniciamos las variables locales
    N            = length(r_indices);            % Número de ondas R
    NECG         = length(ecg);                  % Longitud de la senyal ECG
    offsetR      = round(Fs * (msROffset/1000)); % Número de muestras a cada lado de la onda R    
    offsetWin    = round(Fs * (msWindow /1000)); % Número de muestras a cada lado de la onda R
    
    for i = 1 : N
        
        r_index = r_indices(i);
        index1  = r_index + offsetR;
        index2  =  index1 + offsetWin - 1;
        
        if index1 > 0 && index2 <= NECG
            
            %senyal = ecg(index1 : index2);
            
            [ma, mai] = max(ecg(index1 : index2));
            [mi, mii] = min(ecg(index1 : index2));
            
        
            if(abs(ma) > abs(mi))
                t_indices(i) = r_index + offsetR + mai - 1;
            else
                t_indices(i) = r_index + offsetR + mii - 1;
            end
        end
    end
    
    % Remove empty indices
    t_indices(t_indices == 0) = [];
    
    % Get the t values
    t_values = ecg(t_indices);
        
    % Get the QT interval in ms
    diff_qt_ms = (t_indices - r_indices(1:length(t_indices))) ./ Fs;


end

