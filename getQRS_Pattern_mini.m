function [ patron, per_descartes, rmse, amp ] = getQRS_Pattern_mini( ecg, Fs, qrs_indices, msOffset)
%GETQRS_PATTERN Obtiene un patrón del intervalo R-R

    if nargin < 4
        msOffset = 100;
    end
    
    [n, m] = size(ecg);
    if m > n
        ecg = ecg';
        fprintf('maaaal');
    end
    
    
    % Iniciamos las variables locales
    N          = length(qrs_indices);         % Número de ondas R
    NECG       = length(ecg);                 % Longitud de la senyal ECG
    offset     = round(Fs * (msOffset/1000)); % Número de muestras a cada lado de la onda R
    win        = offset * 2;                  % Tamanyo de la ventana del patron    
    patron     = zeros(win, 1);               % Patrón que devolveremos
    totqrs     = 0;                           % Número de ondas R válidas que entran dentro del patrón
    
    
   
    for i = 1 : N
        
        index1 = qrs_indices(i) - offset;
        index2 = qrs_indices(i) + offset - 1;
        
        if index1 > 0 && index2 <= NECG
            patron = patron + ecg(index1 : index2);
            totqrs = totqrs + 1;        
        end        
    end
    
    % Patrón medio
    if totqrs > 0
        patron = patron ./ totqrs;
    end
    
    
    
    % Repetimos el proceso, pero esta vez filtramos los QRS que no se
    % parezcan al patrón, gracias al factor de correlación
    umbral_c = 0.7; % Umbral de correlación
    patron2  = zeros(win, 1);
    totqrs   = 0;
    for i = 1 : N        

        index1 = qrs_indices(i) - offset;
        index2 = qrs_indices(i) + offset - 1;
        
        if index1 > 0 && index2 <= NECG
            
            % Cogemos el trozo de senyal
            %senyal = ecg(index1 : index2);
            c = corr2(patron, ecg(index1 : index2));
            
            if c >= umbral_c
                patron2 = patron2 + ecg(index1 : index2);
                totqrs = totqrs + 1;
            end        
        end        
    end
    
    % Patrón medio
    if totqrs > 0
        patron = patron2 ./ totqrs;

        % Error cuadrático medio
        rmse     = 0;
        for i = 1 : N        

            index1 = qrs_indices(i) - offset;
            index2 = qrs_indices(i) + offset - 1;

            if index1 > 0 && index2 <= NECG

                % Cogemos el trozo de senyal
                c = corr2(patron, ecg(index1 : index2));

                if c >= umbral_c                
                    rmse = rmse + ((ecg(index1 : index2) - patron).^2);
                end        
            end        
        end

        per_descartes = 1 - (totqrs/N);
        rmse          = sqrt(mean(rmse));
        amp           = max(patron) - min(patron);
        
    else
        per_descartes =  1.0;
        rmse          = 10.0;
        amp           =  0.0;
    end

end

