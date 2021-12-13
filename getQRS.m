%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% getQRS
%%% Get a mask indicating where the R peaks in the ecg signal
%%%
%%% Inputs:
%%% ecg - The input signal
%%% Fs  - Frequency sampling
%%%
%%% Outputs:
%%% qrs         - R mask
%%% qrs_indices - R indexes
%%% qrs_values  - R values
%%%
%%% Author:  Santiago Jiménez-Serrano [sanjiser@upv.es]
%%% Version: 1.0
%%% Date:    2020-03-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ qrs, qrs_indices, qrs_values ] = getQRS(ecg, Fs)
%GETQRS Get a mask indicating where the R peaks in the ecg signal
      
    
	max_bpm    = 210; % Asumimos un ritmo máximo de 210 bpm en humanos    
    bpm_ratio  = max_bpm/60;	
    windowStep = int32(Fs/bpm_ratio); 
    window     = windowStep;
    

    % 1 - Obtenemos los mínimos/máximos
    [qrs] = getRawQRS(ecg, window, windowStep);  
    
    % 2 - Filtramos QRS demasiado cercanos    
    [qrs] = filterQRS_TooClose( ecg, qrs, windowStep );
    
    % 3 - Filtramos por umbral
    [qrs] = filterQRS_Threshold(ecg, qrs);
    
    % 4 - Filtramos hasta quedarnos sin outliers
    %%[qrs] = filterQRS_Outliers(ecg, qrs, Fs); %% este proceso está anulado en la función original
    
    % 5 - Filtramos los QRS que no cumplan las condiciones de sus derivadas
    [qrs] = filterQRS_Derivada(ecg, qrs, Fs);
    
    % 6 - Filtramos los que no cumplan con el patrón medio
    [qrs, qrs_values, qrs_indices] = filterQRS_Pattern( ecg, qrs, Fs );

end

function [ qrs ] = getRawQRS( ecg, window, windowStep )
%GETRAWQRS_BYMINS Obtiene una máscara inicial de qrs con los mínimos de la senyal ecg

    N   = length(ecg);
    qrs = zeros(N, 1);
    
    iIni = 1;
    iFin = window;
    
    if isRWavePositive( ecg, window, windowStep ) == true
        fun = @max;
    else
        fun = @min;
    end
    
    while iFin <= N
        
        % Adquirimos el trozo de senyal y lo analizamos
        [~, ind] = fun(ecg(iIni:iFin));
        
        % Ponemos la marca
        qrs(iIni+ind-1) = 1;

        % Avance hacia la condición de fin
        iIni = iIni + windowStep;
        iFin = iFin + windowStep;

        if iIni < N && iFin > N
            iFin = N;
        end
    end
    
    %% Filtramos los que no superen un umbral razonable
    qrs_values = ecg(qrs==1);
    umbral     = mean(abs(qrs_values)) / 3;    
    qrs(abs(ecg) < umbral) = 0;

end

function [ result ] = isRWavePositive( ecg, window, windowStep )
%ISRWAVEPOSITIVE Devuelve un valor indicando si la onda R es positiva o no para la senyal indicada

    N   = length(ecg);    
        
    avgMin  = 0;
    avgMax  = 0;
    cont    = 0;
    iIni    = 1;
    iFin    = window;
    
    %% Calculamos el valor medio de los Mínimos y los Máximos
    while iFin <= N
        
        % Adquirimos el trozo de senyal y lo analizamos        
        senyal = ecg(iIni:iFin);
        avgMax = avgMax + max(senyal);
        avgMin = avgMin + min(senyal);
        cont   = cont + 1;

        % Avance hacia la condición de fin
        iIni = iIni + windowStep;
        iFin = iFin + windowStep;

        if iIni < N && iFin > N
            iFin = N;
        end
    end
    
    if cont > 0
        avgMax = avgMax / cont; 
        avgMin = avgMin / cont; 
    else
        error('Senyal demasiado pequenya para ese trozo de ventana')
    end
    
    
    %% Establecemos un umbral que han de superar tanto mínimos como máximos
    umbralMax = avgMax / 3;
    umbralMin = avgMin / 3;
    
    
    %% Volvemos a hacer otra pasada, teniendo en cuenta estos umbrales
    avgMin        = 0;
    avgMax        = 0;    
    iIni          = 1;
    iFin          = window;
    
    while iFin <= N
        
        % Adquirimos el trozo de senyal y lo analizamos
        senyal = ecg(iIni:iFin);
        ma     = max(senyal);
        mi     = min(senyal);
        
        if ma >= umbralMax && abs(ma) >= abs(mi)
            avgMax = avgMax + ma;        
        elseif mi <= umbralMin && abs(mi) >= abs(ma)
            avgMin = avgMin + mi;
        else
            cont = cont - 1;
        end

        % Avance hacia la condición de fin
        iIni = iIni + windowStep;
        iFin = iFin + windowStep;

        if iIni < N && iFin > N
            iFin = N;
        end
    end
    
    % Media con los únicos que han superado el umbral
    if cont > 0
        avgMax = avgMax / cont;
        avgMin = avgMin / cont;
    else
        error('No es posible que ninguna onda R no supere alguna vez el umbral');
    end
    
    
    %% La media mayor en amplitud de los valores es la que nos indica el signo de la onda R
    %if num_positivos >= num_negativos
    if abs(avgMax) >= abs(avgMin)
        result = true;
    else
        result = false;
    end

end


function [ qrs ] = filterQRS_TooClose( ecg, qrs, windowStep )
%FILTERQRS_TOOCLOSE Filtra los QRS que están demasiado juntos


    qrs_indices = find(qrs==1);
    FIN = length(qrs_indices);    
    for i = 1 : FIN
        
        valueI = abs(ecg(qrs_indices(i)));
        j = i+1;
        while j <= FIN && qrs_indices(j) - qrs_indices(i) <= windowStep            
            
            valueJ = abs(ecg(qrs_indices(j)));
            if(valueI > valueJ)
                qrs(qrs_indices(j)) = 0;
            else
                qrs(qrs_indices(i)) = 0;
            end
            j = j+1;
        end
    end
end


function [ qrs ] = filterQRS_Threshold( ecg, qrs )
%FILTERQRS_THRESHOLD Filtra los qrs que no superen el 20% de la media

    qrs_values  = ecg(qrs==1);
    qrs_indices = find(qrs==1);
    qrs_mean    = abs(mean(qrs_values));
    umbral      = qrs_mean - (abs(qrs_mean * 0.8));
    FIN         = length(qrs_indices); 
    for i = 1 : FIN
        index = qrs_indices(i);
        v     = abs(ecg(index));
        if v < umbral
            qrs(index) = 0;
        end
    end

end


function [ qrs, qrs_values, qrs_indices ] = filterQRS_Derivada( ecg, qrs, Fs, msOffset )
%FILTERQRS_DERIVADA Filtra los qrs que no cumplan ciertos criterios respecto la derivada
% 
% La idea es quedarte sólo con los QRS que tengan un factor
% de correlación medio con todos los demás superior a un umbral
%
% INPUTS:
%         ecg: Senyal ecg de alguna de las 12 derivaciones corrientes
%         qrs: Máscara de 1s y 0s donde se indica la presencia de ondas R
%          Fs: Frecuencia de Muestreo
%    msOffset: Opcional - Default = 10 - Ventana en ms para obtener el máximo de la derivada respecto al pico R.
%    umbral_c: Opcional - Default = 0.6 - Umbral de correlación medio entre patrones.

    if nargin < 4
        msOffset  = 50;        
    end

    qrs_indices = find(qrs==1);    
    derivada    = diff(ecg);
    N           = length(qrs_indices);
    NDER        = length(derivada);
    offset      = round(Fs * (msOffset/1000)); % Número de muestras a cada lado de la onda R    
    max_DValues = nan(1, N);
    
    
    % Calculamos la máxima derivada en valor absoluto alrededor de las
    % ondas R
    for i = 1 : N
        indexI = qrs_indices(i);
        index1 = indexI - offset;
        index2 = indexI + offset - 1;
        
        if index1 <    1; index1 =    1; end;
        if index2 > NDER; index2 = NDER; end        
        
        senyal         = derivada(index1 : index2); % Cogemos el trozo de derivada
        max_DValues(i) = max(abs(senyal));          % Nos quedamos con el máximo
        
    end
    
    
    umbral = nanmean(max_DValues) / 2;
    for i = 1 : N        
        valueD = max_DValues(i);        
        % Filtramos todas las ondas R cuya derivada máxima no supere el umbral
        if ~isnan(valueD) && valueD < umbral
            indexI = qrs_indices(i);
            qrs(indexI) = 0;
        end        
    end
    
    % Comprobamos el primer y último R, ya que a veces dan un Falso Positivo residual    
    qrs_values  = ecg(qrs==1);
    qrs_indices = find(qrs==1);
    
    if length(qrs_values) > 1
        
        vFin = qrs_values(end);
        vAnt = qrs_values(end-1);
        
        % No es normal que el último valor sea menos de la mitad que el anterior
        if abs(vFin) < abs(vAnt)/2
            qrs(qrs_indices(end)) = 0;  % Marcamos como que no hay R
            qrs_indices(end)      = []; % Eliminamos el último índice y valor
            qrs_values(end)       = [];
        end        
    end
    
    if length(qrs_values) > 1
        
        v1 = qrs_values(1);
        v2 = qrs_values(2);
        
        % No es normal que el primer valor sea menos de la mitad que el siguiente
        if abs(v1) < abs(v2)/2
            qrs(qrs_indices(1)) = 0;  % Marcamos como que no hay R
            qrs_indices(1)      = []; % Eliminamos el último índice y valor
            qrs_values(1)       = [];
        end        
    end

end


function [ qrs, qrs_values, qrs_indices ] = filterQRS_Pattern( ecg, qrs, Fs, msOffset, umbral_c )
%FILTERQRS_PATTERN Filtra los qrs que no cumplan el patrón medio
% 
% La idea es quedarte sólo con los QRS que tengan un factor
% de correlación medio con todos los demás superior a un umbral
%
% INPUTS:
%         ecg: Senyal ecg de alguna de las 12 derivaciones corrientes
%         qrs: Máscara de 1s y 0s donde se indica la presencia de ondas R
%          Fs: Frecuencia de Muestreo
%    msOffset: Opcional - Default = 100  - Ventana en ms para obtener el trozo de senyal de QRS y poder compararlo con los demás.
%    umbral_c: Opcional - Default = 0.75 - Umbral de correlación medio entre patrones.

    if nargin < 4
        msOffset  = 100;        
    end
    if nargin < 5
        umbral_c = 0.6; % Umbral de correlación medio
    end

    qrs_indices = find(qrs==1);
    N           = length(qrs_indices);
    NECG        = length(ecg);
    offset      = round(Fs * (msOffset/1000)); % Número de muestras a cada lado de la onda R    
    
    
    for i = 1 : N        

        indexI = qrs_indices(i);
        index1 = indexI - offset;
        index2 = indexI + offset - 1;
        
        if index1 > 0 && index2 <= NECG
            
            % Cogemos el trozo de senyal
            senyal1 = ecg(index1 : index2);
            numcomp = 0;
            sumcorr = 0;            
            window  = 3;
            
            for j = i-window : i+window
            
                if j < 1, continue, end
                if j > N, continue, end
                
                indexJ = qrs_indices(j);
                
                % Si no es el mismo QRS
                if j~=i && qrs(indexJ) == 1                    
                    
                    index1 = indexJ - offset;
                    index2 = indexJ + offset - 1;
                    
                    if index1 > 0 && index2 <= NECG            
                        % Cogemos el trozo de senyal y correlamos
                        senyal2 = ecg(index1 : index2); 
                        c       = corr2(senyal1, senyal2);
                        if c > 0.2
                            sumcorr = sumcorr + c;
                            numcomp = numcomp + 1;
                        end
                    end
                end
            end
            
            
            if numcomp > 0
                % Calculamos el factor de correlación medio                
                avgcorr = sumcorr / numcomp;
                % Si no supera el umbral, quitamos la marca R
                if avgcorr < umbral_c
                    qrs(indexI) = 0; % Quitamos la marca R
                end
            end
        end        
    end
    
    if nargout > 1; qrs_values  = ecg(qrs==1);  end    
    if nargout > 2; qrs_indices = find(qrs==1); end

end




