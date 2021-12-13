function [ patron, qrs, qrs_indices, qrs_values, duraciones, per_descartes, rmse ] = getQRS_Pattern3( ecg, Fs, msOffset, plot_flag, qrs )
%GETQRS_PATTERN Obtiene un patrón del intervalo R-R

    if nargin < 3; msOffset  = 100;   end
    if nargin < 4; plot_flag = false; end
    
    [n, m] = size(ecg);
    if m > n
        ecg = ecg';
    end
    
    if nargin < 5
        [qrs, qrs_indices, qrs_values] = getQRS(ecg, Fs, plot_flag, false);
    else
        qrs_indices = find(qrs==1);
        qrs_values  =  ecg(qrs==1);
    end
    
    % Iniciamos las variables locales
    N          = length(qrs_indices);         % Número de ondas R
    NECG       = length(ecg);                 % Longitud de la senyal ECG
    duraciones = diff(qrs_indices);           % Intervalos R-R en número de muestras    
    offset     = round(Fs * (msOffset/1000)); % Número de muestras a cada lado de la onda R
    win        = offset * 2;                  % Tamanyo de la ventana del patron    
    patron     = zeros(win, 1);               % Patrón que devolveremos
    totqrs     = 0;                           % Número de ondas R válidas que entran dentro del patrón
    
    
    if plot_flag == true
        T = getTimeVector(Fs, win);
        figure;
        subplot(1, 2, 1);
        hold on;
    end

    
    for i = 1 : N
        
        index1 = qrs_indices(i) - offset;
        index2 = qrs_indices(i) + offset - 1;
        
        if index1 > 0 && index2 <= NECG
            %senyal = ecg(index1 : index2);
            %patron = patron + senyal;            
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
            c      = corr2(ecg(index1 : index2), senyal);
            
            if c >= umbral_c
                patron2 = patron2 + ecg(index1 : index2);
                totqrs = totqrs + 1;

                if plot_flag == true                    
                    plot(T, senyal, 'b-');
                end
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
                %senyal = ecg(index1 : index2);
                c      = corr2(patron, ecg(index1 : index2));

                if c >= umbral_c                
                    rmse = rmse + ((ecg(index1 : index2) - patron).^2);
                end        
            end        
        end

        per_descartes= 1 - (totqrs/N);
        rmse         = sqrt(mean(rmse));  
        
    else
        per_descartes = 1.0;
        rmse          = 10;
    end
    

    
    % Gráficos
    if plot_flag == true       
        plot(T, patron, 'r-');    
        title('Patrón');
        grid;
        hold off;
        
        if ~isempty(duraciones)
            subplot(2, 2, 2);
            boxplot(duraciones./Fs);
            title('Intervalos R-R');
            ylabel('Tiempo R-R [s.]');
            grid;
        end
        
        subplot(2, 2, 4);
        hold on
        plot(T(1:end-1), diff(patron), 'r-');
        plot(T(1:end-2), diff(diff(patron)), 'b-');
        title('Derivada del Patrón');
        ylabel('Derivada');
        legend('Derivada 1', 'Derivada 2');
        grid;
        hold off
        
    end

end

