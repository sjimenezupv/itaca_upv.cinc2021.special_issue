function [G] = plotQRS(ecg, qrs, Fs)
%PLOTQRS Plotea una senyal de ecg con la correspondiente máscara de QRS.

        T = getTimeVector(Fs, length(ecg));        
        qrs_values = ecg(qrs==1);
        qrs_index  = find(qrs==1);
        duraciones = diff(qrs_index) / Fs;
        
        G=figure;
        
        
        max_ecg = max(1, max(ecg));
        subplot(3, 1, 1);
        hold on
        stem(T(qrs==1), qrs(qrs==1) .* max_ecg, 'r-');
        plot(T, ecg, 'b-');
        title('ECG');
        xlabel('Time [s]');
        ylabel('ECG');
        grid;
        hold off;
        
        dn = 1; % Número de derivada a plotear
        subplot(3, 1, 2);
        plot(T(1:end-dn), (diff(ecg, dn)), 'r-');
        title('Derivada(ECG)');
        xlabel('Time [s]');
        ylabel('diff(ECG)');
        grid;
        
                
        subplot(3, 2, 5);
        boxplot(qrs_values);
        title('Valores R');
        xlabel('Pico R');
        ylabel('Amplitud');
        grid;

        subplot(3, 2, 6);
        boxplot(duraciones); 
        title('Intervalos R-R');
        xlabel('R-R');
        ylabel('Tiempo R-R [s.]');
        grid;
end

