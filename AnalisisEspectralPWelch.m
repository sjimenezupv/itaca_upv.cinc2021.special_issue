function [Pxx, F, T, FDom, PercentAreaFDom] = AnalisisEspectralPWelch(x, Fs, plot_flag, WinTitle)
% ANALISISESPECTRALPWELCH Realiza una análisis espectral con pwelch
%
% INPUTS
%           x: Input signal, x
%          Fs: Frecuencia de muestreo
%   plot_flag: Indica si crear o no un plot
%    WinTitle: Título del plot en caso que plot_flag == true
%
% OUTPUTS
%             Pxx: The power spectral density (PSD) estimate, pxx, of the input signal, x
%               F: Vector de Frecuencias
%               t: Vector de Tiempo de la senyal x.
%            FDom: Frecuencia Dominante (Frecuencia cuya Pxx es mayor)
% PercentAreaFDom: Porcentaje que presenta el área de la Frecuencia Dominante (+-0.5Hz)
    
    
    T = getTimeVector(Fs, length(x));
    
    nwindow  = 1024;
    window   = hanning(nwindow);
    noverlap = []; %int32(window/2)
    [Pxx,F] = pwelch(x, nwindow, nwindow/2, 8192, Fs);
    
    [~, maxPI] = max(Pxx);
    FDom = F(maxPI(1));
    
    TotalArea = sum(Pxx(F <= 50));
    AreaPico  = sum(Pxx(F <= 50 & F <= FDom + 0.5 & F >= FDom - 0.5));
    PercentAreaFDom = AreaPico / TotalArea;
    

    
    
    
    if nargin < 3
        plot_flag = false;
    end
    
    if plot_flag == true 
        
        G = figure;
        subplot(2, 1, 1);
        plot(T, x);
        xlabel('Time [s]');
        ylabel('ECG');
        title('Gráfico en función del tiempo');
        grid;

        % Análisis Espectral
        subplot(2, 1, 2);    
        plot(F, Pxx./max(Pxx));
        xlabel('Frecuencia [Hz]');
        ylabel('Magnitud');
        title('Análisis Espectral [Dominio de la Frecuencia]');
        grid;    
        
        if nargin == 4            
            set(G, 'Name', WinTitle, 'NumberTitle', 'off'); 
        end
    end
    
end