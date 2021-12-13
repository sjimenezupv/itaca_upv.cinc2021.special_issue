function [Pxx, F, FDom, PercentAreaFDom, Accum] = AnalisisEspectralPWelch_mini(x, Fs)
% ANALISISESPECTRALPWELCH Realiza una análisis espectral con pwelch
%
% INPUTS
%           x: Input signal, x
%          Fs: Frecuencia de muestreo
%
% OUTPUTS
%             Pxx: The power spectral density (PSD) estimate, pxx, of the input signal, x
%               F: Vector de Frecuencias
%               t: Vector de Tiempo de la senyal x.
%            FDom: Frecuencia Dominante (Frecuencia cuya Pxx es mayor)
% PercentAreaFDom: Porcentaje que presenta el área de la Frecuencia Dominante (+-0.5Hz)
    
    
    [Pxx,F] = pwelch(x, 1024, 512, 8192, Fs);
    
    [~, maxPI] = max(Pxx);
    FDom = F(maxPI(1));
    
    TotalArea = sum(Pxx(F <= 50));
    AreaPico  = sum(Pxx(F <= 50 & F <= FDom + 0.5 & F >= FDom - 0.5));
    PercentAreaFDom = AreaPico / TotalArea;
    
    % Normalize to [0, 1]
    Pxx = Pxx./max(Pxx);
    
    
    Fini=0;
    Fend=2;
    
    Accum = zeros(1, 15);
    for i = 1 : 15
        Accum(i) = sum(Pxx(F >= Fini & F < Fend));
        Fini = Fini + 2;
        Fend = Fend + 2;
    end
    
end