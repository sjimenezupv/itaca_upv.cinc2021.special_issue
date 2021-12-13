function [ T ] = getTimeVector( Fs, NumSamples )
%GETTIMEVECTOR Obtiene el Vector de Tiempo acorde a los parámetros
%indicados
% 
%INPUT 
%            Fs: Frecuencia de Muestreo
%    NumSamples: Número de Muestras


    %T = (0: 1/Fs: (double(NumSamples)/Fs)-(1/Fs))';  % Index 0-based
    T = (1/Fs: 1/Fs: (double(NumSamples)/Fs))';       % Index 1-based

end

