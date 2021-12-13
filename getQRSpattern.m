%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% getQRSpattern
%%% Get the QRS pattern for the given ECG signal
%%%
%%% Inputs: ...
%%%
%%% Outputs: ...
%%%
%%% Author:  Santiago Jiménez-Serrano [sanjiser@upv.es]
%%% Version: 1.0
%%% Date:    2020-03-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [patterns,nQRS,classes,R]=getQRSpattern(sig,R,win_len_ms,Fs)

%% [patterns,nQRS]=getQRSpattern(leads,R)
% Esta función realiza una clasificación morfológica de los complejos QRS y
% obtiene un patrón promedio del ciclo completo para cada tipo de complejo 
% presente en la señal.

%% INPUTS

% sig       -->     Vector que contiene la señal de ECG.
% R         -->     Vector de marcas con las posiciones de los complejos
%                   QRS [número de muestra]
% win_len_ms -->    Duración de la ventana de los patrones en ms. 
%                   (Recomendado/por defecto 1.2*RRms).
% Fs        -->     Frecuencia de muestreo [Hz] (default 360Hz).

%% OUTPUTS

% patterns  -->    Matriz rxc que contiene los r patrones obtenidos. 
% nQRS      -->    Vector de longitud r que contiene el número de ciclos
%                  promediados para la obtención de cada patrón.

%% IMPLEMENTACIÓN
% inicialización de variables
if isempty(sig)
    error('No input signal has been found. The signal is a required input.');
end
if isempty(Fs)
    Fs=300;
end
if isempty(R)
%     [R] = qrs_detect2(sig,0.25,0.6,Fs);
    [QRSon,QRSoff,R]=QRSdelineationHilbert(sig,Fs);
    %error('Sorry, the QRS detector utility has not been implemented yet. Please provide your own anotations.')
end
RRmean=mean(R(2:end)-R(1:end-1));
if isempty(win_len_ms)
    win_len_ms=1.2*RRmean/Fs*1000;
end

m=length(sig);
classes={};
max_delay=round(15*Fs/1000); % tolerancia de 15ms entre marca de la R y pico en la función de correlación

% Clasificación y promediado

r=R;
win_len=round(win_len_ms*Fs/1000/2);
if r(1)<=win_len
    r=r(2:end);
end
if length(sig)-r(end)<=win_len
    r=r(1:end-1);
end
nClasses=1;
while (~isempty(r))
    try
    classes{nClasses}=sig(r(1)-win_len:r(1)+win_len); 
    nQRS(nClasses)=1;
    r(1)=0;
    [cfun,lag]=obtaincfun(sig,classes{nClasses}(1,:),win_len);
%     plot(sig,'k')
%     hold on 
%     plot(cfun*400,'r')
%     hold off
%     pause
%     plot(classes{nClasses}(1,:))
%     pause
    [pks,locs]=findpeaks(cfun,'MinPeakHeight',prctile(cfun,95),'MinPeakDistance',round(0.5*RRmean));
    pks=pks(2:end);
    locs=locs(2:end);
    locs=locs(pks<(mean(pks)+2*std(pks))); 
    pks=pks(pks<(mean(pks)+2*std(pks)));
    for j=1:length(locs)
        idx=findnearest(r,locs(j),max_delay);
        if idx > 0
            try
                classes{nClasses}(nQRS(nClasses)+1,:)=sig(locs(j)-win_len:locs(j)+win_len);
                nQRS(nClasses)=nQRS(nClasses)+1;
            catch
            end
            r(idx)=0;
        end
    end
%     for k=1:nQRS(nClasses)
%         plot(classes{nClasses}(k,:),'k')
%         hold on
%     end
    patterns(nClasses,:)=sum(classes{nClasses},1)/nQRS(nClasses);
%     plot(patterns(nClasses,:));
%     hold off
%     pause
    nClasses=nClasses+1;
    r=r(r>0);
    catch
        r(1)=0;
        r=r(r>0);
    end
end
nClasses=nClasses-1;
%% Refinamiento resultados

% lowPop=find(nQRS<3);
% hiPop=find(nQRS>3);
% for i=lowPop
%     for j=1:length(nQRS)
%         c=corrcoef(patterns(i,:),patterns(j,:));
%         if c(2,1)>0.8 && j~=i
%             patterns(j,:)=(patterns(j,:)*nQRS(j)+patterns(i,:)*nQRS(i))/(nQRS(i)+nQRS(j));
%             nQRS(j)=nQRS(j)+1;
%             classes{j}(nQRS(j):nQRS(j)+nQRS(i)-1,:)=classes{i};
%             nQRS(i)=0;
%             break;
%         end
%     end
% end
% patterns=patterns(nQRS>0,:);
% classes=classes(nQRS>0);
% nQRS=nQRS(nQRS>0);
% nClasses=size(patterns,1);

%%Visualización resultados
% for i=1:nClasses
%     subplot(1,nClasses,i)
%     hold on
%     for j=1:nQRS(i)
%         plot(classes{i}(j,:),'k')
%     end
%     plot(patterns(i,:),'r','LineWidth',2)
%     xlim([1 length(patterns)])
% end
% pause    
    
%% Funciones auxiliares

function [cfun,lag]=obtaincfun(sig,pattern,win_len)
[cfun,lag]=xcorr(sig,pattern);
aux= length(sig)-win_len;
lag=lag(aux:aux+length(sig)-1);
cfun=cfun(aux:aux+length(sig)-1);
cfun=cfun/max(cfun); 

function pos=findnearest(vector,idx,max_delay)
aux=abs(vector-idx);
if min(aux)>max_delay
    pos=-1;
else
    pos=find(aux==min(aux));
end