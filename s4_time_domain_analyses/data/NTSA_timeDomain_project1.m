%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-domain analyses
%      VIDEO: Project 2-1: Quantify the ERP as peak-mean or peak-to-peak
% Instructor: sincxpress.com
%
%%

load sampleEEGdata.mat

% channel to pick
chan2use = 'o1';

% time window for negative peak
% Mike's choices: 
negpeaktime = [  50 110 ];
pospeaktime = [ 110 170 ];


%%% compute ERP
erp = double( mean(EEG.data(chanidx,:,:),3) );

% plot ERP
figure(1), clf
plot(EEG.times,erp,'k','linew',1)
set(gca,'xlim',[-300 1000])

% plot patches over areas
ylim = get(gca,'ylim');
ph = patch(EEG.times(negpeaktime([1 1 2 2])),ylim([1 2 2 1]),'y');
set(ph,'facealpha',.8,'edgecolor','none')



% move the patches to the background
set(gca,'Children',flipud( get(gca,'Children') ))

%% first low-pass filter (windowed sinc function)

lowcut = 15;
filttime = -.3:1/EEG.srate:.3;
filtkern = sin(2*pi*lowcut*filttime) ./ filttime;

% adjust NaN and normalize filter to unit-gain
filtkern(~isfinite(filtkern)) = max(filtkern);
filtkern = filtkern./sum(filtkern);

% windowed sinc filter
filtkern = filtkern .* hann(length(filttime))';



% inspect the filter kernel
figure(2), clf
subplot(211)
plot(,,'k','linew',2)
xlabel('Time (s)')
title('Time domain')


subplot(212)
hz = linspace(0,EEG.srate,length(filtkern));
plot(,,'ks-','linew',2)
set(gca,'xlim',[0 lowcut*3])
xlabel('Frequency (Hz)'), ylabel('Gain')
title('Frequency domain')

%% now filter the ERP and replot

% apply filter


% plot on top of unfiltered ERP


%% peak-to-peak voltages and timings

%%%% first for unfiltered ERP

% find minimum/maximum peak values and peak times


% ERP timings


% get results (peak-to-peak voltage and latency)



%%%% then for low-pass filtered ERP

% find minimum/maximum peak values and peak times


% ERP timings


% get results (peak-to-peak voltage and latency)


%% Report the results in the command window

% clear the screen
clc

fprintf('\nRESULTS FOR PEAK POINT:')
fprintf('\n   Peak-to-peak on unfiltered ERP: %5.4g muV, %4.3g ms span.',erpP2P,erpP2Plat)
fprintf('\n   Peak-to-peak on filtered ERP:   %5.4g muV, %4.3g ms span.\n\n',erpFP2P,erpFP2Plat)

%%

%% repeat for mean around the peak

% time window for averaging (one-sided!!)
win = 10; % in ms
% now convert to indices


%%%% first for unfiltered ERP

% find minimum/maximum peak times


% adjust ERP timings


% now find average values around the peak time



% ERP timings


% get results (peak-to-peak voltage and latency)



%%%% then for low-pass filtered ERP

% find minimum/maximum peak values and peak times


% adjust ERP timings


% now find average values around the peak time


% adjust ERP timings


% get results (peak-to-peak voltage and latency)


%% Report the results in the command window

fprintf('\nRESULTS FOR WINDOW AROUND PEAK:')
fprintf('\n   Peak-to-peak on unfiltered ERP: %5.4g muV, %4.3g ms span.',erpP2P,erpP2Plat)
fprintf('\n   Peak-to-peak on filtered ERP:   %5.4g muV, %4.3g ms span.\n\n',erpFP2P,erpFP2Plat)

%% done.
