%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-domain analyses
%      VIDEO: Event-related potential (ERP)
% Instructor: sincxpress.com
%
%%

%% The theory of an ERP via simulation

% simulation details
srate   = 500;
time    = -1:1/srate:2;
ntrials = 100;
sfreq   =   9; % sine wave frequency in Hz
gpeakt  = .43;
gwidth  =  .2; % gaussian width in seconds

noiseamp = 2; % noise standard deviation



% create signal
swave  = cos( );
gausw  = exp( 4*log(2)*(time-gpeakt) / gwidth^2 );
signal = swave .* gausw;

% create data and multiple channels plus noise
data = repmat(signal,ntrials,1);
data = data + noiseamp * randn(ntrials,length(time));



figure(1), clf
subplot(511)
plot(time,signal,'k','linew',3)
set(gca,'ylim',[-1 1])
title('Pure signal')

subplot(5,1,2:4)
imagesc(data)
set(gca,'clim',[-1 1]*noiseamp*2)
ylabel('Trial')
title('All trials: signal + noise')

subplot(515)
plot(time,mean(data),'k','linew',3)
xlabel('Time (s)')
set(gca,'ylim',[-1 1])
title('Average over trials')

%% now in real data

load v1_laminar.mat
% first time seeing these data? Inspect!!
clear
whos

%% plot some time domain features

% pick a channel to plot
chan2plot = 7;



figure(2), clf
subplot(311), hold on

% plot ERP from selected channel (in one line of code!)
plot(timevec,csd(chan2plot,:,:),'b','linew',2)
set(gca,'xlim',[-.1 1.3])
title([ 'ERP from channel ' num2str(chan2plot) ])
plot([0 0],get(gca,'ylim'),'k--')
plot([0 0]+.5,get(gca,'ylim'),'k--')
plot(get(gca,'clim'),[0 0],'k--') % clim?!
ylabel('Voltage (\muV)')


% now plot all trials from this channel
subplot(3,1,2:3)
imagesc(timevec,[],squeeze(csd(chan2plot,:,:))')
set(gca,'clim',[-1 1]*1e3,'xlim',[-.1 1.3])
xlabel('Time (s)')
ylabel('Trials')
hold on
plot([0 0],get(gca,'ylim'),'k--','linew',3)
plot([0 0]+.5,get(gca,'ylim'),'k--','linew',3)


%% data for all channels

figure(3), clf

% make an image of the ERPs from all channels
contourf(timevec,1:16,squeeze(mean(csd,3)))
set(gca,'xlim',[-.1 1.3],'ydir','reverse')
title('Time-by-depth plot')
xlabel('Time (s)'), ylabel('Channel')
hold on
plot([0 0],get(gca,'ylim'),'k--','linew',3)
plot([0 0]+.5,get(gca,'ylim'),'k--','linew',3)

%% done.
