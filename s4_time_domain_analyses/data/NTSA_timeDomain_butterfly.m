%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-domain analyses
%      VIDEO: Butterfly plot and topo-variance time series
% Instructor: sincxpress.com
%
%%

load sampleEEGdata.mat

figure(1), clf

% make a butterfly plot
subplot(311)
plot(EEG.times,squeeze(mean(EEG.data,3)),'linew',2)
set(gca,'xlim',[-500 1300])
title('Butterfly plot')
xlabel('Time (s)'), ylabel('Voltage (\muV)')
grid on
title('Earlobe reference')


%% compute the average reference

% the fancy way...
data = bsxfun(@minus,EEG.data,mean(EEG.data,1));


subplot(312)
plot(EEG.times,squeeze(mean(data,3)),'linew',2)
set(gca,'xlim',[-500 1300])
title('Butterfly plot')
xlabel('Time (s)'), ylabel('Voltage (\muV)')
grid on
title('Average reference')


%% compute the variance time series

% variance for both earlobe and average reference
var_ts_ER = var( mean(EEG.data,3) );
var_ts_AR = var( mean(data,3) );

subplot(313), hold on
plot(EEG.times,var_ts_ER,'rs-','linew',2)
plot(EEG.times,var_ts_AR,'k-','linew',2)

set(gca,'xlim',[-500 1300])
xlabel('Time (s)'), ylabel('Voltage (\muV)')
grid on
title('Topographical variance time series')

%% done.
