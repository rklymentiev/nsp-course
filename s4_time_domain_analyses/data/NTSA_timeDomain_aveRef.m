%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-domain analyses
%      VIDEO: Compute average reference
% Instructor: sincxpress.com
%
%%

load sampleEEGdata.mat

% initialize new data matrices
[EEG.cardata1,EEG.cardata2] = deal( zeros(size(EEG.data)) );


%% via a double-loop

for triali=1:EEG.trials
    for ti=1:EEG.pnts
        
        % new channel vector is itself minus average over channels
        EEG.cardata1(:,ti,triali) = EEG.data(:,ti,triali) - mean(EEG.data(:,ti,triali));
    end
end

%% via bsxfun

EEG.cardata2 = bsxfun(@minus,EEG.data,mean(EEG.data,1));

%% compare them 

chan2plot = 'poz';


% convert channel label to index
chanidx = strcmpi({EEG.chanlocs.labels},chan2plot);

figure(1), clf, hold on
h(1) = plot(EEG.times,mean(EEG.data(chanidx,:,:),3),'r');
h(2) = plot(EEG.times,mean(EEG.cardata1(chanidx,:,:),3),'k');
h(3) = plot(EEG.times,mean(EEG.cardata2(chanidx,:,:),3),'b');

legend({'Earlobe';'Average (loop)';'Average (bsxfun)'})

% adjust line widths and markers
set(h,'linewidth',2)
set(h(2),'marker','s')

% adjust general plot settings
set(gca,'xlim',[-300 1200])
xlabel('Time (ms)'), ylabel('Voltage (\muV)')

%% done.
