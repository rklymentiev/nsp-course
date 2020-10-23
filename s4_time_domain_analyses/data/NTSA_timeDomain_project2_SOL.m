%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-domain analyses
%      VIDEO: Project 2-2: Solutions
% Instructor: sincxpress.com
%
%%

%% Loop through each channel and find the peak time of the ERP between 100 and 400 ms. 
%   Store these peak times in a separate variable, and then make a
%   topographical plot of the peak times. Repeat for a low-pass filtered ERP.

load sampleEEGdata.mat


% define time boundaries and convert to indices
timeboundaries = [ 100 400 ]; % in ms
timeidx = dsearchn(EEG.times',timeboundaries');

% trick! it doesn't need to be done in a loop
[~,maxERPtime] = max(mean(EEG.data(:,timeidx(1):timeidx(2),:),3),[],2);

% convert indices back to ms
maxERPtime = EEG.times(maxERPtime+timeidx(1)-1);

% make plot
figure(1), clf
subplot(121)
topoplotIndie(maxERPtime,EEG.chanlocs,'numcontour',4,'electrodes','numbers');
title({'ERP peak times';[' (' num2str(timeboundaries(1)) '-' num2str(timeboundaries(2)) ' ms)' ]})
set(gca,'clim',timeboundaries)
colormap hot
colorbar

%% repeat for filtered ERP

% low-pass filter
lowcut = 15;
filttime = -.3:1/EEG.srate:.3;
filtkern = sin(2*pi*lowcut*filttime) ./ filttime;

% adjust NaN and normalize filter to unit-gain
filtkern(~isfinite(filtkern)) = max(filtkern);
filtkern = filtkern./sum(filtkern);

% windowed sinc filter
filtkern = filtkern .* hann(length(filttime))';


% filter
erp = zeros(EEG.nbchan,EEG.pnts);
for chani=1:EEG.nbchan
    erp(chani,:) = filtfilt(filtkern,1,double(mean(EEG.data(chani,:,:),3)));
end

% trick! it doesn't need to be done in a loop
[~,maxERPtime] = max(abs(erp(:,timeidx(1):timeidx(2))),[],2);

% convert indices back to ms
maxERPtime = EEG.times(maxERPtime+timeidx(1)-1);

% make plot
subplot(122)
topoplotIndie(maxERPtime,EEG.chanlocs,'numcontour',4,'electrodes','numbers');
title({'Filtered ERP peak times';[ ' (' num2str(timeboundaries(1)) '-' num2str(timeboundaries(2)) ' ms)' ]})
set(gca,'clim',timeboundaries)
colormap hot
colorbar

%%
