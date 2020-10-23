%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-domain analyses
%      VIDEO: Lowpass filter an ERP
% Instructor: sincxpress.com
%
%%

load v1_laminar

% reduce data for convenience
data = double(squeeze( csd(7,:,:) ));


% cutoff frequency for low-pass filter
lowcut = 20; % in Hz

%% create and inspect the filter kernel

% filter order
filtord = round( 18 * (lowcut*1000/srate) );

% create filter
filtkern = fir1(filtord,lowcut/(srate/2),'low');

% inspect the filter kernel
figure(1), clf
subplot(211)
plot((0:length(filtkern)-1)/srate,filtkern,'k','linew',2)
xlabel('Time (s)')
title('Time domain')


subplot(212)
hz = linspace(0,srate,length(filtkern));
plot(hz,abs(fft(filtkern)).^2,'ks-','linew',2)
set(gca,'xlim',[0 lowcut*3])
xlabel('Frequency (Hz)'), ylabel('Gain')
title('Frequency domain')

%% option 1: filter the ERP

% extract ERP
erp1 = mean(data,2);

% apply filter
erp1 = filtfilt(filtkern,1,erp1);

%% option 2: filter the single trials

erp2 = zeros(size(timevec));

for triali=1:size(data,2)
    erp2 = erp2 + filtfilt(filtkern,1,data(:,triali));
end

% complete the averaging
erp2 = erp2/triali;

%% option 3: concatenate

% make one long trial
supertrial = reshape(data,1,[]);

% apply filter
supertrial = filtfilt(filtkern,1,supertrial);

% reshape back and take average
erp3 = reshape(supertrial,size(data));
erp3 = mean(erp3,2);

%% now compare

figure(2), clf, hold on

c = 'brk';
s = 'so^';

for i=1:3
    eval([ 'plot(timevec,erp' num2str(i) ',[c(i) s(i) ''-'' ],''linew'',2)' ])
end

xlabel('Time (s)'), ylabel('Voltage (\muV)')
legend({'filter ERP';'filter trials';'filter concat'})

% let's take a closer look
zoom on

%% done.
