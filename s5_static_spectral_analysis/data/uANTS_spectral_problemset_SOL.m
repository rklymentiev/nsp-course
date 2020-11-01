%%
%   COURSE: Neural signal processing and analysis: Zero to hero
%  SESSION: Problem set: Spectral analyses of real and simulated data
%  TEACHER: Mike X Cohen, sincxpress.com
%


%% 1) Generate 10 seconds of data at 1 kHz, comprising 4 sine waves with different 
%    frequencies (between 1 and 30 Hz) and different amplitudes.
%    Plot the individual sine waves, each in its own plot. In separate subplots,
%    plot the summed sine waves with (1) a little bit of noise and (2) a lot of noise.
% 

srate  = 1000;
frex   = [ 3  10 15 30 ];
amplit = [ 5  15  5  7 ];
phases = [  pi/8  pi  pi/2  -pi/4 ];
time   = -1:1/srate:1;

% create sine waves
sine_waves = zeros(length(frex),length(time));

for fi=1:length(frex)
    sine_waves(fi,:) =  amplit(fi) * sin(2*pi*time*frex(fi) + phases(fi));
end

littleNoise = randn(1,length(time))*10;
lotsOfNoise = randn(1,length(time))*50;


figure(1), clf

% plot constituent sine waves (without noise)
for snum=1:4
    subplot(4,1,snum)
    plot(time,sine_waves(snum,:))
    title([ 'Sine wave component with frequency ' num2str(frex(snum)) ' Hz' ])
end
xlabel('Time (s)'), ylabel('Amplitude (arb.)')


% plot summed sine waves with little noise
figure(2), clf
subplot(211)
plot(time,sum(sine_waves,1)+littleNoise)
title('Time series with LITTLE noise')

subplot(212)
plot(time,sum(sine_waves,1)+lotsOfNoise)
title('Time series with A LOT of noise')

%% 2) Compute the power spectrum of the simulated time series (use FFT) and plot the results, 
%    separately for a little noise and a lot of noise. Show frequencies 0 to 35 Hz.
%    How well are the frequencies reconstructed, and does this depend on noise?
% 

figure(3), clf

for noisei=1:2
    
    % FFT
    if noisei==1
        f = fft(sum(sine_waves,1) + littleNoise)/length(time);
    else
        f = fft(sum(sine_waves,1) + lotsOfNoise)/length(time);
    end
    
    % compute frequencies in Hz
    hz = linspace(0,srate/2,floor(length(time)/2)+1);
    
    % plot
    subplot(2,1,noisei)
    plot(hz,2*abs(f(1:length(hz))),'k-o')
    xlabel('Frequencies (Hz)'), ylabel('Amplitude')
    set(gca,'xlim',[0 35],'ylim',[0 max(amplit)*1.2])
    
    if noisei==1
        title('FFT with LITTLE noise')
    else
        title('FFT with LOTS OF noise')
    end
end


%% 3) Compute the power spectrum of data from electrode 7 in the laminar V1 data. 
%    First, compute the power spectrum separately for each trial and then average the power 
%    results together. Next, average the trials together and then compute the power spectrum. 
%    Do the results look similar or different, and are you surprised? Why might they look 
%    similar or different?
% 

% load the LFP data
load v1_laminar.mat

% pick which channel
chan2use = 7;

% FFT of all trials individually (note that we can do it in one line)
powspectSeparate = fft(squeeze(csd(chan2use,:,:)))/length(timevec);
powspectSeparate = mean(2*abs(powspectSeparate),2); % average over trials, not over frequency!

% now FFT of all trials after averaging together
powspectAverage  = fft(squeeze(mean(csd(chan2use,:,:),3)))/length(timevec);
powspectAverage  = 2*abs(powspectAverage);

% frequencies in Hz
hz = linspace(0,srate/2,floor(length(timevec)/2)+1);


% now plot
figure(4), clf
set(gcf,'name',[ 'Results from electrode ' num2str(chan2use) ])
subplot(211)
plot(hz,powspectSeparate(1:length(hz)))
set(gca,'xlim',[0 100])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Averaging done after FFT on each trial')

subplot(212)
plot(hz,powspectAverage(1:length(hz)))
xlabel('Frequency (Hz)'), ylabel('Amplitude')
set(gca,'xlim',[0 100])
title('FFT done on trial average')

%% 4) Do the same as above but for electrode 1. Plot the results on top of the previous ones.
%    How do these results compare to the results from channel 7, and does this depend on
%    whether you average first in the time-domain or average the individual power spectra?
% ANATOMICAL NOTE: channel 7 is around L4; channel 1 is in the hippocampus.

% This code is just copied from the previous cell. I changed the channel number,
%  I set 'hold on' when calling each subplot, and I specified a different color and a legend.

chan2use = 1;

% FFT of all trials individually (note that we can do it in one line)
powspectSeparate = fft(squeeze(csd(chan2use,:,:)))/length(timevec);
powspectSeparate = mean(2*abs(powspectSeparate),2); % average over trials, not over frequency!

% now FFT of all trials after averaging together
powspectAverage  = fft(squeeze(mean(csd(chan2use,:,:),3)))/length(timevec);
powspectAverage  = 2*abs(powspectAverage);

% frequencies in Hz
hz = linspace(0,srate/2,floor(length(timevec)/2)+1);


% now plot
set(gcf,'name',[ 'Results from electrode ' num2str(chan2use) ])
subplot(211), hold on
plot(hz,powspectSeparate(1:length(hz)),'r')
set(gca,'xlim',[0 100])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Averaging done after FFT on each trial')
legend({'channel 7';'channel 1'})

subplot(212), hold on
plot(hz,powspectAverage(1:length(hz)),'r')
xlabel('Frequency (Hz)'), ylabel('Amplitude')
set(gca,'xlim',[0 100])
title('FFT done on trial average')

%% 5) Fourier transform from scratch!
% Hey, wouldn't it be fun to program the discrete-time Fourier transform
% from scratch! Yes, of course it would be. Let's do that.
% Generate a 20-element vector of random numbers.
% Use the hints below to help you write the Fourier transform.
% Next, use the fft function on the same data to verify that your FT was accurate.


N       = 20;         % length of sequence
signal  = randn(1,N); % data
fTime   = (0:N-1)/N;  % "time" used in Fourier transform

% initialize Fourier output matrix
fourierCoefs = zeros(size(signal)); 

% loop over frequencies
for fi=1:N
    
    % create sine wave for this frequency
    fourierSine = exp( -1i*2*pi*(fi-1).*fTime );
    
    % compute dot product as sum of point-wise elements
    fourierCoefs(fi) = sum( fourierSine.* signal );
end

% divide by N to scale coefficients properly
fourierCoefs = fourierCoefs / N;

figure(5), clf
subplot(211)
plot(signal)
title('Data')

subplot(212)
plot(abs(fourierCoefs)*2,'*-')
xlabel('Frequency (a.u.)')

% use the fft function on the same data
fourierCoefsF = fft(signal) / N;

% plot the results on top. Do they look similar? (Should be identical!)
hold on
plot(abs(fourierCoefsF)*2,'ro')
legend({'Manual FT';'FFT'})

%% 6) zero-padding and interpolation
% Compute the power spectrum of channel 7 from the V1 dataset. Take the
% power spectrum of each trial and then average the power spectra together.
% But don't use a loop over trials! And use only the data from 0-.5 sec. 
% What is the frequency resolution?
% Repeat this procedure, but zero-pad the data to increase frequency
% resolution. Try some different zero-padding numbers. At what multiple of the native nfft
% does the increased frequency resolution have no visible effect on the results?

tidx = [dsearchn(timevec',0) dsearchn(timevec',.5)];

% set nfft to be multiples of the length of the data
nfft = 10 * (diff(tidx)+1);

powspect = mean( abs( fft(squeeze(csd(7,tidx(1):tidx(2),:)),nfft,1)/(diff(tidx)+1) ).^2 ,2);
hz = linspace(0,srate/2,floor(nfft/2)+1);

figure(6), clf
plot(hz,powspect(1:length(hz)),'k-o','linew',3)
xlabel('Frequency (Hz)'), ylabel('Power')
set(gca,'xlim',[0 120])
title([ 'Frequency resolution is ' num2str(mean(diff(hz))) ' Hz' ])

%% 7) Poor man's filter via frequency-domain manipulations.
% The goal of this exercise is to see how a basic frequency-domain filter
%  works: Take the FFT of a signal, zero-out some Fourier coefficients,
%  take take the IFFT. 
% Here you will do this by generating a 1/f noise signal and adding 50 Hz
%  line noise.

% Generate 1/f noise with 50 Hz line noise. 
srate = 1234;
npnts = srate*3;
time  = (0:npnts-1)/srate;


% the key parameter of pink noise is the exponential decay (ed)
ed = 50; % try different values!
as = rand(1,npnts) .* exp(-(0:npnts-1)/ed);
fc = as .* exp(1i*2*pi*rand(size(as)));

signal = real(ifft(fc)) * npnts;

% now add 50 Hz line noise
signal = signal + sin(2*pi*50*time);


% compute its spectrum
hz = linspace(0,srate/2,floor(npnts/2)+1);
signalX = fft(signal);



%%% plot the signal and its power spectrum
figure(7), clf
subplot(211)
plot(time,signal,'k')
xlabel('Time (s)'), ylabel('Activity')
title('Time domain')

subplot(212)
plot(hz,2*abs(signalX(1:length(hz)))/npnts,'k')
set(gca,'xlim',[0 80])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Frequency domain')

%% zero-out the 50 Hz component

% find the index into the frequencies vector at 50 hz
hz50idx = dsearchn(hz',50);

% create a copy of the frequency vector
signalXf = signalX; % f=filter

% zero out the 50 Hz component
signalXf(hz50idx) = 0;

% take the IFFT
signalf = real(ifft(signalXf));

% take FFT of filtered signal
signalXf = fft(signalf);


% plot on top of original signal
figure(8), clf
subplot(211), hold on
plot(time,signal,'k')
plot(time,signalf,'r')
xlabel('Time (s)'), ylabel('Activity')
title('Time domain')


subplot(212), hold on
plot(hz,2*abs(signalX(1:length(hz)))/npnts,'k')
plot(hz,2*abs(signalXf(1:length(hz)))/npnts,'ro-','markerfacecolor','r')
set(gca,'xlim',[0 80])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Frequency domain')


%%% QUESTION: Why do you need to take the real part of the ifft?
% 
%%% QUESTION: Why doesn't this procedure get rid of the line noise?!?!?!?
% 


%% now fix the problem ;)

% Notice that the filter didn't work: It attenuated but did not eliminate
% the line noise. Why did this happen? Use plotting to confirm your
% hypothesis! Then fix the problem in this cell ;)

% find the index into the frequencies vector at 50 hz
hz50idx = dsearchn(hz',50);

% create a copy of the frequency vector
signalXf = signalX; % f=filter

% zero out the 50 Hz component
signalXf(hz50idx) = 0;
signalXf(end-hz50idx+2) = 0;

% take the IFFT
signalff = real(ifft(signalXf));

% take FFT of filtered signal
signalXf = fft(signalff);


% plot all three versions
figure(9), clf
subplot(211), hold on
plot(time,signal,'k')
plot(time,signalf,'b')
plot(time,signalff,'r')
xlabel('Time (s)'), ylabel('Activity')
legend({'Original';'Half-filtered';'Filtered'})
title('Time domain')

subplot(212), hold on
plot(hz,2*abs(signalX(1:length(hz)))/npnts,'k')
plot(hz,2*abs(signalXf(1:length(hz)))/npnts,'ro-','markerfacecolor','r')
set(gca,'xlim',[0 80])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('Frequency domain')

%% 8) Exploring why the filter in #7 isn't a good filter


% number of time points
N = 1000;

% generate a flat Fourier spectra
fourspect1 = ones(N,1);

% copy it and zero-out some fraction
fourspect2 = fourspect1;
fourspect2(round(N*.1):round(N*.2)) = 0;


% create time-domain signals via IFFT of the spectra
signal1 = real( ifft(fourspect1) );
signal2 = real( ifft(fourspect2) );
time = linspace(0,1,N);


% and plot!
figure(10), clf
subplot(211)
plot(time,fourspect1, time,fourspect2,'linew',2)
set(gca,'ylim',[0 1.05])
xlabel('Frequency (a.u.)')
title('Frequency domain')
legend({'Flat spectrum';'With edges'})


subplot(212)
plot(time,signal1, time,signal2,'linew',2)
set(gca,'ylim',[-1 1]*.05)
xlabel('Time (a.u.)')
title('Time domain')
legend({'Flat spectrum';'With edges'})


%% done.
