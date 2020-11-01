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

srate  = 
frex   = [  ];
amplit = [  ];
phases = [  ];
time   = -1:1/srate:1;

% create sine waves, first initialize to correct size
sine_waves = zeros(,);

for fi=1:length(frex)
    sine_waves(fi,:) =  amplit * sin(i*time*frex(fi) + fi);
end

littleNoise = randn
lotsOfNoise = 


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
plot(time,littleNoise)
title('Time series with LITTLE noise')

subplot(212)
plot(time,
title('Time series with A LOT of noise')

%% 2) Compute the power spectrum of the simulated time series (use FFT) and plot the results, 
%    separately for a little noise and a lot of noise. Show frequencies 0 to 35 Hz.
%    How well are the frequencies reconstructed, and does this depend on noise?
% 

figure(3), clf

for noisei=1:2
    
    % FFT
    if noisei==1
        f = fft(
    else
        
    end
    
    % compute frequencies in Hz
    hz = linspace(
    
    % plot the amplitude spectrum
    subplot(2,1,noisei)
    plot(hz
    xlabel('Frequencies (Hz)'), ylabel('Amplitude')
    set(gca,'xlim',[0 1],'ylim',[0 max(amplit)*1.2]) % is this a good choice for x-axis limit?!?!?!
    
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

% FFT of all trials individually (note that you can do it in one line!)
powspectSeparate = 
% Then average the single-trial spectra together (average over trials, not over frequencies)
powspectSeparate = mean(2*abs(powspectSeparate),1);


% now average first, then take the FFT of the trial average
powspectAverage  = 
powspectAverage  = 

% frequencies in Hz
hz = linspace(0,srate/2,floor(length(timevec)/2)+1);


% now plot
figure(4), clf
set(gcf,'name',[ 'Results from electrode ' chan2use ])
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

%% 4) Do the same as above but for electrode 1. 
%    How do these results compare to the results from channel 7, and does this depend on
%    whether you average first in the time-domain or average the individual power spectra?
% ANATOMICAL NOTE: channel 7 is around L4; channel 1 is in the hippocampus.



%% 5) Fourier transform from scratch!
% Hey, wouldn't it be fun to program the discrete-time Fourier transform
% from scratch! Yes, of course it would be. Let's do that.
% Generate a 20-element vector of random numbers.
% Use the hints below to help you write the Fourier transform.
% Next, use the fft function on the same data to verify that your FT was accurate.


N       = 20;          % length of sequence
signal  = randn(1,N);  % data
fTime   = ; % "time" used in Fourier transform

% initialize Fourier output matrix
fourierCoefs = zeros(size(signal)); 

% loop over frequencies
for fi=1:N
    
    % create sine wave for this frequency
    fourierSine = 
    
    % compute dot product as sum of point-wise elements
    fourierCoefs(fi) = 
end

% divide by N to scale coefficients properly



figure(5), clf
subplot(211)
plot(signal)
title('Data')

subplot(212)
plot(abs(fourierCoefs)*2,'*-')
xlabel('Frequency (a.u.)')

% for comparison, use the fft function on the same data
fourierCoefsF = 

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
% does the increased frequency resolution have no appreciable visible effect on the results?

tidx(1) = dsearchn(timevec',0);
tidx(2) = dsearchn(timevec',.5);

% set nfft to be multiples of the length of the data
nfft = 1 * (diff(tidx)+1);

powspect = 
hz = linspace(0,srate/2,floor(nfft/2)+1);

figure(6), clf
plot(hz,powspect(1:length(hz)),'k-o')
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
time  = :(


% the key parameter of pink noise is the exponential decay (ed)
ed = 50; % try different values!
as = rand(1,npnts) .* exp(-(0:npnts-1);
fc = as .* exp(1i*2*pi*rand(size(as)));

signal = real(ifft(fc)) * npnts;

% now add 50 Hz line noise
signal = signal + ;


% compute its spectrum
hz = linspace(0,srate/2,floor(npnts/2)+1);
signalX = fft();



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
hz50idx = dsearchn

% create a copy of the frequency vector
signalXf = signalX; % f=filter

% zero out the 50 Hz component
signalXf(hz50idx)

% take the IFFT
signalf = real(ifft());


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
hz50idx = dsearchn(

% create a copy of the frequency vector
signalXf = signalX; % f=filter

% zero out the 50 Hz component
signalXf(hz50idx) = 0;
signalXf(end- % hint: the negative frequencies

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
signal1 = 
signal2 = 
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
