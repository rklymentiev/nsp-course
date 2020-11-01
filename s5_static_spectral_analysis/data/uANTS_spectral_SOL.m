%%
%   COURSE: Neural signal processing and analysis: Zero to hero
%  SESSION: Static spectral analyses
%  TEACHER: Mike X Cohen, sincxpress.com
%


%% 
% 
%  VIDEO: Sine waves and their parameters
% 
% 

%%% The goal of this cell is to create a signal by summing together sine waves.

% define a sampling rate
srate = 1000;

% list some frequencies
frex = [ 3   10   5   15   35 ];

% list some random amplitudes... make sure there are the same number of
% amplitudes as there are frequencies!
amplit = [ 5   15   10   5   7 ];

% phases... list some random numbers between -pi and pi
phases = [  pi/7  pi/8  pi  pi/2  -pi/4 ];

% define time...
time = -1:1/srate:1;


% now loop through frequencies and create sine waves
sine_waves = zeros(length(frex),length(time));
for fi=1:length(frex)
    sine_waves(fi,:) = amplit(fi) * sin(2*pi*time*frex(fi) + phases(fi));
end


%%% now plot
figure(1), clf

for sinei=1:length(amplit)
    subplot(length(amplit),1,sinei)

    % should be one sine wave per subplot
    plot(time,sine_waves(sinei,:),'k')
end

figure(2), clf
subplot(211)
plot(time,sum(sine_waves,1),'k')
xlabel('Time (s)')

%% sine wave parameter intuition via GUI

% a little interactive GUI to give you
% a better sense of sine wave parameters
sinewave_from_params;


%% 
% 
%  VIDEO: Complex numbers and Euler's formula
% 
% 

% several ways to create a complex number
z = 4 + 3i;
z = 4 + 3*1i;
z = 4 + 3*sqrt(-1);
z = complex(4,3);

disp([ 'Real part is ' num2str(real(z)) ' and imaginary part is ' num2str(imag(z)) '.' ])


% beware of a common programming error:
i = 2;
zz = 4 + 3*i;

% note that it's not possible to set 1i to a variable:
1i = 2;


% plot the complex number
figure(3), clf
plot(real(z),imag(z),'s','markersize',12,'markerfacecolor','k')

% make plot look nicer
set(gca,'xlim',[-5 5],'ylim',[-5 5])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Real axis')
ylabel('Imaginary axis')
title([ 'Number (' num2str(real(z)) ' ' num2str(imag(z)) 'i) on the complex plane' ])

%% Euler's formula and the complex plane

% use Euler's formula to plot vectors
m = 4;
k = pi/3;
compnum = m*exp( 1i*k );

% extract magnitude and angle
mag = abs(compnum);
phs = angle(compnum);

% plot a red dot in the complex plane corresponding to the
% real (x-axis) and imaginary (y-axis) parts of the number.
figure(4), clf
plot([0 real(compnum)],[0 imag(compnum)],'ro','linew',2,'markersize',10,'markerfacecolor','r')

% make plot look nicer
set(gca,'xlim',[-5 5],'ylim',[-5 5])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Real axis')
ylabel('Imaginary axis')

% draw a line using polar notation
h = polar([0 phs],[0 mag],'r');
set(h,'linewidth',2)


% also draw a unit circle
x = linspace(-pi,pi,100);
h = plot(cos(x),sin(x));
set(h,'color',[1 1 1]*.7) % light gray

% title
title([ 'Rectangular: [' num2str(real(compnum)) ' ' num2str(imag(compnum)) 'i ], ' ...
        'Polar: ' num2str(mag) 'e^{i' num2str(phs) '}' ])
    

%% 
% 
%  VIDEO: The dot product and sine waves
% 
% 


% create two vectors
v1 = [ 2 4 2 1 5 3 1 ];
v2 = [ 4 2  2 -3 2 5 0 ];

% two (among other) ways to create the dot product
dp1 = sum( v1.*v2 )
dp2 = dot(v1,v2)

%% the dot product with sine waves

%%% In this section, you will create a signal (wavelet)
%   and then compute the dot product between that signal
%   and a series of sine waves.


% phase of signal
theta = 1*pi/4;


% simulation parameters
srate = 1000;
time  = -1:1/srate:1;

% here is the signal (don't change this line)
signal = sin(2*pi*5*time + theta) .* exp( (-time.^2) / .1);

% sine wave frequencies (Hz)
sinefrex = 2:.5:10;


% plot signal
figure(4), clf
subplot(211)
plot(time,signal,'k','linew',3)
xlabel('Time (sec.)'), ylabel('Amplitude (a.u.)')
title('Signal')

dps = zeros(size(sinefrex));
for fi=1:length(dps)

    % create a real-valued sine wave. Note that the amplitude should be 1 and the phase should be 0
    sinew = sin(2*pi*sinefrex(fi)*time);

    % compute the dot product between sine wave and signal
    % normalize by the number of time points
    dps(fi) = dot(sinew,signal) / length(time);
end

% and plot
subplot(212)
stem(sinefrex,dps,'k','linew',3,'markersize',10,'markerfacecolor','w')
set(gca,'xlim',[sinefrex(1)-.5 sinefrex(end)+.5],'ylim',[-.2 .2])
xlabel('Sine wave frequency (Hz)')
ylabel('Dot product (signed magnitude)')
title([ 'Dot product of signal and sine waves (' num2str(theta) ' rad. offset)' ])

%%% Question: Try changing the 'theta' parameter. What is the effect on the spectrum of dot products?
%
%

%% 
% 
%   VIDEO: Complex sine waves
% 
% 


%%% The goal here is to create a complex-valued sine wave

% general simulation parameters
srate = 500; % sampling rate in Hz
time  = 0:1/srate:2-1/srate; % time in seconds

% sine wave parameters
freq = 5;    % frequency in Hz
ampl = 2;    % amplitude in a.u.
phas = pi/3; % phase in radians

% generate the complex sine wave. Note that exp(x) means e^x
csw = ampl*exp( 2*1i*freq*time + phas );


% plot in 2D
figure(5), clf
subplot(211)
plot(time,real(csw), time,imag(csw),'linew',2)
xlabel('Time (sec.)'), ylabel('Amplitude')
title('Complex sine wave projections')
legend({'real';'imag'})

% plot in 3D
subplot(212)
plot3(time,real(csw),imag(csw),'k','linew',3)
xlabel('Time (sec.)'), ylabel('real part'), zlabel('imag. part')
set(gca,'ylim',[-1 1]*ampl*3,'zlim',[-1 1]*ampl*3)
axis square
rotate3d on


%% 
% 
%   VIDEO: The complex dot product
% 
% 


% two vectors
v1 = [  3i  4  5 -3i ];
v2 = [ -3i 1i  1  0  ];

% notice the dot product is a complex number
sum(v1.*v2)

%% complex dot product with wavelet

% 1) copy/paste the code from cell "the dot product with sine waves"
% 2) Change the sine waves into complex sine waves
% 3) Plot the magnitude of the dot product ( abs(dps) )



% phase of signal
theta = 1*pi/4;


% simulation parameters
srate = 1000;
time  = -1:1/srate:1;

% here is the signal (don't change this line)
signal = sin(2*pi*5*time + theta) .* exp( (-time.^2) / .1);

% sine wave frequencies (Hz)
sinefrex = 2:.5:10;


% plot signal
figure(4), clf
subplot(211)
plot(time,signal,'k','linew',3)
xlabel('Time (sec.)'), ylabel('Amplitude (a.u.)')
title('Signal')

dps = zeros(size(sinefrex));
for fi=1:length(dps)

    % create a complex-valued sine wave. Note that the amplitude should be 1 and the phase should be 0
    sinew = exp( 1i*2*pi*sinefrex(fi)*time );

    % compute the dot product between sine wave and signal
    % normalize by the number of time points
    dps(fi) = dot(sinew,signal) / length(time);
end

% and plot
subplot(212)
stem(sinefrex,abs(dps),'k','linew',3,'markersize',10,'markerfacecolor','w')
set(gca,'xlim',[sinefrex(1)-.5 sinefrex(end)+.5],'ylim',[-.2 .2])
xlabel('Sine wave frequency (Hz)')
ylabel('Dot product (signed magnitude)')
title([ 'Dot product of signal and sine waves (' num2str(theta) ' rad. offset)' ])



%%% Question: Is the dot product spectrum still dependent on the phase of the signal?

%%

%% A movie showing why complex sine waves are phase-invariant

% no need to change the code; just run and enjoy!

% create complex sine wave
csw = exp( 1i*2*pi*5*time );
rsw = cos(    2*pi*5*time );

% specify range of phase offsets for signal
phases = linspace(0,7*pi/2,100);


% setup the plot
figure(6), clf
subplot(223)
ch = plot(0,0,'ro','linew',2,'markersize',10,'markerfacecolor','r');
set(gca,'xlim',[-.2 .2],'ylim',[-.2 .2])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Cosine axis')
ylabel('Sine axis')
title('Complex plane')

% and then setup the plot for the real-dot product axis
subplot(224)
rh = plot(0,0,'ro','linew',2,'markersize',10,'markerfacecolor','r');
set(gca,'xlim',[-.2 .2],'ylim',[-.2 .2],'ytick',[])
grid on, hold on, axis square
plot(get(gca,'xlim'),[0 0],'k','linew',2)
plot([0 0],get(gca,'ylim'),'k','linew',2)
xlabel('Real axis')
title('Real number line')


for phi=1:length(phases)

    % create signal
    signal = sin(2*pi*5*time + phases(phi)) .* exp( (-time.^2) / .1);

    % compute complex dot product
    cdp = sum( signal.*csw ) / length(time);

    % compute real-valued dot product
    rdp = sum( signal.*rsw ) / length(time);

    % plot signal and real part of sine wave
    subplot(211)
    plot(time,signal, time,rsw,'linew',2)
    title('Signal and sine wave over time')

    % plot complex dot product
    subplot(223)
    set(ch,'XData',real(cdp),'YData',imag(cdp))

    % draw normal dot product
    subplot(224)
    set(rh,'XData',rdp)

    % wait a bit
    pause(.1)
end


%%
% 
%   VIDEO: The discrete-time Fourier transform
% 
% 


%% generate multi-component sine wave (exactly as before)

% define a sampling rate
srate = 1000;

% some parameters
frex   = [ 3   10   5   15   35 ];
amplit = [ 5   15   10   5   7 ];

% define time...
time = -1:1/srate:1;

% now loop through frequencies and create sine waves
signal = zeros(1,length(time));
for fi=1:length(frex)
    signal = signal + amplit(fi)*sin(2*pi*time*frex(fi));
end

%% The Fourier transform in a loop

%%% The goal of this section of code is to implement
%   the Fourier transform in a loop, as described in lecture.

fourierTime = (0:N-1)/N;    % "time" used for sine waves
nyquist     = srate/2;        % Nyquist frequency -- the highest frequency you can measure in the data
N           = length(signal); % length of sequence

% initialize Fourier output matrix
fourierCoefs = zeros(size(signal));

% These are the actual frequencies in Hz that will be returned by the
% Fourier transform. The number of unique frequencies we can measure is
% exactly 1/2 of the number of data points in the time series (plus DC).
frequencies = linspace(0,nyquist,floor(N/2)+1);


% loop over frequencies
for fi=1:N

    % create complex-valued sine wave for this frequency
    fourierSine = exp( 1i*2*pi*(fi-1)*fourierTime );

    % compute dot product between sine wave and signal (created in the previous cell)
    fourierCoefs(fi) = dot(fourierSine,signal);

end

% scale Fourier coefficients to original scale
fourierCoefs = fourierCoefs / N;


figure(7), clf
subplot(221)
plot(real(exp( -2*pi*1i*(10).*fourierTime )))
xlabel('time (a.u.)'), ylabel('Amplitude')
title('One sine wave from the FT (real part)')

subplot(222)
plot(signal)
title('Data')

subplot(212)
plot(frequencies,abs(fourierCoefs(1:length(frequencies)))*2,'*-')
xlabel('Frequency (Hz)')
ylabel('Power (\muV)')
title('Power spectrum derived from discrete Fourier transform')

%% The fast-Fourier transform (FFT)

%%% The "slow" FT is important to understand and see implemented,
%   but in practice you should always use the FFT.
%   In this code you will see that they produce the same results.

% Compute fourier transform and scale
fourierCoefsF = fft(signal) / N;

subplot(212), hold on
plot(frequencies,abs(fourierCoefsF(1:length(frequencies)))*2,'ro')
set(gca,'xlim',[0 40])


%% 
% 
%   VIDEO: Fourier coefficients as complex numbers
% 
% 


%%% Fourier coefficients are difficult to interpret 'numerically', that is,
%   it's difficult to extract the information in a Fourier coefficient
%   simply by looking at the real and imaginary parts printed out.
%   Instead, you can understand them by visualizing them (in the next cell!).


srate = 1000;
time  = (0:srate-1)/srate;
freq  = 6;

% create sine waves that differ in power and phase
sine1 = 3 * cos(2*pi*freq*time + 0 );
sine2 = 2 * cos(2*pi*freq*time + pi/6 );
sine3 = 1 * cos(2*pi*freq*time + pi/3 );

% compute Fourier coefficients
fCoefs1 = fft(sine1) / length(time);
fCoefs2 = fft(sine2) / N;
fCoefs3 = fft(sine3) / N;

hz = linspace(0,srate/2,floor(length(time)/2)+1);

% find the frequency of our sine wave
hz6 = dsearchn(hz',freq);

% let's look at the coefficients for this frequency
disp([ '6 Hz Fourier coefficient for sin1: ' num2str(fCoefs1(hz6)) ])
disp([ '6 Hz Fourier coefficient for sin2: ' num2str(fCoefs2(hz6)) ])
disp([ '6 Hz Fourier coefficient for sin3: ' num2str(fCoefs3(hz6)) ])

%% complex numbers as vectors in a polar plot

%%% Explore the concept that the Fourier coefficients are complex numbers,
%   and can be represented on a complex plane.
%   The magnitude is the length of the line, and the phase is the angle of that line.


% make polar plots of fourier coefficients
figure(9), clf
h(1) = polarplot([0 angle(fCoefs1(hz6))],[0 2*abs(fCoefs1(hz6))],'r');
hold on
h(2) = polarplot([0 angle(fCoefs2(hz6))],[0 2*abs(fCoefs2(hz6))],'b');
h(3) = polarplot([0 angle(fCoefs3(hz6))],[0 2*abs(fCoefs3(hz6))],'m');

% adjust the plots a bit
set(h,'linewidth',5)
legend({'sine1';'sine2';'sine3'})

%% phase and power information can be extracted via Euler's formula

% extract amplitude using Pythagorian theorem
amp1 = 2*sqrt( real(fCoefs1).^2 + imag(fCoefs1).^2 ); % hint: use the functions "imag" and "real"
amp2 = 2*sqrt( real(fCoefs2).^2 + imag(fCoefs2).^2 );
amp3 = 2*sqrt( real(fCoefs3).^2 + imag(fCoefs3).^2 );

% extract amplitude using the Matlab function abs
% amp1 = abs( fCoefs1 );
% amp2 = abs( fCoefs2 );
% amp3 = abs( fCoefs3 );

% yet another possibility (the complex number times its conjugate)
%amp1 = fCoefs1.*conj(fCoefs1);


figure(9), clf

% plot amplitude spectrum
subplot(211)
plot(hz,2*amp1(1:length(hz)),'ro-','linew',3), hold on
plot(hz,2*amp2(1:length(hz)),'bp-','linew',3)
plot(hz,2*amp3(1:length(hz)),'ks-','linew',3)
set(gca,'xlim',[freq-3 freq+3])
xlabel('Frequency (Hz)'), ylabel('Amplitude')

%% and now for phase...

% extract phase angles using trigonometry
phs1 = atan2( imag(fCoefs1),real(fCoefs1) );
phs2 = atan2( imag(fCoefs2),real(fCoefs2) );
phs3 = atan2( imag(fCoefs3),real(fCoefs3) );

% extract phase angles using Matlab function angle
% phs1 = angle( fCoefs1 );
% phs2 = angle( fCoefs2 );
% phs3 = angle( fCoefs3 );


% plot phase spectrum
subplot(212)
plot(hz,phs1(1:length(hz)),'ro-','linew',3), hold on
plot(hz,phs2(1:length(hz)),'bp-','linew',3)
plot(hz,phs3(1:length(hz)),'ks-','linew',3)
set(gca,'xlim',[freq-3 freq+3])
xlabel('Frequency (Hz)'), ylabel('Phase (rad.)')



%% 
% 
%   VIDEO: Positive/negative spectrum; amplitude scaling
% 
% 

% This cell will run a video that shows the complex sine waves used
% in the Fourier transform. The blue line corresponds to the real part
% and the red line corresponds to the imaginary part. The title tells
% you the frequency as a fraction of the sampling rate (Nyquist = .5).
% Notice what happens as the sine waves go into the negative frequencies.


% setup parameters
N     = 100;
fTime = ((1:N)-1)/N;
whichhalf = {'negative';'positive'};

% generate the plot
figure(10), clf
ploth = plot(fTime,ones(2,N),'linew',2);
titleh= title('.');
set(ploth(1),'color','b')
set(ploth(2),'color','r','linestyle','--')
xlabel('Time (norm.)'), ylabel('Amplitude')


% loop over frequencies
for fi=1:N

    % create sine wave for this frequency
    fourierSine = exp( -1i*2*pi*(fi-1).*fTime );

    % update graph
    set(ploth(1),'ydata',real(fourierSine));
    set(ploth(2),'ydata',imag(fourierSine));
    set(titleh,'String',[ num2str(fi/N) ' of fs (' whichhalf{(fi<N/2)+1} ')' ])

    pause(.1)
end

%%% QUESTION: What happens to the sine waves above f=.5?
%
%%% QUESTION: What happens to the sine waves as they approach f=.5?
%

%% scaling of Fourier coefficients

%%% The goal of this section of code is to understand the necessity and logic
%   behind the two normalization factors that get the Fourier coefficients
%   in the same scale as the original data.


% create the signal
srate  = 1000; % hz
time   = (0:3*srate-1)/srate; % time vector in seconds
pnts   = length(time); % number of time points
signal = 2.5 * sin( 2*pi*4*time );


% prepare the Fourier transform
fourTime = (0:pnts-1)/pnts;
fCoefs   = zeros(size(signal));

for fi=1:pnts
    % create complex sine wave and compute dot product with signal
    csw = exp( -1i*2*pi*(fi-1)*fourTime );
    fCoefs(fi) = sum( signal.*csw );
end

% extract amplitudes
ampls = abs(fCoefs/pnts);

ampls = 2*ampls;

% compute frequencies vector
hz = linspace(0,srate/2,floor(pnts/2)+1);

figure(11), clf
stem(hz,ampls(1:length(hz)),'ks-','linew',3,'markersize',10,'markerfacecolor','w')

% make plot look a bit nicer
set(gca,'xlim',[0 10])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')

%%% QUESTION: Does the amplitude of 4 Hz in the figure match the simulated signal?
% 
%%% QUESTION: Does the amplitude also depend on the length of the signal?
%             Apply the two normalization factors discussed in lecture.
%             Test whether the scaling is robust to signal length.


%% DC reflects the mean offset

figure(12), clf

% NOTE: below is the same signal with (1) small, (2) no, (3) large DC
%       Is the amplitude spectrum accurate at 0 Hz???
signalX1 = fft(signal+2)             / pnts;
signalX2 = fft(signal-mean(signal))  / pnts;
signalX3 = fft(signal+10)            / pnts;

% extract amplitude properly
amplSigX1 = [ abs(signalX1(1)) 2*abs(signalX1(2:length(hz))) ];
amplSigX2 = [ abs(signalX2(1)) 2*abs(signalX2(2:length(hz))) ];
amplSigX3 = [ abs(signalX3(1)) 2*abs(signalX3(2:length(hz))) ];


% plot signals in the time domain
subplot(211)
plot(ifft(signalX1)*pnts,'bo-'), hold on
plot(ifft(signalX2)*pnts,'rd-'), hold on
plot(ifft(signalX3)*pnts,'k*-'), hold on
xlabel('Tme (ms)'), ylabel('Amplitude')


% plot signals in the frequency domain
subplot(212)
plot(hz,amplSigX1,'bo-','linew',2,'markersize',8), hold on
plot(hz,amplSigX2,'rd-','linew',2,'markersize',8)
plot(hz,amplSigX3,'k*-','linew',2,'markersize',8)

xlabel('Frequencies (Hz)'), ylabel('Amplitude')
legend({'original';'de-meaned';'increased mean'})
set(gca,'xlim',[0 10])



%%% QUESTION: Can you adapt the code for accurate scaling of all three signals??
%             (Yes, of course you can! Do it!)

%%


%% 
% 
%  VIDEO: Spectral analysis of resting-state EEG
% 
% 

% The goal of this cell is to plot a power spectrum of resting-state EEG data.

clear
load EEGrestingState.mat

% create a time vector that starts from 0
npnts = length(eegdata);
time = (0:npnts-1)/srate;


% plot the time-domain signal
figure(13), clf
plot(time,eegdata)
xlabel('Time (s)'), ylabel('Voltage (\muV)')
zoom on


% static spectral analysis
hz = linspace(0,srate/2,floor(npnts/2)+1);
ampl = 2*abs(fft(eegdata)/npnts);
powr = ampl.^2;

figure(14), clf, hold on
plot(hz,ampl(1:length(hz)),'k','linew',2)
plot(hz,powr(1:length(hz)),'r','linew',2)

xlabel('Frequency (Hz)')
ylabel('Amplitude or power')
legend({'Amplitude';'Power'})

% optional zooming in
%set(gca,'xlim',[0 30])


%%% QUESTION: What are the three most prominent features of the EEG spectrum?
% 
%%% QUESTION: What do you notice about the difference between the amplitude
%             and power spectra?
% 
%%% QUESTION: Can you see the ~10 Hz oscillation in the raw time series data?
% 


%% 
% 
%  VIDEO: Quantify alpha power over the scalp
% 
% 

clear
load restingstate64chans.mat

% These data comprise 63 "epochs" of resting-state data. Each epoch is a
%   2-second interval cut from ~2 minutes of resting-state.
% The goal is to compute the power spectrum of each 2-second epoch
%   separately, then average together.
%   Then, extract the average power from 8-12 Hz (the "alpha band") and
%   make a topographical map of the distribution of this power.

% convert to double-precision
EEG.data = double(EEG.data);


% can do it in a loop, or all at once, specifying to compute the FFT over
% the second dimension (time)
chanpowr = ( 2*abs( fft(EEG.data,[],2) )/EEG.pnts ).^2;

% then average over trials
chanpowr = mean(chanpowr,3);

% vector of frequencies
hz = linspace(0,EEG.srate/2,floor(EEG.pnts/2)+1);


% do some plotting
% plot power spectrum of all channels
figure(15), clf
plot(hz,chanpowr(:,1:length(hz)),'linew',2)
xlabel('Frequency (Hz)'), ylabel('Power (\muV)')

set(gca,'xlim',[0 30],'ylim',[0 50])

%% now to extract alpha power

% boundaries in hz
alphabounds = [ 8 12 ];

% convert to indices
freqidx = dsearchn(hz',alphabounds');


% extract average power
alphapower = mean(chanpowr(:,freqidx(1):freqidx(2)),2);

% and plot
figure(16), clf
topoplotIndie(alphapower,EEG.chanlocs,'numcontour',0);
set(gca,'clim',[0 6])
colormap hot


%% 
% 
%  VIDEO: Reconstruct a signal via inverse Fourier transform
% 
% 

%% create the signal

% define a sampling rate and time vector
srate = 1000;
time  = -1:1/srate:1;

% frequencies
frex = [ 3 10 5 15 35 ];


% now loop through frequencies and create sine waves
signal = zeros(1,length(time));
for fi=1:length(frex)
    signal = signal + fi * sin(2*pi*time*frex(fi));
end

%% on to the ift!

%%% Here you will invert the Fourier transform,
%   by starting from Fourier coefficients and getting back into the time domain.

N           = length(signal); % length of sequence
fourierTime = (0:N-1)/N;      % "time" used for sine waves

reconSignal = zeros(size(signal));
fourierCoefs = fft(signal)/N;

% loop over frequencies
for fi=1:N

    % create coefficient-modulated sine wave for this frequency
    % Note: this is a complex sine wave without the minus sine in the exponential.
    fourierSine = fourierCoefs(fi) * exp( 1i*2*pi*(fi-1)*fourierTime );

    % continue building up signal...
    reconSignal = reconSignal + fourierSine;
end

% note: in practice, the inverse Fourier transform should be done using:
%reconSignal = ifft(fourierCoefs) * N;

figure(17), clf
plot(real(reconSignal),'.-')
hold on
plot(signal,'ro')
zoom on % inspect the two signals

legend({'reconstructed';'original'})


%% 
% 
%   VIDEO: Frequency resolution and zero-padding
% 
% 


%%% We start by investigating the difference between sampling rate and
%   number of time points for Fourier frequencies.

% temporal parameters
srates  = [100 100 1000];
timedur = [  1  10    1];


freq    =     5; % in Hz
colors  = 'kmb';
symbols = 'op.';


figure(18), clf
legendText = cell(size(timedur));
for parami=1:length(colors)

    % define sampling rate in this round
    srate = srates(parami); % in Hz

    % define time
    time = -1:1/srate:timedur(parami);

    % create signal (Morlet wavelet)
    signal = cos(2*pi*freq.*time) .* exp( (-time.^2) / .05 );

    % compute FFT and normalize
    signalX = fft(signal)/length(signal);
    signalX = signalX./max(signalX);

    % define vector of frequencies in Hz
    hz = linspace(0,srate/2,floor(length(signal)/2)+1);


    % plot time-domain signal
    subplot(211)
    plot(time,signal,[colors(parami) symbols(parami) '-'],'markersize',10,'markerface',colors(parami)), hold on
    set(gca,'xlim',[-1 1])
    xlabel('Time (s)'), ylabel('Amplitude')
    title('Time domain')

    % plot frequency-domain signal
    subplot(212), hold on
    plot(hz,abs(signalX(1:length(hz))),[colors(parami) symbols(parami) '-'],'markersize',10,'markerface',colors(parami))
    xlabel('Frequency (Hz)'), ylabel('Amplitude')
    title('Frequency domain')

    legendText{parami} = [ 'srate=' num2str(srates(parami)) ', N=' num2str(timedur(parami)+1) 's' ];
end

legend(legendText)
zoom

%% zero-padding, spectral resolution, and sinc interpolation

%%% Explore the effects of zero-padding.
%   Note: I created the numbers here to show rather strong effects of zero-padding.
%    You can try with other numbers; the effects will be more mild.

figure(19), clf

signal  = [ 1 0 1 2 -3 1 2 0 -3 0 -1 2 1 -1];

% compute the FFT of the same signal with different DC offsets
signalX1 = fft(signal,length(signal))     / length(signal);
signalX2 = fft(signal,length(signal)+10)  / length(signal);
signalX3 = fft(signal,length(signal)+100) / length(signal);% zero-pad to 100 plus length of signal!

% define frequencies vector
frex1 = linspace( 0, .5, floor(length(signalX1)/2)+1 );
frex2 = linspace( 0, .5, floor(length(signalX2)/2)+1 );
frex3 = linspace( 0, .5, floor(length(signalX3)/2)+1 );


% plot signals in the time domain
subplot(211)
plot(ifft(signalX1)*length(signal),'bo-'), hold on
plot(ifft(signalX2)*length(signal),'rd-'), hold on
plot(ifft(signalX3)*length(signal),'k*-'), hold on
xlabel('Time points (arb. units)')

% plot signals in the frequency domain
subplot(212)
plot(frex1,2*abs(signalX1(1:length(frex1))),'bo-'), hold on
plot(frex2,2*abs(signalX2(1:length(frex2))),'rd-')
plot(frex3,2*abs(signalX3(1:length(frex3))),'k*-')

xlabel('Normalized frequency units'), ylabel('Amplitude')
legend({'"Native" N';'N+10';'N+100'})


%% 
% 
%   VIDEO: Examples of sharp non-stationarities on power spectra
% 
% 

%% sharp transitions


% simulation parameters
srate = 1000;
t = 0:1/srate:10;
n = length(t);

a = [10 2 5 8];
f = [3 1 6 12];

timechunks = round(linspace(1,n,length(a)+1));

signal = 0;
for i=1:length(a)
    signal = cat(2,signal,a(i)* sin(2*pi*f(i)*t(timechunks(i):timechunks(i+1)-1) ));
end

signalX = fft(signal)/n;
hz = linspace(0,srate/2,floor(n/2)+1);

figure(20), clf
subplot(211)
plot(t,signal)
xlabel('Time'), ylabel('amplitude')

subplot(212)
plot(hz,2*abs(signalX(1:length(hz))),'s-','markerface','k')
xlabel('Frequency (Hz)'), ylabel('amplitude')
set(gca,'xlim',[0 20])

%% edges and edge artifacts

x = (linspace(0,1,n)>.5)+0; % +0 converts from boolean to number

% uncommenting this line shows that nonstationarities
% do not prevent stationary signals from being easily observed
x = x + .08*sin(2*pi*6*t);

% plot
figure(21), clf
subplot(211)
plot(t,x)
set(gca,'ylim',[-.1 1.1])
xlabel('Time (s.)'), ylabel('Amplitude (a.u.)')

subplot(212)
xX = fft(x)/n;
plot(hz,2*abs(xX(1:length(hz))))
set(gca,'xlim',[0 20],'ylim',[0 .1])
xlabel('Frequency (Hz)'), ylabel('Amplitude (a.u.)')


%% 
% 
%   VIDEO: Examples of smooth non-stationarities on power spectra
% 
% 


srate = 1000;
t = 0:1/srate:10;
n = length(t);
f = 3; % frequency in Hz

% sine wave with time-increasing amplitude
ampl1 = linspace(1,10,n);
% ampl1 = abs(interp1(linspace(t(1),t(end),10),10*rand(1,10),t,'spline'));
ampl2 = mean(ampl1);

signal1 = ampl1 .* sin(2*pi*f*t);
signal2 = ampl2 .* sin(2*pi*f*t);


% obtain Fourier coefficients and Hz vector
signal1X = fft( signal1 )/n;
signal2X = fft( signal2 )/n;
hz = linspace(0,srate,n);

figure(22), clf
subplot(211)
plot(t,signal2,'r','linew',2), hold on
plot(t,signal1,'linew',2)
xlabel('Time'), ylabel('amplitude')

subplot(212)
plot(hz,2*abs(signal2X(1:length(hz))),'ro-','markerface','r','linew',2)
hold on
plot(hz,2*abs(signal1X),'s-','markerface','k','linew',2)

xlabel('Frequency (Hz)'), ylabel('amplitude')
set(gca,'xlim',[1 7])
legend({'Stationary';'Non-stationary'})

%% frequency non-stationarity


f  = [2 10];
ff = linspace(f(1),mean(f),n);
signal1 = sin(2*pi.*ff.*t);
signal2 = sin(2*pi.*mean(ff).*t);


signal1X = fft(signal1)/n;
signal2X = fft(signal2)/n;
hz = linspace(0,srate/2,floor(n/2));

figure(23), clf

subplot(211)
plot(t,signal1), hold on
plot(t,signal2,'r')
xlabel('Time'), ylabel('amplitude')
set(gca,'ylim',[-1.1 1.1])

subplot(212)
plot(hz,2*abs(signal1X(1:length(hz))),'.-'), hold on
plot(hz,2*abs(signal2X(1:length(hz))),'r.-')
xlabel('Frequency (Hz)'), ylabel('amplitude')
set(gca,'xlim',[0 20])


%% examples of rhythmic non-sinusoidal time series

% parameters
srate = 1000;
time  = (0:srate*6-1)/srate;
npnts = length(time);
hz = linspace(0,srate,npnts);

% various mildly interesting signals to test
signal = detrend( sin( cos(2*pi*time)-1 ) );
signal = sin( cos(2*pi*time) + time );
signal = detrend( cos( sin(2*pi*time).^4 ) );


% plot!
figure(24), clf

% time domain
subplot(211)
plot(time,signal,'k','linew',3)
xlabel('Time (s)')

% frequency domain
subplot(212)
plot(hz,abs(fft(signal)),'k','linew',3)
set(gca,'xlim',[0 20])
ylabel('Frequency (Hz)')


%% 
% 
%   VIDEO: Welch's method on phase-slip data
% 
% 

srate = 1000;
time  = (0:srate-1)/srate;

signal = [ sin(2*pi*10*time) sin(2*pi*10*time(end:-1:1)) ];

figure(25), clf
subplot(211)
plot(signal)

subplot(223)
bar(linspace(0,srate,length(signal)),2*abs(fft(signal))/length(signal))
set(gca,'xlim',[5 15])
title('Static FFT')
xlabel('Frequency (Hz)')



%% Now for Welch's method

% parameters
winlen = 500; % window length in points (same as ms if srate=1000!)
skip = 100; % also in time points;

% vector of frequencies for the small windows
hzL = linspace(0,srate/2,floor(winlen/2)+1);

% initialize time-frequency matrix
welchspect = zeros(1,length(hzL));

% Hann taper
hwin = .5*(1-cos(2*pi*(1:winlen) / (winlen-1)));



% loop over time windows
nbins = 1;
for ti=1:skip:length(signal)-winlen
    
    % extract part of the signal
    tidx    = ti:ti+winlen-1;
    tmpdata = signal(tidx);
    
    % FFT of these data (does the taper help?)
    x = fft(hwin.*tmpdata)/winlen;
    
    % and put in matrix
    welchspect = welchspect + 2*abs(x(1:length(hzL)));
    nbins = nbins + 1;
end

% divide by nbins to complete average
welchspect = welchspect/nbins;


subplot(224)
bar(hzL,welchspect)
set(gca,'xlim',[5 15])
title('Welch''s method')
xlabel('Frequency (Hz)')

%% 
% 
%   VIDEO: Welch's method on resting-state EEG data
% 
% 


% load data
load EEGrestingState.mat
N = length(eegdata);

% time vector
timevec = (0:N-1)/srate;

% plot the data
figure(26), clf
plot(timevec,eegdata,'k')
xlabel('Time (seconds)'), ylabel('Voltage (\muV)')

%% one big FFT (not Welch's method)

% "static" FFT over entire period, for comparison with Welch
eegpow = abs( fft(eegdata)/N ).^2;
hz = linspace(0,srate/2,floor(N/2)+1);

%% "manual" Welch's method

% window length in seconds*srate
winlength = 1*srate;

% number of points of overlap
nOverlap = round(srate/2);

% window onset times
winonsets = 1:nOverlap:N-winlength;

% note: different-length signal needs a different-length Hz vector
hzW = linspace(0,srate/2,floor(winlength/2)+1);

% Hann window
hannw = .5 - cos(2*pi*linspace(0,1,winlength))./2;

% initialize the power matrix (windows x frequencies)
eegpowW = zeros(1,length(hzW));

% loop over frequencies
for wi=1:length(winonsets)
    
    % get a chunk of data from this time window
    datachunk = eegdata(winonsets(wi):winonsets(wi)+winlength-1);
    
    % apply Hann taper to data
    datachunk = datachunk .* hannw;
    
    % compute its power
    tmppow = abs(fft(datachunk)/winlength).^2;
    
    % enter into matrix
    eegpowW = eegpowW  + tmppow(1:length(hzW));
end

% divide by N
eegpowW = eegpowW / length(winonsets);


%%% plotting
figure(27), clf, subplot(211), hold on

plot(hz,eegpow(1:length(hz)),'k','linew',2)
plot(hzW,eegpowW/10,'r','linew',2)
set(gca,'xlim',[0 40])
xlabel('Frequency (Hz)')
legend({'"Static FFT';'Welch''s method'})
title('Using FFT and Welch''s')

%% MATLAB pwelch

subplot(212)

% create Hann window
winsize = 2*srate; % 2-second window
hannw = .5 - cos(2*pi*linspace(0,1,winsize))./2;

% number of FFT points (frequency resolution)
nfft = srate*100;

pwelch(eegdata,hannw,round(winsize/4),nfft,srate);
set(gca,'xlim',[0 40])


%% 
% 
%   VIDEO: Welch's method on v1 laminar data
% 
% 

clear
load v1_laminar.mat
csd = double(csd);

% specify a channel for the analyses
chan2use = 7;


% create Hann window
hannw = .5 - cos(2*pi*linspace(0,1,size(csd,2)))./2;

% Welch's method using MATLAB pwelch
[pxx,hz] = pwelch(squeeze(csd(chan2use,:,:)),hannw,round(size(csd,2)/10),1000,srate);

figure(28), clf
subplot(211)
plot(timevec,mean(csd(chan2use,:,:),3),'linew',2)
set(gca,'xlim',timevec([1 end]))
xlabel('Time (s)'), ylabel('Voltage (\muV)')

subplot(212)
plot(hz,mean(pxx,2),'linew',2)
set(gca,'xlim',[0 140])
xlabel('Frequency (Hz)')
ylabel('Power (\muV^2)')


%% done.
