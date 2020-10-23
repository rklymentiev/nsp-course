function plot_simEEG(varargin)
% plot_simEEG - plot function for MXC's course on EEG simulation
% 
% INPUTS:  EEG : eeglab structure
%         chan : channel to plot (default = 1)
%       fignum : figure to plot into (default = 1)


if isempty(varargin)
    help plot_simEEG
    error('No inputs!')
elseif length(varargin)==1
    EEG = varargin{1};
    [chan,fignum] = deal(1);
elseif length(varargin)==2
    EEG = varargin{1};
    chan = varargin{2};
    fignum = 1;
elseif length(varargin)==3
    EEG = varargin{1};
    chan = varargin{2};
    fignum = varargin{3};
end


figure(fignum), clf

%% ERP

subplot(211), hold on
h = plot(EEG.times,squeeze(EEG.data(chan,:,:)),'linew',.5);
set(h,'color',[1 1 1]*.75)
plot(EEG.times,squeeze(mean(EEG.data(chan,:,:),3)),'k','linew',3);
xlabel('Time (s)'), ylabel('Activity')
title([ 'ERP from channel ' num2str(chan) ])


%% static power spectrum

hz = linspace(0,EEG.srate,EEG.pnts);
if numel(size(EEG.data))==3
    pw = mean((2*abs(fft(squeeze(EEG.data(chan,:,:)),[],1)/EEG.pnts)).^2,2);
else
    pw = (2*abs(fft(EEG.data(chan,:))/EEG.pnts)).^2;
end
subplot(223)
plot(hz,pw,'linew',2)
set(gca,'xlim',[0 40])
xlabel('Frequency (Hz)'), ylabel('Power')
title('Static power spectrum')


%% time-frequency analysis

% frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
frex  = linspace(2,30,40);

% number of wavelet cycles (hard-coded to 3 to 10)
waves = 2*(linspace(3,10,length(frex))./(2*pi*frex)).^2;

% setup wavelet and convolution parameters
wavet = -2:1/EEG.srate:2;
halfw = floor(length(wavet)/2)+1;
nConv = EEG.pnts*EEG.trials + length(wavet) - 1;

% initialize time-frequency matrix
tf = zeros(length(frex),EEG.pnts);

% spectrum of data
dataX = fft(reshape(EEG.data(chan,:,:),1,[]),nConv);

% loop over frequencies
for fi=1:length(frex)
    
    % create wavelet
    waveX = fft( exp(2*1i*pi*frex(fi)*wavet).*exp(-wavet.^2/waves(fi)),nConv );
    waveX = waveX./max(waveX); % normalize
    
    % convolve
    as = ifft( waveX.*dataX );
    % trim and reshape
    as = reshape(as(halfw:end-halfw+1),[EEG.pnts EEG.trials]);
    
    % power
    tf(fi,:) = mean(abs(as),2);
end

% show a map of the time-frequency power
subplot(224)
contourf(EEG.times,frex,tf,40,'linecolor','none')
xlabel('Time'), ylabel('Frequency (Hz)')
title('Time-frequency plot')

%%

