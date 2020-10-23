%%
%     COURSE: Solved problems in neural time series analysis
%    SECTION: Time-domain analyses
%      VIDEO: Project 2-2: ERP peak latency topoplot
% Instructor: sincxpress.com
%
%%

%% Loop through each channel and find the peak time of the ERP between 100 and 400 ms. 
%   Store these peak times in a separate variable, and then make a
%   topographical plot of the peak times. Repeat for a low-pass filtered ERP.

load sampleEEGdata.mat

%%
