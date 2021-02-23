% this is the script that performs the basic simulation for optomotor resp
clear all
close all

% first we need to make the eye_filters and the patterns
make_eye_filters; % makes 72 receptor eye map
make_4_pix_wide_pattern_triple_res;
lp_tc = 30e-3;
sample_rate = 1000; % simulate 1000 samples per second

pause_time = 0.1;   num_pts_pause = pause_time*sample_rate;
OL_time = 1;      num_pts_OL = OL_time*sample_rate;
t = [1:(num_pts_pause + num_pts_OL)]/sample_rate;

num_X = size(Pats,3);
num_Y = size(Pats,4);

%imagesc(squeeze(Pats(:,:,1,:)))
exp_test_rates = [3 6 12 24];
num_speeds = length(exp_test_rates);

for j=1:num_speeds
    % first channel is constant, the FOE position
    frame_positions(1:(num_pts_pause + num_pts_OL),1) = 1;
    %second channel, pause first, then expansion at a given speed    
    frame_positions(1:num_pts_pause, 2) = -1;    
    ifi = exp_test_rates(j)/sample_rate; %inter-frame interv    
    frame_positions((num_pts_pause+1):(num_pts_pause+num_pts_OL),2) = ...
    mod( round( ([1:num_pts_OL] - 1)*ifi), num_Y) + 1;

    [sim_data(j).eye_sample, sim_data(j).HR_Motion] = OL_arena_simulation(eye_filt, Pats, frame_positions, sample_rate, lp_tc);
end

%% now calcualte a few things from the results
lp_taus = [0.05 0.1 0.2 0.4];
h = 1/sample_rate;
for j = 1:num_speeds
    HR_mean(j) = mean(mean(sim_data(j).HR_Motion(num_pts_pause+800:end,:),2));
    sim_data(j).HR_sum_R = sum(sim_data(j).HR_Motion(:,36:70)');
    sim_data(j).HR_sum_L = sum(sim_data(j).HR_Motion(:,1:25)');
    for k = 1:length(lp_taus)
        A_lp = 1 - (2*lp_taus(k))/h;
        B_lp = 1 + (2*lp_taus(k))/h;

        InMat_1 = 0; FiltMat_1 = 0;
        for i = 1:length(t)
            InMat = sim_data(j).HR_sum_R(i);
            FiltMat(i) = ( InMat + InMat_1 - A_lp*FiltMat_1 ) / B_lp ;  
            InMat_1   = InMat; FiltMat_1 = FiltMat(i);  
        end
        sim_data(j).filt(k).FM = FiltMat;
    end
end


figure(9)
for j = 1:num_speeds
    subplot(num_speeds, 1, j)
    plot(t, sim_data(j).HR_sum_R)
    hold all
    for k =1:length(lp_taus)
        plot(t, sim_data(j).filt(k).FM);
    end
    axis([0 t(end) 0 12])
    set(gca, 'Ytick', [0 12]);
    set(gca, 'Xtick', [0 1.1]);
end
xlabel('time (seconds)')
set(9, 'Position', [100 100 500 600])
%fixfig_subplot([1], [1 2 3 4], [0.9 3.5], 0.025, 0.03)
%MR_thesis_plot_Helvetica(11, 'EMD_response_lp_tc.eps');

% 
% set(gca,'xscale','log');
% xlim([0.2 30]);
% ylim([0 180]);
% set(gca, 'XTick', [0.2, 0.5, 1, 2, 3, 5, 10, 20]);
% set(gca, 'XTickLabel', [0.2, 0.5, 1, 2, 3, 5, 10, 20]);
% xlabel('temporal frequency (Hz) of optomotor stimulus');
% ylabel('EMD response');
% legend('6 pix', '8 pix', '12 pix', '16 pix');
% title('mean EMD response');
% box off