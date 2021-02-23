%% generate position on circle data based on just geometry
% close all
% clear all

%if speed is 23 cm/sec then 0.1 cm step size should take 1/300 second
step_size = 0.05; % in cm
speed_sec = 23; % cm/sec
del_t = step_size/speed_sec
num_ang_samples = 960; % eye_filt has 960 points to sample

% EMD setups information
make_eye_filters; % create the 72 array of eye filters - 10 points per ommat.
% eye_filt is 960 x 72
% eye_filt(i,j) is the sensitivity of ommatidium j to angle theta(i)
% thus, this will be a somewhat diagonalish matrix
% angles start at 3*pi/2 and decrease -- clockwise samples around the fly
% starting from directly behind it, and since the matrix is diagonalish
% this is also the case for oma ordering

lp_Tau_HR = 30e-3;
lp_Tau_O = 0.2;

% first simulation, centered path
simD.Y_pos = -15; 
simD.X_pos = 0;
simD.Th_pos = pi/2; % pointing up (I hope)
simD.num_positions = 1201; %(go from -15 to + 45, so 60/0.05 = 1200 + a few)
simD.step_size = step_size;

num_sims = 1;

[sim_data1, mean_resp1, X_positions1, Y_positions1] = run_flyorama_EMD_simulation_open_loop(...
    simD, del_t, eye_filt, lp_Tau_HR, lp_Tau_O, num_sims);

% figure(1)
% imagesc(mean_resp.HR_mean_resp)

%% SECOND SIMULATION
clear simD
step_size = 0.05; % in cm

simD.Y_pos = -15; 
simD.X_pos = 10; %20;
simD.Th_pos = pi/2; % pointing up (I hope)
simD.num_positions = 1201;
simD.step_size = step_size;

num_sims = 500;

[sim_data2, mean_resp2, X_positions2, Y_positions2] = run_flyorama_EMD_simulation_open_loop(...
    simD, del_t, eye_filt, lp_Tau_HR, lp_Tau_O, num_sims);

% figure(1)
% imagesc(mean_resp.HR_mean_resp)


%% THIRD SIMULATION
simD.Y_pos = -15; 
simD.X_pos = 20;
simD.Th_pos = pi/2; % pointing up (I hope)
simD.num_positions = 1201;
simD.step_size = step_size;

num_sims = 500;

[sim_data3, mean_resp3, X_positions3, Y_positions3] = run_flyorama_EMD_simulation_open_loop(...
    simD, del_t, eye_filt, lp_Tau_HR, lp_Tau_O, num_sims);

save EMD_flight_sim sim_data1 mean_resp1 X_positions1 Y_positions1 ...
    sim_data2 mean_resp2 X_positions2 Y_positions2...
    sim_data3 mean_resp3 X_positions3 Y_positions3


%% plotting for sim 1

% generate some colors, red to black
num_pts = 6;
start_pt = 0.33; % the point at which we start blending black with the other color
step_size = (1-start_pt)/(num_pts - 2);
CMAP = flipud([[0 start_pt:step_size:1]', zeros(num_pts,1), zeros(num_pts,1)]);

axis_lims = [1 70 -0.31 0.31];

plot_pts = [100:200:1201];
%plot_pts = [200:200:1201];
%plot_pts = [151:150:1081];
%plot_pts = [111 201:100:1001 1081];
figure(4)
subplot(131)
arena_radius = 50; % in units of cm
X_arena_vals = arena_radius*cos(-pi:0.01:pi);
Y_arena_vals = arena_radius*sin(-pi:0.01:pi);
plot(X_arena_vals, Y_arena_vals, 'r', 'LineWidth', 1.5)
hold on
plot(X_positions1, Y_positions1);
for j = 1:length(plot_pts)
    plot(X_positions1(plot_pts(j)), Y_positions1(plot_pts(j)), '.', 'MarkerSize', 20, 'color', CMAP(j,:));
end

circ_50 = 18.8; % 50% saccade location
circ_75 = 26.076; % 75% saccade location
X_50 = circ_50*cos(-pi:0.01:pi);
Y_50 = circ_50*sin(-pi:0.01:pi);
X_75 = circ_75*cos(-pi:0.01:pi);
Y_75 = circ_75*sin(-pi:0.01:pi);

%plot(X_50, Y_50, 'k', 'LineWidth', 0.75)
plot(X_75, Y_75, 'k', 'LineWidth', 0.75)
axis square
axis off

subplot(133)
imagesc(-flipud(mean_resp1.HR_mean_filt_resp)) % minus sign is for MD who wanted a flipped color scheme?
caxis([-0.3 0.3]);
set(gca, 'XTick', [1 18 35.5 53 70]);
set(gca, 'XTickLabel', {'rear', 'left', 'front', 'right', 'rear'});
set(gca, 'YTick', [plot_pts]);
set(gca, 'YTickLabel', [1:6]);

subplot(132)
for j = 1:length(plot_pts)
    plot(mean_resp1.HR_mean_filt_resp(plot_pts(j),:)', 'color', CMAP(j,:))
    hold on
    %pause
end
axis(axis_lims);
set(gca, 'XTick', [1 18 35.5 53 70]);
set(gca, 'XTickLabel', {'rear', 'left', 'front', 'right', 'rear'});
set(gca, 'YTick', [-0.3 0 0.3]);
plot(xlim, 0.1347*[1 1], 'k', xlim, -0.1347*[1 1], 'k', xlim, 0.1905*[1 1], 'k', xlim, -0.1905*[1 1], 'k', xlim, 0*[1 1], 'k');

set(4,'Position', [100 400 950 300]);
%fixfig_subplot([1 2], [1 2], [1 1], 0.03, 0.0175)
%MR_thesis_plot_Helvetica(10, 'EMD_flight1.eps');

% make some plots for sim 2
%plot_pts = [200:200:1201];
%plot_pts = [151:150:1001];
%plot_pts = [111 201:100:1001];
figure(5)
subplot(131)
arena_radius = 50; % in units of cm
plot(X_arena_vals, Y_arena_vals, 'r', 'LineWidth', 1.5)
hold on
plot(X_positions2, Y_positions2);
for j = 1:length(plot_pts)
    plot(X_positions2(plot_pts(j)), Y_positions2(plot_pts(j)), '.', 'MarkerSize', 20, 'color', CMAP(j,:));
end

%plot(X_50, Y_50, 'k', 'LineWidth', 0.75)
plot(X_75, Y_75, 'k', 'LineWidth', 0.75)
axis square
axis off

subplot(133)
imagesc(-flipud(mean_resp2.HR_mean_filt_resp))
caxis([-0.3 0.3]);
set(gca, 'XTick', [1 18 35.5 53 70]);
set(gca, 'XTickLabel', {'rear', 'left', 'front', 'right', 'rear'});
set(gca, 'YTick', [plot_pts]);
set(gca, 'YTickLabel', [1:6]);

subplot(132)
for j = 1:length(plot_pts)
    plot(mean_resp2.HR_mean_filt_resp(plot_pts(j),:)', 'color', CMAP(j,:))
    hold on
    %pause
end
axis(axis_lims);
set(gca, 'XTick', [1 18 35.5 53 70]);
set(gca, 'XTickLabel', {'rear', 'left', 'front', 'right', 'rear'});
set(gca, 'YTick', [-0.3 0 0.3]);
plot(xlim, 0.1347*[1 1], 'k', xlim, -0.1347*[1 1], 'k', xlim, 0.1905*[1 1], 'k', xlim, -0.1905*[1 1], 'k', xlim, 0*[1 1], 'k');

set(5,'Position', [100 300 950 300]);
%fixfig_subplot([1 2], [1 2], [1 1], 0.03, 0.0175)
%MR_thesis_plot_Helvetica(10, 'EMD_flight2.eps');

% plots for sim3
%plot_pts = [151:150:901];
%plot_pts = [200:200:1201];

%plot_pts = [111 201:100:801];
figure(6)
subplot(131)
arena_radius = 50; % in units of cm
plot(X_arena_vals, Y_arena_vals, 'r', 'LineWidth', 1.5)
hold on
plot(X_positions3, Y_positions3);
for j = 1:length(plot_pts)
    plot(X_positions3(plot_pts(j)), Y_positions3(plot_pts(j)), '.', 'MarkerSize', 20, 'color', CMAP(j,:));
    hold all
end

%plot(X_50, Y_50, 'k', 'LineWidth', 0.75)
plot(X_75, Y_75, 'k', 'LineWidth', 0.75)
axis square
axis off

subplot(133)
imagesc(-flipud(mean_resp3.HR_mean_filt_resp))
caxis([-0.3 0.3]);
set(gca, 'XTick', [1 18 35.5 53 70]);
set(gca, 'XTickLabel', {'rear', 'left', 'front', 'right', 'rear'});
set(gca, 'YTick', [plot_pts]);
set(gca, 'YTickLabel', [1:6]);

subplot(132)
for j = 1:length(plot_pts)
    plot(mean_resp3.HR_mean_filt_resp(plot_pts(j),:)', 'color', CMAP(j,:))
    hold on
    %pause
end
axis(axis_lims);
plot(xlim, 0.1347*[1 1], 'k', xlim, -0.1347*[1 1], 'k', xlim, 0.1905*[1 1], 'k', xlim, -0.1905*[1 1], 'k', xlim, 0*[1 1], 'k');
set(gca, 'XTick', [1 18 35.5 53 70]);
set(gca, 'XTickLabel', {'rear', 'left', 'front', 'right', 'rear'});
set(gca, 'YTick', [-0.3 0 0.3]);

set(6,'Position', [100 200 950 300]);
%fixfig_subplot([1 2], [1 2], [1 1], 0.03, 0.0175)
%MR_thesis_plot_Helvetica(10, 'EMD_flight3.eps');

% figure(7)
% imagesc(mean_resp3.HR_mean_filt_resp)
% caxis([-0.3 0.3]);
% colorbar



