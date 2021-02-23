function [sim_data, mean_resp, X_positions, Y_positions, Th_positions] = run_flyorama_EMD_simulation_open_loop(...
    simD, del_t, eye_filt, lp_Tau_HR, lp_Tau_O, num_sims)
% function for running an OL EMD simulation in flyorama

doplot = true;
dosavevideo = true;

[num_ang_samples, num_receptors] = size(eye_filt);

% inititalize LP filter on HR output
h = del_t;
AA_lp = 1 - (2*lp_Tau_O)/h; BB_lp = 1 + (2*lp_Tau_O)/h;

% initializations for HR model
A_lp = 1 - (2*lp_Tau_HR)/h; B_lp = 1 + (2*lp_Tau_HR)/h;

% how many receptors per eye?
rec_pe = num_receptors/2; % currently assume same number per eye, 
% deal with separately if this is not the case

num_positions = simD.num_positions;
Y_pos = simD.Y_pos;
X_pos = simD.X_pos;
Th_pos = simD.Th_pos;
step_size = simD.step_size;

X_positions = zeros(num_positions,1); Y_positions = zeros(num_positions,1); 
Th_positions = zeros(num_positions,1);

if isfield(simD,'Th_dot')
    Th_dot = simD.Th_dot;
else
    Th_dot = 999;
end
    
% collect circle pos data, for all view angles, then try on different wall
% patterns.

for i = 1:num_positions  
    
    % get current eye positions
    if (Th_dot == 999)
        X_pos = X_pos + step_size*cos(Th_pos);
        Y_pos = Y_pos + step_size*sin(Th_pos);
    else
        X_pos = X_pos;
        Y_pos = Y_pos;
        Th_pos = Th_pos + Th_dot*del_t;
    end
    
    if (mod(i,100) == 0)
        [i X_pos Y_pos]
    end
    
    
    X_positions(i) = X_pos; Y_positions(i) = Y_pos; Th_positions(i) = Th_pos;
    % circle_pos(i,:) is where along the arena wall each sampled line of
    % sight intersects the arena when the fly is in the position defined by
    % X_pos, Y_pos, Th_pos
    % these correspond to clockwise samples around the fly starting from
    % directly behind it. 
    [circle_pos(i,:)] = sample_wall_positions_flyorama_circ(X_pos, Y_pos, num_ang_samples, 0, Th_pos);
    
end

%% now simulate views from circle_pos
for k = 1:num_sims
    if (mod(k,50) == 0)
        k
    end
    
    % 5 deg patterns -> 72 pixels
    pixel_size = 2*pi/72;
    %pattern_breaks = 0:1:71;
    %pixel_vals = repmat([1 1 -1 -1], 1, 18);
    %pixel_vals = repmat([1 -1], 1, 36);
    
    % this goes from 0 to 2*pi-epsilon ccw
    sim_data(k).pixel_vals = 1 - 2*(rand(1,72) > 0.5);
%     tmp = 1;
%     sim_data(k).pixel_vals(1) = -1;
%     j0 = 2;
%     while true,
%       val = -sim_data(k).pixel_vals(j0-1);
%       j1 = j0+tmp-1;
%       if j1 > 72,
%         dostop = true;
%         j1 = 72;
%       else
%         dostop = false;
%       end
%       sim_data(k).pixel_vals(j0:j1) = val;
%       tmp = tmp + 1;
%       j0 = j1+1;
%       if dostop,
%         break;
%       end
%     end
      
    % change from rand to uniform....gives much larger responses, and
    % movement of peak position more clear, but a much less fair
    % comparison to experimental..
    %if mod(k,2)
    %    sim_data(k).pixel_vals = mod(1:72,2)*2 - 1;
    %else
    %    sim_data(k).pixel_vals = mod(1:72,2)*-2 + 1;
    %end
    
    sim_data(k).HR_resp = zeros(num_positions, num_receptors - 2); % 70 EMDs
    InMat     = zeros(1,num_receptors);  InMat_1   = zeros(1,num_receptors);
    FiltMat   = zeros(1,num_receptors); FiltMat_1 = zeros(1,num_receptors);    

    HR_In_1 = zeros(1,num_receptors - 2); HR_Filt_1 = zeros(1,num_receptors - 2);
    
    % also shift the orientation by upto one pixel
    Th_shift = rand(1)*pixel_size;
    sim_data(k).num_shifts = -round(num_ang_samples*Th_shift./(2*pi));
        
    for i = 1:num_positions
        cur_circ_pos = circshift(circle_pos(i,:), [0 sim_data(k).num_shifts]);
        %current_indices = floor(circle_pos(i,:)./pixel_size) + 1;
        current_indices = floor(cur_circ_pos./pixel_size) + 1;

        sim_views = sim_data(k).pixel_vals(current_indices); % gets the values;
        filtered_views = sim_views*eye_filt;    
        sim_data(k).f_view(i,:) = filtered_views;
        
        % get image (fly's eye view)
        InMat = filtered_views;
        % compute HR motion - 
        % (x(t) + x(t-1) - a*F(t-1)) / b
        % = (x(t) + x(t-1) - a*( x(t-1) + x(t-2) - F(t-2) )/b )/b
        % = x(t)/b + (1/b + a/b^2)*x(t-1) + (a/b^2 + a^2/b^3) x(t-2) + ...
        % + a^(n-1)/b^n * (1+a/b) x(t-n) ...
        % a < 0, b > 0, |b| > |a|, 1 >> 1 + a/b > 0, (a/b)^n -> 0
        % let A = |a|
        % so this is
        % .035 x(t) + .0024 x(t-1) - .0023 x(t-2) + .0021 x(t-3) - .0020 x(t-4) ...
        % note that to use these constants, the sampling frequency h must
        % match 
        FiltMat = ( InMat + InMat_1 - A_lp*FiltMat_1 ) / B_lp ;
        InMat_1   = InMat; FiltMat_1 = FiltMat;                              
        %HR_Motion = (FiltMat(1:end-1).*InMat(2:end) - FiltMat(2:end).*InMat(1:end-1));
        % past(i) * current(i+1) - past(i+1) * current(i)
        HR_Motion(1:(rec_pe-1)) = (FiltMat(1:(rec_pe-1)).*InMat(2:rec_pe) - FiltMat(2:rec_pe).*InMat(1:(rec_pe-1)));    
        HR_Motion((rec_pe):(2*rec_pe - 2)) = -((FiltMat((rec_pe+2):end).*InMat((rec_pe+1):end-1) - FiltMat((rec_pe+1):end-1).*InMat((rec_pe+2):end)));        
        sim_data(k).HR_resp(i,:) = HR_Motion;

        % now lp filter the HR output
        HR_In = HR_Motion;
        sim_data(k).HR_Filt(i,:) = ( HR_In + HR_In_1 - AA_lp*HR_Filt_1 ) / BB_lp ;  
        HR_In_1   = HR_In; HR_Filt_1 = sim_data(k).HR_Filt(i,:);  
        %[exp_filter(i), cent_filter(i), combo_filter(i), FOE_est(i)] = turn_filter(scale_fact*HR_Filt(i,:));
        %Th_pos = Th_pos + combo_filter(i)*0.5;
        
        if doplot,
          PlotViews;
          set(hfig,'Name',sprintf('time = %d',i));
          %pause(.1);
          drawnow;
          if dosavevideo,
            if i == 1,
              vidobj = VideoWriter(sprintf('flyorama_emd_simulation_%d.avi',k),'Motion JPEG AVI');
              open(vidobj);
            end
            writeVideo(vidobj,getframe(hfig));
            if i == num_positions,
              close(vidobj);
            end
          end
        end
        
    end
end

HR_mean_resp = zeros(size(sim_data(1).HR_resp));
HR_mean_filt_resp = zeros(size(sim_data(1).HR_Filt));
for k = 1:num_sims
    HR_mean_resp = HR_mean_resp + sim_data(k).HR_resp;
    HR_mean_filt_resp = HR_mean_filt_resp + sim_data(k).HR_Filt;
%     figure
%     imagesc(sim_data(k).HR_resp)
end
mean_resp.HR_mean_resp = HR_mean_resp/num_sims;
mean_resp.HR_mean_filt_resp = HR_mean_filt_resp/num_sims;

