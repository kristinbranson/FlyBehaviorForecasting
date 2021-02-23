function [Filt_resp,HR_resp,oma_angles,ts] = compute_otherflies_motion(td,varargin)

num_ang_samples = 960; % eye_filt has 960 points to sample

%if speed is 23 cm/sec then 0.1 cm step size should take 1/300 second
step_size = 0.05; % in cm
speed_sec = 23; % cm/sec
del_t = step_size/speed_sec;
lp_Tau_HR = 30e-3;
lp_Tau_O = 0.2;
nwallpixels = 72;
wallshade_amplitude = .5;
wallspeckle_amplitude = .5;
arena_radius = 65;

[num_ang_samples,del_t,lp_Tau_HR,lp_Tau_O,doplot,T0,T1,dosavevideo,flies,...
  nwallpixels,wallshade_amplitude,wallspeckle_amplitude,arena_radius] = ...
  myparse(varargin,...
  'num_ang_samples',num_ang_samples,...
  'del_t',del_t,...
  'lp_Tau_HR',lp_Tau_HR,...
  'lp_Tau_0',lp_Tau_O,...
  'doplot',true,...
  'T0',1,...
  'T1',inf,...
  'dosavevideo',false,...
  'flies',[],...
  'nwallpixels',nwallpixels,...
  'wallshade_amplitude',wallshade_amplitude,...
  'wallspeckle_amplitude',wallspeckle_amplitude,...
  'arena_radius',arena_radius);

if isempty(flies),
  flies = 1:numel(td.trx);
end

T0 = max(T0,min([td.trx.firstframe]));
T1 = min(T1,max([td.trx.endframe]));
assert(T1>=T0);

% EMD setups information
make_eye_filters; % create the 72 array of eye filters - 10 points per ommat.
num_receptors = size(eye_filt,2);

% eye_filt is 960 x 72
% eye_filt(i,j) is the sensitivity of ommatidium j to angle theta(i)
% thus, this will be a somewhat diagonalish matrix
% angles start at 3*pi/2 and decrease -- clockwise samples around the fly
% starting from directly behind it, and since the matrix is diagonalish
% this is also the case for oma ordering

% inititalize LP filter on HR output
h = del_t;
AA_lp = 1 - (2*lp_Tau_O)/h; BB_lp = 1 + (2*lp_Tau_O)/h;
% initializations for HR model
A_lp = 1 - (2*lp_Tau_HR)/h; B_lp = 1 + (2*lp_Tau_HR)/h;
% how many receptors per eye?
rec_pe = num_receptors/2; % currently assume same number per eye, 
% deal with separately if this is not the case

% resample data so that timestep difference is del_t
dt_video = median(diff(td.timestamps));
trxr = td.trx;
ts0all = (0:max([td.trx.endframe]))*dt_video;
ts1all = 0:del_t:ts0all(end);
fns = {'x','y','theta','a','b','xwingl','ywingl','xwingr','ywingr','x_mm','y_mm','a_mm','b_mm','theta_mm'};
for fly = 1:numel(trxr),
  firstframer = find(ts1all>=ts0all(td.trx(fly).firstframe),1);
  endframer = find(ts1all<=ts0all(td.trx(fly).endframe),1,'last');
  trxr(fly).firstframe = firstframer;
  trxr(fly).endframe = endframer;
  trxr(fly).off = 1-firstframer;
  trxr(fly).nframes = endframer-firstframer+1;
  trxr(fly).fps = 1/del_t;
  trxr(fly).timestamps = ts1all(firstframer:endframer);
  trxr(fly).dt = diff(trxr(fly).timestamps);
  
  ts0 = ts0all(td.trx(fly).firstframe:td.trx(fly).endframe);
  ts1 = ts1all(firstframer:endframer);
  for j = 1:numel(fns),
    fn = fns{j};
    if ismember(fn,{'theta','theta_mm'}),
      dtheta = modrange(diff(td.trx(fly).(fn),1,2),-pi,pi);
      theta = td.trx(fly).(fn)(1)+cumsum([0,dtheta]);
      trxr(fly).(fn)= interp1(ts0,theta,ts1);
    else
      trxr(fly).(fn)= interp1(ts0,td.trx(fly).(fn),ts1);
    end
  end
end

[~,T0r] = min(abs(ts1all-ts0all(T0)));
[~,T1r] = min(abs(ts1all-ts0all(T1)));
Tr = T1r-T0r+1;
ts = ts1all(T0r:T1r);

% initialize running means
InMat_1   = zeros(1,num_receptors);
FiltMat_1 = zeros(1,num_receptors);
HR_In_1 = zeros(1,num_receptors - 2); HR_Filt_1 = zeros(1,num_receptors - 2);

Filt_resp = nan(num_receptors,Tr,numel(flies));
HR_resp = nan(num_receptors-2,Tr,numel(flies));

hax = [];
% "pattern" on arena wall
pixel_vals = 1-max(0,min(2,wallshade_amplitude*sin(linspace(0,pi,nwallpixels)) + ...
  wallspeckle_amplitude*rand(1,nwallpixels)));
pixel_size = 2*pi/nwallpixels;
%pixel_vals = linspace(-1,1,nwallpixels);


theta0 = -pi;
dtheta = pi/num_ang_samples*2;
simangles = theta0:dtheta:pi-dtheta;
[~,oma_idx] = max(eye_filt,[],1);
oma_angles = simangles(oma_idx);

tic;
for flyi = 1:numel(flies),
  fly = flies(flyi);
  
  t0curr = max(trxr(fly).firstframe,T0r);
  t1curr = min(trxr(fly).endframe,T1r);

  for t = t0curr:t1curr,
    % theta = -pi:dtheta:pi-dtheta;
    % sim_angles = modrange(-pi/2-linspace(0,2*pi,numel(sim_views)+1),0,2*pi)';
    i = t-trxr(fly).firstframe+1;
    x1 = trxr(fly).x_mm(i) + 2*trxr(fly).a_mm(i)*cos(trxr(fly).theta_mm(i));
    y1 = trxr(fly).y_mm(i) + 2*trxr(fly).a_mm(i)*sin(trxr(fly).theta_mm(i));
    cur_circ_pos = sample_wall_positions_flyorama_circ(x1,y1,num_ang_samples,0,trxr(fly).theta_mm(i),arena_radius,simangles);
    current_indices = mod(round((cur_circ_pos-theta0)/pixel_size),nwallpixels)+1;
    wall_sim_views = pixel_vals(current_indices); % gets the values;

    [sim_views,~,psi1s,psi2s] = compute_otherflies_simview(trxr,t,fly,'num_ang_samples',num_ang_samples,'theta',simangles,...
      'simview_wall',wall_sim_views);
    
    filtered_views = sim_views*eye_filt;
    Filt_resp(:,t-T0r+1,flyi) = filtered_views;
    %sim_data(k).f_view(i,:) = filtered_views;
        
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
    HR_resp(:,t-T0r+1,flyi) = HR_Motion;

    % now lp filter the HR output
    HR_In = HR_Motion;
    HR_Filt = ( HR_In + HR_In_1 - AA_lp*HR_Filt_1 ) / BB_lp ;
    HR_In_1   = HR_In; HR_Filt_1 = HR_Filt;
    
    if doplot,
      updatehandles = t ~= t0curr;
      hax = PlotOtherFliesViews(trxr,fly,t,sim_views,psi1s,psi2s,pixel_vals,arena_radius,simangles,eye_filt,filtered_views,FiltMat,HR_Motion,hax,updatehandles);
      %hax = PlotOtherFliesViews(trxr,fly,t,wall_sim_views,psi1s,psi2s,pixel_vals,arena_radius,simangles,eye_filt,filtered_views,FiltMat,HR_Motion,hax,updatehandles);
      hfig = get(hax(1),'Parent');
      set(hfig,'Name',sprintf('fly = %d, time = %f s = %.2f fr',fly,ts1all(t),t*del_t/dt_video));
      %pause(.1);
      drawnow;
      if dosavevideo,
        if t == t0curr,
          vidobj = VideoWriter(sprintf('otherflies_simulation_fly%d_%s.avi',fly,datestr(now,'yyyymmddTHHMMSS')),'Motion JPEG AVI'); %#ok<TNMLP>
          vidobj.Quality = 90;
          open(vidobj);
        end
        writeVideo(vidobj,getframe(hfig));
        if t == t1curr,
          close(vidobj);
        end
      end
    else
      if toc > 10,
        fprintf('fly %d / %d, frame %d / %d\n',flyi,numel(flies),t-t0curr+1,t1curr-t0curr+1);
        tic;
      end
    end
    
  end
end
if false,
  expdir = '/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/GMR_71G01_AE_01_TrpA_Rig1Plate15BowlA_20120316T144027'; %#ok<UNRCH>
  trxfile = fullfile(expdir,'registered_trx.mat');
  td = load(trxfile);
  compute_otherflies_motion(td);
  compute_otherflies_motion(td,'T0',3776,'T1',4275,'flies',1,'dosavevideo',false,'doplot',true);
end

