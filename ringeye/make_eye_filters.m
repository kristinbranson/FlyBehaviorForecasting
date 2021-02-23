%% make_eye_filters.m

% now get eye projection - 
% simulate a 4.5 degree ommatidium

% rough approximation. follows from caption of Fig. 18, Buchner, 1981 (in Ali)
delta_rho = 5*pi/180; % angular half width
% to approx. interommatidial angle of 4.5, use 12 points, bc.
% 360/(960/12) is 4.5 degrees, close enough to 4.6

theta = -pi:pi/480:pi - pi/480;
% From Snyder (1979) as cited in Burton & Laughlin (2003)
filt = exp( -4.*log(2).*abs(theta).^2 ./ delta_rho^2 );
filt = filt./sum(filt);

eye_filt(:,37) = circshift(filt, [0 -1]);
cnt = 1;
for j = 38:72 % simulate 34 right side ommatidia
    eye_filt(:,j) = circshift(eye_filt(:,37), [cnt*12 0]);
    cnt = cnt + 1;
end

eye_filt(:,36) = filt; %circshift(filt, [0 -3]);
cnt = 1;
for j = 35:-1:1 % simulate 34 left side ommatidia
    eye_filt(:,j) = circshift(eye_filt(:,36), [-cnt*12 0]);
    cnt = cnt + 1;
end


%% make a nice plot of eye filters
% RLIMIT = [.02 0.24];
% RTICKS = [ 0 0.15];
% 
% h = mmpolar(repmat(theta, 72, 1)', [0.15 + eye_filt], 'Style','compass', 'TTickDelta', 10, ...
%     'RLimit', RLIMIT, 'TTickLabelVisible', 'off', 'Border', 'off', ...
%     'RGridLineStyle', '--', 'RTickValue', RTICKS, 'RTickLabelVisible', 'on', ...
%     'RGridLineWidth', 1.5);
% MR_thesis_plot_Helvetica(10, 'eye_map.eps');
% 
% 
% %% just for testing
% for j = 1:68
%     hold all
%     plot(eye_filt (:,j), '.')
%     [a,b] = max(eye_filt(:,j));
%     [j b]
%     pause
% end
