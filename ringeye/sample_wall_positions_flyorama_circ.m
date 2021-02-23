% find where sight lines from the fly intersect the circular arena for each
% sampled direction. directions sampled are from 0 to 2*pi evenly for
% num_ang_samples with a direction defined by cur_Th (I think it starts
% behind the fly).
% circle_pos is the angle in the arena's coordinate system of the
% intersection point
function [circle_pos] = sample_wall_positions_flyorama_circ...
    (X_position, Y_position, num_ang_samples, p_val, cur_Th, arena_radius, theta)
% same as other, but this one only returns the circ position, and also this
% function shifts the image using cur_Th;
%% simulate the fly-o-rama specs

if ~exist('arena_radius','var'),
  arena_radius = 50; % in units of cm
end

isthetain = true;
if ~exist('theta','var'),
  isthetain = false;
  theta = 0:(2*pi)/num_ang_samples:2*pi - (1/num_ang_samples);
  one_quarter = (num_ang_samples/4);
  theta = [theta(one_quarter + 1:end) theta(1:one_quarter)];
  % theta starts at behind then goes counterclockwise, so flip this -
  theta = fliplr(theta);
end

if p_val
    figure(2)
    X_arena_vals = arena_radius*cos(-pi:0.01:pi);
    Y_arena_vals = arena_radius*sin(-pi:0.01:pi);
    plot(X_arena_vals, Y_arena_vals);
    hold all
end

for j = 1:length(theta)
    % intersction of circle with origin [0,0], radius arena_radius and line 
    % through point X_position, Y_position in direction theta. num_int is
    % the number of intersections and p are these intersection points 
    [num_int, p] = circle_imp_line_par_int_2d ( arena_radius, [0 0], X_position, Y_position, cos(theta(j)), sin(theta(j)) ); 
    x_circle_pos(j) = p(1,1);
    y_circle_pos(j) = p(2,1);
    if p_val
        plot([X_position p(1,1)], [Y_position p(2,1)]);
        axis equal
    end
end

circle_positions = (atan2(y_circle_pos, x_circle_pos)); % relative to origin
circle_pos = mod((circle_positions)*180/pi, 360)*pi/180; %% all angles are zero to 2*pi
circle_pos(find(circle_pos == 2*pi)) = 0;

if isthetain,
  cur_Th = -pi/2 - mod(cur_Th, 2*pi); % keep it between 0 and 2*pi;
  num_shifts = round(num_ang_samples*cur_Th./(2*pi));
else
  cur_Th = pi/2 - mod(cur_Th, 2*pi); % keep it between 0 and 2*pi;
  num_shifts = -round(num_ang_samples*cur_Th./(2*pi));
end
circle_pos = circshift(circle_pos, [0 num_shifts]);
