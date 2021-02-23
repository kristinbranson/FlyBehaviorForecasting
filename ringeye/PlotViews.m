arena_radius = 50;

pixel_angs = linspace(0,2*pi,numel(sim_data(k).pixel_vals)+1)';
pixel_angs = pixel_angs(1:end-1);
sim_angles = modrange(-pi/2-linspace(0,2*pi,numel(sim_views)+1),0,2*pi)';
sim_angles = sim_angles(1:end-1);
oma_idx = round(sum(eye_filt.*(1:960)',1)./sum(eye_filt,1));
oma_angles = sim_angles(oma_idx);

if ~exist('hax','var') || ~all(ishandle(hax(:))),
  hfig = gcf;
  clf(hfig);
  hax = createsubplots(2,3,.025);
  hax = reshape(hax,[2,3]);
  hax = hax';
end

axes(hax(1));
hold(hax(1),'off');
plot(hax(1),X_positions(i),Y_positions(i),'bo');
axis(hax(1),'equal');
hold(hax(1),'on');
plot(hax(1),X_positions(i)+[0,cos(Th_positions(i))]*2,Y_positions(i)+[0,sin(Th_positions(i))]*2,'b-');
drawellipse(X_positions(i),Y_positions(i),0,5,5,'b-','Parent',hax(1));
drawellipse(0,0,0,arena_radius,arena_radius,'Parent',hax(1));
scatter(arena_radius*cos(pixel_angs),arena_radius*sin(pixel_angs),[],sim_data(k).pixel_vals','filled','Parent',hax(1));
title(hax(1),'pixel_vals','interpreter','none');
colorbar('peer',hax(1));
set(hax(1),'CLim',[-1,1]);

hold(hax(2),'off');
drawellipse(0,0,0,5,5,'b-','Parent',hax(2));
axis(hax(2),'equal');
hold(hax(2),'on');
plot(hax(2),0,0,'bo');
plot(hax(2),[0,cos(pi/2)]*2,[0,sin(pi/2)]*2,'b-');
scatter(5*cos(sim_angles),5*sin(sim_angles),[],sim_views','filled','Parent',hax(2));
title(hax(2),'sim_views','interpreter','none');
colorbar('peer',hax(2));
set(hax(2),'CLim',[-1,1]);

hold(hax(3),'off');
drawellipse(0,0,0,5,5,'b-','Parent',hax(3));
axis(hax(3),'equal');
hold(hax(3),'on');
plot(hax(3),0,0,'bo');
plot(hax(3),[0,cos(pi/2)]*2,[0,sin(pi/2)]*2,'b-');
% oma_angles = modrange(-pi/2-linspace(0,2*pi,numel(filtered_views)+1),0,2*pi)';
% oma_angles = oma_angles(1:end-1);
scatter(5*cos(oma_angles),5*sin(oma_angles),[],filtered_views','filled','Parent',hax(3));
colorbar('peer',hax(3));
set(hax(3),'CLim',[-1,1]);
title(hax(3),'filtered_views','interpreter','none');

hold(hax(4),'off');
drawellipse(0,0,0,5,5,'b-','Parent',hax(4));
axis(hax(4),'equal');
hold(hax(4),'on');
plot(hax(4),0,0,'bo');
plot(hax(4),[0,cos(pi/2)]*2,[0,sin(pi/2)]*2,'b-');
scatter(5*cos(oma_angles),5*sin(oma_angles),[],FiltMat','filled','Parent',hax(4));
title(hax(4),'FiltMat','interpreter','none');
colorbar('peer',hax(4));
set(hax(4),'CLim',[-1,1]);

hold(hax(5),'off');
drawellipse(0,0,0,5,5,'b-','Parent',hax(5));
axis(hax(5),'equal');
hold(hax(5),'on');
plot(hax(5),0,0,'bo');
plot(hax(5),[0,cos(pi/2)]*2,[0,sin(pi/2)]*2,'b-');
scatter(5*cos(oma_angles(2:end-1)),5*sin(oma_angles(2:end-1)),[],HR_Motion','filled','Parent',hax(5));
title(hax(5),'HR_Motion','interpreter','none');
colorbar('peer',hax(5));
set(hax(5),'CLim',[-.3,.3]);


hold(hax(6),'off');
drawellipse(0,0,0,5,5,'b-','Parent',hax(6));
axis(hax(6),'equal');
hold(hax(6),'on');
plot(hax(6),0,0,'bo');
plot(hax(6),[0,cos(pi/2)]*2,[0,sin(pi/2)]*2,'b-');
scatter(5*cos(oma_angles(2:end-1)),5*sin(oma_angles(2:end-1)),[],sim_data(k).HR_Filt(i,:)','filled','Parent',hax(6));
title(hax(6),'HR_Filt','interpreter','none');
colorbar('peer',hax(6));
set(hax(6),'CLim',[-.3,.3]);
