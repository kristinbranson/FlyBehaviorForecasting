function hax = PlotOtherFliesViews(trx,fly,t,simview,psi1s,psi2s,pixel_vals,arena_radius,theta,eye_filt,filtered_views,FiltMat,HR_Motion,hax,updatehandles)

persistent hflies hsub hsc axlim hqu
% pixel_angs = linspace(0,2*pi,numel(sim_data(k).pixel_vals)+1)';
% pixel_angs = pixel_angs(1:end-1);
% sim_angles = modrange(-pi/2-linspace(0,2*pi,numel(sim_views)+1),0,2*pi)';
% sim_angles = sim_angles(1:end-1);
doplotwall = ~isempty(pixel_vals);

if ~exist('hax','var') || ~all(ishandle(hax(:))) || numel(hax)~=6,
  hfig = gcf;
  clf(hfig);
  hax = createsubplots(2,3,.025);
  hax = reshape(hax,[2,3]);
  hax = hax';
  colormap(flipud(jet));
  updatehandles = false;
else
  if ~exist('updatehandles','var'),
    updatehandles = true;
  end
  updatehandles = updatehandles && ~isempty(hflies) && ...
    (get(hflies(1,1),'Parent') == hax(1)) && size(hflies,1)==numel(trx);
end

if ~updatehandles,
  hflies = gobjects(numel(trx),numel(hax));
  hsub = gobjects([numel(trx),2,numel(hax)]);
  hsc = gobjects(numel(hax),1);
end

if doplotwall,
  nwallpixels = numel(pixel_vals);
else
  nwallpixels = 72;
end
dtheta = 2*pi/numel(pixel_vals);
pixel_angs = (0:dtheta:2*pi-dtheta)'-pi/2;

idx = find([trx.firstframe]<=t & [trx.endframe]>= t);

i = t - trx(fly).firstframe + 1;
if ~updatehandles,
  axlim = max(max(abs([trx.x_mm])),max(abs([trx.y_mm])));
  if ~isempty(arena_radius),
    axlim = max(axlim,arena_radius);
  end
end
th = trx(fly).theta_mm(i)-pi/2;
x1 = trx(fly).x_mm(i) + 2*trx(fly).a_mm(i)*cos(trx(fly).theta_mm(i));
y1 = trx(fly).y_mm(i) + 2*trx(fly).a_mm(i)*sin(trx(fly).theta_mm(i));

x0 = arena_radius*cos(pixel_angs(:));
y0 = arena_radius*sin(pixel_angs(:));
c0 = pixel_vals(:);

if updatehandles,
  updatefly(hflies(fly,1),trx(fly).x_mm(i),trx(fly).y_mm(i),trx(fly).theta_mm(i),trx(fly).a_mm(i),trx(fly).b_mm(i));
  if doplotwall,
%     set(hsc(1),'XData',arena_radius*cos(pixel_angs),...
%       'YData',arena_radius*sin(pixel_angs),...
%       'CData',pixel_vals');
    UpdateInterpColorLine(hsc(1),'x',x0,'y',y0,'colors',c0);
  end

else
  hold(hax(1),'off');
  hflies(fly,1) = drawflyo(trx(fly).x_mm(i),trx(fly).y_mm(i),trx(fly).theta_mm(i),trx(fly).a_mm(i),trx(fly).b_mm(i),'Color','w','LineWidth',2,'Parent',hax(1));
  hold(hax(1),'on');
  if doplotwall,
    hsc(1) = PlotInterpColorLine(x0,y0,c0,[],'Parent',hax(1),'LineWidth',2);
    set(hsc(1),'EdgeColor','flat');
    %hsc(1) = scatter(arena_radius*cos(pixel_angs),arena_radius*sin(pixel_angs),[],pixel_vals','filled','Parent',hax(1));
    set(hax(1),'CLim',[-1,1]);
    colorbar('peer',hax(1));
  end
end

colororder = get(hax(1),'ColorOrder');
coloridx = 1;
for j1 = 1:numel(idx),
  ofly = idx(j1);
  if ofly == fly,
    continue;
  end
  oi = t-trx(ofly).firstframe+1;
  if updatehandles,
    updatefly(hflies(ofly,1),trx(ofly).x_mm(oi),trx(ofly).y_mm(oi),trx(ofly).theta_mm(oi),trx(ofly).a_mm(oi),trx(ofly).b_mm(oi));
  else    
    hflies(ofly,1) = drawflyo(trx(ofly).x_mm(oi),trx(ofly).y_mm(oi),trx(ofly).theta_mm(oi),trx(ofly).a_mm(oi),trx(ofly).b_mm(oi),'Color',colororder(coloridx,:),'Parent',hax(1));
    %text(trx(ofly).x_mm(oi),trx(ofly).y_mm(oi),num2str(ofly),'Color',colororder(coloridx,:),'Parent',hax(1));
  end
  coloridx = mod(coloridx,size(colororder,1))+1;
end
if ~updatehandles,
  axis(hax(1),'equal');
  set(hax(1),'XLim',axlim*[-1,1],'YLim',axlim*[-1,1])
  title(hax(1),'Trajectories');
end

x = (x0-x1)*cos(th) + (y0-y1)*sin(th);
y = (y0-y1)*cos(th) - (x0-x1)*sin(th);
axlimr = [min(x),max(x),min(y),max(y)];
if updatehandles,
  updatefly(hflies(fly,2),0,-2*trx(fly).a_mm(i),trx(fly).theta_mm(i)-th,trx(fly).a_mm(i),trx(fly).b_mm(i));
  %set(hsc(2),'XData',x,'YData',y,'CData',pixel_vals');
  UpdateInterpColorLine(hsc(2),'x',x,'y',y,'colors',c0);
else
  hold(hax(2),'off');
  hflies(fly,2) = drawflyo(0,-2*trx(fly).a_mm(i),trx(fly).theta_mm(i)-th,trx(fly).a_mm(i),trx(fly).b_mm(i),'Color','w','LineWidth',2,'Parent',hax(2));
  hold(hax(2),'on');
  hsc(2) = PlotInterpColorLine(x,y,c0,[],'Parent',hax(2),'LineWidth',2);
  set(hsc(2),'EdgeColor','flat');
end

coloridx = 1;
for j1 = 1:numel(idx),
  ofly = idx(j1);
  if ofly == fly,
    continue;
  end
  oi = t-trx(ofly).firstframe+1;
  x = (trx(ofly).x_mm(oi)-x1)*cos(th) + (trx(ofly).y_mm(oi)-y1)*sin(th);
  y = (trx(ofly).y_mm(oi)-y1)*cos(th) - (trx(ofly).x_mm(oi)-x1)*sin(th);
  
  if updatehandles,
    hflies(ofly,2) = ellipsedraw(2*trx(ofly).a_mm(oi),2*trx(ofly).b_mm(oi),x,y,trx(ofly).theta_mm(oi)-th,'-','hEllipse',hflies(ofly,2));
  else
    hflies(ofly,2) = ellipsedraw(2*trx(ofly).a_mm(oi),2*trx(ofly).b_mm(oi),x,y,trx(ofly).theta_mm(oi)-th,'-','Parent',hax(2));
    set(hflies(ofly,2),'Color',colororder(coloridx,:));
  end
  dcurr = sqrt(x^2+y^2);
  if updatehandles,
    set(hsub(ofly,1,2),'XData',[0,cos(psi1s(j1)+pi/2)*dcurr*1.05],...
      'YData',[0,sin(psi1s(j1)+pi/2)*dcurr*1.05]);
    set(hsub(ofly,2,2),'XData',[0,cos(psi2s(j1)+pi/2)*dcurr*1.05],...
      'YData',[0,sin(psi2s(j1)+pi/2)*dcurr*1.05]);
  else
    hsub(ofly,1,2) = plot(hax(2),[0,cos(psi1s(j1)+pi/2)*dcurr*1.05],[0,sin(psi1s(j1)+pi/2)*dcurr*1.05],'Color',colororder(coloridx,:));
    hsub(ofly,2,2) = plot(hax(2),[0,cos(psi2s(j1)+pi/2)*dcurr*1.05],[0,sin(psi2s(j1)+pi/2)*dcurr*1.05],'Color',colororder(coloridx,:));
  end
  coloridx = mod(coloridx,size(colororder,1))+1;
end
if ~updatehandles,
  axis(hax(2),'equal');
  title(hax(2),'Angles subtended');
  set(hax(2),'CLim',[-1,1]);
  colorbar('peer',hax(2));
end
% axlims = axlim*[-1,-1;-1,1;1,-1;1,1];
% axlimsx = (axlims(:,1)-x1)*cos(th) + ...
%   (axlims(:,2)-y1)*sin(th);
% axlimsy = (axlims(:,2)-y1)*cos(th) - ...
%   (axlims(:,1)-x1)*sin(th);
% set(hax(2),'XLim',[min(axlimsx(:)),max(axlimsx(:))],'YLim',[min(axlimsy(:)),max(axlimsy(:))]);
%set(hax(2),'XLim',axlimr(1:2),'YLim',axlimr(3:4));
set(hax(2),'XLim',axlim*[-1,1]*2,'YLim',axlim*[-1,1]*2);
if updatehandles,
  updatefly(hflies(fly,3),0,-2*trx(fly).a_mm(i),pi/2,trx(fly).a_mm(i),trx(fly).b_mm(i));
else
  hold(hax(3),'off');
  hflies(fly,3) = drawflyo(0,-2*trx(fly).a_mm(i),pi/2,trx(fly).a_mm(i),trx(fly).b_mm(i),'Color','w','LineWidth',2,'Parent',hax(3));
  hold(hax(3),'on');
end
rplot = 5;
if updatehandles,
  set(hsc(3),'XData',rplot*cos(pi/2+theta),...
    'YData',rplot*sin(pi/2+theta),...
    'CData',simview');
else
  hsc(3) = scatter(rplot*cos(pi/2+theta),rplot*sin(pi/2+theta),[],simview','filled','Parent',hax(3));
end
coloridx = 1;
for j1 = 1:numel(idx),
  ofly = idx(j1);
  if ofly == fly,
    continue;
  end
  if updatehandles,
    set(hsub(ofly,1,3),'XData',[0,cos(psi1s(j1)+pi/2)*rplot,cos(psi2s(j1)+pi/2)*rplot,0],...
      'YData',[0,sin(psi1s(j1)+pi/2)*rplot,sin(psi2s(j1)+pi/2)*rplot,0]);
  else
    hsub(ofly,1,3) = plot(hax(3),[0,cos(psi1s(j1)+pi/2)*rplot,cos(psi2s(j1)+pi/2)*rplot,0],[0,sin(psi1s(j1)+pi/2)*rplot,sin(psi2s(j1)+pi/2)*rplot,0],'Color',colororder(coloridx,:));
  end
  coloridx = mod(coloridx,size(colororder,1))+1;
end
if ~updatehandles,
  axis(hax(3),'equal');
  set(hax(3),'XLim',[-rplot-.1,rplot+.1],'YLim',[-rplot-.1,rplot+.1],'CLim',[-1,1]);
  title(hax(3),'Sim view');
  colorbar('peer',hax(3));
end

[~,oma_idx] = max(eye_filt,[],1);
oma_angles = theta(oma_idx);

if ~updatehandles,
  hold(hax(4),'off');
  hcirc(4) = drawellipse(0,0,0,rplot,rplot,'b-','Parent',hax(4));
  axis(hax(4),'equal');
  hold(hax(4),'on');
end
if updatehandles,
  updatefly(hflies(fly,4),0,-2*trx(fly).a_mm(i),pi/2,trx(fly).a_mm(i),trx(fly).b_mm(i));
  set(hsc(4),'XData',rplot*cos(pi/2+oma_angles),...
    'YData',rplot*sin(pi/2+oma_angles),...
    'CData',filtered_views');
else
  hflies(fly,4) = drawflyo(0,-2*trx(fly).a_mm(i),pi/2,trx(fly).a_mm(i),trx(fly).b_mm(i),'Color','w','LineWidth',2,'Parent',hax(4));
  hsc(4) = scatter(rplot*cos(pi/2+oma_angles),rplot*sin(pi/2+oma_angles),[],filtered_views','filled','Parent',hax(4));
end

coloridx = 1;
for j1 = 1:numel(idx),
  ofly = idx(j1);
  if ofly == fly,
    continue;
  end
  if updatehandles,
    set(hsub(ofly,1,4),'XData',[0,cos(psi1s(j1)+pi/2)*rplot,cos(psi2s(j1)+pi/2)*rplot,0],...
      'YData',[0,sin(psi1s(j1)+pi/2)*rplot,sin(psi2s(j1)+pi/2)*rplot,0]);
  else
    hsub(ofly,1,4) = plot(hax(4),[0,cos(psi1s(j1)+pi/2)*rplot,cos(psi2s(j1)+pi/2)*rplot,0],[0,sin(psi1s(j1)+pi/2)*rplot,sin(psi2s(j1)+pi/2)*rplot,0],'Color',colororder(coloridx,:));
  end
  coloridx = mod(coloridx,size(colororder,1))+1;
end
if ~updatehandles,
  colorbar('peer',hax(4));
  set(hax(4),'XLim',[-rplot-.1,rplot+.1],'YLim',[-rplot-.1,rplot+.1],'CLim',[-1,1]);
  title(hax(4),'Ommatidia','interpreter','none');
end

if ~updatehandles,
  hold(hax(5),'off');
  hcirc(5) = drawellipse(0,0,0,rplot,rplot,'b-','Parent',hax(5));
  axis(hax(5),'equal');
  hold(hax(5),'on');
end
if updatehandles,
  updatefly(hflies(fly,5),0,-2*trx(fly).a_mm(i),pi/2,trx(fly).a_mm(i),trx(fly).b_mm(i));
  set(hsc(5),'XData',rplot*cos(pi/2+oma_angles),'YData',rplot*sin(pi/2+oma_angles),...
    'CData',FiltMat');
else
  hflies(fly,5) = drawflyo(0,-2*trx(fly).a_mm(i),pi/2,trx(fly).a_mm(i),trx(fly).b_mm(i),'Color','w','LineWidth',2,'Parent',hax(5));
  hsc(5) = scatter(rplot*cos(pi/2+oma_angles),rplot*sin(pi/2+oma_angles),[],FiltMat','filled','Parent',hax(5));
  title(hax(5),'Time average','interpreter','none');
  colorbar('peer',hax(5));
  set(hax(5),'XLim',[-rplot-.1,rplot+.1],'YLim',[-rplot-.1,rplot+.1],'CLim',[-1,1]);
end

xo = rplot*cos(pi/2+oma_angles(2:end-1));
yo = rplot*sin(pi/2+oma_angles(2:end-1));
dx = -sin(pi/2+oma_angles(2:end-1));
dy = cos(pi/2+oma_angles(2:end-1));
% xu = xo + dx.*HR_Motion;
% yv = yo + dy.*HR_Motion;
% xd = [xo;xu;nan(size(xo))];
% yd = [yo;yv;nan(size(yo))];
% c = repmat(HR_Motion,[3,1]);
idxpos = HR_Motion >= 0;

if ~updatehandles,
  hold(hax(6),'off');
  drawellipse(0,0,0,rplot,rplot,'b-','Parent',hax(6));
  axis(hax(6),'equal');
  hold(hax(6),'on');
  set(hax,'Color','k');
end
if updatehandles,
  updatefly(hflies(fly,6),0,-2*trx(fly).a_mm(i),pi/2,trx(fly).a_mm(i),trx(fly).b_mm(i));
  set(hsc(6),'XData',rplot*cos(pi/2+oma_angles(2:end-1)),...
    'YData',rplot*sin(pi/2+oma_angles(2:end-1)),...
    'CData',HR_Motion');
  %UpdateInterpColorLine(hqu,'x',xd(:),'y',yd(:),'colors',c(:));
  set(hqu(1),'XData',xo(idxpos),'YData',yo(idxpos),'UData',dx(idxpos).*HR_Motion(idxpos),'VData',dy(idxpos).*HR_Motion(idxpos));
  set(hqu(2),'XData',xo(~idxpos),'YData',yo(~idxpos),'UData',dx(~idxpos).*HR_Motion(~idxpos),'VData',dy(~idxpos).*HR_Motion(~idxpos));
else
  hflies(fly,6) = drawflyo(0,-2*trx(fly).a_mm(i),pi/2,trx(fly).a_mm(i),trx(fly).b_mm(i),'Color','w','LineWidth',2,'Parent',hax(6));
  hqu(1) = quiver(xo(idxpos),yo(idxpos),dx(idxpos).*HR_Motion(idxpos),dy(idxpos).*HR_Motion(idxpos),0,'Color',[.75,.75,1],'Parent',hax(6),'LineWidth',2);
  hqu(2) = quiver(xo(~idxpos),yo(~idxpos),dx(~idxpos).*HR_Motion(~idxpos),dy(~idxpos).*HR_Motion(~idxpos),0,'Color',[1,.75,.75],'Parent',hax(6),'LineWidth',2);
  %hqu = PlotInterpColorLine(xd(:),yd(:),c(:),[],'Parent',hax(6),'LineWidth',2);
  %set(hqu,'EdgeColor','flat');
  hsc(6) = scatter(rplot*cos(pi/2+oma_angles(2:end-1)),rplot*sin(pi/2+oma_angles(2:end-1)),[],HR_Motion','.','Parent',hax(6));
  title(hax(6),'HR Motion','interpreter','none');
  colorbar('peer',hax(6));
  set(hax(6),'XLim',[-rplot-.1,rplot+.1],'YLim',[-rplot-.1,rplot+.1],'CLim',[-.3,.3]);
end