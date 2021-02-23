function [simview,theta,psi1s,psi2s] = compute_otherflies_simview(trx,t,fly,varargin)

[num_ang_samples,theta,debug,simview_wall] = myparse(varargin,'num_ang_samples',960,...
  'theta',[],'debug',false,'simview_wall',[]);

if isempty(theta),
  theta0 = -pi;
  dtheta = pi/num_ang_samples*2;
  theta = theta0:dtheta:pi-dtheta;
  %theta = -pi:dtheta:pi-dtheta;
%   % if i want theta = 3*pi/2:-dtheta:-pi/2+dtheta
%   theta0 = 3*pi/2;
%   dtheta = -pi/num_ang_samples*2;
%   theta = theta0:dtheta:-pi/2-dtheta;
  
else
  num_ang_samples = numel(theta);
  theta0 = theta(1);
  dtheta = theta(2)-theta(1);
end
if isempty(simview_wall),
  simview = ones(1,num_ang_samples);
else
  simview = simview_wall;
end
idx = find([trx.firstframe]<=t & [trx.endframe]>= t);
assert(ismember(fly,idx));

i = t-trx(fly).firstframe+1;

% compute angle subtended by each other fly
psi1s = nan(1,numel(idx));
psi2s = nan(1,numel(idx));
for j1 = 1:numel(idx),
  ofly = idx(j1);
  if ofly == fly,
    continue;
  end
  oi = t-trx(ofly).firstframe+1;

  if any(isnan([trx(ofly).x_mm(oi),trx(ofly).y_mm(oi),...
      trx(ofly).a_mm(oi)*2,trx(ofly).b_mm(oi)*2,trx(ofly).theta_mm(oi)])),
    continue;
  end

  
  % here -pi and pi are behind the fly, 0 is in front, pi/2 to the left,
  % -pi/2 to the right -- I think this matches make_eye_filters
  [psi1,psi2] = anglesubtended_limits(trx(fly).x_mm(i),trx(fly).y_mm(i),...
    trx(fly).a_mm(i)*2,trx(fly).b_mm(i)*2,trx(fly).theta_mm(i),...
    trx(ofly).x_mm(oi),trx(ofly).y_mm(oi),...
    trx(ofly).a_mm(oi)*2,trx(ofly).b_mm(oi)*2,trx(ofly).theta_mm(oi));
  psi1s(j1) = psi1;
  psi2s(j1) = psi2;
%   dcurr = sqrt((trx(fly).x_mm(i)-trx(ofly).x_mm(oi)).^2 + ...
%     (trx(fly).y_mm(i)-trx(ofly).y_mm(oi)).^2);
  % convert to index
  psi1i = mod(round((psi1-theta0)/dtheta),num_ang_samples)+1;
  if psi2-psi1 >= 2*pi - abs(dtheta)/2,
    psi2i = psi1i - 1;
  else
    psi2i = mod(round((psi2-theta0)/dtheta),num_ang_samples)+1;
  end
    
%   if ~(argmin(abs(modrange(theta-psi1,-pi,pi))) == psi1i) || ...
%       ~(argmin(abs(modrange(theta-psi2,-pi,pi))) == psi2i),
%     keyboard;
%   end
  if dtheta < 0,
    tmp = psi2i;
    psi2i = psi1i;
    psi1i = tmp;
  end
  %psi1i = mod(round((psi1+pi)/dtheta),num_ang_samples)+1;
  %psi2i = mod(round((psi2+pi)/dtheta),num_ang_samples)+1;
  if psi1i <= psi2i,
    simview(psi1i:psi2i) = -1; % not using distance
  else
    simview(psi1i:end) = -1;
    simview(1:psi2i) = -1; 
  end
end

if debug,
  axlim = max(max(abs([trx.x_mm])),max(abs([trx.y_mm])));
  th = trx(fly).theta_mm(i)-pi/2;
  x1 = trx(fly).x_mm(i) + 2*trx(fly).a_mm(i)*cos(trx(fly).theta_mm(i));
  y1 = trx(fly).y_mm(i) + 2*trx(fly).a_mm(i)*sin(trx(fly).theta_mm(i));
  

  clf;
  hax = createsubplots(1,3,.05);

  axes(hax(1));
  drawflyo(trx(fly).x_mm(i),trx(fly).y_mm(i),trx(fly).theta_mm(i),trx(fly).a_mm(i),trx(fly).b_mm(i),'Color','k','LineWidth',2);
  hold on;
  colororder = get(hax(1),'ColorOrder');
  coloridx = 1;
  for j1 = 1:numel(idx),
    ofly = idx(j1);
    if ofly == fly,
      continue;
    end
    oi = t-trx(ofly).firstframe+1;
    drawflyo(trx(ofly).x_mm(oi),trx(ofly).y_mm(oi),trx(ofly).theta_mm(oi),trx(ofly).a_mm(oi),trx(ofly).b_mm(oi),'Color',colororder(coloridx,:));
    text(trx(ofly).x_mm(oi),trx(ofly).y_mm(oi),num2str(ofly),'Color',colororder(coloridx,:));
    coloridx = mod(coloridx,size(colororder,1))+1;
  end
  axis equal;
  set(gca,'XLim',axlim*[-1,1],'YLim',axlim*[-1,1])
  
  axes(hax(2));
  hold off;
  drawflyo(0,-2*trx(fly).a_mm(i),trx(fly).theta_mm(i)-th,trx(fly).a_mm(i),trx(fly).b_mm(i),'Color','k','LineWidth',2);
  hold on;
  
  coloridx = 1;
  for j1 = 1:numel(idx),
    ofly = idx(j1);
    if ofly == fly,
      continue;
    end
    oi = t-trx(ofly).firstframe+1;
    x = (trx(ofly).x_mm(oi)-x1)*cos(th) + (trx(ofly).y_mm(oi)-y1)*sin(th);
    y = (trx(ofly).y_mm(oi)-y1)*cos(th) - (trx(ofly).x_mm(oi)-x1)*sin(th);    
    h = ellipsedraw(2*trx(ofly).a_mm(oi),2*trx(ofly).b_mm(oi),x,y,trx(ofly).theta_mm(oi)-th);
    set(h,'Color',colororder(coloridx,:));
    dcurr = sqrt(x^2+y^2);
    plot([0,cos(psi1s(j1)+pi/2)*dcurr*1.05],[0,sin(psi1s(j1)+pi/2)*dcurr*1.05],'Color',colororder(coloridx,:));
    plot([0,cos(psi2s(j1)+pi/2)*dcurr*1.05],[0,sin(psi2s(j1)+pi/2)*dcurr*1.05],'Color',colororder(coloridx,:));
    coloridx = mod(coloridx,size(colororder,1))+1;
  end
  axis equal;
  
  axlims = axlim*[-1,-1;-1,1;1,-1;1,1];
  axlimsx = (axlims(:,1)-trx(fly).x_mm(i))*cos(th) + ...
    (axlims(:,2)-trx(fly).y_mm(i))*sin(th);
  axlimsy = (axlims(:,2)-trx(fly).y_mm(i))*cos(th) - ...
    (axlims(:,1)-trx(fly).x_mm(i))*sin(th);
  set(gca,'XLim',[min(axlimsx(:)),max(axlimsx(:))],'YLim',[min(axlimsy(:)),max(axlimsy(:))]);

  axes(hax(3));
  hold off;
  drawflyo(0,-2*trx(fly).a_mm(i),pi/2,trx(fly).a_mm(i),trx(fly).b_mm(i),'Color','k','LineWidth',2);
  hold on;
  rplot = 5;
  scatter(rplot*cos(pi/2+theta),rplot*sin(pi/2+theta),[],simview','filled','Parent',hax(3));
  coloridx = 1;
  for j1 = 1:numel(idx),
    ofly = idx(j1);
    if ofly == fly,
      continue;
    end
    plot([0,cos(psi1s(j1)+pi/2)*rplot,cos(psi2s(j1)+pi/2)*rplot,0],[0,sin(psi1s(j1)+pi/2)*rplot,sin(psi2s(j1)+pi/2)*rplot,0],'Color',colororder(coloridx,:));
    coloridx = mod(coloridx,size(colororder,1))+1;
  end
  
  axis equal;
  set(gca,'XLim',[-rplot-.1,rplot+.1],'YLim',[-rplot-.1,rplot+.1],'CLim',[-1,1]);

  
end
