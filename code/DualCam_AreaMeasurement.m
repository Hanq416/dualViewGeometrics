clear all; %#ok<*CLALL>
close all;
clc;
% Area measurement assistant, dual Canon+Sigma DSLR system.
% Hankun Li, University of Kansas, 04/11/2021
% Note: image processing tool box and computer vision tool box required to
% using auto detection, for manual function: CV is not required.
%% Camera value...
sx = 22.3; sy = 14.9; % camera sensor size
f = 4.5; % lens focal length
Lfov = 180; % lens fov
% Camera value end...
% Notes: Sigma lens is an equisolidangle projection lens

%% step1: load image
[imEL,imER] = imgLoad(Lfov,f,sy);

%% step2: find target, define the point of shape
% currently only support triangle and parallelogram
% note: change [fov] to zoom, fov range:[30-180], Canon
fov = 90;
[tarSet,sides] = targetFind(imEL,imER,fov);

%% step3: calculate distance, get coordinate in the real 3d space (! not image space)
% Initial compensation for dual camera system... 
% if both cameras were not paralle installed !
% if paralle installed, skip this step, put zero or quote them below
leftHC = -1.5; % in deg, optional
rightHC = -8.5; % in deg, optional
% ...end...
camD = 2.0; % camera distance, in ft, required
pointSet = disCalculate(camD, tarSet, sides, leftHC, rightHC);

%% step4: get area of a parallelogram or triangle
[area,proj_area] = getArea(pointSet);

%% [dev function 1]image tape-measure
[~] = imgTape(pointSet(1),pointSet(2)); % change point number

%% [dev function 2] peojected area the on image plane
[~] = CameraProjectedArea(proj_area,pointSet,f);

%% [dev function 3] get rolling angle of projected surface
[~] = getProjRoll(pointSet);

%% [dev function 4] using left camera only to measure light
[~] = getArea_leftMajorCam(pointSet,camD);

%% main function end here.
% TEST FUNCTION CAN BE WRITE HERE


















% END of Main
%% Functions
% dev function
function [area,proj_area] = getArea_leftMajorCam(pointSet,camD)
i = 1;
while i <= size(pointSet,2)
    a(i) = pointSet(i).Results.alpha;
    b(i) = pointSet(i).Results.beta;
    d(i) = pointSet(i).Results.distance;
    coordinate3(i,:) = [-cosd(b(i))*d(i)*sind(a(i)),...
        cosd(b(i))*d(i)*cosd(a(i)),-sign(b(i))*sind(abs(b(i)))*d(i)];
    i = i + 1;
end
if size(pointSet,2) == 3
    u1 = coordinate3(2,:) - coordinate3(1,:);
    v1 = coordinate3(1,:) - coordinate3(3,:);
    w1 = coordinate3(3,:) - coordinate3(2,:);
    a1 = norm(cross(u1,v1)); a2 = norm(cross(u1,w1));
    a3 = norm(cross(v1,w1)); area = round(mean([a1,a2,a3])/2,3);
    tarNul = null([u1;w1;v1]);
    tilt = round(asind(tarNul(3)/norm(tarNul)),2);
    pan = round(asind(tarNul(1)/norm(tarNul(1:2))),2);
else
    u1 = coordinate3(2,:) - coordinate3(1,:); 
    v1 = coordinate3(4,:) - coordinate3(1,:);
    u2 = coordinate3(2,:) - coordinate3(3,:);
    v2 = coordinate3(4,:) - coordinate3(3,:);
    u3 = coordinate3(1,:) - coordinate3(4,:);
    v3 = coordinate3(3,:) - coordinate3(4,:);
    u4 = coordinate3(1,:) - coordinate3(2,:); 
    v4 = coordinate3(3,:) - coordinate3(2,:);
    a1 = norm(cross(u1,v1)); a2 = norm(cross(u2,v2));
    a3 = norm(cross(u3,v3)); a4 = norm(cross(u4,v4));
    area = round(mean([a1,a2,a3,a4]),3);
    tarNul = null([coordinate3(1,:)-coordinate3(3,:);
        coordinate3(4,:)-coordinate3(2,:)]);
    tilt = round(asind(tarNul(3)/norm(tarNul)),2);
    pan = round(asind(tarNul(1)/norm(tarNul(1:2))),2);
end
fprintf('Target surface area is:\n%.3f sq.ft.\n%.3f m2\n',...
    area, round(area/10.764,2));
fprintf('Tilt and pan of the target area,\npan: %.2f deg .\ntilt: %.2f deg\n',...
    pan, tilt);
% projected area calculation
pan_cam = mean(a); tilt_cam = -mean(b); dis_cam = mean(d);
pan_cam = 90 - atand(dis_cam*sind(90-pan_cam)/(dis_cam*cosd(90-pan_cam)-camD/2));
[proj_area] = getProjArea(coordinate3,size(pointSet,2),...
    pan_cam,tilt_cam);
proj_tilt = acosd(proj_area/area);
fprintf('[*]Target projected area is:\n%.3f sq.ft.\n%.3f m2\n',...
    proj_area, round(proj_area/10.764,2));
fprintf('[*]Tilt angle of the projected area,\ntilt: %.2f deg\n',...
    proj_tilt);
end

% dev function
function [proj_roll] = getProjRoll(pointSet)
uiwait(msgbox({'two point on the shape, which you know the vector formed by them is 0-deg';
    'e.g., for a rect shape, p(1)p(2) or p(3)p(4)'},'Notice!'));
i = 1;
while i <= size(pointSet,2)
    a(i) = pointSet(i).Results.alpha;
    b(i) = pointSet(i).Results.beta;
    d(i) = pointSet(i).Results.distance;
    coordinate3(i,:) = [-cosd(b(i))*d(i)*sind(a(i)),...
        cosd(b(i))*d(i)*cosd(a(i)),...
        -sign(b(i))*sind(abs(b(i)))*d(i)];
    i = i + 1;
end
pan_cam = mean(a); tilt_cam = -mean(b); [p1,p2] = uigetPoint();
coordinate3_new = xyzrotate(coordinate3,[pan_cam,0,tilt_cam]);
v = coordinate3_new(p2,:) - coordinate3_new(p1,:);
proj_roll = round(atand(v(3)/v(1)),2);
fprintf('[Dev function]Roll angle of the projected area,\nroll: %.2f deg\n',...
    proj_roll);
end

function [ans1,ans2] = uigetPoint()
prompt = {'Point 1';'Point 2'};
dlgtitle = 'Two points of the shape'; dims = [1 50];definput = {'1','2'};
answer = str2double(inputdlg(prompt,dlgtitle,dims,definput));
if isempty(answer)
    ans1 = 1; ans2 = 2;
else
    ans1 = answer(1); ans2 = answer(2);
end
end

% dev function
function [ap_area_image] = CameraProjectedArea(ap_area,pointSet,f)
for i = 1:size(pointSet,2)
    disData(i) = pointSet(i).Results.distance; %#ok<*SAGROW>
end
ap_area_image = (ap_area/10.764*1000^2)*(f/(mean(mink(disData,2))/3.281*1000-f))^2; % in mm2
fprintf('projected area on the image plane: %.4f mm2\n', ap_area_image);
end

function yn = yn_dialog(ques)
opts.Interpreter = 'tex'; opts.Default = 'No';
yn = questdlg(ques,'Dialog Window',...
    'Yes','No',opts);
end

% DEV function, image based tape measure.
function [dis] = imgTape(point1,point2)
% input p1,p2 should be 3d polar coordinates.
% p format: [alpha, beta, distance]
p1 = [point1.Results.alpha,...
    point1.Results.beta,...
    point1.Results.distance];
p2 = [point2.Results.alpha,...
    point2.Results.beta,...
    point2.Results.distance];
v1 = [-cosd(p1(2))*p1(3)*sind(p1(1)),...
    cosd(p1(2))*p1(3)*cosd(p1(1)),...
    -sign(p1(2))*sind(abs(p1(2)))*p1(3)];
v2 = [-cosd(p2(2))*p2(3)*sind(p2(1)),...
    cosd(p2(2))*p2(3)*cosd(p2(1)),...
    -sign(p2(2))*sind(abs(p2(2)))*p2(3)];
dis = round(norm(v2-v1),2);
fprintf('Measured distance is:\n%.1f ft\n%.2f m\n',...
    dis, dis/3.28);
end

function [area,proj_area] = getArea(pointSet)
i = 1;
while i <= size(pointSet,2)
    a(i) = pointSet(i).Results.alpha;
    b(i) = pointSet(i).Results.beta;
    d(i) = pointSet(i).Results.distance;
    coordinate3(i,:) = [-cosd(b(i))*d(i)*sind(a(i)),...
        cosd(b(i))*d(i)*cosd(a(i)),...
        -sign(b(i))*sind(abs(b(i)))*d(i)];
    i = i + 1;
end
if size(pointSet,2) == 3
    u1 = coordinate3(2,:) - coordinate3(1,:);
    v1 = coordinate3(1,:) - coordinate3(3,:);
    w1 = coordinate3(3,:) - coordinate3(2,:);
    a1 = norm(cross(u1,v1)); a2 = norm(cross(u1,w1));
    a3 = norm(cross(v1,w1)); area = round(mean([a1,a2,a3])/2,3);
    tarNul = null([u1;w1;v1]);
    tilt = round(asind(tarNul(3)/norm(tarNul)),2);
    pan = round(asind(tarNul(1)/norm(tarNul(1:2))),2);
else
    u1 = coordinate3(2,:) - coordinate3(1,:); v1 = coordinate3(4,:) - coordinate3(1,:);
    u2 = coordinate3(2,:) - coordinate3(3,:); v2 = coordinate3(4,:) - coordinate3(3,:);
    u3 = coordinate3(1,:) - coordinate3(4,:); v3 = coordinate3(3,:) - coordinate3(4,:);
    u4 = coordinate3(1,:) - coordinate3(2,:); v4 = coordinate3(3,:) - coordinate3(2,:);
    a1 = norm(cross(u1,v1)); a2 = norm(cross(u2,v2));
    a3 = norm(cross(u3,v3)); a4 = norm(cross(u4,v4));
    area = round(mean([a1,a2,a3,a4]),3);
    tarNul = null([coordinate3(1,:)-coordinate3(3,:);
        coordinate3(4,:)-coordinate3(2,:)]);
    tilt = round(asind(tarNul(3)/norm(tarNul)),2);
    pan = round(asind(tarNul(1)/norm(tarNul(1:2))),2);
end
fprintf('Target surface area is:\n%.3f sq.ft.\n%.3f m2\n',...
    area, round(area/10.764,2));
fprintf('Tilt and pan of the target area,\npan: %.2f deg .\ntilt: %.2f deg\n',...
    pan, tilt);
% projected area calculation
roZ = mean(a); roX = -mean(b);
[proj_area] = getProjArea(coordinate3,size(pointSet,2),...
    roZ,roX);
proj_tilt = acosd(proj_area/area);
fprintf('[*]Target projected area is:\n%.3f sq.ft.\n%.3f m2\n',...
    proj_area, round(proj_area/10.764,2));
fprintf('[*]Tilt angle of the projected area,\ntilt: %.2f deg\n',...
    proj_tilt);
end

function [proj_area] = getProjArea(coordinate3,side,roZ,roX)
coordinate3_new = xyzrotate(coordinate3,[roX,0,roZ]);
if side == 3
    v1 = coordinate3_new(2,:) - coordinate3_new(1,:); 
    v2 = coordinate3_new(3,:) - coordinate3_new(2,:); 
    v3 = coordinate3_new(1,:) - coordinate3_new(3,:);
    proj1 = norm(cross([v1(1),0,v1(3)],[v2(1),0,v2(3)]))/2;
    proj2 = norm(cross([v2(1),0,v2(3)],[v3(1),0,v3(3)]))/2;
    proj3 = norm(cross([v3(1),0,v3(3)],[v1(1),0,v1(3)]))/2;
    proj_area = round(mean([proj1,proj2,proj3]),3);
elseif side == 4
    v1 = coordinate3_new(2,:) - coordinate3_new(1,:); 
    v2 = coordinate3_new(3,:) - coordinate3_new(2,:); 
    v3 = coordinate3_new(4,:) - coordinate3_new(3,:); 
    v4 = coordinate3_new(1,:) - coordinate3_new(4,:);
    proj1 = norm(cross([v1(1),0,v1(3)],[v2(1),0,v2(3)]));
    proj2 = norm(cross([v2(1),0,v2(3)],[v3(1),0,v3(3)]));
    proj3 = norm(cross([v3(1),0,v3(3)],[v4(1),0,v4(3)]));
    proj4 = norm(cross([v4(1),0,v4(3)],[v1(1),0,v1(3)]));
    proj_area = round(mean([proj1,proj2,proj3,proj4]),3);
else
    error('Sides of shape can only be 3 or 4...\n');
end
end

function [pointSet,pt] = targetFind(imEL,imER,fov)
uiwait(msgbox('current version only support area calcualtion of trianle or rectangular shape...','Notice!'));
uiwait(msgbox('find target view in the [left] camera.','Notice!'));
while true
    [vL, hL, imgpL] = findView(imEL,fov); %#ok<*ASGLU>
    yn = yn_dialog('confirm target in the view?');
    if ismember(yn, ['Yes', 'yes'])
        break; end
end
uiwait(msgbox('find target view in the [right] camera.','Notice!'));
while true
    [vR, hR, imgpR] = findView(imER,fov); %#ok<*ASGLU>
    yn = yn_dialog('confirm target in the view?');
    if ismember(yn, ['Yes', 'yes'])
        break; end
end
yn = yn_dialog('Rectangular area[Y]? Triangle area[N]');
if ismember(yn, ['Yes', 'yes'])
    pt = 4;
else
    pt = 3;
end
amkey = 0;
yn = yn_dialog('select target using ROI function[Y],or manual[N]?');
if ismember(yn, ['Yes', 'yes'])
    amkey = 1;
end
if amkey
    uiwait(msgbox('select target in the [LEFT] view.','Notice!'));
    pointSet_left_cam = roiGetPoints(imgpL,vL,hL,fov,pt);
    uiwait(msgbox('select target in the [RIGHT] view.','Notice!'));
    pointSet_right_cam = roiGetPoints(imgpR,vR,hR,fov,pt);
    i = 1;
    while i <= pt
        p = inputParser;
        addOptional(p,'Hleft',pointSet_left_cam(i,1));
        addOptional(p,'Vleft',pointSet_left_cam(i,2));
        addOptional(p,'Hright',pointSet_right_cam(i,1));
        addOptional(p,'Vright',pointSet_right_cam(i,2));
        parse(p); pointSet(i) = p;
        i = i + 1; clear p;
    end
    return
end
uiwait(msgbox({'Using [Tool-Data Cursor]';'get pixel coordinate in both image view';},'Notice'));
figure(1); 
sb1 = subplot(1,2,1); image(imgpL); axis equal tight; grid on;
sb1.Position = sb1.Position.*[0.5 0.5 1.2 1.2]; 
title('Left view','Color','red','FontSize',16) 
sb2 = subplot(1,2,2);image(imgpR); axis equal tight; grid on;
sb2.Position = sb2.Position.*[1.0 0.5 1.2 1.2]; 
title('Right view','Color','blue','FontSize',16) 
clear sb1 sb2;
uiwait(msgbox('pixel coordinates ready?','Notice!'));
i = 1;
while i <= pt
    uiwait(msgbox('target pixel coordinate in the [left] view.','Notice!'));
    [tarXl, tarYl] = angularOffset(imgpL,vL,hL,fov);
    uiwait(msgbox('target pixel coordinate in the [right] view.','Notice!'));
    [tarXr, tarYr] = angularOffset(imgpR,vR,hR,fov);
    p = inputParser;
    addOptional(p,'Hleft',tarXl); addOptional(p,'Vleft',tarYl);
    addOptional(p,'Hright',tarXr); addOptional(p,'Vright',tarYr);
    parse(p); 
    pointSet(i) = p; i = i + 1; clear p;
end
end


function [targetPoint] = roiGetPoints(imgp,ang1,ang2,fov,total_point)
img = rgb2gray(imgp);
figure(1); shape = roipoly(img); close(gcf);
if isempty(shape)
    msgbox('No ROI selected!','Error','error'); close(gcf);
    error('no ROI selected\n..');
end
[pY, pX] = find(shape);
corners = detectMinEigenFeatures(shape,'FilterSize', 17,...
    'ROI', [min(pX),min(pY),max(pX)-min(pX)+50,max(pY)-min(pY)+50],...
    'MinQuality', 0.2);
f = figure(2); 
sb1 = subplot(1,2,1); imshow(imgp); axis equal tight; grid on;
hold on; plot(corners.selectStrongest(total_point)); hold off;
sb1.Position = sb1.Position.*[0.5 0.5 1.2 1.2]; 
title('corner detection results.',...
    'Color','red','FontSize',16) 
sb2 = subplot(1,2,2);imshow(shape); axis equal tight; grid on;
hold on; plot(corners.selectStrongest(total_point)); hold off;
sb2.Position = sb2.Position.*[1.0 0.5 1.2 1.2];
title('selected area','Color','blue','FontSize',16) 
clear sb1 sb2; uiwait(f);
yn = yn_dialog('corner detection correct?');
if ~ismember(yn, ['Yes', 'yes'])
    error('corners not detected, terminated...\n');
end
points_tmp = selectStrongest(corners, total_point);
points_tmp = round(points_tmp.Location);
midp = [mean(points_tmp(:,1)),mean(points_tmp(:,2))];
delta = points_tmp - midp; 
points_tmp(:,3) = atan2d(delta(:,2),delta(:,1));
points_tmp = sortrows(points_tmp,3); 
points = points_tmp(:,1:2);
for i = 1:size(points,1)
    cx = size(imgp,1); 
    x_p = points(i,1);
    y_p = points(i,2);
    xf = 2*((x_p-1)/(cx-1)-0.5); yf = 2*((y_p-1)/(cx-1)-0.5); 
    [xe,ye] = fish2equ(xf,yf,ang1,ang2,0,fov);
    Xe = round((xe+1)/2*(2*cx-1)+1); Ye = round((ye+1)/2*(cx-1)+1);
    currX = -round((Xe-cx)/cx*180,1); currY = round((Ye-cx/2)/cx*180,1);
    currXCali = angCalib(currX); currYCali = angCalib(currY);
    targetPoint(i,:) = [currXCali, currYCali];
end
end

function [imEL,imER] = imgLoad(fov,f,sy)
uiwait(msgbox('Load image from the left camera?','Notice!'));
[fn,pn]=uigetfile('*.jpg','load 180 FOV image of the Left camera');
str=[pn,fn]; imFL = imread(str);
uiwait(msgbox('Load image from the right camera?','Notice!'));
[fn,pn]=uigetfile('*.jpg','load 180 FOV image of the Right camera');
str=[pn,fn]; imFR = imread(str); clear pn fn str;
[y,x] = size(imFL(:,:,1)); pr180 = round(sind(45)*2*f/sy*y/2*1)*2;
imcL = circularCrop(imFL,x,y,pr180); imcR = circularCrop(imFR,x,y,pr180);
imEL = imfish2equ(imcL,fov); imER = imfish2equ(imcR,fov);
fprintf('Image loading...Ok...\n');
end

function imgE = imfish2equ(imgF,varargin)
% Reference: 
% [1] 360-degree-image-processing (https://github.com/k-machida/360-degree-image-processing), GitHub. Retrieved August 18, 2020.
% [2] Tuan Ho, Madhukar Budagavi,  "2DUAL-FISHEYE LENS STITCHING FOR 360-DEGREE IMAGING"
% Copyright of Original Function: Kazuya Machida (2020)
p = inputParser; addRequired(p,'imgF');
addOptional(p,'fov' ,180); % defaul value of fov
addOptional(p,'roll',  0); % defaul value of roll
addOptional(p,'tilt',  0); % defaul value of tilt
addOptional(p,'pan' ,  0); % defaul value of pan
parse(p,imgF,varargin{:});
wf = size(imgF,2); hf = size(imgF,1); ch = size(imgF,3);
we = wf*2; he = hf;
fov  = p.Results.fov; roll = p.Results.roll;
tilt = p.Results.tilt; pan  = p.Results.pan;
[xe,ye] = meshgrid(1:we,1:he);
xe = 2*((xe-1)/(we-1)-0.5); ye = 2*((ye-1)/(he-1)-0.5); 
[xf,yf] = equ2fish(xe,ye,fov,roll,tilt,pan); idx = sqrt(xf.^2+yf.^2) <=1; 
xf = xf(idx); yf = yf(idx); xe = xe(idx); ye = ye(idx);
Xe = round((xe+1)/2*(we-1)+1); Ye = round((ye+1)/2*(he-1)+1); 
Xf = round((xf+1)/2*(wf-1)+1); Yf = round((yf+1)/2*(hf-1)+1); 
Ie = reshape(imgF,[],ch); If = zeros(he*we,ch,'uint8');
idnf = sub2ind([hf,wf],Yf,Xf); idne = sub2ind([he,we],Ye,Xe);
If(idne,:) = Ie(idnf,:);imgE = reshape(If,he,we,3);
end

function [xf,yf] = equ2fish(xe,ye,fov,roll,tilt,pan)
thetaE = xe*180; phiE = ye*90; cosdphiE = cosd(phiE); 
xs = cosdphiE.*cosd(thetaE); ys = cosdphiE.*sind(thetaE); zs = sind(phiE);   
xyzsz = size(xs); xyz = xyzrotate([xs(:),ys(:),zs(:)],[roll tilt pan]);
xs = reshape(xyz(:,1),xyzsz(1),[]); 
ys = reshape(xyz(:,2),xyzsz(1),[]);
zs = reshape(xyz(:,3),xyzsz(1),[]);
thetaF = atan2d(zs,ys); 
% r = 2*atan2d(sqrt(ys.^2+zs.^2),xs)/fov; % equidistant proj
r = 2*(sind(atan2d(sqrt(ys.^2+zs.^2),xs)/2))/(2*sind(fov/4)); % equisolid-angle proj
xf = r.*cosd(thetaF); yf = r.*sind(thetaF);
end

function imgF = imequ2fish(imgE,varargin)
% Reference: 
% [1] 360-degree-image-processing (https://github.com/k-machida/360-degree-image-processing), GitHub. Retrieved August 18, 2020.
% [2] Tuan Ho, Madhukar Budagavi,  "2DUAL-FISHEYE LENS STITCHING FOR 360-DEGREE IMAGING"
% Copyright of Original Function: Kazuya Machida (2020)
p = inputParser;
addRequired(p,'imgE');
addOptional(p,'fov' ,  180); % defaul value of fov
addOptional(p,'roll',  0); % defaul value of roll
addOptional(p,'tilt',  0); % defaul value of tilt
addOptional(p,'pan' ,  0); % defaul value of pan
parse(p,imgE,varargin{:});
we = size(imgE,2); he = size(imgE,1); ch = size(imgE,3);
wf = round(we/2); hf = he;
roll = p.Results.roll; tilt = p.Results.tilt; 
pan  = p.Results.pan; fov = p.Results.fov;
[xf,yf] = meshgrid(1:wf,1:hf);
xf = 2*((xf-1)/(wf-1)-0.5); yf = 2*((yf-1)/(hf-1)-0.5); 
idx = sqrt(xf.^2+yf.^2) <= 1; xf = xf(idx); yf = yf(idx);
[xe,ye] = fish2equ(xf,yf,roll,tilt,pan,fov);
Xe = round((xe+1)/2*(we-1)+1); % rescale to 1~we
Ye = round((ye+1)/2*(he-1)+1); % rescale to 1~he
Xf = round((xf+1)/2*(wf-1)+1); % rescale to 1~wf
Yf = round((yf+1)/2*(hf-1)+1); % rescale to 1~hf
Ie = reshape(imgE,[],ch); If = zeros(hf*wf,ch,'uint8');
idnf = sub2ind([hf,wf],Yf,Xf); idne = sub2ind([he,we],Ye,Xe);
If(idnf,:) = Ie(idne,:); imgF = reshape(If,hf,wf,3);
end

function [xe,ye] = fish2equ(xf,yf,roll,tilt,pan,fov)
thetaS = atan2d(yf,xf);
% phiS = sqrt(yf.^2+xf.^2)*fov/2; % equidistant proj
phiS = 2*asind(sqrt(yf.^2+xf.^2)*sind(fov/4)); % equisolidangle proj
sindphiS = sind(phiS);
xs = sindphiS.*cosd(thetaS); ys = sindphiS.*sind(thetaS); zs = cosd(phiS);
xyzsz = size(xs);
xyz = xyzrotate([xs(:),ys(:),zs(:)],[roll tilt pan]);
xs = reshape(xyz(:,1),xyzsz(1),[]); 
ys = reshape(xyz(:,2),xyzsz(1),[]);
zs = reshape(xyz(:,3),xyzsz(1),[]);
thetaE = atan2d(xs,zs); phiE = atan2d(ys,sqrt(xs.^2+zs.^2));
xe = thetaE/180; ye = 2*phiE/180;
end

function [va,ha,IF] = findView(Iequ,fov)
[va,ha] = ui_fv();
IF = imequ2fish(Iequ,fov,va,ha); % fov, h, v, rotate
[ix,iy] = size(IF(:,:,1)); imshow(IF);
drawcircle('Center',[iy/2,iy/2],'Radius',round(ix/100),'Color','Green');
drawcircle('Center',[iy/2,iy/2],'Radius',round(ix/15),'Color','Yellow');
end

function [ans1,ans2] = ui_fv()
prompt = {'Your selection of vertical angle ?     [-90 to 90 degree]',...
    'Your selection of horizontal angle ? [-90 to 90 degree]'};
dlgtitle = 'User Input'; dims = [1 50];definput = {'0','0'};
answer = str2double(inputdlg(prompt,dlgtitle,dims,definput));
if isempty(answer)
    ans1 = 0; ans2 = 0;
else
    ans1 = answer(1); ans2 = answer(2);
end
end

function [pointSet] = disCalculate(camD,tarSet,sides,varargin)
p = inputParser;
addRequired(p,'camD'); % distance of two canon camera, in ft
addRequired(p,'tarSet'); % angle set of the target in both camera
addRequired(p,'sides'); % total sides of shape, 3 or 4
addOptional(p,'leftCompen' ,  0); % defaul 0-deg compensation for left camera
addOptional(p,'rightCompen',  0); % defaul 0-deg compensation for right camera
parse(p,camD,tarSet,sides,varargin{:});
lC = p.Results.leftCompen;
rC = p.Results.rightCompen;
camD = p.Results.camD;
workload = p.Results.sides;
i = 1;
while i <= workload
    hL = p.Results.tarSet(i).Results.Hleft;
    hR = p.Results.tarSet(i).Results.Hright;
    vL = p.Results.tarSet(i).Results.Vleft;
    vR = p.Results.tarSet(i).Results.Vright;
    hL = hL + lC; hR = hR + rC; beta = (vL + vR)/2;
    hlD = camD/sind(hR - hL)*sind(90-hR); %SinLaw
    hrD = camD/sind(hR - hL)*sind(90+hL);
    Dis2 = (0.5*(hlD^2+hrD^2-0.5*camD^2))^0.5;% Euclid
    Dis3 = round(Dis2/cosd(beta),2);
    alpha = 90 - acosd(((camD/2)^2+Dis2^2-hlD^2)/(camD*Dis2));
    fprintf('point# %d\nTarget distance:\n%.2f ft\n%.2f m\n', i,Dis3, Dis3/3.28);
    fprintf('Target direction:\nAlpha, %.2f deg\nBeta, %.2f deg\n\n', alpha, beta);
    tmp = inputParser;
    addOptional(tmp,'distance',Dis3);
    addOptional(tmp,'alpha',alpha);
    addOptional(tmp,'beta',beta);
    parse(tmp); pointSet(i) = tmp; 
    i = i + 1; clear tmp;
end
end

function Icropped = circularCrop(I,x,y,r)
xc = round(x/2);yc = round(y/2);
c = zeros(y,x); [L(:,1),L(:,2)] = find(c==0);
L(:,3) = sqrt((L(:,1) - yc).^2 + (L(:,2) - xc).^2);
L(L(:, 3) > r, :) = [];
for i = 1: size(L,1)
   c(y+1-L(i,1),L(i,2)) = 1;
end
msk = imbinarize(c,0);
ir = uint8(double(I(:,:,1)).*msk);
ig = uint8(double(I(:,:,2)).*msk);
ib = uint8(double(I(:,:,3)).*msk);
Icc = cat(3,ir,ig,ib);
[mski(:,1), mski(:,2)] = find(msk==1);
Icropped = imcrop(Icc,[min(mski(:,2)),min(mski(:,1)),...
    max(mski(:,2))-min(mski(:,2)),max(mski(:,1))-min(mski(:,1))]);
end

function [currXCali, currYCali] = angularOffset(imgp,ang1,ang2,fov)
cx = size(imgp,1);
roi_flg = 0; % roi function disabled here.
if roi_flg
    [x_p, y_p] = roi_select(imgp);
else
    x_p = input('X-pixel coordinate of target ?\n');
    y_p = input('Y-pixel coordinate of target ?\n');
end
xf = 2*((x_p-1)/(cx-1)-0.5); yf = 2*((y_p-1)/(cx-1)-0.5); 
[xe,ye] = fish2equ(xf,yf,ang1,ang2,0,fov);
Xe = round((xe+1)/2*(2*cx-1)+1); Ye = round((ye+1)/2*(cx-1)+1);
currX = -round((Xe-cx)/cx*180,1); currY = round((Ye-cx/2)/cx*180,1);
fprintf('\nIn the image space:');
fprintf('\nCurrent point:\n Horizontal: %d\n Vertical: %d \n', ang2, ang1);
fprintf('New point:\n Horizontal: %.2f\n Vertical: %.2f \n', currX, currY);
fprintf('\nIn the real space: [Calibrated viewing direction]\n');
currXCali = angCalib(currX); currYCali = angCalib(currY);
fprintf('Horizontal: %.2f\nVertical: %.2f\n\n',currXCali, currYCali);
end

% Important: using calibration file to get the lens calibration function first!
function[xc] = angCalib(x)
% CanonT2i-SigmaF4.5 calibration function, deg3-polynomial
xc = sign(x)*round(1.09842e-05.*abs(x).^(3) -...
    9.89333e-04.*abs(x).^(2)+ 1.02156.*abs(x) - 0.04525,1);
end

function [cx, cy] = roi_select(img_p) %#ok<*DEFNU,*REDEF>
img_p = rgb2gray(img_p);
figure(1);ROI = roipoly(img_p);close(gcf);lookup = []; 
if isempty(ROI)
    msgbox('No ROI selected!','Error','error'); close(gcf);
    error('Error_003: no ROI selected, try again!'); return; %#ok<*UNRCH>
end
[lookup(:,2),lookup(:,1)] = find(ROI);
cy = round(mean(lookup(:,2))); cx = round(mean(lookup(:,1)));
end

function [xyznew] = xyzrotate(xyz,thetaXYZ)
tX =  thetaXYZ(1); tY =  thetaXYZ(2); tZ =  thetaXYZ(3);
T = [ cosd(tY)*cosd(tZ),- cosd(tY)*sind(tZ), sind(tY); ...
      cosd(tX)*sind(tZ) + cosd(tZ)*sind(tX)*sind(tY), cosd(tX)*cosd(tZ) - sind(tX)*sind(tY)*sind(tZ), -cosd(tY)*sind(tX); ...
      sind(tX)*sind(tZ) - cosd(tX)*cosd(tZ)*sind(tY), cosd(tZ)*sind(tX) + cosd(tX)*sind(tY)*sind(tZ),  cosd(tX)*cosd(tY)];
xyznew = xyz*T;
end