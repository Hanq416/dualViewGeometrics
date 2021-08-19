clear all; %#ok<*CLALL>
close all;
clc;
% Distance measurement with dual Canon+Sigma DSLR camera.
% Hankun Li, University of Kansas, 04/11/2021
% Camera value...
sx = 22.3; sy = 14.9; % camera sensor size
f = 4.5; % lens focal length
Lfov = 180; % lens fov
% Camera value end...
% Notes: Sigma lens is equisolidangle projection !!
%% step1: load image
[imEL,imER] = imgLoad(Lfov,f,sy);

%% step2: find target
% note: change [fov] to zoom, fov range of canon+sigma fisheye:[0-180]
fov = 90;
tarSet = targetFind(imEL,imER,fov);

%% step3: Distance caclculate, using viewing direction in the real space!
% !! Initial compensation for two camera
leftC = -1.5; % in deg, optional
rightC = -8.5; % in deg, optional
camD = 2.0; % camera distance, in ft, required
[alpha, beta, ds] = disCalculate(camD, tarSet, leftC, rightC);

%% optional: view check, using viewing direction in the [image space].
[~] = findView(imER,fov);





%Main function end here.
%% Functions
% ui dialog: selection
function yn = yn_dialog(ques)
opts.Interpreter = 'tex'; opts.Default = 'No';
yn = questdlg(ques,'Dialog Window',...
    'Yes','No',opts);
end

% find the target from both camera.
function [p] = targetFind(imEL,imER,fov)
uiwait(msgbox('find target view in the left camera.','Notice!'));
while true
    [vL, hL, imgpL] = findView(imEL,fov); %#ok<*ASGLU>
    yn = yn_dialog('confirm target in the view?');
    if ismember(yn, ['Yes', 'yes'])
        break; end
end
uiwait(msgbox('find target view in the right camera.','Notice!'));
while true
    [vR, hR, imgpR] = findView(imER,fov); %#ok<*ASGLU>
    yn = yn_dialog('confirm target in the view?');
    if ismember(yn, ['Yes', 'yes'])
        break; end
end
uiwait(msgbox({'Using [Tool-Data Cursor]';'get pixel coordinate in both image view';...
    'Optional, skip this to use ROI selection'},'Notice'));
yn = yn_dialog('using pixel method? or skip to use ROI');
if ismember(yn, ['Yes', 'yes'])
    f = figure(1); 
    sb1 = subplot(1,2,1); image(imgpL); axis equal tight; grid on;
    sb1.Position = sb1.Position.*[0.5 0.5 1.2 1.2]; 
    title('Left view','Color','red','FontSize',16) 
    sb2 = subplot(1,2,2);image(imgpR); axis equal tight; grid on;
    sb2.Position = sb2.Position.*[1.0 0.5 1.2 1.2]; 
    title('Right view','Color','blue','FontSize',16) 
    clear sb1 sb2;
    uiwait(f);
end
uiwait(msgbox('target pixel coordinate in the [left] view.','Notice!'));
[tarXl, tarYl] = angularOffset(imgpL,vL,hL,fov);
uiwait(msgbox('target pixel coordinate in the [right] view.','Notice!'));
[tarXr, tarYr] = angularOffset(imgpR,vR,hR,fov); close all;
p = inputParser;
addOptional(p,'Hleft',tarXl); addOptional(p,'Vleft',tarYl);
addOptional(p,'Hright',tarXr); addOptional(p,'Vright',tarYr);
parse(p);
end

% loading function
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
% Copyright of the Original Function: Kazuya Machida (2020)
% Modified by Hankun Li for KU LRL lab use.
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
% Get index of valid fisyeye image area
idx = sqrt(xf.^2+yf.^2) <= 1; xf = xf(idx); yf = yf(idx);
[xe,ye] = fish2equ(xf,yf,roll,tilt,pan,fov);
Xe = round((xe+1)/2*(we-1)+1); % rescale to 1~we
Ye = round((ye+1)/2*(he-1)+1); % rescale to 1~he
Xf = round((xf+1)/2*(wf-1)+1); % rescale to 1~wf
Yf = round((yf+1)/2*(hf-1)+1); % rescale to 1~hf
Ie = reshape(imgE,[],ch); If = zeros(hf*wf,ch,'uint8');
idnf = sub2ind([hf,wf],Yf,Xf);idne = sub2ind([he,we],Ye,Xe);
If(idnf,:) = Ie(idne,:);imgF = reshape(If,hf,wf,3);
end

function [xe,ye] = fish2equ(xf,yf,roll,tilt,pan,fov)
thetaS = atan2d(yf,xf);
% phiS = sqrt(yf.^2+xf.^2)*fov/2; % equidistant proj
phiS = 2*asind(sqrt(yf.^2+xf.^2)*sind(fov/4)); % equisolidangle proj
sindphiS = sind(phiS);
xs = sindphiS.*cosd(thetaS); ys = sindphiS.*sind(thetaS); zs = cosd(phiS);
xyzsz = size(xs);
xyz = xyzrotate([xs(:),ys(:),zs(:)],[roll tilt pan]);
xs = reshape(xyz(:,1),xyzsz(1),[]); ys = reshape(xyz(:,2),xyzsz(1),[]);
zs = reshape(xyz(:,3),xyzsz(1),[]);
thetaE = atan2d(xs,zs); phiE = atan2d(ys,sqrt(xs.^2+zs.^2));
xe = thetaE/180; ye = 2*phiE/180;
end

% find a view with specified viewing direction and FOV.
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

% distance calculation
function [alpha, beta, Dis3] = disCalculate(camD,tarSet,varargin)
p = inputParser;
addRequired(p,'camD'); % distance of two canon camera, in ft
addRequired(p,'tarSet'); % angle set of the target in both camera
addOptional(p,'leftCompen' ,  0); % defaul 0-deg compensation for left camera
addOptional(p,'rightCompen',  0); % defaul 0-deg compensation for right camera
parse(p,camD,tarSet,varargin{:});
lC = p.Results.leftCompen; rC = p.Results.rightCompen;
hL = p.Results.tarSet.Results.Hleft;
hR = p.Results.tarSet.Results.Hright;
vL = p.Results.tarSet.Results.Vleft;
vR = p.Results.tarSet.Results.Vright;
%%
hL = hL + lC; hR = hR + rC; beta = (vL + vR)/2;
hlD = camD/sind(hR - hL)*sind(90-hR); %SinLaw
hrD = camD/sind(hR - hL)*sind(90+hL);
Dis2 = (0.5*(hlD^2+hrD^2-0.5*camD^2))^0.5;% Euclid
Dis3 = round(Dis2/cosd(beta),2);
alpha = 90 - acosd(((camD/2)^2+Dis2^2-hlD^2)/(camD*Dis2));
fprintf('Target distance:\n%.2f ft\n%.2f m\n', Dis3, Dis3/3.2808);
fprintf('Target direction:\nAlpha, %.2f deg\nBeta, %.2f deg\n', alpha, beta);
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
cx = size(imgp,1); roi_flg = 1;
yn1 = yn_dialog('[YES] use ROI selection, [NO] input pixel coordinate, more accurate.');
if ~ismember(yn1, ['Yes', 'yes'])
    roi_flg = 0;
end
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

function [cx, cy] = roi_select(img_p)
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