% Use LQR for X and Y direction.
% Use NN for Z direction.


clear all
close all
clc

global SIASclient FLAG Azd Bzd Qz R r_set Counts t0
global  Error CommandInput SS SS2 commandArray_Idf Cont GoalPt CI
global deltaT inc drone_mode ContGoal xyDesired vertDesired yawDesired DESIREDPOINT
global p state state2 xyDesired vertDesired rollDesired pitchDesired yawDesired DESIRED
global SIASclient Error CommandInput ContGoal Time netx nety netz
global deltaT drone_mode Cont GoalPt DESIRED1 Kx Ky Kz x_r x_init INIT_POS_ADJUST
global p state state2 inc deltaT  xyDesired vertDesired rollDesired pitchDesired yawDesired DESIREDPOINT
 

%% State space matrices.

deltaT=0.2;

alpha_x = 0.0527;
alpha_y = 0.0187;
alpha_z = 1.7873;
alpha = [alpha_x,alpha_y,alpha_z];

beta_x = -5.4779;
beta_y = -7.0608;
beta_z = -1.7382;
beta = [beta_x,beta_y,beta_z];


%% LQR and REFERENCE settings
x_init = [[0.;0],[0.;0],[1.5;0]];
% variable x_r is applied only in LQR at this moment.
x_r = [[0;0],[0;0]];

% option one: sin wave
Counts = 60000;

% r_set = 0.5*sin(0.011*(1:Counts)) + 1.5;

% option two: step 
r_set=[1*ones(1,Counts/4), 2*ones(1,Counts/4), 1.2*ones(1,Counts/4),1.7*ones(1,Counts/4) ];

INIT_POS_ADJUST = 1;
Q = [1 0;0 0.1];
R = 1;
Qz = [2 0;0 1];
C = [1 0];
D=0;

% X-direction
Ax = [0 1 ; 0 -alpha(1)];
Bx=[0;beta(1)];
Gx = ss(Ax,Bx,C,D);
Gxd = c2d(Gx,deltaT);
Axd=Gxd.A;
Bxd=Gxd.B;

Ay = [0 1 ; 0 -alpha(2)];
By=[0;beta(2)];
Gy = ss(Ay,By,C,D);
Gyd = c2d(Gy,deltaT);
Ayd=Gyd.A;
Byd=Gyd.B;

Az = [0 1 ; 0 -alpha(3)];
Bz=[0;beta(3)];
Gz = ss(Az,Bz,C,D);
Gzd = c2d(Gz,deltaT);
Azd=Gzd.A;
Bzd=Gzd.B;

% GET FEEDBACK GAINS
[Kx,Sx,CLP]=dlqr(Axd,Bxd,[2 0;0 1],R);
[Ky,Sy,CLP]=dlqr(Ayd,Byd,[2 0;0 1],R);
[Kz,Sz,CLP]=dlqr(Azd,Bzd,Qz,R);

% Kx=0.1*Kx;
% Ky=0.1*Ky;
% Kz=0.1*Kz;
%% Load ONNX model--Z
netz = importNetworkFromONNX("dense_center_S40_vshape_Iter20_Epoch102.onnx");
% model = importONNXNetwork('model.onnx');

netz.Layers(1).InputInformation;
X = dlarray(rand(1, 3), 'UU');
netz = initialize(netz, X);
summary(netz)



%%

SIASclient = natnet;
SIASclient.connect; 
pause(2)

fprintf('\n\nConnecting to Drone...\n') 
p = parrot(); 
fprintf('Connected to %s\n', p.ID) 
fprintf('Battery Level is %d%%\n', p.BatteryLevel)
takeoff(p); 
pause(0.5)







 
 Error=zeros(4,1);
 CommandInput=zeros(4,1);
inc=1;


t0=double(SIASclient.getFrame.fTimestamp);
while(1>0)


Time(inc,1)=double(SIASclient.getFrame.fTimestamp);

    Position=double([SIASclient.getFrame.RigidBodies(1).x;SIASclient.getFrame.RigidBodies(1).y;SIASclient.getFrame.RigidBodies(1).z]);
    q=quaternion( SIASclient.getFrame.RigidBodies(1).qw, SIASclient.getFrame.RigidBodies(1).qx, SIASclient.getFrame.RigidBodies(1).qy, SIASclient.getFrame.RigidBodies(1).qz );
    eulerAngles=quat2eul(q,'xyz')*180/pi;
    Angle=[eulerAngles(1);eulerAngles(2);eulerAngles(3)];
    state=[Position;Angle];
    [errorArray]=ControlCommand;
    if FLAG(inc)==1
        SS(:,inc)=state;
        CI(:,inc)=commandArray_Idf;
        Error(:,inc)=errorArray;
        inc=inc+1;
    end
% end

end


function [errorArray]=ControlCommand
 
global p state  inc FLAG SS CI Error t0 ...
    Kx Ky Kz x_init x_r INIT_POS_ADJUST...
    commandArray_Idf  Time  netz Azd Bzd Qz R r_set Counts


%% Define desired tolerances and gains



 if inc==1
     dt=0.008301;
 else
     dt=Time(inc,1)-Time(inc-1,1);
 end

if dt < eps 
   dt = 0.008301; % Average calculated time step
end

% LQR inputs

if inc==1
    x = [state(1);0];
    y = [state(2);0];
    z = [state(3);0];
else
    x = [state(1);(state(1)-SS(1,inc-1))/dt];
    y = [state(2);(state(2)-SS(2,inc-1))/dt];
    z = [state(3);(state(3)-SS(3,inc-1))/dt];
end

% adjust init position
    ux = -Kx*(x-x_r(:,1));
    uy = -Ky*(y-x_r(:,2));

    % adjust control input on Z

    % LQR
    % uz = -Kz*(z-x_r(:,3));

    % Varying reference

    tt=Time(inc,1)-t0;
    % if tt < 10
    %     r = 1;
    % elseif tt>=10 && tt< 20
    %     r = 2;
    % elseif tt>=20 && tt<30
    %     r = 1.2;
    % else 
    %     r = 1.7;
    % end

    r = 0.5*sin(0.2*tt) + 1.5;

    r_set(:,inc) = r;

    % r = r_set(mod(inc,Counts)+1); 

    % MPC
    xr = [r;0];
    tic;
    uN = mpc_fun(Azd, Bzd, Qz,R,double(z),xr,10);
    uz = uN(1);
    
    % disp('MPC')

    % NN
    % if inc==1
    %     example_z = dlarray([state(3),0,r], 'UU');
    % else
    %     example_z = dlarray([state(3),(state(3)-SS(3,inc-1))/dt,r], 'UU');
    % end
    % output_z = predict(netz, example_z);
    % uz = extractdata(output_z(1));


%#####################LQR initial state adjust:#########################
thresh_error = 0.5;
if INIT_POS_ADJUST < 20
    % adjust init position
    ux = -Kx*(x-x_init(:,1));
    uy = -Ky*(y-x_init(:,2));
    uz = -Kz*(z-x_init(:,3));
    % uz=0;
    
    init_error = norm(state(1:3)-x_init(1,:)');
    % wait unit error is less than thresh_error enough.
    if init_error < thresh_error
        INIT_POS_ADJUST = INIT_POS_ADJUST + 1;
    end
    disp('LQR')
end

toc
%##################### LQR initial state adjust END ####################

if uz>=0.6
    uz=0.6;
elseif uz<=-0.6
    uz=-0.6;
end
if uy>=0.06
    uy=0.06;
elseif uy<=-0.06
    uy=-0.06;
end
if ux>=0.06
    ux=0.06;
elseif ux<=-0.06
    ux=-0.06;
end

 if inc == 1 


 old_yaw_Error = 0; 


 else
     old_yaw_Error=Error(1,inc-1);

 end


 yawActual = deg2rad(state(6)); 


 
 % to rotate X,Y from world frame to robot frame
 Tw2r = [cos(yawActual), sin(yawActual); -sin(yawActual), cos(yawActual)]; 


 % 
 
 
 
 % Compute the errors
 % Yaw Error
 if inc==1
     dt=0.008301;
 else
     dt=Time(inc,1)-Time(inc-1,1);
 end

if dt < eps 
   dt = 0.008301; % Average calculated time step
end

% dt

yawe1=(deg2rad(0) - yawActual);
 yawError = wrapToPi(yawe1);
 yawD_Error = (yawError-old_yaw_Error)/dt;
 
kYaw =-0.15; 
kD_Yaw =-0.12;

 % compute the yaw commands
 yawCmd = kYaw*yawError+kD_Yaw*yawD_Error;
 
 if abs(yawCmd) > 3.4 
 yawCmd = sign(yawCmd)*3.4; 
 end 


 

 errorArray = [yawError]; 


  if inc>1 
      if Time(inc,1)-Time(inc-1,1)>eps
          commandArray_Idf= [ux; uy; yawCmd; uz];
          FLAG(inc)=1;
  else
      commandArray_Idf= [CI(1,inc-1); CI(2,inc-1); CI(3,inc-1); CI(4,inc-1)];
      FLAG(inc)=0;
      end
  else
      commandArray_Idf= [ux; uy; yawCmd; uz];
       FLAG(inc)=1;
  end


move(p, 0.2, 'RotationSpeed', commandArray_Idf(3),'VerticalSpeed', commandArray_Idf(4),'roll', commandArray_Idf(2), 'pitch', commandArray_Idf(1));







end


