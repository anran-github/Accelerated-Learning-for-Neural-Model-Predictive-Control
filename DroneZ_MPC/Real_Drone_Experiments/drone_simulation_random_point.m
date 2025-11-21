%=====================================
% Simulation: 
% Aftering transfering NN model in 
% MATLAB format, verify models 
% availbility with a given random point.
%=====================================

clc
close all
clear all


%% theta Parameter settings

% Load ONNX model
net = importNetworkFromONNX("/home/anranli/code/physic_inform_onestep/Physic-informed-nn-controller/semi_supervison/DroneZ_MPC_weights/dense_center_S40_vshape_Iter20_Epoch10.onnx");
% model = importONNXNetwork('model.onnx');

net.Layers(1).InputInformation;

X = dlarray(rand(1, 3), 'UU');
net = initialize(net, X);
summary(net)


theta = 1;
direction = 3;
save_name = strcat('trajectory_opt_drone_theta',num2str(theta),'.csv');

alpha_x = 0.0527;
alpha_y = 0.0187;
alpha_z = 1.7873;
alpha = [alpha_x,alpha_y,alpha_z];

beta_x = -5.4779;
beta_y = -7.0608;
beta_z = -1.7382;
beta = [beta_x,beta_y,beta_z];

num_compare = 600;
result_p = zeros(1,2,num_compare);
result_u = zeros(1,num_compare);
cnt = 1;

% read A,B,C,D matrices:
A = [0 1 ; 0 -alpha(direction)];
B=[0;beta(direction)];
C = [1 0];
D=0;
G=ss(A,B,C,D);

Ts = 0.2;
Gd=c2d(G,Ts);
Ad=Gd.A;
Bd=Gd.B;

% MPC Setting
Q = [2 0;0 1];  % State cost (penalize deviation of states)
R = 1;  
horizon = 10;

% x1 = 1.;
% x2 = 1;
% r = 0;

% init point and ref for z direction.
x1 = 1.55;
x2 = -0.5;





% given a random point, go to reference r. with num_compare steps.
p_tt = eye(2);
tic; 

% option one: sin wave
r_set = 0.5*sin(0.011*(1:num_compare)) + 1.5;

% option two: step 
% N = num_compare;  % Number of discrete time steps
% stage = 4;
% r_set=[1*ones(1,N/4), 2*ones(1,N/4), 1.2*ones(1,N/4),1.7*ones(1,N/4) ];

while cnt <= num_compare

    r = r_set(cnt); 
    xr = [r;0];
    % OPtimization part
    if cnt == 1
        x = [x1;x2];
    else
        x = xtt;
    end
 
    example_x = dlarray([x(1,1),x(2,1),r], 'UU');
    
    tic;
    % NN Method
    output = predict(net, example_x);   
    opt_u = extractdata(output(1));      

    % MPC Method
    % uN = mpc_fun(Ad,Bd,Q,R,x,xr,horizon);
    % opt_u = uN(1);



    T_run = toc;

    % display 
    disp(['Step: ', num2str(cnt), ',  t: ', num2str(T_run), ', x1: ', num2str(x(1)), ', r: ', num2str(r), ', opt_u: ', num2str(opt_u)]);


    xtt = Ad*x +Bd*opt_u;



    result_input(cnt,:) = x;
    uset(cnt,:) = opt_u;
    % next x(t+1)
    % xtt = [x(2);-9.8*sin(x(1))] +[0;1]*opt_u;

    cnt = cnt + 1;

end


elapsed_time = toc;

disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
% writematrix(opt_result, save_name);

%%  display

MPC_data = load('/home/anranli/code/physic_inform_onestep/Physic-informed-nn-controller/DroneZ_MPC/Simulation_Result_MPC_SinW.mat');

time = (0:cnt-2) * Ts;

figure;
subplot(3, 1, 1);
plot(time, result_input(:,1), 'r', 'LineWidth', 2);
hold on
plot(time, MPC_data.result_input(:,1), 'b--', 'LineWidth', 2);
plot(time,r_set,'c--', 'LineWidth', 3);
xlabel('Time [s]');
ylabel('State x_1');
legend('NN', 'MPC', 'Reference');
grid("on");
title('State x_1');

subplot(3, 1, 2);
plot(time, result_input(:,2), 'r', 'LineWidth', 2);
hold on
plot(time, MPC_data.result_input(:,2), 'b--', 'LineWidth', 2);
% plot(time, x_hat_store(2, :), 'r--', 'LineWidth', 2);

xlabel('Time [s]');
ylabel('State x_2');
legend('NN', 'MPC');
grid("on");
title('State x_2');

subplot(3, 1, 3);
plot(time, uset, 'k', 'LineWidth', 2);
hold on
plot(time, MPC_data.uset, 'b--', 'LineWidth', 2);
legend('NN', 'MPC');
grid("on");
xlabel('Time [s]');
ylabel('Control input u');
title('Control input u');
% exportgraphics(gcf, 'heater_res.png', 'Resolution', 300);