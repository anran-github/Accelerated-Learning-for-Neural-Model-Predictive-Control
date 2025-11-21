close all
clear all

% ===============SS MODEL and Observer settings===========================
% give canonical form of matrices:
% These values come from "heater_new_model.m": 
A = [0 -0.08091/159; 1 -15.34/159];
B = [0.06545/159; -0.72/159];
C = [0 1];
D = 0;

sys_d=ss(A,B,C,D);

% Discretize the transfer function with Ts = 3 seconds
Ts = 5;
sys_ss_d = c2d(sys_d, Ts);

% Get the state-space matrices of the discrete system
[A_d, B_d, C_d, D_d] = ssdata(sys_ss_d);

% Desired observer poles: should be less than unit circle in Discrete-time
observer_poles = [0.85 0.9];

% Place the poles of the observer
L = place(A_d', C_d', observer_poles)';
% eig(A_d-L*C_d)


% =================LQR SETTINGS===================

% Define LQR parameters
Q = [0.1 0;0 100];  % State cost (penalize deviation of states)
R = 0.001;       % Control effort cost (penalize large control inputs)

% Compute the LQR controller gain matrix K
K = dlqr(A_d, B_d, Q, R);

% Initial conditions
ambient_t = arduino_lab1(0);

x_observer = 0;      % Estimated state (observer)

G = inv(C_d * inv(eye(2) - A_d + B_d * K) * B_d);
% ref = inv(eye(2) - A_d + B_d * K) * B_d*G*desired_t
% xr = inv(eye(2) - A_d + B_d * K) * B_d*G*r;




% ===================LQR END========================


% ================== NN SETTINGS =========================
% Load ONNX model
net = importONNXNetwork("Heater_Results/uniform_S50_constant_Iter40_Epoch10.onnx", 'InputDataFormats', {'BC'}, 'OutputDataFormats', {'BC'});

% ==================NN setting end =========================


% Simulation parameters
T_final = 600;    % Final simulation time
N = T_final / Ts;  % Number of discrete time steps
stage = 4;
r=[15*ones(1,N/stage), 30*ones(1,N/stage), 20*ones(1,N/stage), 40*ones(1,N/stage) ];

% Arrays to store simulation results
x_store = zeros(2, N);        % Store actual state
x_hat_store = zeros(2, N);    % Store estimated state
u_store = zeros(1, N);        % Store control input


%% CORE SECTION:

for k = 1:N

    tic;

    if k == 1
        xt = [x_observer;0];
    end

    xr = [x_observer(1);r(k)];

    % different control laws for discrete-time system: 
    
    % LQR
    % u = -K * xt +G*desired_t;

    % MPC
    % horizon = 30;
    % uN = mpc_fun(A_d,B_d,Q,R,xt,xr,horizon);
    % u = uN(1);

    % YALMIP
    % [u,p_init,theta_init] = heater_original_optimation(xt,xr,A_d, B_d,R,Q,K,G);
    
    % NN MODEL
    data_input = [xt(1),xt(2),xr(2)];
    output = predict(net, data_input);
    u = output(1)*100;


    if u > 100
        u = 100;
    elseif u < 0
        u = 0;
    end


    y = C_d * xt;  % Measurement
    if k == 1
        x_observer = A_d * xt + B_d * u + L * (y - C_d * xt);  % Observer update
    else
        x_observer = A_d * x_hat_store(:,k-1) + B_d * u + L * (y - C_d * x_hat_store(:,k-1));  % Observer update
    end
    x_hat_store(:, k) = x_observer;


    elapsedTime = toc;  % Stop the timer and return the elapsed time

    time_rest = Ts - elapsedTime;
    if time_rest > 0
        pause(time_rest-0.028);
        % pause(time_rest);
    end

    x_2 = arduino_lab1(u)-ambient_t;  % read actural temperature.
    xt = [x_observer(1);x_2];

    % Store results
    x_store(:, k) = xt;
    u_store(k) = u;

    total_time = toc;


    disp(['Computing time: ', num2str(elapsedTime), ' seconds; ', ...
        ' Current input u: ', num2str(u), ' ; ', ...
        ' T_current | T_ref: ', num2str(y), '|', num2str(xr(2)), ' Celsius']);

end


%% Plot the results--SHOW UP IN PAPER

% TO DISPLAY PAPER FIGURE: LOAD SAVED .mat DATA. uncomment following:
% clear
% load("lqr.mat");
% load("result30_NN2.mat")
% x_store_lqr = load('mpc_heater_varyXr_X.mat');
% u_store_lqr = load('mpc_heater_varyXr_U.mat');
data_mpc = load('heater_mpc_T5.mat');


time = (0:N-1) * Ts;

% figure;
% subplot(2, 1, 1);
% plot(time, x_store(1, :), 'r', 'LineWidth', 2);
% hold on
% plot(time, data_mpc.x_store(1, 1:N), 'b', 'LineWidth', 2);
% xlabel('Time [s]');
% ylabel('State x_1');
% legend('NMPC', 'MPC','Location','southeast');
% grid("on");
% title('State x_1');
% 

% define figure size 800x300
figure('Position', [100, 100, 900, 600]);
subplot(2, 1, 1);
plot(time, x_store(2, :)+ambient_t, 'r', 'LineWidth', 2);
hold on;
% plot(time, x_hat_store(2, :), 'r--', 'LineWidth', 2);
% plot reference
plot(time, data_mpc.x_store(2, :)+ambient_t, 'b', 'LineWidth', 2);
plot(time,r+ambient_t,'--','Color', [0.6 0.6 0.6], 'LineWidth', 3);
% make x-axis and y-axis font size bigger
set(gca, 'FontSize', 16);
xlabel('Time [s]');
ylabel('Temperature [°C]');
legend('Neural MPC', 'MPC','Reference','Location','southeast');
grid("on");
% title('Temperature [°C]');

subplot(2, 1, 2);
plot(time, u_store, 'k', 'LineWidth', 2);
hold on
plot(time, data_mpc.u_store, 'm', 'LineWidth', 2);
% plot upper and lower bounds
yline(100, 'r--', 'LineWidth', 1.5);
yline(0, 'r--', 'LineWidth', 1.5);
% set y-axis range
ylim([-10 110]);
set(gca, 'FontSize', 16);
% label with u_max and u_min
% legend('Neural MPC','MPC', 'u_{max}', 'u_{min}','Location','southeast');
legend('Neural MPC','MPC', 'Constraint','Location','southeast');
grid("on");
xlabel('Time [s]');
ylabel('Control input');
% title('Control input u');
exportgraphics(gcf, 'heater_res.png', 'Resolution', 300);


% compute tracking error and control effort
tracking_error_mpc = norm(data_mpc.x_store(2,:) - r, 2);
tracking_error_nmpc = norm(x_store(2,:) - r, 2);
control_effort_mpc = norm(data_mpc.u_store, 2);
control_effort_nmpc = norm(u_store, 2);
disp(['Tracking Error NMPC: ', num2str(tracking_error_nmpc)]);
disp(['Tracking Error MPC: ', num2str(tracking_error_mpc)]);
disp(['Control Effort NMPC: ', num2str(control_effort_nmpc)]);
disp(['Control Effort MPC: ', num2str(control_effort_mpc)]);
