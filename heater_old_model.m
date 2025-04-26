
%% Model Heater System:
close all
clear all

% Define the transfer function

% These values come from "arduino_lab_R2016a_update_2023a.slx" linear
% model.
num = [0.72];   % Numerator coefficients
den = [159 0.89]; % Denominator coefficients
sys = tf(num, den);  % Create the transfer function

% Define a time delay of 3 seconds
delay = 22;

A = [0 -0.08091/159; 1 -15.34/159];
B = [0.06545/159; -0.72/159];
C = [0 1];
D = [0];



% Use Pade approximation to represent the time delay
[pade_num, pade_den] = pade(delay, 1);  % Pade approximation
delay_sys = tf(pade_num, pade_den);  % Create the delay system

% Combine the system and the delay
sys_with_delay = series(sys, delay_sys);  % Series combination of the transfer function and the delay

% Convert the discretized transfer function to state-space form
sys_d = ss(sys_with_delay,'minimal');
sys_d=ss(A,B,C,D);

% Discretize the transfer function with Ts = 3 seconds
Ts = 3;
sys_ss_d = c2d(sys_d, Ts);



% Get the state-space matrices of the discrete system
[A_d, B_d, C_d, D_d] = ssdata(sys_ss_d);

% check observerbility:
rank([C_d;C_d*A_d])

% Desired observer poles: should be less than unit circle in Discrete-time
observer_poles = [0.5 0.7];

% Place the poles of the observer
L = place(A_d', C_d', observer_poles)';
eig(A_d-L*C_d)

% The observer gain L will estimate the second state x2




%% Simulation:


% Define LQR parameters
Q = [1 0;0 20];  % State cost (penalize deviation of states)
R = 1;       % Control effort cost (penalize large control inputs)

% Compute the LQR controller gain matrix K
K = dlqr(A_d, B_d, Q, R);

% Initial conditions
ambient_t = 22;          % Actual state
desired_t = 85;
x_hat = 0;      % Estimated state (observer)


% Simulation parameters
T_final = 510;    % Final simulation time
N = T_final / Ts;  % Number of discrete time steps

% Arrays to store simulation results
x_store = zeros(2, N);        % Store actual state
x_hat_store = zeros(2, N);    % Store estimated state
u_store = zeros(1, N);        % Store control input


G = inv(C_d * inv(eye(2) - A_d + B_d * K) * B_d);

ref = inv(eye(2) - A_d + B_d * K) * B_d*G*desired_t;             % set x_dot = 0 to find x1 value.

% Simulation loop (discrete-time LQR and observer)
for k = 1:N

    if k == 1
        xt = [x_hat;ambient_t];
    end

    % LQR control law for discrete-time system: 
    u = -K * xt +G*desired_t;  


    y = C_d * xt;  % Measurement
    x_hat = A_d * xt + B_d * u + L * (y - C_d * xt);  % Observer update

    % Discrete-time system dynamics: x[k+1] = A_d * x[k] + B_d * u[k]
    xt = A_d * [x_hat(1);y] + B_d * u;
    xt = A_d * xt + B_d * u;

    % Store results
    x_store(:, k) = xt;
    x_hat_store(:, k) = x_hat;
    u_store(k) = u;
    
    end

%% Plot the results
time = (0:N-1) * Ts;

figure;
subplot(3, 1, 1);
plot(time, x_store(1, :), 'b', 'LineWidth', 2);
hold on;
plot(time, x_hat_store(1, :), 'r--', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('State x1');
legend('True x1', 'Estimated x1');
title('State x1 and its estimate');

subplot(3, 1, 2);
plot(time, x_store(2, :), 'b', 'LineWidth', 2);
hold on;
plot(time, x_hat_store(2, :), 'r--', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('State x2');
legend('True x2', 'Estimated x2');
title('State x2 and its estimate');

subplot(3, 1, 3);
plot(time, u_store, 'k', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('Control input u');
title('Control input u (LQR)');