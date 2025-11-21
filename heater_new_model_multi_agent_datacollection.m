close all
clear all

% CODE REVISED FROM heater_new_model_hardware.m

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
horizon = 30;
sys_ss_d = c2d(sys_d, Ts);
% Get the state-space matrices of the discrete system
[A_d, B_d, C_d, D_d] = ssdata(sys_ss_d);

% Desired observer poles: should be less than unit circle in Discrete-time
observer_poles = [0.85 0.9];

% Place the poles of the observer
L = place(A_d', C_d', observer_poles)';
% eig(A_d-L*C_d)


% =================MULTI-AGENT PARAM SETTINGS===================

% Define LQR parameters
Q = [0.1 0;0 100];  % State cost (penalize deviation of states)
R = 0.001;       % Control effort cost (penalize large control inputs)

% filename for saving dataset
% Compute the LQR controller gain matrix K
K = dlqr(A_d, B_d, Q, R);

x_observer = 0;      % Estimated state (observer)

% try to find out x1_ref
G = inv(C_d * inv(eye(2) - A_d + B_d * K) * B_d);



% saving file setting
file_name = 'mpc_data_heater_10.csv';


% open file
M = readmatrix('heater_sampling_points_10.csv');

%% DATA COLLECTION SECTION:


for i=1:size(M,1) % reference

    error = 100;
    
    % Initial conditions
    x_observer = 0;      % Estimated state (observer)
    k = 1;

    % find x_r with LQR;
    % xr = inv(eye(2) - A_d + B_d * K) * B_d*G*r;
    xr=[M(i,1);M(i,3)];
    xt = M(i,1:2)';
    % continue iterations until error reached.
    cache_matrix = zeros(1,horizon+3);
    tic;

    uN = mpc_fun(A_d,B_d,Q,R,xt,xr,horizon);
    T_end = toc;

    disp(['MPC Computation Time for reference ', num2str(M(i,3)), ': ', num2str(T_end), ' seconds.']);

    result_matrix = [M(i,:),uN];

    writematrix(result_matrix, file_name,'WriteMode','append');

    
    % while error > 0.005 && k <= 200
    % 
    %     % if k == 1
    %     %     xt = [x_observer;0];
    %     %     % generate points start from random init temperature
    %     %     if rand() < 0.4
    %     %         tmp_t = rand()*30 + 5;
    %     %         xt = [tmp_t*0.1;tmp_t]; % between 5-35
    %     %     end
    %     % end
    % 
    %     % Select different control law for the discrete-time system: 
    %     uN = mpc_fun(A_d,B_d,Q,R,xt,xr,horizon);
    %     u = uN(1);
    % 
    % 
    %     % update observer
    %     y = C_d * xt;  
    %     if k == 1
    %         x_observer = A_d * xt + B_d * u + L * (y - C_d * xt);  % Observer update
    %     else
    %         x_observer = A_d * x_hat_store(:,k-1) + B_d * u + L * (y - C_d * x_hat_store(:,k-1));  % Observer update
    %     end
    % 
    %     x_hat_store(:, k) = x_observer;
    % 
    %     % Store results
    %     % ========== Data Format ===========
    %     %  x1   x2(Th)  r  u1   u2, ... , uN
    %     % ==================================
    % 
    %     % SAVE RESULT
    %     cache_matrix(1,1:2) = xt;
    %     cache_matrix(1,3) = r;
    %     cache_matrix(1,4:horizon+3) = uN;
    %     if k==1
    %         result_matrix = cache_matrix;
    %     else
    %         result_matrix = cat(1,result_matrix, cache_matrix);
    %     end
    % 
    %     % update temperature with SS model.
    %     x_2 = A_d * xt + B_d * u;   % Adx+Bdu for discrete-time 
    %     xt = [x_observer(1);x_2(2)];
    % 
    %     error = abs(x_2(2) - r);
    %     k = k+1;
    % end
    % 
    % % save data
    % writematrix(result_matrix, file_name,'WriteMode','append');
    % clear result_matrix x_hat_store 

end
