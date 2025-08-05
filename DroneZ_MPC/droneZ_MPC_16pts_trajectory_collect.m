clc; clear;
close all;
yalmip('clear');


%% System matrices
A = [0,1;0,-1.7873];
B = [0;-1.7382];
C = [1,0];
D=0;
G=ss(A,B,C,D);

Gd=c2d(G,0.1);
Ad=Gd.A;
Bd=Gd.B;
Cd=Gd.C;
Dd=Gd.D;
% Ad = [1.0000 0.0916;
%       0      0.8363];
% Bd = [-0.0082;
%       -0.1592];
% Cd = [1 0];
% Dd = 0;

%% MPC settings
N = 50;              % Prediction horizon
Q = [20 0;0 10];          % State cost
R = 0.1;            % Input cost

%% Initial Point Selection

filename = 'droneZ_MPC_16trajectory.csv';
% method 1: uniformly sampled
init_pts = zeros(16,2);
cnt = 1;
for r=linspace(0.5,2.5,4)
    for c=linspace(-1,1,4)
        init_pts(cnt,:) = [r,c];
        cnt = cnt + 1;
    end
end


%% Collect Trajectory Data
for cnt=1:size(init_pts,1)

    % Initial state
    x0 = init_pts(cnt,:)';
    
    % Simulation length
    Tsim = 100;
    
    % Reference point
    xref = [1.5; 0];
    
    % Variables to store result
    xHist = zeros(2, Tsim+1);
    uHist = zeros(1, Tsim);
    
    xHist(:,1) = x0;
    
    % Input constraints (optional)
    u_max = 0.6;
    u_min = -0.6;
    
    for k = 1:Tsim
        % Decision variables
    
        u = sdpvar(1, N);
        
        % Objective and constraints
        cost =(xHist(:,k)-xref)'*Q*(xHist(:,k)-xref) + u(:,1)'*R*u(:,1);
        constraints = [u_min <= u(:,1), u(:,1)<= u_max];
       
        
        
        for i = 1:N
            if i==1
                x(:,i) = Ad*xHist(:,k) + Bd*u(:,i);
            else
                x(:,i) = Ad*x(:,i-1) + Bd*u(:,i);
            end
            cost = cost + (x(:,i)-xref)'*Q*(x(:,i)-xref) + u(:,i)'*R*u(:,i);
            constraints = [constraints,u_min <= u(:,i), u(:,i) <= u_max];
        end
    
    
        % % add terminal conditions:
        [K, P, ~] = dlqr(Ad, Bd, Q, R);
    
        % Add terminal cost
        cost = cost + (x(:,N)-xref)' * P * (x(:,N)-xref);
    
        % Terminal feasibility using 20-step LQR rollout
        x_terminal = x(:,N);
        for t = 1:20
            u_lqr = -K * x_terminal;
            constraints = [constraints, u_min <= u_lqr, u_lqr <= u_max];
            x_terminal = Ad * x_terminal + Bd * u_lqr;
        end
    
            
        % Set options and solve
        % options = sdpsettings('solver','quadprog','verbose',0);
        options = sdpsettings('solver','fmincon');
        diagnostics = optimize(constraints, cost, options);
        
        if diagnostics.problem ~= 0
            error('The optimization problem was not solved!');
        end
        
        % Apply control and simulate system
        uHist(k) = value(u(:,1));
        xHist(:,k+1) = Ad * xHist(:,k) + Bd * uHist(k);

        % save cache
        result_matrix(k,:) = [xHist(1,k),xHist(2,k),xref(1),value(u)];
    end

    % save trajectory:
    % format: [x1,x2,xr,u]
    writematrix(result_matrix, filename,'WriteMode','append');
    clear result_matrix


end


%% Plot results
figure;
subplot(2,1,1);
plot(0:Tsim, xHist(1,:), 'b', 'LineWidth', 2); hold on;
plot(0:Tsim, xHist(2,:), 'r', 'LineWidth', 2);
yline(xref(1), '--b');
legend('x_1','x_2');
xlabel('Time Step'); ylabel('States');
title('MPC State Trajectory using YALMIP');

subplot(2,1,2);
stairs(0:Tsim-1, uHist, 'k', 'LineWidth', 2);
xlabel('Time Step'); ylabel('Control Input');
title('MPC Control Input');
