clc; clear;

% System matrices
Ad = [1.0000    0.0916;
         0    0.8363];

Bd = [-0.0082;
      -0.1592];

Cd = [1 0];
Dd = 0;

% Dimensions
nx = size(Ad,1);
nu = size(Bd,2);

% MPC parameters
N = 10;  % prediction horizon
Q = eye(nx);       % state weighting
R = 0.01;          % input weighting

% Simulation settings
Tsim = 50;
x = zeros(nx, Tsim+1);
x(:,1) = [0; 0];  % initial state
u = zeros(nu, Tsim);
xref = [1.5; 0];  % target state

% MPC loop
for k = 1:Tsim
    % Build cost function: H, f
    H = blkdiag(kron(eye(N), Q), kron(eye(N), R));
    f = zeros(N*nx + N*nu, 1);
    
    % Build prediction matrices
    % x = Fx0 + Gu
    Fx0 = zeros(N*nx, nx);
    G = zeros(N*nx, N*nu);
    
    for i = 1:N
        Fx0((i-1)*nx+1:i*nx,:) = Ad^i;
        for j = 1:i
            G((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = Ad^(i-j)*Bd;
        end
    end
    
    % Objective: minimize (x - xref)'Q(x - xref) + u'Ru
    % So we build z = [x1; x2; ...; xN; u0; u1; ...; u_{N-1}]
    % and express x in terms of x0 and U (decision variable)
    
    % Target state trajectory
    Xref = repmat(xref, N, 1);
    
    % Linear equality constraint: x = Fx0 + G*u
    % Let z = [u]
    Aeq = [G];     % x = Fx0 + G*u
    beq = Xref - Fx0 * x(:,k);
    
    % Cost: minimize ||x - Xref||_Q + ||u||_R
    % So cost becomes: (Gu - (Xref - Fx0 x0))'Q(Gu - (Xref - Fx0 x0)) + u'Ru
    % Build total H and f accordingly
    H = G'*kron(eye(N), Q)*G + kron(eye(N), R);
    f = -G'*kron(eye(N), Q)*(Xref - Fx0*x(:,k));
    
    % Solve QP: minimize (1/2)*u'Hu + f'u
    % Constraints can be added here (e.g., input bounds)
    options = optimoptions('quadprog','Display','off');
    [U_opt,~,exitflag] = quadprog(2*H, f, [], [], [], [], [], [], [], options);
    
    if exitflag ~= 1
        warning('QP did not converge');
        break;
    end
    
    % Apply only the first control input
    u(:,k) = U_opt(1:nu);
    x(:,k+1) = Ad*x(:,k) + Bd*u(:,k);
end

% Plot results
figure;
subplot(2,1,1);
plot(0:Tsim, x(1,:), 'b', 'LineWidth', 2); hold on;
plot(0:Tsim, x(2,:), 'r', 'LineWidth', 2);
yline(xref(1), '--b');
legend('x_1','x_2');
xlabel('Time Step'); ylabel('States');
title('MPC State Trajectory');

subplot(2,1,2);
stairs(0:Tsim-1, u, 'k', 'LineWidth', 2);
xlabel('Time Step'); ylabel('Control Input');
title('MPC Control Input');
