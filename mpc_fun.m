function u_seq = mpc_fun(Ad, Bd, Q, R, x0, xref, N)
% Simplified finite-horizon MPC (1-step receding horizon)
% Inputs:
%   Ad, Bd - discrete system matrices
%   Q, R   - state/input cost matrices
%   x0     - current state
%   xref   - desired state
%   N      - prediction horizon
% Output:
%   u_seq  - optimal input sequence [u(1)...u(N)]

    % Define decision variables
    nx = size(Ad,1);
    nu = size(Bd,2);

    x = sdpvar(nx, N+1);
    u = sdpvar(nu, N);

    % Constraints and objective initialization
    constraints = [];
    cost = 0;

    % Input constraints
    u_min = 0;
    u_max = 100;

    % Initial condition
    constraints = [constraints, x(:,1) == x0];

    % Build cost and constraints
    for k = 1:N
        cost = cost + (x(:,k)-xref)'*Q*(x(:,k)-xref) + u(:,k)'*R*u(:,k);
        constraints = [constraints, x(:,k+1) == Ad*x(:,k) + Bd*u(:,k)];
        constraints = [constraints, u_min <= u(:,k) <= u_max];
    end

    % Add terminal cost
    [~, P, ~] = dlqr(Ad, Bd, Q, R);
    cost = cost + (x(:,N+1)-xref)' * P * (x(:,N+1)-xref);

    % Solve optimization
    % options = sdpsettings('solver','quadprog','verbose',0);
    options = sdpsettings('solver','fmincon','verbose',0);


    diagnostics = optimize(constraints, cost, options);

    if diagnostics.problem ~= 0
        warning('MPC optimization did not fully converge.');
    end

    % Extract first control action sequence
    u_seq = value(u);
end
