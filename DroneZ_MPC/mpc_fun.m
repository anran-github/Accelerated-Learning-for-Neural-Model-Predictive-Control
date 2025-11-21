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

    % clear yalmip
    

    % Define decision variables
    nx = size(Ad,1);
    nu = size(Bd,2);

    % x = sdpvar(nx, N+1);
    u = sdpvar(nu, N);

    % Constraints and objective initialization
    constraints = [];
    cost = 0;

    % Input constraints
    u_min = -0.6;
    u_max = 0.6;

    % % Initial condition
    % x(:,1) = x0;
    

    % Build cost and constraints
    for k = 0:N-1
        if k==0
             cost = cost + (x0-xref)'*Q*(x0-xref) + u(:,k+1)'*R*u(:,k+1);
             x(:,k+1) = Ad*x0  + Bd*u(:,k+1);
             constraints = [constraints; u_min <= u(:,k+1); u(:,k+1) <= u_max];
        else
           cost = cost + (x(:,k)-xref)'*Q*(x(:,k)-xref) + u(:,k+1)'*R*u(:,k+1);
           x(:,k+1) = Ad*x(:,k)  + Bd*u(:,k+1);
           constraints = [constraints; u_min <= u(:,k+1); u(:,k+1) <= u_max];
        end
    end

    % Add terminal cost
    [Kcl, P, ~] = dlqr(Ad, Bd, Q, R);
    cost = cost + (x(:,N)-xref)' * P * (x(:,N)-xref);

   % Terminal feasibility using 20-step LQR rollout
      x_terminal = x(:,N);
       for t = 1:20
           if t==1
            u_lqr = -Kcl *(x(:,N)-xref) ;
            constraints = [constraints; u_min <= u_lqr; u_lqr <= u_max];
            x_terminal = Ad * x_terminal + Bd * u_lqr;
           else
            u_lqr = -Kcl *(x_terminal-xref) ;
            constraints = [constraints; u_min <= u_lqr; u_lqr <= u_max];
            x_terminal = Ad * x_terminal + Bd * u_lqr;
           end
        end


    % Solve optimization
    options = sdpsettings('solver','quadprog','verbose',0);
    % options = sdpsettings('solver','fmincon','verbose',0);


    diagnostics = optimize(constraints, cost, options);

    if diagnostics.problem ~= 0
        warning('MPC optimization did not fully converge.');
    end

    % Extract first control action sequence
    u_seq = value(u);

end