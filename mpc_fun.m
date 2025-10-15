function [uHist] = mpc_fun(Ad, Bd, Q, R, x0, xref,N)
        
    
    % Simulation length
    Tsim = 1;
    
    
    % Variables to store result
    xHist = zeros(2, Tsim+1);
    uHist = zeros(1, Tsim);
    
    xHist(:,1) = x0;
    
    % Input constraints (optional)
    u_max = 100;
    u_min = -100;
    
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
        for t = 1:30
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
        uHist = value(u);
        % xHist(:,k+1) = Ad * xHist(:,k) + Bd * uHist(k);
    end
    



end

