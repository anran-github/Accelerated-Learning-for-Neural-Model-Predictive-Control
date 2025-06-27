clc
close all
clear all


%% STATE SPACE Define
% read A,B,C,D matrices:
dt = 0.1;
Ad = [1, dt;0, 1];
Bd= [0;0];

% Save the data to a text file
filename = 'dataset/MM_DiffSys_dataset_trajectories.csv';

% read data of MM solved:
dataset = readmatrix('dataset/MM_DiffSys_dataset.csv');
% set initial state and reference
x = [2,-1]';
x_r = [0,0]';

% Qx = [2,0;0,2];

% init errors, start while loop until stop threshold is satisfied.
delta = 0.001;
delta_p = 0.01;
iteration = 1;
steps = 50;

x_set = zeros(steps,1,2);
x_set(1,:,:)=x;

for i=2:steps+1

    % update linearized SS matrices:
    Ad(2,1) = 3*dt*x(1,1)^2;
    Bd(2,1) = dt*(x(2,1)^2+1);

    % get init values from origional one-step ahead method:
    % [u_init,p_init,theta_init] = mm_original_optimation(x,x_r,Ad, Bd);


    % ============= get init values from collected dataset:================
    % find min distance between x:
    diff = size(zeros(size(dataset, 1)/2),1);
    for idx=1:size(dataset, 1)/2
        diff(idx) = norm(dataset(idx*2-1:idx*2,1) - x);
    end
    [~, idx_min] = min(diff);
    init_val = dataset(idx_min*2-1:idx_min*2,:);

    % Retrieve the closest data point
    u_init     = init_val(1,4);
    p_init     = init_val(1:2,2:3);
    theta_init = init_val(2,4);
    % ===============================END===================================
    
 
    % init errors, start while loop until stop threshold is satisfied.
    delta = 0.01;
    error1 = 1;
    error2 = 1;
    error3 = 1;
    error_cost = 1;
    iteration = 1;
    % ========= MAIN LOOP OF MULTI-AGENT ===========
    while iteration <= 1e10
        
        % if error1 <= delta && error2 <= delta && error3 <= delta
        if error_cost <= delta
            break
        end
        % start with optimize u      
        if iteration == 1
            pit = p_init;
            ui = u_init;
            thetai=theta_init;
    
        end
        % ==================OPT U=======================
        clear yalmip
        u = sdpvar(1,1,'full');
        theta=sdpvar(1,1,'full');
        % slack = sdpvar(1,1,'full');
        
            
        Objective1 = 0.1*(norm(u))^2+ 2*(norm(Ad*x+Bd*u-x_r)^2)+((x-x_r)'*pit*(x-x_r))+exp(-theta);
        Constraints = [theta>=1e-10;((x-x_r)'*pit*(x-x_r))>=((1.5*theta)^2)*(x-x_r)'*(x-x_r);
            ((Ad*x+Bd*u-x_r)'*pit*(Ad*x+Bd*u-x_r))<=((0.5*theta)^2)*(x-x_r)'*(x-x_r)];
    
        opt=sdpsettings('solver','fmincon');
        sol=optimize(Constraints,Objective1,opt)
    
        % update u with optimized solution
        u_ii = double(u);
        theta_ii = double(theta);
    
        % ==================OPT P=======================
        clear yalmip
        P=sdpvar(2,2,'symmetric');
        Slack=sdpvar(1,1,'full');
        
        Objective2 = 0.1*(norm(ui))^2+ 2*(norm(Ad*x+Bd*ui-x_r)^2)+((x-x_r)'*P*(x-x_r))+exp(-thetai);
        Constraints = [P>=1e-10;
        ((x-x_r)'*P*(x-x_r))>=((1.5*thetai)^2)*(x-x_r)'*(x-x_r);
        ((Ad*x+Bd*ui-x_r)'*P*(Ad*x+Bd*ui-x_r))<=((0.5*thetai)^2)*(x-x_r)'*(x-x_r)];
    
    
        sol=optimize(Constraints,Objective2,opt)
        % cost2(iteration) = double(Objective2);
        
        % update p with optimal solution
        p_ii = double(P);
    
           
        % ================== update pi, ui, thetai with optimal w ===========
        w1 = sdpvar(1,1,'full');
        w2 = sdpvar(1,1,'full');
        w3 = sdpvar(1,1,'full');
    
        Slack=sdpvar(1,1,'full');
        ui = w1*u_ii+(1-w1)*ui;
        pit = w2*p_ii+(1-w2)*pit;
        thetai=w3*theta_ii+(1-w3)*thetai;
        Objective3 = 0.1*(norm(ui))^2+ 2*(norm(Ad*x+Bd*(ui)-x_r)^2)+((x-x_r)'*pit*(x-x_r))+exp(-thetai);
        Constraints = [pit>=Slack;Slack>=1e-10;
            w1>=0;w1<=1;w2>=0;w2<=1;w3>=0;w3<=1;
        ((x-x_r)'*pit*(x-x_r))>=((1.5*thetai)^2)*(x-x_r)'*(x-x_r);
        ((Ad*x+Bd*ui-x_r)'*pit*(Ad*x+Bd*ui-x_r))<=((0.5*thetai)^2)*(x-x_r)'*(x-x_r)];
        sol=optimize(Constraints,Objective3,opt)
    
        double([w1,w2,w3])
        ui = double(ui);
        pit = double(pit);
        thetai = double(thetai);

         % update errors:
        error1 = norm(u_ii-ui);
        error2 = norm([p_ii(1,1);p_ii(1,2);p_ii(2,2)]-[pit(1,1);pit(1,2);pit(2,2)]);
        % error2 = norm(eig(p_ii/norm(p_ii)) - eig(pit/norm(pit)));
        error3 = norm(theta_ii-thetai);
        
        Cost(iteration)=0.1*(norm(ui))^2+ 2*(norm(Ad*x+Bd*ui-x_r)^2)+((x-x_r)'*pit*(x-x_r))+exp(-thetai);
        if iteration > 1
            error_cost = abs(Cost(iteration) - Cost(iteration-1));
        else
            error_cost = abs(Cost(iteration));
        end
        [error_cost,error1,error2,error3]
    
    
        iteration = iteration + 1;   
    end
    


    % update x with f() and g()
    x1_new = x(1,1) + dt * x(2,1);
    x2_new = x(2,1) + dt * (x(1,1)^3+(x(2,1)^2+1)*ui);
    x = [x1_new,x2_new]';
    x_set(i,:,:)=x;
 
end

%% Plot phase protrait x:

plot(x_set(1:i-1,1,1),x_set(1:i-1,:,2))
grid
xlabel('x1', 'FontName', 'Times New Roman');
ylabel('x2', 'FontName', 'Times New Roman');
set(gca, 'FontName', 'Times New Roman');
title(strcat('Phase Protrait x'))
