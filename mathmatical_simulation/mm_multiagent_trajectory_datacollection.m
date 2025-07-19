clc
close all
clear all


%% Specify data save path and loading initial values

% Save the data to a text file
filename = 'dataset/MM_DiffSys_dataset_trajectories_uniform.csv';


% =================read data of MM solved:=========================
dataset = readmatrix('dataset/MM_DiffSys_dataset.csv');

% ==================Generate INIT Points======================
% total 16 init points, each genreating separate trajectory

% method 1: uniformly sampled
% init_pts = zeros(16,2);
% cnt = 1;
% for r=-3:2:3
%     for c=-3:2:3
%         init_pts(cnt,:) = [r,c];
%         cnt = cnt + 1;
%     end
% end

% method 2: boundary only
x = linspace(-1, 1, 5);  % 4 points along each edge
y = linspace(-1, 1, 5);
% Get boundary points
bottom = [x; -1*ones(1,5)];
top    = [x;  1*ones(1,5)];
left   = [-1*ones(1,3); y(2:4)];  % exclude corners already in top/bottom
right  = [1*ones(1,3); y(2:4)];
% Combine all
init_pts = [bottom, top, left, right]';   % optional: make it 16x2
plot(init_pts(:,1),init_pts(:,2),'*')

pts_nums = size(init_pts,1);

%% ==================Start Looping for Data Collection======================
% define A,B,C,D matrices:
dt = 0.1;
Ad = [1, dt;0, 1];
Bd= [0;0];
C = [1 0];           % Output matrix
D = 0;               % Direct transmission matrix

% weights of optimization problem
Q = [2,0;0,2];
R = 0.1;
x_r = [0,0]';

for pts_num=1:pts_nums

    % set initial state and reference
    x = init_pts(pts_num,:)';
    
    
    % init errors, start while loop until stop threshold is satisfied.
    delta = 0.001;
    delta_p = 0.01;
    iteration = 1;
    steps = 30;
    
    
    for i=1:steps
    
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
    
            Objective1 = R*(norm(u))^2+ (Ad*x+Bd*u-x_r)'*Q*(Ad*x+Bd*u-x_r)+((x-x_r)'*pit*(x-x_r))+exp(-theta);
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
            % Slack=sdpvar(1,1,'full');
    
            Objective2 = R*(norm(ui))^2+ (Ad*x+Bd*ui-x_r)'*Q*(Ad*x+Bd*ui-x_r)+((x-x_r)'*P*(x-x_r))+exp(-thetai);
            Constraints = [P>=1e-10;
            ((x-x_r)'*P*(x-x_r))>=((1.5*thetai)^2)*(x-x_r)'*(x-x_r);
            ((Ad*x+Bd*ui-x_r)'*P*(Ad*x+Bd*ui-x_r))<=((0.5*thetai)^2)*(x-x_r)'*(x-x_r)];
    
    
            sol=optimize(Constraints,Objective2,opt)
    
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
            Objective3 = R*(norm(ui))^2+ (Ad*x+Bd*(ui)-x_r)'*Q*(Ad*x+Bd*(ui)-x_r)+((x-x_r)'*pit*(x-x_r))+exp(-thetai);
            Constraints = [pit>=Slack;Slack>=1e-10;
                w1>=0;w1<=1;w2>=0;w2<=1;w3>=0;w3<=1;
            ((x-x_r)'*pit*(x-x_r))>=((1.5*thetai)^2)*(x-x_r)'*(x-x_r);
            ((Ad*x+Bd*ui-x_r)'*pit*(Ad*x+Bd*ui-x_r))<=((0.5*thetai)^2)*(x-x_r)'*(x-x_r)];
            sol=optimize(Constraints,Objective3,opt)
    
            double([w1,w2,w3]);
            ui = double(ui);
            pit = double(pit);
            thetai = double(thetai);
    
             % update errors:
            error1 = norm(u_ii-ui);
            error2 = norm([p_ii(1,1);p_ii(1,2);p_ii(2,2)]-[pit(1,1);pit(1,2);pit(2,2)]);
            % error2 = norm(eig(p_ii/norm(p_ii)) - eig(pit/norm(pit)));
            error3 = norm(theta_ii-thetai);
    
            Cost(iteration)=R*(norm(ui))^2+ (Ad*x+Bd*ui-x_r)'*Q*(Ad*x+Bd*ui-x_r)+((x-x_r)'*pit*(x-x_r))+exp(-thetai);
            if iteration > 1
                error_cost = abs(Cost(iteration) - Cost(iteration-1));
            else
                error_cost = abs(Cost(iteration));
            end
            [error_cost,error1,error2,error3]
    
    
            iteration = iteration + 1;   
        end
    
        % ========= END OF MULTI-AGENT ====================
    
        % save solved data
        chache_matrix(:,1) = x;
        % eig(p)
        chache_matrix(:,2:3) = pit;
        chache_matrix(1,4) = ui;
        chache_matrix(2,4) = thetai;
        result_matrix(:,:,i) = chache_matrix;
       
        % update x with f() and g()
        x1_new = x(1,1) + dt * x(2,1);
        x2_new = x(2,1) + dt * (x(1,1)^3+(x(2,1)^2+1)*ui);
        x = [x1_new,x2_new]';

    end


    % save result matrix and clear it. recount numbers
    writematrix(result_matrix, filename,'WriteMode','append');
    clear result_matrix


end


