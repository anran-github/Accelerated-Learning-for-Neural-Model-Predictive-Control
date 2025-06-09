%% This is the new system: Inverted Pedulum system

clc
close all
clear all

%%  Basic Settings

% Save the data to a text file
filename = 'dataset/MM_DiffSys_dataset.csv';

% read A,B,C,D matrices:
dt = 0.1;
Ad = [1, dt;0, 1];
Bd= [0;0];

%% Load NOM dataset.
NOMdata = load('dataset/DifSYS_NOM_Dataset_0.1.txt');

% load current existing .csv file (if it exists, otherwise comment out):
% [column1:x1, x2 col2-3:P, col4: K]
existing_data = readmatrix(filename);
last_i = length(existing_data)/2;

% count = 1;
save_index = 1;
for count=(last_i+1):length(NOMdata)

    % get init values from NOM:
    % NOM data format: x1,x2,u,p1,p2,p3,r        
    x1 = NOMdata(count,1);
    x2 = NOMdata(count,2);
    
    p1 = NOMdata(count,4);
    p2 = NOMdata(count,5);
    p3 = NOMdata(count,6);


    % set initial state and reference
    x = [x1,x2]';
    u_init = NOMdata(count,3);
    p_init = [p1,p2;p2,p3];        
    theta_init = 0.1;
    x_r = [NOMdata(count,7),0]';
            
    % update Ad Bd with linearied term:
    Ad(2,1) = 3*dt*x(1,1)^2;
    Bd(2,1) = dt*(x(2,1)^2+1);
    
    
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
    
    % SAVE RESULT
    chache_matrix(:,1) = x;
    % eig(p)
    chache_matrix(:,2:3) = pit;
    chache_matrix(1,4) = ui;
    chache_matrix(2,4) = thetai;
    result_matrix(:,:,save_index) = chache_matrix;
    
    if mod(length(result_matrix),1)==0 || count>(length(NOMdata)-100)
        % save result matrix and clear it. recount numbers
        writematrix(result_matrix, filename,'WriteMode','append');
        clear result_matrix
        save_index = 1;      
    end
   
    % continue for next line
    count = count+1;
    
            

end






