clc
close all
clear all


%% STATE SPACE Define
% read A,B,C,D matrices:
dt = 0.1;
Ad = [1, dt;0, 1];
Bd= [0;0];

% set initial state and reference
x = [-2,0.3]';

r = [0,0]';

% update Ad Bd with linearied term:
Ad(2,1) = 3*dt*x(1,1)^2;
Bd(2,1) = dt*(x(2,1)^2+1);
% get init values from origional one-step ahead method:
[u0,p0,theta0] = mm_original_optimation(x,r,Ad, Bd);

% data from NOM:
% theta0 = 0.1;
% u0 = -0.11808618903160095;
% p0 = [0.0015205196104943752, -0.021180329844355583; -0.021180329844355583, 1.4490597248077393];

p_init = p0;
u_init = u0;
theta_init=theta0;
x_r = r;


% init errors, start while loop until stop threshold is satisfied.
delta = 0.001;
error1 = 1;
error2 = 1;
error3 = 1;
iteration = 1;

while iteration <= 1e10
    
    if error1 <= delta && error2 <= delta && error3 <= delta
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

    opt=sdpsettings('solver','fmincon','MaxIter',3000);
    sol=optimize(Constraints,Objective1,opt)

    % cost1(iteration) = double(Objective1);
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

   
    % ============== update pi and ui with fixed w1 ====================
    % w1=0.5;
    % w2=0.5;
    % ui = w1*u_ii+w2*ui;
    % pit = w1*p_ii+w2*pit;
    % thetai=w1*theta_ii+w2*thetai;


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


    % =====================use only one w =========================
    % w1 = sdpvar(1,1,'full');
    % Slack=sdpvar(1,1,'full');
    % w2 = 1 - w1;
    % ui = w1*u_ii+w2*ui;
    % pit = w1*p_ii+w2*pit;
    % thetai=w1*theta_ii+w2*thetai;
    % Objective3 = 0.1*(norm(ui))^2+ 2*(norm(Ad*x+Bd*(ui)-x_r)^2)+((x-x_r)'*pit*(x-x_r))+exp(-thetai);
    % Constraints = [w1>=0;w1<=1;pit>=Slack;Slack>=1e-10;
    % ((x-x_r)'*pit*(x-x_r))>=((1.5*thetai)^2)*(x-x_r)'*(x-x_r);
    % ((Ad*x+Bd*ui-x_r)'*pit*(Ad*x+Bd*ui-x_r))<=((0.5*thetai)^2)*(x-x_r)'*(x-x_r)];
    % sol=optimize(Constraints,Objective3,opt)
    % 
    % double(w1)
    % ui = double(ui);
    % pit = double(pit);
    % thetai = double(thetai);

    % ==========================Weight Calculate End==================================


    % update errors:
    error1 = norm(u_ii-ui);
    error2 = norm([p_ii(1,1);p_ii(1,2);p_ii(2,2)]-[pit(1,1);pit(1,2);pit(2,2)]);
    % error2 = norm(eig(p_ii/norm(p_ii)) - eig(pit/norm(pit)));
    error3 = norm(theta_ii-thetai);
    [error1,error2,error3]

    Cost2(iteration)=0.1*(norm(ui))^2+ 2*(norm(Ad*x+Bd*ui-x_r)^2)+((x-x_r)'*pit*(x-x_r))+exp(-thetai);

    iteration = iteration + 1;   
end


%% plot cost function
% two initial points: x = [0.1,0.1] and x = [-2,0.3]
clear all
close all

load('numerical_test1.mat');
load('numerical_test2.mat');

subplot(2,1,1);
plot(1:size(Cost,2),Cost/max(Cost))
grid
xlabel('Iteration', 'FontName', 'Times New Roman');
ylabel('Cost Function Magnitude', 'FontName', 'Times New Roman');
ylim([0.83,1]);
set(gca, 'FontName', 'Times New Roman');
title(strcat('x_0=[0.1,0.1]'))
subplot(2,1,2);
plot(1:size(Cost2,2),Cost2/max(Cost2))
grid
xlabel('Iteration', 'FontName', 'Times New Roman');
ylabel('Cost Function Magnitude', 'FontName', 'Times New Roman');
set(gca, 'FontName', 'Times New Roman');
title('x_0=[-2,0.3]')

saveas(gcf, 'numerical_init.png');  % Save as PNG
saveas(gcf, 'numerical_init.fig');  % Save as MATLAB FIG file