%% This is the new system: Inverted Pedulum system

clc
close all
clear all


%%  Basic Settings
% % theta
% theta = 0.01;
Q = [100 0;0 10];  % State cost (penalize deviation of states)
R = 0.1;       % Control effort cost (penalize large control inputs)

% Save the data to a text file
filename = "drone_mpc_z.csv";


%% System settings

alpha_x = 0.0527;
alpha_y = 0.0187;
alpha_z = 1.7873;
alpha = [alpha_x,alpha_y,alpha_z];

beta_x = -5.4779;
beta_y = -7.0608;
beta_z = -1.7382;
beta = [beta_x,beta_y,beta_z];

%% Loop starts

steps = 0.02;

% [column1:x1, x2 col2-3:P, col4: K]

% resume from previous dataset points:

Resume_Flag = 0;

if Resume_Flag == 1
    % Read the CSV file into a matrix
    data = csvread(filename);
    
    % Find the total number of rows
    num_rows = size(data, 1);
    
    % Extract the last two rows of the first column
    last_two_rows = data((num_rows-1):num_rows, 1);
    x1_resume = last_two_rows(1);
    x1_dot_resume = last_two_rows(2);
end


direction=3;

count = 1;

% read A,B,C,D matrices:
A = [0 1 ; 0 -alpha(direction)];
B=[0;beta(direction)];
C = [1 0];
D=0;
G=ss(A,B,C,D);

Gd=c2d(G,0.1);
Ad=Gd.A
Bd=Gd.B
Cd=Gd.C
Dd=Gd.D

% reference pose
x_r = [0;0];
if Resume_Flag == 1
    x1_range = x1_resume:steps:1;
else
    x1_range = -1:steps:1;
end

if direction == 3
    x_r = [1.5;0];
    Q = [20 0;0 0.1];
    if Resume_Flag == 1
        x1_range = x1_resume:steps:2.5;
    else
        x1_range = 0.5:steps:2.5;
    end
end


for px = x1_range

    x2_range = -1:steps:1;
    if Resume_Flag == 1 % run only once.
        x2_range = x1_dot_resume:steps:1;
    end

    for px_dot = x2_range

        % save only for next data point
        if Resume_Flag == 1
            Resume_Flag=0;
            continue
        end

        % current state
        x = [px;px_dot];

        % optimize
        [u,P,theta] = multi_agent_algorithm(x,x_r,Ad,Bd,R,Q);
        

        chache_matrix(:,1) = x;
        p = double(P);
        % eig(p)
        chache_matrix(:,2:3) = p;
        chache_matrix(1,4) = double(u);
        chache_matrix(2,4) = theta;
        result_matrix(count:count+1,:) = chache_matrix;
    

        % save data every count numbers.
        if count == 9 || (px==x1_range(size(x1_range,2)) && px_dot == 1) 
            writematrix(result_matrix, filename,'WriteMode','append');
            clear result_matrix chache_matrix
            count = 1;            % Clear variables
        else
            count = count+2;
        end


        clear('yalmip')
   
    end
end





