%% This is the new system: Inverted Pedulum system

clc
close all
clear all


%%  Basic Settings

% Save the data to a text file
filename = "drone_mpc_z.csv";



% System matrices
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

% reference pose
x_r = [1.5;0];

% MPC weights
N = 50;              % Prediction horizon
Q = [20 0;0 10];          % State cost
R = 0.1;            % Input cost


%% Resume data collection task

% [x1,x2,xr,u]

% resume from previous dataset points:

Resume_Flag = 0;

if Resume_Flag == 1
    % Read the CSV file into a matrix
    data = csvread(filename);
    
    % Find the total number of rows
    num_rows = size(data, 1);
    
    % Extract the last two rows of the first column
    last_two_rows = data(num_rows, 1:2);
    x1_resume = last_two_rows(1);
    x1_dot_resume = last_two_rows(2);
end



%% Loop starts

steps = 0.02;

if Resume_Flag == 1
    x1_range = x1_resume:steps:2.5;
else
    x1_range = 0.5:steps:2.5;
end

count = 1;
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
        [u] = mpc_fun(Ad,Bd,Q,R,x,x_r,N);
        

        chache_matrix(1,1:2) = x';
        chache_matrix(1,3) = x_r(1);
        chache_matrix(1,4:3+N) = u;
        

        result_matrix(count,:) = chache_matrix;
    

        % save data every count numbers.
        if count == 9 || (px==x1_range(size(x1_range,2)) && px_dot == 1) 
            writematrix(result_matrix, filename,'WriteMode','append');
            clear result_matrix chache_matrix
            count = 1;            % Clear variables
        else
            count = count+1;
        end


        clear('yalmip')
   
    end
end





