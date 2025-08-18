clc; clear;
close all;
yalmip('clear');


%% System matrices
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
% Ad = [1.0000 0.0916;
%       0      0.8363];
% Bd = [-0.0082;
%       -0.1592];
% Cd = [1 0];
% Dd = 0;

%% MPC settings
N = 50;              % Prediction horizon
Q = [20 0;0 10];          % State cost
R = 0.1;            % Input cost

%% Initial Point Selection

filename = 'droneZ_MPC_16pts_uniform_multi_xr.csv';
% method 1: uniformly sampled
init_pts = zeros(16,2);
cnt = 1;
for r=linspace(0.5,2.5,4)
    for c=linspace(-1,1,4)
        init_pts(cnt,:) = [r,c];
        cnt = cnt + 1;
    end
end

tic
%% Collect Trajectory Data
count = 1;
for r=1:0.1:2    
    for cnt=1:size(init_pts,1)
    
        % Initial state
        x0 = init_pts(cnt,:)';
                
        % Reference point
        xref = [r; 0];
                
                
        [u] = mpc_fun(Ad,Bd,Q,R,x0,xref,N);
    
        % save trajectory:
        % format: [x1,x2,xr,u]
        chache_matrix(1,1:2) = x0';
        chache_matrix(1,3) = xref(1);
        chache_matrix(1,4:3+N) = u;

        result_matrix(count,:) = chache_matrix;
        count = count + 1;
        clear('yalmip')
    
    end

    % save data every count numbers.
    writematrix(result_matrix, filename,'WriteMode','append');
    clear result_matrix chache_matrix
    count = 1;            % Clear variables

end

toc