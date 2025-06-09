function [u_ii, p_ii, theta_ii] = mm_original_optimation(x0,r, Ad, Bd)
    
    % unit: need radian
    x = x0;
    x_r = r;
    
    % variables to optimize
    u = sdpvar(1,1,'full');
    P=sdpvar(2,2,'symmetric');
    theta=sdpvar(1,1,'full');

    
    Objective = 0.1*(norm(u))^2+2*(norm(Ad*x+Bd*u-x_r)^2)+((x-x_r)'*P*(x-x_r))+exp(-theta);
    Constraints = [P==[(1.5*theta)^2 0;0 (1.5*theta)^2];theta==1;
    ((x-x_r)'*P*(x-x_r))>=((1.5*theta)^2)*(x-x_r)'*(x-x_r);
    ((Ad*x+Bd*u-x_r)'*P*(Ad*x+Bd*u-x_r))<=((0.5*theta)^2)*(x-x_r)'*(x-x_r)];
    
    % opt=sdpsettings('solver','bmibnb');
    opt=sdpsettings('solver','fmincon','MaxIter',1000);
    sol=optimize(Constraints,Objective,opt)
    u_ii = double(u);
    p_ii = double(P);
    theta_ii=double(theta);

end