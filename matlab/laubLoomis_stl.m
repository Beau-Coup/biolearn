function dx = laubLoomis_stl(x,u)
% laubLoomis - dynamic equation for the Laub-Loomis benchmark 
%              (see Sec. 3.2 in [1])
%
% Syntax:
%    dx = laubLoomis(x,u)
%
% Inputs:
%    x - state vector
%    u - input vector
%
% Outputs:
%    dx - time-derivate of the system state
% 
% References:
%    [1] F. Immler, "ARCH-COMP19 Category Report: Continuous and Hybrid 
%        Systems with Nonlinear Dynamics", 2019

% Authors:       Niklas Kochdumper
% Written:       19-June-2020
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

    pos = @(x) exp(x);
    
    % [Array([ 0.99747209, -0.57499656, -0.5789802 , -1.83342623, -0.79213622,
    % -1.99736468, -2.27500342], dtype=float64), Array([-0.30890153, -2.15500641,  1.05173124, -0.73161133, -1.68358933,
    % -0.82089658, -1.94029467], dtype=float64)]
    
    k_1 =1.4;%pos(0.99747209);% 1.4;
    k_3 =pos(-0.57499656);% 2.5;
    k_5 =0.6;%pos(-0.5789802);%0.6;
    k_7 =pos(-1.83342623);% 2.0;
    k_9 =pos(-0.79213622);%0.7;
    k_11 =pos(-1.99736468);%0.3;
    k_13 =pos(-2.27500342);%1.8;
    
    k_2 = 0.9;%pos(-0.30890153);%0.9;
    k_4 = pos(-2.15500641);%1.5;
    k_6 = 0.8;%pos(1.05173124);%0.8;
    k_8 = pos(-0.73161133);%1.3;
    k_10 = pos(-1.68358933);%1.0;
    k_12 = pos(-0.82089658);%3.1;
    k_14 = pos(-1.94029467);%1.5;

    dx(1,1) = k_1*x(3)-k_2*x(1);
    dx(2,1) = k_3*x(5)-k_4*x(2);
    dx(3,1) = k_5*x(7)-k_6*x(3)*x(2);
    dx(4,1) = k_7-k_8*x(4)*x(3);
    dx(5,1) = k_9*x(1)-k_10*x(4)*x(5);
    dx(6,1) = k_11*x(1)-k_12*x(6);
    dx(7,1) = k_13*x(6)-k_14*x(7)*x(2);

% ------------------------------ END OF CODE ------------------------------