function dx = quadrocopterControlledSimplified(x,u)

% ------------------------------ BEGIN CODE -------------------------------

% x_1 = p_n
% x_2 = p_e
% x_3 = h

% x_7 = u
% x_8 = v
% x_9 = w

% x_4 = phi
% x_5 = theta
% x_6 = psi

% x_10 = p
% x_11 = q
% x_12 = r

% u_1 desired height

% parameters
g = 9.81; %[m/s^2], gravity constant
R = 0.1;
l = 0.5;
M_rotor = 0.1;
M = 1;
P = 10;
D = 3;
m = M + 4*M_rotor;
u_1 = 1; % u_1 desired height

% auxiliary parameters
J_x = 2*M*R^2/5 + 2*l^2*M_rotor;
J_y = J_x;
J_z = 2*M*R^2/5 + 4*l^2*M_rotor;

% height control (PD)
F_temp = m*g - P*(x(3) - u_1) - D*x(9);
%F = clip(F_temp, 0.5 * m * g, 1.5 * m * g);
amplitude = 0.5 * m*g;
F = m*g + amplitude * tanh((F_temp - m*g) / amplitude);

% desired 

% roll control (PD)
tau_phi = -x(4) - x(10);

% pitch control (PD)
tau_theta = -x(5) - x(11);

% heading is uncontrolled:
tau_psi = 0;

dx(1,1) = x(7);
dx(2,1) = x(8);
dx(3,1) = x(9);

dx(4,1) = x(10); %dot{phi}
dx(5,1) = x(11); %dot{theta}
dx(6,1) = x(12); %dot{psi}

dx(7,1) = F/m *(-cos(x(4))*sin(x(5))*cos(x(6))-sin(x(4))*sin(x(6)));
dx(8,1) = F/m * (-cos(x(4))*sin(x(5))*sin(x(6))+sin(x(4))*cos(x(6)));
dx(9,1) = F/m * (cos(x(4))*cos(x(5))) - g;

dx(10,1) = (1/J_x)*tau_phi;
dx(11,1) = (1/J_y)*tau_theta;
dx(12,1) = (1/J_z)*tau_psi;

% ------------------------------ END OF CODE ------------------------------
