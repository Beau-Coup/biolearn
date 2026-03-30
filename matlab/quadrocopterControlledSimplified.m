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

pos = @(x) exp(x);

% with settle vel -- Array(-1.52199224, dtype=float64), Array(-0.94857276, dtype=float64), Array(-0.39972714, dtype=float64), Array(1.10494175, dtype=float64), Array(12.51213242, dtype=float64), Array(4.85958017, dtype=float64)
R = pos(1.10494175);
l = pos(-0.39972714);
M_rotor = pos(-0.94857276);
M = pos(-1.52199224);
P = 12.51213242;
D = 4.85958017;


% Raw model parameters for model 0: [Array(-1.300103, dtype=float64), Array(-1.1083545, dtype=float64), Array(-1.1319697, dtype=float64), Array(-1.60407105, dtype=float64), Array(10.89245698, dtype=float64), Array(4.62627563, dtype=float64)]

% R = pos(-1.60407105);
% l = pos(-1.1319697);
% M_rotor = pos(-1.1083545);
% M = pos(-1.300103);
% P = 10.89245698;
% D = 4.62627563;

% Raw model parameters for model 1: [Array(-1.67330438, dtype=float64), Array(-1.0996063, dtype=float64), Array(-0.56773639, dtype=float64), Array(0.87149174, dtype=float64), Array(11.65271952, dtype=float64), Array(4.93742236, dtype=float64)]

% R = pos(0.87149174);
% l = pos(-0.56773639);
% M_rotor = pos(-1.0996063);
% M = pos(-1.67330438);
% P = 11.65271952;
% D = 4.93742236;

% Raw model parameters for model 2: [Array(-1.34559882, dtype=float64), Array(-1.03720905, dtype=float64), Array(2.10316167, dtype=float64), Array(2.79825807, dtype=float64), Array(8.69069675, dtype=float64), Array(3.28781862, dtype=float64)]
% R = pos(2.79825807);
% l = pos(2.10316167);
% M_rotor = pos(-1.03720905);
% M = pos(-1.34559882);
% P = 8.69069675;
% D = 3.28781862;

% Raw model parameters for model 3: [Array(-0.7953955, dtype=float64), Array(-1.26824576, dtype=float64), Array(-0.02588137, dtype=float64), Array(-0.32847564, dtype=float64), Array(10.7821653, dtype=float64), Array(4.41285643, dtype=float64)]
% R = pos(-0.32847564);
% l = pos(-0.02588137);
% M_rotor = pos(-1.26824576);
% M = pos(-0.7953955);
% P = 10.7821653;
% D = 4.41285643;

% Raw model parameters for model 4: [Array(-3.72496084, dtype=float64), Array(-0.91149542, dtype=float64), Array(-0.29272793, dtype=float64), Array(0.24697439, dtype=float64), Array(10.28873488, dtype=float64), Array(4.24895886, dtype=float64)]
% R = pos(0.24697439);
% l = pos(-0.29272793);
% M_rotor = pos(-0.91149542);
% M = pos(-3.72496084);
% P = 10.28873488;
% D = 4.24895886;

% Raw model parameters for model 5: [Array(-1.16145026, dtype=float64), Array(-1.06504571, dtype=float64), Array(1.85149298, dtype=float64), Array(3.17897958, dtype=float64), Array(7.38028073, dtype=float64), Array(2.70775927, dtype=float64)]
% R = pos(3.17897958);
% l = pos(1.85149298);
% M_rotor = pos(-1.06504571);
% M = pos(-1.16145026);
% P = 7.38028073;
% D = 2.70775927;

% Raw model parameters for model 6: [Array(-0.55751984, dtype=float64), Array(-1.35954175, dtype=float64), Array(-2.02351583, dtype=float64), Array(-2.02856028, dtype=float64), Array(7.7354948, dtype=float64), Array(3.24146696, dtype=float64)]
% R = pos(-2.02856028);
% l = pos(-2.02351583);
% M_rotor = pos(-1.35954175);
% M = pos(-0.55751984);
% P = 7.7354948;
% D = 3.24146696;

% Raw model parameters for model 7: [Array(-0.90511603, dtype=float64), Array(-1.20299027, dtype=float64), Array(2.44587236, dtype=float64), Array(1.21903293, dtype=float64), Array(6.95172766, dtype=float64), Array(2.81546965, dtype=float64)]
% R = pos(1.21903293);
% l = pos(2.44587236);
% M_rotor = pos(-1.20299027);
% M = pos(-0.90511603);
% P = 6.95172766;
% D = 2.81546965;

% Raw model parameters for model 8: [Array(-0.97415357, dtype=float64), Array(-1.19142689, dtype=float64), Array(-2.013086, dtype=float64), Array(-0.69811511, dtype=float64), Array(7.22585676, dtype=float64), Array(2.99239581, dtype=float64)]
% R = pos(-0.69811511);
% l = pos(-2.013086);
% M_rotor = pos(-1.19142689);
% M = pos(-0.97415357);
% P = 7.22585676;
% D = 2.99239581;

% Raw model parameters for model 9: [Array(-0.95436256, dtype=float64), Array(-1.05458121, dtype=float64), Array(-2.54677401, dtype=float64), Array(0.74984644, dtype=float64), Array(13.85406783, dtype=float64), Array(4.5587215, dtype=float64)]
% R = pos(0.74984644);
% l = pos(-2.54677401);
% M_rotor = pos(-1.05458121);
% M = pos(-0.95436256);
% P = 13.85406783;
% D = 4.5587215;

% Raw model parameters for model 10: [Array(-1.49088358, dtype=float64), Array(-0.99215, dtype=float64), Array(-0.64202692, dtype=float64), Array(1.30840094, dtype=float64), Array(9.38132348, dtype=float64), Array(3.62408822, dtype=float64)]
% R = pos(1.30840094);
% l = pos(-0.64202692);
% M_rotor = pos(-0.99215);
% M = pos(-1.49088358);
% P = 9.38132348;
% D = 3.62408822;

% Raw model parameters for model 11: [Array(-1.78912011, dtype=float64), Array(-0.9180163, dtype=float64), Array(-0.45851308, dtype=float64), Array(-0.10714696, dtype=float64), Array(14.62045225, dtype=float64), Array(4.61983502, dtype=float64)]
% R = pos(-0.10714696);
% l = pos(-0.45851308);
% M_rotor = pos(-0.9180163);
% M = pos(-1.78912011);
% P = 14.62045225;
% D = 4.61983502;

% Raw model parameters for model 12: [Array(-1.71205352, dtype=float64), Array(-1.05087897, dtype=float64), Array(-0.91916635, dtype=float64), Array(0.76721482, dtype=float64), Array(13.02232404, dtype=float64), Array(5.38251872, dtype=float64)]
% R = pos(0.76721482);
% l = pos(-0.91916635);
% M_rotor = pos(-1.05087897);
% M = pos(-1.71205352);
% P = 13.02232404;
% D = 5.38251872;

% Raw model parameters for model 13: [Array(-1.88927683, dtype=float64), Array(-1.00585707, dtype=float64), Array(-2.10295298, dtype=float64), Array(-0.64936173, dtype=float64), Array(8.52566197, dtype=float64), Array(3.35978971, dtype=float64)]
% R = pos(-0.64936173);
% l = pos(-2.10295298);
% M_rotor = pos(-1.00585707);
% M = pos(-1.88927683);
% P = 8.52566197;
% D = 3.35978971;

% Raw model parameters for model 14: [Array(-2.22334251, dtype=float64), Array(-0.96269552, dtype=float64), Array(-2.49266384, dtype=float64), Array(-1.73768025, dtype=float64), Array(7.3373064, dtype=float64), Array(2.90552414, dtype=float64)]
% R = pos(-1.73768025);
% l = pos(-2.49266384);
% M_rotor = pos(-0.96269552);
% M = pos(-2.22334251);
% P = 7.3373064;
% D = 2.90552414;


% Raw model parameters for model 15: [Array(-1.41669971, dtype=float64), Array(-1.0436958, dtype=float64), Array(-1.89715546, dtype=float64), Array(-0.52683709, dtype=float64), Array(5.76632829, dtype=float64), Array(2.2198597, dtype=float64)]
% R = pos(-0.52683709);
% l = pos(-1.89715546);
% M_rotor = pos(-1.0436958);
% M = pos(-1.41669971);
% P = 5.76632829;
% D = 2.2198597;








% R = pos(0.76721482); %0.1;
% l = pos(-0.91916635); %0.5;
% M_rotor = pos(-1.05087897); %0.1;
% M = pos(-1.71205352); %1;
% P = 13.02232404; %10;
% D = 5.38251872; %3;

% v3 -- seed 12 -- Raw model parameters for model 12: [Array(-1.71205352, dtype=float64), Array(-1.05087897, dtype=float64), Array(-0.91916635, dtype=float64), Array(0.76721482, dtype=float64), Array(13.02232404, dtype=float64), Array(5.38251872, dtype=float64)]

% v2 [Array(-1.256092, dtype=float64), Array(-0.81904369, dtype=float64), Array(0.21673936, dtype=float64), Array(0.00037989, dtype=float64), Array(8.39692474, dtype=float64), Array(3.86830183, dtype=float64)]
% 
% v1 [Array(-2.05464906, dtype=float64), Array(-0.51968967, dtype=float64), Array(1.06936761, dtype=float64), Array(-0.83801818, dtype=float64), Array(3.56293838, dtype=float64), Array(1.25085725, dtype=float64)]
% 
%     log_body_mass: jax.Array
%     log_rotor_mass: jax.Array
%     log_length: jax.Array
%     log_radius: jax.Array
%     kp: jax.Array
%     kd: jax.Array

m = M + 4*M_rotor;
u_1 = 1; % u_1 desired height

% auxiliary parameters
J_x = 2/5*M*R^2 + 2*(l^2)*M_rotor;
J_y = J_x;
J_z = 2/5*M*R^2 + 4*(l^2)*M_rotor;

% height control (PD)
F_temp = m*g - P*(x(3) - u_1) - D*x(9);
%F = clip(F_temp, 0.5 * m * g, 1.5 * m * g);
amplitude = 10.0; %0.5 * m*g;%15.0; %0.5 * m*g;
center = 20.0; %m*g; %12.0; %m*g;
F = center + (amplitude * tanh((F_temp - center) / amplitude));

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
