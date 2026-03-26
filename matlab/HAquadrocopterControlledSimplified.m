function HA = HAquadrocopterControlledSimplified()

% ------------------------------ BEGIN CODE -------------------------------

% Define constants
u_1 = 1; % Replace with your actual desired height constant
M_rotor = 0.1;
M = 1;
P = 10;
D = 3;
m = M + 4*M_rotor;
g = 9.81;

% Initialize empty 1x12 matrices for C
C_upper = zeros(1, 12);
C_lower = zeros(1, 12);

% Map the coefficients to x_3 and x_9
C_upper(3) = -P;   C_upper(9) = -D;
C_lower(3) = P;    C_lower(9) = D;

% Define the d limits
d_upper = 0.5 * m * g - P * u_1;
d_lower = 0.5 * m * g + P * u_1;

% ---------------------------------------------------------
% 1. NORMAL MODE INVARIANT (Bounded by both upper and lower)
% ---------------------------------------------------------
C_inv_norm = [C_upper; C_lower];
d_inv_norm = [d_upper; d_lower];
inv_normal = polytope(C_inv_norm, d_inv_norm);

% Create an identity reset (the states do not jump during saturation)
id_reset = linearReset(eye(12));

% ---------------------------------------------------------
% 2. MAX SATURATION INVARIANT & GUARDS
% ---------------------------------------------------------
inv_max = polytope(-C_upper, -d_upper); 

% Guard: Normal -> Max (Trigger when F >= 1.5mg)
guard_norm_to_max = polytope(-C_upper, -d_upper); 
trans_norm_to_max = transition(guard_norm_to_max, id_reset, 2);

% Guard: Max -> Normal (Trigger when F <= 1.5mg)
guard_max_to_norm = polytope(C_upper, d_upper);
trans_max_to_norm = transition(guard_max_to_norm, id_reset, 1);

% ---------------------------------------------------------
% 3. MIN SATURATION INVARIANT & GUARDS
% ---------------------------------------------------------
inv_min = polytope(-C_lower, -d_lower);

% Guard: Normal -> Min (Trigger when F <= 0.5mg)
guard_norm_to_min = polytope(-C_lower, -d_lower);
trans_norm_to_min = transition(guard_norm_to_min, id_reset, 3);

% Guard: Min -> Normal (Trigger when F >= 0.5mg)
guard_min_to_norm = polytope(C_lower, d_lower);
trans_min_to_norm = transition(guard_min_to_norm, id_reset, 1);

% dynamics
sys_min = nonlinearSys(@aux_minForce_quadrocopterControlledSimplified);
sys_normal = nonlinearSys(@aux_normal_quadrocopterControlledSimplified);
sys_max = nonlinearSys(@aux_maxForce_quadrocopterControlledSimplified);

% ---------------------------------------------------------
% 4. ASSEMBLE THE LOCATIONS
% ---------------------------------------------------------
% Syntax: location(name, invSet, trans, sys)
% The names '1', '2', and '3' correspond to the target numbers in your transitions.

% Location 1: Normal (Linear) Mode
% Transitions: Can go to Max or Min
loc_normal = location('1', inv_normal, [trans_norm_to_max, trans_norm_to_min], sys_normal);

% Location 2: Max Saturation Mode
% Transitions: Can only go back to Normal
loc_max = location('2', inv_max, trans_max_to_norm, sys_max);

% Location 3: Min Saturation Mode
% Transitions: Can only go back to Normal
loc_min = location('3', inv_min, trans_min_to_norm, sys_min);

% ---------------------------------------------------------
% 5. BUILD THE HYBRID AUTOMATON
% ---------------------------------------------------------
% Combine all three locations into the final HA object.
% The order here [1, 2, 3] defines the target indices used in your transitions!
HA = hybridAutomaton([loc_normal, loc_max, loc_min]);
    
end

function dx = aux_minForce_quadrocopterControlledSimplified(x,u)

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
F = 0.5 * m * g;

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

end


function dx = aux_normal_quadrocopterControlledSimplified(x,u)

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
F = m*g - P*(x(3) - u_1) - D*x(9);
%F = clip(F_temp, 0.5 * m * g, 1.5 * m * g);

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

end

function dx = aux_maxForce_quadrocopterControlledSimplified(x,u)

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
F = 1.5 * m * g;

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
end
