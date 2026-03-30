function res = example_stl_quadrotor()

%------------- BEGIN CODE --------------

% --- Tanh Sharpness Visualization ---
% m = 1.4; 
% g = 9.81;
% mg = m * g;
% F_temp = linspace(0, 30, 500);
% 
% % 1. The Hard Clip
% lower_bound = 0.5 * mg;
% upper_bound = 1.5 * mg;
% F_clip = max(lower_bound, min(F_temp, upper_bound));
% 
% % 2. Tanh with different sharpness factors (k)
% center = mg;
% amplitude = 0.5 * mg;
% 
% F_tanh_standard = center + amplitude * tanh((F_temp - center) / amplitude);         % k = 1
% F_tanh_sharp    = center + amplitude * tanh(2 * (F_temp - center) / amplitude);     % k = 2
% F_tanh_sharper  = center + amplitude * tanh(4 * (F_temp - center) / amplitude);     % k = 4
% F_tanh_alex = ((tanh(F_temp) + 2) * center) / 2;
% 
% % Plotting
% figure;
% hold on; grid on;
% plot(F_temp, F_clip, 'k--', 'LineWidth', 2, 'DisplayName', 'Hard Clip');
% plot(F_temp, F_tanh_standard, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Tanh (k=1)');
% plot(F_temp, F_tanh_sharp, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Tanh (k=2)');
% plot(F_temp, F_tanh_sharper, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Tanh (k=4)');
% plot(F_temp, F_tanh_alex, 'y-', 'LineWidth', 1.5, 'DisplayName', 'jax');
% 
% title('Adjusting Tanh Sharpness (k)');
% xlabel('Requested PD Thrust: F_{temp} (N)');
% ylabel('Actual Motor Thrust: F (N)');
% legend('Location', 'northwest');
% xlim([0 30]);
% ylim([0 25]);


% Parameter ---------------------------------------------------------------
dim = 12;
params.tFinal = 5; 
x0 = zeros(dim,1);
% x0(1,1)=-0.2;
% x0(2,1)=-0.2;
x0(3,1)=-0.2;
% x0(7,1)=-0.2;
% x0(8,1)=-0.2;
x0(9,1)=-0.2;
params.R0 = zonotope(x0, diag([0.2*ones(3,1); 0.00*ones(3,1); 0.2*ones(3,1); 0.02*ones(3,1)]));

% System Dynamics ---------------------------------------------------------
ContSys = nonlinearSys(@quadrocopterControlledSimplified, 12,0);

% Simulation ---------------------------------------------------------
% simOpt.points = 20;
% traj = simulateRandom(ContSys,params,simOpt);
% % %traj = simulateRandom(ContSys,params,simOpt);
% % 
% % 
% figure;
% hold on;
% 
% g = 9.81; %[m/s^2], gravity constant
% 
% pos = @(x) exp(x);
% 
% R = pos(0.00037989); %0.1;
% l = pos(0.21673936); %0.5;
% M_rotor = pos(-0.81904369); %0.1;
% M = pos(-1.256092); %1;
% P = 8.39692474; %10;
% D = 3.86830183; %3;
% 
% m = M + 4*M_rotor;
% 
% for i = 1:length(traj)
%     t_vec = traj(i).t;
%     F_temp = m*g - P*(traj(i).x(3, :) - 1) - D*traj(i).x(9, :);
%     amplitude =0.5 * m*g;
%     center = m*g;
%     F1 = center + amplitude * tanh((F_temp - center) / amplitude);
%     %F1 = clip(F_temp, 0.5 * m * g, 1.5 * m * g);
%     amplitude = 5.0; %0.5 * m*g;
%     center = 10.0; %m*g;
%     F2 = center + amplitude * tanh((F_temp - center) / amplitude);
% 
%     % Plot state dim
%     %plot(t_vec, F1, 'LineWidth', 1.5,'DisplayName', 'F relative');
%     %plot(t_vec, F2, 'LineWidth', 1.5,'DisplayName', 'F absolute');
%     % plot(t_vec, traj(i).x(9, :), 'LineWidth', 1.5,'DisplayName', 'State 9');
%     plot(t_vec, traj(i).x(3, :), 'LineWidth', 1.5,'DisplayName', 'State 3');
%     % plot(t_vec, traj(i).x(2, :), 'LineWidth', 1.5,'DisplayName', 'State 2');
%     % plot(t_vec, traj(i).x(1, :), 'LineWidth', 1.5,'DisplayName', 'State 1');
% end
% 
% % Add the threshold line
% yline(1.4, 'r--', 'Threshold (1.4)', 'LineWidth', 2);
% yline(0.9, 'r--', 'Threshold (0.9)', 'LineWidth', 2);
% 
% hold off;
% xlabel('Time t [s]');
% ylabel('Height h [m]');
% grid on;

% Reachability Analysis ---------------------------------------------------

options.timeStep = 0.05;
options.taylorTerms = 4;
options.zonotopeOrder = 100;
options.intermediateOrder = 50;
options.errorOrder = 10;
options.alg = 'poly-adaptive';%'lin-adaptive';
options.tensorOrder = 3;
%options.maxError =0.1*ones(12,1);

% PolyZonotope specific settings (from your documentation)
options.polyZono.maxDepGenOrder = 20;               % Keep at default 20
options.polyZono.maxPolyZonoRatio = 10;             % Change from Inf to force restructuring 
options.polyZono.restructureTechnique = 'reduceGirard'; % Reliable default technique

% options.timeStep = 0.005;
% options.taylorTerms = 3;
% options.zonotopeOrder = 100;
% 
% options.alg = 'poly';
% options.tensorOrder = 3;
% options.errorOrder = 10;
% options.intermediateOrder = 50;

%options.timeStepInner = 0.1;
%options.algInner = 'scale';
%options.algInner = 'parallelo'; 
%params.tFinal = 1.0; 

tic
R = reach(ContSys, params, options);
%R_inner = reachInner(ContSys, params, options);
tComp = toc;
disp(['computation time: ',num2str(tComp)]);


% Verification -- ARCH Style ------------------------------------------------------------
res = 1;

for i = 1:length(R.timeInterval.set)
    
    int = interval(project(R.timeInterval.set{i},3));

    int_hdot = interval(project(R.timeInterval.set{i},9));
    t = infimum(R.timeInterval.time{i});

    % check if height is below 1.4 for all times
    if supremum(int) >= 1.4
       res = 0;
       break;
    end

    % check if height is above 0.9 after 1 second
    if t > 1 && infimum(int) < 0.9
       res = 0;
       break;
    end

    % check if height is above 0.9 after 1 second
    if t > 3 && (infimum(int_hdot) < - 0.1 || supremum(int_hdot) > 0.1)
       res = 0;
       break;
    end
end

disp(['verified: ',num2str(res)]);

% Visualization -----------------------------------------------------------

figure; hold on; box on;

% plot results over time
plotOverTime(R,3,'FaceColor','b','EdgeColor','none');
yline(1.4, 'r--', 'Threshold (1.4)', 'LineWidth', 2);
yline(0.9, 'r--', 'Threshold (0.9)', 'LineWidth', 2);
    
% label plot
xlabel('t');
ylabel('h');

axis([0,5,-0.8,1.5]);

figure; hold on; box on;

% plot results over time
plotOverTime(R,9,'FaceColor','g','EdgeColor','none');
yline(-0.1, 'r--', 'Threshold ', 'LineWidth', 2);
yline(0.1, 'r--', 'Threshold ', 'LineWidth', 2);
    
% label plot
xlabel('t');
ylabel('h');

axis([0,5,-0.5,2]);

%------------- END OF CODE --------------