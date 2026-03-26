function res = example_stl_quadrotor()

%------------- BEGIN CODE --------------



% Parameter ---------------------------------------------------------------
dim = 12;
params.tFinal = 5; 
x0 = zeros(dim,1);
params.R0 = zonotope(x0, 0.1*diag([ones(3,1); zeros(3,1); ones(3,1); zeros(3,1)]));

% System Dynamics ---------------------------------------------------------
ContSys = nonlinearSys(@quadrocopterControlledSimplified, 12,0);

% Simulation ---------------------------------------------------------
simOpt.points = 20;
traj = simulateRandom(ContSys,params,simOpt);
% %traj = simulateRandom(ContSys,params,simOpt);
% 
% 
figure;
hold on;

for i = 1:length(traj)
    t_vec = traj(i).t;

    % Plot state dim
    % plot(t_vec, traj(i).x(6, :), 'LineWidth', 1.5,'DisplayName', 'State 6');
    % plot(t_vec, traj(i).x(5, :), 'LineWidth', 1.5,'DisplayName', 'State 5');
    % plot(t_vec, traj(i).x(4, :), 'LineWidth', 1.5,'DisplayName', 'State 4');
    plot(t_vec, traj(i).x(3, :), 'LineWidth', 1.5,'DisplayName', 'State 3');
    % plot(t_vec, traj(i).x(2, :), 'LineWidth', 1.5,'DisplayName', 'State 2');
    % plot(t_vec, traj(i).x(1, :), 'LineWidth', 1.5,'DisplayName', 'State 1');
end

% Add the threshold line
yline(1.4, 'r--', 'Threshold (1.4)', 'LineWidth', 2);
yline(0.9, 'r--', 'Threshold (0.9)', 'LineWidth', 2);

hold off;
xlabel('Time t [s]');
ylabel('Height h [m]');
grid on;

% Reachability Analysis ---------------------------------------------------

options.timeStep = 0.1;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.intermediateOrder = 5;
options.errorOrder = 1;
options.alg = 'lin-adaptive';
options.tensorOrder = 2;

tic
R = reach(ContSys, params, options);
tComp = toc;
disp(['computation time: ',num2str(tComp)]);


% Verification -- ARCH Style ------------------------------------------------------------

goal = interval(0.98,1.02);
t = options.timeStep;
res = 1;

for i = 1:length(R.timeInterval.set)
    
    int = interval(project(R.timeInterval.set{i},3));

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
end

%check if final reachable set is in goal region
% if ~in(goal,project(R.timePoint.set{end},3))
%    res = 0; 
% end

disp(['verified: ',num2str(res)]);

% Visualization -----------------------------------------------------------

figure; hold on; box on;

% plot results over time
plotOverTime(R,3,'FaceColor','b','EdgeColor','none');
    
% label plot
xlabel('t');
ylabel('x_3');
axis([0,5,-0.8,1.5]);

%------------- END OF CODE --------------