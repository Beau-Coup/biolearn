function res = example_stl_Hill
% example_stl_BNN - example of signal temporal logic
% checking of the room heating model
%
% Syntax:
%    res = example_stl_BNN()
%
% Inputs:
%    no
%
% Outputs:
%    res - boolean

% ------------------------------ BEGIN CODE -------------------------------
%y_test = BNN([7.34164577e-05, 2.33403951e-04, 1.70125806e-04, 1.64744608e-04, 2.10874017e-06, 1.51597852e-06, 0.0, 0.1, 0.3]);

% Parameters --------------------------------------------------------------

params.tFinal = 20;                                      % final time

options.zonotopeOrder = 50;       
options.intermediateOrder = 50;
options.errorOrder = 20;

% reachability algorithm
options.alg = 'poly';
options.tensorOrder = 3;

options.timeStep = 0.1;
options.taylorTerms = 4;


% System Dynamics ---------------------------------------------------------

% system with uncertain parameters
Hill = nonlinearSys(@BioTransmission);

%%%%%%%%%%% TESTING
simOpt.points = 1;

params.R0 = zonotope([0.25; 0.25; 0.25; 0.2; 0.95; 0.95], diag([0.15*ones(4,1);0.05*ones(2,1)])); 
params.tFinal = 40;   


% testing initial conditions to ensure currect code translation
%params.R0 = zonotope([0.2; 0.2; 0.1; 0.1; 0.9; 0.9]);
traj = simulateRandom(Hill,params,simOpt);

figure;
hold on;

for i = 1:length(traj)
    t_vec = traj(i).t;
    %plot(t_vec, traj(i).x(7, :), 'LineWidth', 1.5);
    plot(t_vec, traj(i).x(6, :), 'LineWidth', 1.5,'DisplayName', 'State 6');
    plot(t_vec, traj(i).x(5, :), 'LineWidth', 1.5,'DisplayName', 'State 5');
    plot(t_vec, traj(i).x(4, :), 'LineWidth', 1.5,'DisplayName', 'State 4');
    plot(t_vec, traj(i).x(3, :), 'LineWidth', 1.5,'DisplayName', 'State 3');
    plot(t_vec, traj(i).x(2, :), 'LineWidth', 1.5,'DisplayName', 'State 2');
    plot(t_vec, traj(i).x(1, :), 'LineWidth', 1.5,'DisplayName', 'State 1');
end

% Add the threshold line

hold off;
xlabel('Time (h)');
ylabel('Species concentration');
title('Hill model');
grid on;


% Reachability Analysis ---------------------------------------------------        

% compute reachable set with uncertain parameters
options.intermediateTerms = 4;
timerVal = tic;
RcontParam = reach(Hill, params, options);
tComp = toc(timerVal);
disp(['computation time of reachable set with uncertain parameters: ',num2str(tComp)]);


% Visualization -----------------------------------------------------------
figure;
dims = {[1,2],[3,4],[5,6]};

% plot different projections
for i = 1:length(dims)

    figure; hold on; box on;
    projDims = dims{i};
    useCORAcolors("CORA:contDynamics", 2)

    % plot reachable sets
    plot(RcontParam,projDims,'DisplayName','parametric');

    % plot initial set
    plot(RcontParam.R0,projDims,'DisplayName','Initial set');

    % plot simulation results
    plot(traj,projDims,'DisplayName','Simulations');

    % label plot
    xlabel(['x_{',num2str(projDims(1)),'}']);
    ylabel(['x_{',num2str(projDims(2)),'}']);
    legend();
end


end

% ------------------------------ END OF CODE ------------------------------
