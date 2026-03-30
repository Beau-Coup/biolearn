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

params.tFinal = 25;                                      % final time

options.zonotopeOrder = 100;       
options.intermediateOrder = 50;
options.errorOrder = 20;

% reachability algorithm

options.alg = 'poly-adaptive';
options.tensorOrder = 3;

options.timeStep = 0.05; %0.1;
options.taylorTerms = 4;


% System Dynamics ---------------------------------------------------------

% system with uncertain parameters
Hill = nonlinearSys(@BioTransmission);

%%%%%%%%%%% TESTING
simOpt.points = 1;

params.R0 = zonotope([0.2; 0.2; 0.2; 0.2; 0.95; 0.95], diag([0.2*ones(4,1);0.05*ones(2,1)])); 
params.tFinal = 25;   


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
%options.intermediateTerms = 4;
timerVal = tic;
R = reach(Hill, params, options);
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
    plot(R,projDims,'DisplayName','parametric');

    % plot initial set
    plot(R.R0,projDims,'DisplayName','Initial set');

    % plot simulation results
    plot(traj,projDims,'DisplayName','Simulations');

    % label plot
    xlabel(['x_{',num2str(projDims(1)),'}']);
    ylabel(['x_{',num2str(projDims(2)),'}']);
    legend();
end


% Specification -----------------------------------------------------------

x = stl('x', 6);

% x6 - 1 > 0 | x(3) - 0.48*x(1) <= 0.1
%T1 = stl('T1', atomicProposition(polytope([-0.48 0 1 0 0 0 0], 0.1)));
%T2 = stl('T2', atomicProposition(polytope([0.48 0 -1 0 0 0 0], -0.1)));

    %(x(4) < 0.6 | finally(globally(x(3) <= 0.3 ,interval(0,20)), interval(0,20))) % check always time bounds

phi = {
    finally(x(1) > 0.2 & x(2) > 0.3, stlInterval(0,25)) 
    (finally(x(3) >= 0.5 ,interval(0,20)) & finally(globally(x(4) >= 0.9 ,interval(0,10)), interval(0,10)))
    globally(x(1) <= 1.5, stlInterval(0,25))
    globally(x(2) <= 1.5, stlInterval(0,25))
    globally(x(3) <= 1.5, stlInterval(0,25))
    globally(x(4) <= 1.5, stlInterval(0,25))
};


% Verification ------------------------------------------------------------

res = true;
alg = {'signals','incremental'};

for j = 1:length(alg)
    disp('-');
    disp(['checking alg ',alg{j}]);

    tFull = 0;

    % run all formulas for the selected algorithm
    for i = 1:length(phi)
        timerVal = tic;
        valid = modelChecking(R,phi{i},alg{j});
        tComp = toc(timerVal);

        disp(['computation time for ',num2str(i),' is ',num2str(tComp)]);

        tFull = tFull + tComp;

        % all tested fomulas are valid
        if ~valid
            disp('false negative');
        end

        res = res && valid;
    end

    disp(['computation time with ',alg{j},': ',num2str(tFull)]);
end


% --- CSV EXPORT ---
disp('Extracting state bounds for CSV export...');
num_states = 6;
num_pts = length(R.timePoint.set);

% 1. Extract raw bounds
local_t = zeros(num_pts, 1);
local_min = zeros(num_pts, num_states);
local_max = zeros(num_pts, num_states);

for i = 1:num_pts
    local_t(i) = R.timePoint.time{i};
    for k = 1:num_states
        int_k = interval(project(R.timePoint.set{i}, k));
        local_min(i, k) = infimum(int_k);
        local_max(i, k) = supremum(int_k);
    end
end

final_min = local_min; 
final_max = local_max; 

disp('Plotting visual inspection...');
inspect_dim = 1; % Change this from 1 to 6 to inspect a different state

figure('Name', ['Visual Inspection: State ', num2str(inspect_dim)]);
hold on; grid on;

% Grab the valid (non-NaN) data for the chosen dimension
valid_idx = ~isnan(final_min(:, inspect_dim));
t_plot = local_t(valid_idx);
min_plot = final_min(valid_idx, inspect_dim);
max_plot = final_max(valid_idx, inspect_dim);

% Draw the shaded tube and the thick boundary lines
fill([t_plot; flip(t_plot)], [min_plot; flip(max_plot)], ...
    'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Bounds Tube');
plot(t_plot, min_plot, 'b-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(t_plot, max_plot, 'b-', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Time (h)');
ylabel(['State ', num2str(inspect_dim), ' Concentration']);
title(['Extracted Min/Max Bounds for State ', num2str(inspect_dim)]);
legend('Location', 'best');

% 4. Build table and export
varNames = {'Time'};
data_matrix = local_t;

for k = 1:num_states
    % Append column names dynamically
    varNames{end+1} = ['Min_state_', num2str(k)];
    varNames{end+1} = ['Max_state_', num2str(k)];
    % Append data columns dynamically
    data_matrix = [data_matrix, final_min(:, k), final_max(:, k)];
end




export_data = array2table(data_matrix, 'VariableNames', varNames);
writetable(export_data, 'hill_model_bounds.csv');
disp('Successfully saved all 6 state bounds to hill_model_bounds.csv');



end

% ------------------------------ END OF CODE ------------------------------
