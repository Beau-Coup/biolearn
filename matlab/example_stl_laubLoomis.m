function text = example_stl_laubLoomis()
% benchmark_nonlinear_reach_ARCH23_laubLoomis - example of 
%    nonlinear reachability analysis
%
% Syntax:
%    benchmark_nonlinear_reach_ARCH23_laubLoomis
%
% Inputs:
%    ---
%
% Outputs:
%    res - true/false
%

% Authors:       Matthias Althoff, Niklas Kochdumper
% Written:       27-March-2019
% Last update:   25-March-2023 (MW, use adaptive algorithm)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------


dim_x = 7;

% Parameters --------------------------------------------------------------

params.tFinal = 20;
x0 = [1.2; 1.05; 1.5; 2.4; 1; 0.1; 0.45];
laubloomis = nonlinearSys(@laubLoomis_stl); 


% Reachability Analysis ---------------------------------------------------

% Initial set W = 0.1
W = 0.1;
params.R0 = polyZonotope(x0,diag([0.1*ones(2,1); 0.1*ones(2,1); 0.1*ones(3,1)]));


%_______________
simOpt.points = 20;
simOpt.fracVert = 0;
traj = simulateRandom(laubloomis,params,simOpt);

% figure;
% hold on;
% 
% for i = 1:length(traj)
%     t_vec = traj(i).t;
%     %plot(t_vec, traj(i).x(7, :), 'LineWidth', 1.5);
%     plot(t_vec, traj(i).x(7, :), 'LineWidth', 1.5,'DisplayName', 'State 7');
%     plot(t_vec, traj(i).x(6, :), 'LineWidth', 1.5,'DisplayName', 'State 6');
%     plot(t_vec, traj(i).x(5, :), 'LineWidth', 1.5,'DisplayName', 'State 5');
%     plot(t_vec, traj(i).x(4, :), 'LineWidth', 1.5,'DisplayName', 'State 4');
%     plot(t_vec, traj(i).x(3, :), 'LineWidth', 1.5,'DisplayName', 'State 3');
%     plot(t_vec, traj(i).x(2, :), 'LineWidth', 1.5,'DisplayName', 'State 2');
%     plot(t_vec, traj(i).x(1, :), 'LineWidth', 1.5,'DisplayName', 'State 1');
% end
% 
% hold off;
% xlabel('Time (h)');
% ylabel('Species concentration');
% title('Laub Loomis');
% grid on;

%%%%%%%%%% CSV export of traj %%%%%%%%%%%%%

% Create a unified time grid (e.g., from 0 to 25 with 0.05 steps)
% This ensures all 20 trajectories line up perfectly in the rows
num_states = 7;
num_trajs = length(traj);
common_time = (0:0.05:params.tFinal)'; 

for k = 1:num_states
    % Initialize the column names and data matrix for this state's CSV
    varNames = {'Time'};
    data_matrix = common_time;

    
    for i = 1:num_trajs
        % Extract time and the k-th state data for the i-th trajectory
        t_vec = traj(i).t(:);
        x_vec = traj(i).x(k, :)';
        
        % Remove duplicates (ODE solvers sometimes hit the same time step twice)
        [t_unq, idx] = unique(t_vec);
        x_unq = x_vec(idx);
        
        % Interpolate this trajectory onto the common time grid
        % Using 'extrap' ensures that if a solver stopped at 24.999, it cleanly fills the 25.0 row
        interp_x = interp1(t_unq, x_unq, common_time, 'linear', 'extrap');
        
        % Append the interpolated column to our matrix
        data_matrix = [data_matrix, interp_x];
        
        % Add the column header dynamically (traj_1, traj_2, etc.)
        varNames{end+1} = ['traj_', num2str(i)];
    end

    % --- VISUAL INSPECTION PLOT ---
    if k == 2
        figure('Name', ['Interpolated Trajectories: State ', num2str(k)]);
        hold on; grid on;
        % Plot all 20 trajectories at once using the interpolated matrix
        % (Columns 2 through the end contain the trajectory data)
        plot(common_time, data_matrix(:, 2:end), 'LineWidth', 1.2);
        
        xlabel('Time (h)');
        ylabel(['State ', num2str(k), ' Concentration']);
        title(['Interpolated Trajectories for State ', num2str(k)]);
    end
    % -----------------------------
    
    % Build the table and export to a specific CSV for this state
    filename = ['laubloomis_traj_state_', num2str(k), '.csv'];
    export_data = array2table(data_matrix, 'VariableNames', varNames);
    writetable(export_data, filename);
    disp(['Saved: ', filename]);
end

disp('All 7 trajectory CSVs generated successfully.');


%_______________
options.zonotopeOrder = 100;       
options.intermediateOrder = 50;
%options.errorOrder = 20;

% reachability algorithm

options.alg = 'poly-adaptive';
%options.tensorOrder = 3;

%options.timeStep = 0.05; %0.1;
%options.taylorTerms = 4;

%options1.alg = 'poly-adaptive';
% options1.verbose = true;



timerVal = tic;
R = reach(laubloomis, params, options);
tComp1 = toc(timerVal);

width1 = 2*rad(interval(project(R.timePoint.set{end},4)));
disp(['computation time of reachable set (W=0.1): ',num2str(tComp1)]);
disp(['width of final reachable set (W=0.1): ',num2str(width1)]);
disp(' ');

% Visualization -----------------------------------------------------------

figure; hold on; box on;

% plot results over time
plotOverTime(R,4,'FaceColor',colorblind('b'));

% specs
%plot([0,20],[4.5,4.5],'r--');

xlabel('$t$','interpreter','latex');
ylabel('$x_4$','interpreter','latex');
%axis([0,20,1.5,5.5]);

% Specification -----------------------------------------------------------

x = stl('x', 7);

% x6 - 1 > 0 | x(3) - 0.48*x(1) <= 0.1
%T1 = stl('T1', atomicProposition(polytope([-0.48 0 1 0 0 0 0], 0.1)));
%T2 = stl('T2', atomicProposition(polytope([0.48 0 -1 0 0 0 0], -0.1)));

phi = {
    globally(x(4) <= 4.5, stlInterval(0,20))
    finally(globally(x(3) - 0.55 <= 0.1 & x(3) - 0.55 >= -0.1,stlInterval(0,10)), stlInterval(0,10))
    finally(globally(x(2) - 0.35 <= 0.1 & x(2) - 0.35 >= -0.1,stlInterval(0,10)), stlInterval(0,10))
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
num_states = 7;
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
inspect_dim = 4; % Change this from 1 to 6 to inspect a different state

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
writetable(export_data, 'laubloomis_model_bounds.csv');
disp('Successfully saved all 7 state bounds to laubloomis_model_bounds.csv');


% ------------------------------ END OF CODE ------------------------------
