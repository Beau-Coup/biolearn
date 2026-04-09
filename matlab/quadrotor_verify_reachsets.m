% --- 1. Load the Data ---
disp('Loading reachable sets...');
load('quadrotor_reach_data.mat'); % Loads reachableSets and centerList
disp(['Loaded ', num2str(length(reachableSets)), ' reachable sets.']);

safe_tiles = 0;
unsafe_tiles = 0;

% --- 2. Initialize Arrays for Global Bounds ---
% We assume all tiles have the same number of time steps
global_time = (0:0.01:5)'; 
num_grid_points = length(global_time);

global_min_z  = inf(num_grid_points, 1);
global_max_z  = -inf(num_grid_points, 1);
global_min_vz = inf(num_grid_points, 1);
global_max_vz = -inf(num_grid_points, 1);


% --- 3. Loop and Verify ---
for j = 1:length(reachableSets)
    R = reachableSets{j};
    tile_center = centerList(j, :);
    
    % flag to track if this specific tile failed
    is_safe = true; 

    % Get the number of actual adaptive time points for THIS specific tile
    num_local_points = length(R.timePoint.set);
    
    % Temporary arrays to hold this tile's specific data
    local_t   = zeros(num_local_points, 1);
    local_min_z  = zeros(num_local_points, 1);
    local_max_z  = zeros(num_local_points, 1);
    local_min_vz = zeros(num_local_points, 1);
    local_max_vz = zeros(num_local_points, 1);

    for i = 1:length(R.timeInterval.set)
        
        int = interval(project(R.timeInterval.set{i},3));
        int_point = interval(project(R.timePoint.set{i},3));
    
        int_hdot = interval(project(R.timeInterval.set{i},9));
        int_hdot_point = interval(project(R.timePoint.set{i},9));
        t = infimum(R.timeInterval.time{i});

        t2 = R.timePoint.time{i};
        
        % Store locally
        local_t(i)   = t2;
        local_min_z(i)  = infimum(int_point);
        local_max_z(i)  = supremum(int_point);
        local_min_vz(i) = infimum(int_hdot_point);
        local_max_vz(i) = supremum(int_hdot_point);
    
        % check if height is below 1.4 for all times
        if supremum(int) >= 1.4
           disp(['Spec 3 Violation found in tile ', num2str(j), ' (Started at z=', num2str(tile_center(3)), ')']);
           is_safe = false;
        end
    
        % check if height is above 0.9 after 1 second
        if t > 1 && infimum(int) < 0.9
           disp(['Spec 2 Violation found in tile ', num2str(j), ' (Started at z=', num2str(tile_center(3)), ')']);
           is_safe = false;
        end
    
        % check if height is above 0.9 after 1 second
        if t > 3 && (infimum(int_hdot) < - 0.1 || supremum(int_hdot) > 0.1)
           disp(['Spec 3 Violation found in tile ', num2str(j), ' (Started at z=', num2str(tile_center(3)), ')']);
           is_safe = false;
        end
    end

    % --- FIX: MERGE DUPLICATE TIME POINTS ---
    % Find unique time points and the index mapping of duplicates
    [unique_t, ~, idx] = unique(local_t);
    
    % Use accumarray to group duplicates and take the absolute min/max
    unique_min_z  = accumarray(idx, local_min_z,  [], @min);
    unique_max_z  = accumarray(idx, local_max_z,  [], @max);
    unique_min_vz = accumarray(idx, local_min_vz, [], @min);
    unique_max_vz = accumarray(idx, local_max_vz, [], @max);
    
    % --- INTERPOLATE TO MASTER GRID ---
    % Using NaN stops early-terminating sets from returning garbage
    z_min_interp  = interp1(unique_t, unique_min_z,  global_time, 'linear', NaN);
    z_max_interp  = interp1(unique_t, unique_max_z,  global_time, 'linear', NaN);
    vz_min_interp = interp1(unique_t, unique_min_vz, global_time, 'linear', NaN);
    vz_max_interp = interp1(unique_t, unique_max_vz, global_time, 'linear', NaN);

    % --- Update Global CSV Bounds ---
    global_min_z  = min(global_min_z,  z_min_interp);
    global_max_z  = max(global_max_z,  z_max_interp);
    global_min_vz = min(global_min_vz, vz_min_interp);
    global_max_vz = max(global_max_vz, vz_max_interp);
    
    if is_safe
        safe_tiles = safe_tiles + 1;
    else
        unsafe_tiles = unsafe_tiles + 1;
    end
end

disp('--- Verification Complete ---');
disp(['Safe starting tiles: ', num2str(safe_tiles)]);
disp(['Unsafe/Unknown tiles: ', num2str(unsafe_tiles)]);

%%%%%%%%%%%%%%

subplot(1, 2, 1);
hold on; grid on;

% Shade the safe reachable region
fill([global_time; flipud(global_time)], [global_min_z; flipud(global_max_z)], ...
    [0.7 0.8 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% Plot the outer boundary lines
plot(global_time, global_max_z, 'b', 'LineWidth', 1.5);
plot(global_time, global_min_z, 'b', 'LineWidth', 1.5);

% Plot the STL Specifications for Z
yline(1.4, 'r--', 'LineWidth', 1.5, 'Label', 'Max Z (1.4)'); % Spec 1: z < 1.4 for all t
plot([1, max(global_time)], [0.9, 0.9], 'r--', 'LineWidth', 1.5); % Spec 2: z > 0.9 after t=1
text(1, 0.9, ' Min Z (0.9) after t=1s ', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'Color', 'r');

xlabel('Time (s)');
ylabel('Z Position (m)');
title('Global Reach Tube: Height (Z)');
ylim([min(global_min_z)-0.1, max(1.5, max(global_max_z)+0.1)]); % Keep bounds in view
hold off;


% Subplot 2: Z Velocity (State 9)
subplot(1, 2, 2);
hold on; grid on;

% Shade the safe reachable region
fill([global_time; flipud(global_time)], [global_min_vz; flipud(global_max_vz)], ...
    [1.0 0.8 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% Plot the outer boundary lines
plot(global_time, global_max_vz, 'r', 'LineWidth', 1.5);
plot(global_time, global_min_vz, 'r', 'LineWidth', 1.5);

% Plot the STL Specifications for Vz (Only active after t=3)
plot([3, max(global_time)], [0.1, 0.1], 'r--', 'LineWidth', 1.5);
plot([3, max(global_time)], [-0.1, -0.1], 'r--', 'LineWidth', 1.5);
text(3, 0.1, ' Max Vz (0.1) ', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'Color', 'r');
text(3, -0.1, ' Min Vz (-0.1) ', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', 'Color', 'r');

xlabel('Time (s)');
ylabel('Z Velocity (m/s)');
title('Global Reach Tube: Velocity (Vz)');
hold off;

%%%%%%%%%%%%%%%

% --- 4. Export to CSV ---
disp('Exporting global bounds to CSV...');
export_data = table(global_time, global_min_z, global_max_z, global_min_vz, global_max_vz, ...
    'VariableNames', {'Time_s', 'Min_h', 'Max_h', 'Min_h_dot', 'Max_h_dot'});
writetable(export_data, 'reachable_bounds_export.csv');
disp('Successfully saved bounds to reachable_bounds_export.csv');
