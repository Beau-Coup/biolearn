% --- 1. Load the Data ---
disp('Loading reachable sets...');
load('quadrotor_reach_data.mat'); % Loads reachableSets and centerList
disp(['Loaded ', num2str(length(reachableSets)), ' reachable sets.']);

safe_tiles = 0;
unsafe_tiles = 0;

% --- 2. Initialize Arrays for Global Bounds ---
% We assume all tiles have the same number of time steps
num_time_steps = length(reachableSets{1}.timeInterval.set);

global_time   = zeros(num_time_steps, 1);
global_min_z  = inf(num_time_steps, 1);
global_max_z  = -inf(num_time_steps, 1);
global_min_vz = inf(num_time_steps, 1);
global_max_vz = -inf(num_time_steps, 1);

% --- 3. Loop and Verify ---
for j = 1:length(reachableSets)
    R = reachableSets{j};
    tile_center = centerList(j, :);
    
    % flag to track if this specific tile failed
    is_safe = true; 
    for i = 1:length(R.timeInterval.set)
        
        int = interval(project(R.timeInterval.set{i},3));
    
        int_hdot = interval(project(R.timeInterval.set{i},9));
        t = infimum(R.timeInterval.time{i});

        % --- Update Global CSV Bounds ---
        if j == 1 % Record time vector once
            global_time(i) = t;
        end
        global_min_z(i)  = min(global_min_z(i),  infimum(int));
        global_max_z(i)  = max(global_max_z(i),  supremum(int));
        global_min_vz(i) = min(global_min_vz(i), infimum(int_hdot));
        global_max_vz(i) = max(global_max_vz(i), supremum(int_hdot));
    
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
    
    if is_safe
        safe_tiles = safe_tiles + 1;
    else
        unsafe_tiles = unsafe_tiles + 1;
    end
end

disp('--- Verification Complete ---');
disp(['Safe starting tiles: ', num2str(safe_tiles)]);
disp(['Unsafe/Unknown tiles: ', num2str(unsafe_tiles)]);

% --- 4. Export to CSV ---
disp('Exporting global bounds to CSV...');
export_data = table(global_time, global_min_z, global_max_z, global_min_vz, global_max_vz, ...
    'VariableNames', {'Time_s', 'Min_h', 'Max_h', 'Min_h_dot', 'Max_h_dot'});
writetable(export_data, 'reachable_bounds_export.csv');
disp('Successfully saved bounds to reachable_bounds_export.csv');