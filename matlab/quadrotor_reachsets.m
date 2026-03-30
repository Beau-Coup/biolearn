params.tFinal = 5; 
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

% --- 1. Define the Grid ---
dim = 12;
split_dims = [1, 2, 3, 7, 8, 9]; % The 6 dimensions you are splitting
total_width = 0.8;
num_bins = 2; % Number of splits per dimension (Start with 2!)

% Calculate the radius and centers for the sub-intervals
sub_radius = (total_width / num_bins) / 2;
centers_1d = linspace(-total_width/2 + sub_radius, total_width/2 - sub_radius, num_bins);

% Generate the 6D grid of centers
C = cell(1, length(split_dims));
[C{:}] = ndgrid(centers_1d);
grid_centers = zeros(numel(C{1}), length(split_dims));
for i = 1:length(split_dims)
    grid_centers(:, i) = C{i}(:);
end

num_tiles = size(grid_centers, 1);
disp(['Total reachability runs to compute: ', num2str(num_tiles)]);

% --- 2. Setup System and Arrays ---
ContSys = nonlinearSys(@quadrocopterControlledSimplified, dim, 0);
reachableSets = cell(num_tiles, 1);

% PRE-COMPUTE CENTER LIST OUTSIDE THE LOOP FOR PARFOR SAFETY
centerList = zeros(num_tiles, dim);
for i = 1:num_tiles
    centerList(i, split_dims) = grid_centers(i, :);
end

% Create a baseline R0 diagonal with your non-split dimensions
% (Make sure to keep the 1e-6 trick for dimensions with 0 uncertainty!)
base_diag = [0.2*ones(3,1); 0.00*ones(3,1); 0.2*ones(3,1); 0.02*ones(3,1)]; % Update this with your actual baseline uncertainties
base_diag(split_dims) = sub_radius; 

% --- 3. The Parallel Computation Loop ---
tic;
% Use 'parfor' if you have the Parallel Computing Toolbox, otherwise use 'for'
parfor i = 1:num_tiles 
    % Copy options/params inside the loop for parallel worker compatibility
    local_params = params; % Assuming 'params' is defined above
    local_options = options; % Assuming 'options' is defined above
    
    x0_local = centerList(i, :)';
    local_params.R0 = zonotope(x0_local, diag(base_diag));
    
    % Compute Reachability (Suppressing command window output keeps parfor clean)
    R_tile = reach(ContSys, local_params, local_options);
    reachableSets{i} = R_tile;
    
    disp(['Completed tile ', num2str(i), ' of ', num2str(num_tiles)]);
end
computation_time = toc;
disp(['Total computation time: ', num2str(computation_time/60), ' minutes']);

% --- 4. Save to .mat file ---
save('quadrotor_reach_data.mat', 'reachableSets', 'centerList', 'sub_radius', '-v7.3');
disp('Data successfully saved to quadrotor_reach_data.mat');