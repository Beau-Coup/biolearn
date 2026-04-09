function res = example_stl_Gene

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


end

% ------------------------------ END OF CODE ------------------------------
