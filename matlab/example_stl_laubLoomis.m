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
params.R0 = polyZonotope(x0,W*eye(dim_x));


%_______________
simOpt.points = 1;
traj = simulateRandom(laubloomis,params,simOpt);

figure;
hold on;

for i = 1:length(traj)
    t_vec = traj(i).t;
    %plot(t_vec, traj(i).x(7, :), 'LineWidth', 1.5);
    plot(t_vec, traj(i).x(7, :), 'LineWidth', 1.5,'DisplayName', 'State 7');
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
title('Laub Loomis');
grid on;

%_______________

options1.alg = 'poly-adaptive';
% options1.verbose = true;



timerVal = tic;
R = reach(laubloomis, params, options1);
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
plot([0,20],[4.5,4.5],'r--');

xlabel('$t$','interpreter','latex');
ylabel('$x_4$','interpreter','latex');
axis([0,20,1.5,5.5]);

% Specification -----------------------------------------------------------

x = stl('x', 7);

% x6 - 1 > 0 | x(3) - 0.48*x(1) <= 0.1
%T1 = stl('T1', atomicProposition(polytope([-0.48 0 1 0 0 0 0], 0.1)));
%T2 = stl('T2', atomicProposition(polytope([0.48 0 -1 0 0 0 0], -0.1)));

phi = {
    globally(x(4) <= 4.5, stlInterval(0,20))
    globally(x(4) < 3.0 | finally(x(4) <= 3.0 ,interval(0,4)), interval(0,16))
    finally(globally(x(3) - 0.48*x(1) <= 0.1 & x(3) - 0.48*x(1) >= -0.1,stlInterval(0,10)), stlInterval(0,10))
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


% ------------------------------ END OF CODE ------------------------------
