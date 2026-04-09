function res = example_stl_quadrotor()

%------------- BEGIN CODE --------------


% Parameter ---------------------------------------------------------------
dim = 12;
params.tFinal = 5; 
x0 = zeros(dim,1);
x0(3,1)=-0.2;
x0(9,1)=-0.2;
params.R0 = zonotope(x0, diag([0.2*ones(3,1); 0.00*ones(3,1); 0.2*ones(3,1); 0.02*ones(3,1)]));

% System Dynamics ---------------------------------------------------------
ContSys = nonlinearSys(@quadrocopterControlledSimplified, 12,0);

% Reachability Analysis ---------------------------------------------------

options.timeStep = 0.05;
options.taylorTerms = 4;
options.zonotopeOrder = 100;
options.intermediateOrder = 50;
options.errorOrder = 10;
options.alg = 'poly-adaptive';%'lin-adaptive';
options.tensorOrder = 3;

% PolyZonotope specific settings 
options.polyZono.maxDepGenOrder = 20;               
options.polyZono.maxPolyZonoRatio = 10;             
options.polyZono.restructureTechnique = 'reduceGirard'; 

tic
R = reach(ContSys, params, options);
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
