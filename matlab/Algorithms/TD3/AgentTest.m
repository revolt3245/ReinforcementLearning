clear; clc; close;

trial = 'trial2';

if ~isfolder("LearningResult\" + trial)
    mkdir("LearningResult\" + trial);
end

Env = CartPoleContinuousAction2;
load("AgentResult\" + trial + "\agent_saver.mat");

SimOpts = rlSimulationOptions('MaxSteps', 500);

plot(Env);

SimResult = sim(Env, Agent, SimOpts);

Observation = SimResult.Observation.CartPoleStates;
Action = SimResult.Action.CartPoleAction;

L = length(Observation.Data);
X = reshape(Observation.Data(1,:,:), 1, L);
Xdot = reshape(Observation.Data(2,:,:), 1, L);
T = reshape(Observation.Data(3,:,:), 1, L);
Tdot = reshape(Observation.Data(4,:,:), 1, L);

t = tiledlayout(2, 2);

Title = title(t, "Observations", 'FontSize', 15);

ax_X = nexttile;
ax_Xdot = nexttile;
ax_T = nexttile;
ax_Tdot = nexttile;

plot(ax_X, X, 'Color', [0 0 1]); ax_X.XLim = [0 500];
plot(ax_Xdot, Xdot, 'Color', [0 0 1]); ax_Xdot.XLim = [0 500];
plot(ax_T, T, 'Color', [0 0 1]); ax_T.XLim = [0 500];
plot(ax_Tdot, Tdot, 'Color', [0 0 1]); ax_Tdot.XLim = [0 500];

title(ax_X, '$x$', 'Interpreter', 'latex', 'FontSize', 12);
title(ax_Xdot, '$\dot{x}$', 'Interpreter', 'latex', 'FontSize', 12);
title(ax_T, '$\theta$', 'Interpreter', 'latex', 'FontSize', 12);
title(ax_Tdot, '$\dot{\theta}$', 'Interpreter', 'latex', 'FontSize', 12);

xlabel(ax_X, 'step', 'FontSize', 11);
xlabel(ax_Xdot, 'step', 'FontSize', 11);
xlabel(ax_T, 'step', 'FontSize', 11);
xlabel(ax_Tdot, 'step', 'FontSize', 11);

ylabel(ax_X, 'position [m]', 'FontSize', 11);
ylabel(ax_Xdot, 'velocity [m/s]', 'FontSize', 11);
ylabel(ax_T, 'angle [rad]', 'FontSize', 11);
ylabel(ax_Tdot, 'angular velocity [rad/s]', 'FontSize', 11);

saveas(gcf, "LearningResult\" + trial + "\ObservationGraph.fig");