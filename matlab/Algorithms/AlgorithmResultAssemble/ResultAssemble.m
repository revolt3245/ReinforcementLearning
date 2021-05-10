clear; clc; close;

if ~isfile('CartPoleDiscreteAction2.m')
    copyfile('..\DQN\CartPoleDiscreteAction2.m', pwd);
end

if ~isfile('CartPoleContinuousAction2.m')
    copyfile('..\..\CustomEnvironment\CartPole\CartPoleContinuousAction2.m', pwd);
end

if ~isfolder('LearningResult')
    mkdir('LearningResult')
end

AgentDQN = load("..\DQN\LearningResult\trial5\agent_saver.mat");
AgentDDPG = load("..\DDPG\agent_saver.mat");
AgentPPO = load("..\PPO\Trial\Trial(Default)\agent_saver.mat");

EnvContinuous1 = CartPoleContinuousAction2;
EnvContinuous2 = CartPoleContinuousAction2;
EnvDiscrete = CartPoleDiscreteAction2;

SimOpts = rlSimulationOptions('MaxSteps', 500);

plot(EnvContinuous1);
plot(EnvContinuous2);
plot(EnvDiscrete);

SimResultDQN = sim(EnvDiscrete, AgentDQN.agent2, SimOpts);
SimResultDDPG = sim(EnvContinuous1, AgentDDPG.Agent, SimOpts);
SimResultPPO = sim(EnvContinuous2, AgentPPO.Agent, SimOpts);

ObservationDQN = SimResultDQN.Observation.CartPoleStates;
ObservationDDPG = SimResultDDPG.Observation.CartPoleStates;
ObservationPPO = SimResultPPO.Observation.CartPoleStates;

LDQN = length(ObservationDQN.Data);
XDQN = reshape(ObservationDQN.Data(1,:,:), 1, LDQN);
XdotDQN = reshape(ObservationDQN.Data(2,:,:), 1, LDQN);
TDQN = reshape(ObservationDQN.Data(3,:,:), 1, LDQN);
TdotDQN = reshape(ObservationDQN.Data(4,:,:), 1, LDQN);

LDDPG = length(ObservationDDPG.Data);
XDDPG = reshape(ObservationDDPG.Data(1,:,:), 1, LDDPG);
XdotDDPG = reshape(ObservationDDPG.Data(2,:,:), 1, LDDPG);
TDDPG = reshape(ObservationDDPG.Data(3,:,:), 1, LDDPG);
TdotDDPG = reshape(ObservationDDPG.Data(4,:,:), 1, LDDPG);

LPPO = length(ObservationPPO.Data);
XPPO = reshape(ObservationPPO.Data(1,:,:), 1, LPPO);
XdotPPO = reshape(ObservationPPO.Data(2,:,:), 1, LPPO);
TPPO = reshape(ObservationPPO.Data(3,:,:), 1, LPPO);
TdotPPO = reshape(ObservationPPO.Data(4,:,:), 1, LPPO);

t = tiledlayout(2, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

Title = title(t, "Observations", 'FontSize', 30, 'FontWeight', 'bold');

ax_X = nexttile;
ax_Xdot = nexttile;
ax_T = nexttile;
ax_Tdot = nexttile;

hold(ax_X, 'on'); box(ax_X, 'on');
plot(ax_X, XDQN, 'Color', [0 0 1]); ax_X.XLim = [0 500];
plot(ax_X, XDDPG, 'Color', [0 1 0]);
plot(ax_X, XPPO, 'Color', [1 0 0]);

hold(ax_Xdot, 'on'); box(ax_Xdot, 'on');
plot(ax_Xdot, XdotDQN, 'Color', [0 0 1]); ax_Xdot.XLim = [0 500];
plot(ax_Xdot, XdotDDPG, 'Color', [0 1 0]);
plot(ax_Xdot, XdotPPO, 'Color', [1 0 0]);

hold(ax_T, 'on'); box(ax_T, 'on');
plot(ax_T, TDQN, 'Color', [0 0 1]); ax_T.XLim = [0 500];
plot(ax_T, TDDPG, 'Color', [0 1 0]);
plot(ax_T, TPPO, 'Color', [1 0 0]);

hold(ax_Tdot, 'on'); box(ax_Tdot, 'on');
plot(ax_Tdot, TdotDQN, 'Color', [0 0 1]); ax_Tdot.XLim = [0 500];
plot(ax_Tdot, TdotDDPG, 'Color', [0 1 0]);
plot(ax_Tdot, TdotPPO, 'Color', [1 0 0]);

xlabel(ax_X, 'step', 'FontSize', 15, 'FontWeight', 'bold');
xlabel(ax_Xdot, 'step', 'FontSize', 15, 'FontWeight', 'bold');
xlabel(ax_T, 'step', 'FontSize', 15, 'FontWeight', 'bold');
xlabel(ax_Tdot, 'step', 'FontSize', 15, 'FontWeight', 'bold');

ylabel(ax_X, 'position [m]', 'FontSize', 15, 'FontWeight', 'bold');
ylabel(ax_Xdot, 'velocity [m/s]', 'FontSize', 15, 'FontWeight', 'bold');
ylabel(ax_T, 'angle [rad]', 'FontSize', 15, 'FontWeight', 'bold');
ylabel(ax_Tdot, 'angular velocity [rad/s]', 'FontSize', 15, 'FontWeight', 'bold');

legend(ax_X, ["DQN", "DDPG", "PPO"], 'FontSize', 12, 'FontWeight', 'bold', 'Location', 'best');
legend(ax_Xdot, ["DQN", "DDPG", "PPO"], 'FontSize', 12, 'FontWeight', 'bold', 'Location', 'best');
legend(ax_T, ["DQN", "DDPG", "PPO"], 'FontSize', 12, 'FontWeight', 'bold', 'Location', 'best');
legend(ax_Tdot, ["DQN", "DDPG", "PPO"], 'FontSize', 12, 'FontWeight', 'bold', 'Location', 'best');

saveas(gcf, "LearningResult\ObservationGraph.fig");