clear; clc; close;

env1 = rlPredefinedEnv("CartPole-Discrete");
env2 = CartPoleDiscreteAction2;

plot(env1);plot(env2);
load("agent_saver.mat");

simOptions = rlSimulationOptions('MaxSteps', 500);

experience1 = sim(env1, agent1, simOptions);
experience2 = sim(env2, agent2, simOptions);

%% env1
f1 = figure(3);
plot(experience1.Observation.CartPoleStates);
h1 = legend("$x$", "$\dot{x}$", "$\theta$", "$\dot{\theta}$");
h1.Interpreter = 'latex';
title("Observation of 2-action-state environment");

f2 = figure(4);
plot(experience1.Action.CartPoleAction);
h1 = legend("$F$");
h1.Interpreter = 'latex';
title("Action of 2-action-state environment");

%%env2
f3 = figure(5);
plot(experience2.Observation.CartPoleStates);
h2 = legend("$x$", "$\dot{x}$", "$\theta$", "$\dot{\theta}$");
h2.Interpreter = 'latex';
title("Observation of 5-action-state environment");

f4 = figure(6);
plot(experience2.Action.CartPoleAction);
h2 = legend("$F$");
h2.Interpreter = 'latex';
title("Action of 5-action-state environment");

saveas(f1, "ExperienceResult\trial1\2state_obs.png");
saveas(f2, "ExperienceResult\trial1\2state_act.png");
saveas(f3, "ExperienceResult\trial1\5state_obs.png");
saveas(f4, "ExperienceResult\trial1\5state_act.png");