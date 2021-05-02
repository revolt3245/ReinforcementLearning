clear; clc; close;

env1 = rlPredefinedEnv("CartPole-Discrete");
env2 = CartPoleDiscreteAction2;

plot(env1);plot(env2);
load("agent_saver.mat");

simOptions = rlSimulationOptions('MaxSteps', 500);

experience1 = sim(env1, agent1, simOptions);
experience2 = sim(env2, agent2, simOptions);

hold on;
figure(3);
plot(experience1.Observation.CartPoleStates);
h1 = legend("$x$", "$\dot{x}$", "$\theta$", "$\dot{\theta}$");
h1.Interpreter = 'latex';
figure(4);
plot(experience2.Observation.CartPoleStates);
h2 = legend("$x$", "$\dot{x}$", "$\theta$", "$\dot{\theta}$");
h2.Interpreter = 'latex';