clear; clc; close;

Env = CartPoleContinuousAction2;

load('agent_saver.mat');

SimOpts = rlSimulationOptions('MaxSteps', 500);

plot(Env);

SimResult = sim(Env, Agent, SimOpts);

Observation = SimResult.Observation.CartPoleStates;
Action = SimResult.Action.CartPoleAction;

plot(Observation);