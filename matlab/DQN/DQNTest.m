clear; clc; close;

env = rlPredefinedEnv("CartPole-Discrete");

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

load('MATLABCartpoleDQNMulti.mat','agent')

h = plot(env);

simOptions = rlSimulationOptions('MaxSteps',500);
experience = sim(env,agent,simOptions);