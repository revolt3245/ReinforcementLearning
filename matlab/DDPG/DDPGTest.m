clear; clc; close;

env = rlPredefinedEnv("CartPole-Continuous");

obsInfo = env.getObservationInfo;
actInfo = env.getActionInfo;

rng(0);

dnn = [
    featureInputLayer(obsInfo.Dimension(1),'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(24, 'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(actInfo.Dimension(1), 'Name','output')];
