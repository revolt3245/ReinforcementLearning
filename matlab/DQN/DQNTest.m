clear; clc; close;

env1 = rlPredefinedEnv("CartPole-Discrete");
env2 = CartPoleDiscreteAction2;

obsInfo1 = getObservationInfo(env1);
actInfo1 = getActionInfo(env1);

rng(0)

episode_2state = [];
episode_5state = [];

%% Network
dnn1 = [
    featureInputLayer(obsInfo1.Dimension(1),'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(24, 'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(length(actInfo1.Elements),'Name','output')];

criticOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',1, 'UseDevice', "gpu");
critic = rlQValueRepresentation(dnn1,obsInfo1,actInfo1,'Observation',{'state'},criticOpts);

agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false, ...    
    'TargetSmoothFactor',1, ...
    'TargetUpdateFrequency',4, ...   
    'ExperienceBufferLength',100000, ...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',256);

agent1 = rlDQNAgent(critic,agentOpts);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000, ...
    'MaxStepsPerEpisode',500, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',480);

for i=1:5
    trainingStats = train(agent1, env1, trainOpts);
    episode_2state = [episode_2state length(trainingStats.EpisodeIndex)];
    agent1 = rlDQNAgent(critic,agentOpts);
    close all;
end

obsInfo2 = getObservationInfo(env2);
actInfo2 = getActionInfo(env2);

rng(0)
dnn2 = [
    featureInputLayer(obsInfo2.Dimension(1),'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(24, 'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(length(actInfo2.Elements),'Name','output')];

criticOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',1, 'UseDevice', "gpu");
critic = rlQValueRepresentation(dnn2,obsInfo2,actInfo2,'Observation',{'state'},criticOpts);

agent2 = rlDQNAgent(critic,agentOpts);

for i=1:5
    trainingStats = train(agent2, env2, trainOpts);
    episode_5state = [episode_5state length(trainingStats.EpisodeIndex)];
    agent2 = rlDQNAgent(critic,agentOpts);
    close all;
end

save("episode_number.mat", "episode_2state", "episode_5state");