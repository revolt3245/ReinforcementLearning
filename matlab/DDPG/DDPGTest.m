clear; clc; close;

% Env = rlPredefinedEnv("CartPole-Continuous");
Env = CartPoleContinuousAction2;
% Env.PenaltyForFalling = -5;

ObsInfo = Env.getObservationInfo;
ActInfo = Env.getActionInfo;

rng(1);

%% Critic Network
StatePath = [
    featureInputLayer(ObsInfo.Dimension(1), 'Name', 'ObsInput', 'Normalization', 'none')
    fullyConnectedLayer(24, 'Name', 'ObsL1')
    reluLayer('Name', 'ObsReLU1')
    fullyConnectedLayer(24, 'Name', 'ObsL2')
    ];

ActionPath = [
    featureInputLayer(ActInfo.Dimension(1), 'Name', 'ActInput', 'Normalization', 'none')
    fullyConnectedLayer(24, 'Name', 'ActL1', 'BiasLearnRateFactor', 0)
    ];

CommonPath = [
    additionLayer(2, 'Name', 'Add')
    reluLayer('Name', 'CommonReLU')
    fullyConnectedLayer(1, 'Name', 'CriticOutput')
    ];

CriticNetwork = layerGraph(StatePath);
CriticNetwork = CriticNetwork.addLayers(ActionPath).addLayers(CommonPath);
CriticNetwork = CriticNetwork.connectLayers('ObsL2', 'Add/in1').connectLayers('ActL1', 'Add/in2');


CriticOption = rlRepresentationOptions('LearnRate', 1e-3, 'GradientThreshold', 1, 'UseDevice', 'gpu');
Critic = rlQValueRepresentation(CriticNetwork, ObsInfo, ActInfo,...
    'Observation', {'ObsInput'}, 'Action', {'ActInput'}, CriticOption);

%% Actor Network
ActorNetwork = [
    featureInputLayer(ObsInfo.Dimension(1), 'Name', 'ActorInput')
    fullyConnectedLayer(24, 'Name', 'ActorL1')
    reluLayer('Name', 'ActorReLU1')
    fullyConnectedLayer(24, 'Name', 'ActorL2')
    reluLayer('Name', 'ActorReLU2')
    fullyConnectedLayer(1, 'Name', 'ActorL3')
    tanhLayer('Name', 'ActorTanh1')
    scalingLayer('Name', 'ActorScaling', 'Scale', ActInfo.UpperLimit)
    ];

ActorOption = rlRepresentationOptions('LearnRate', 5e-4, 'GradientThreshold', 1, 'UseDevice', 'gpu');

Actor = rlDeterministicActorRepresentation(ActorNetwork, ObsInfo, ActInfo,...
    'Observation', {'ActorInput'}, 'Action', {'ActorScaling'}, ActorOption);

%% Agent
AgentOptions = rlDDPGAgentOptions(...
    'SampleTime', Env.Ts,...
    'TargetSmoothFactor', 5e-2,...
    'ExperienceBufferLength', 1e6,...
    'MiniBatchSize', 256);

Agent = rlDDPGAgent(Actor, Critic, AgentOptions);

%% Train
TrainOpts = rlTrainingOptions(...
    'MaxEpisodes',10000, ...
    'MaxStepsPerEpisode',500, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',480);

plot(Env);
TraningStats = train(Agent, Env, TrainOpts);

save('agent_saver.mat', 'Agent');