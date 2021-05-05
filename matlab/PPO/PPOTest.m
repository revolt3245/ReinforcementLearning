clear; clc; close;

Env = CartPoleContinuousAction2;

ObsInfo = Env.getObservationInfo;
ActInfo = Env.getActionInfo;

rng(1);

%% Critic Network
CriticNetwork = [
    featureInputLayer(ObsInfo.Dimension(1), 'Name', 'CriticInput')
    fullyConnectedLayer(24, 'Name', 'CriticL1')
    reluLayer('Name', 'CriticReLU1')
    fullyConnectedLayer(24, 'Name', 'CriticL2')
    reluLayer('Name', 'CriticReLU2')
    fullyConnectedLayer(1, 'Name', 'CriticOutput')
    ];

CriticOption = rlRepresentationOptions(...
    'LearnRate', 1e-3,...
    'GradientThreshold', 1,...
    'UseDevice', 'gpu');

Critic = rlValueRepresentation(CriticNetwork, ObsInfo, 'Observation', {'CriticInput'}, CriticOption);

%% Actor Network
ActorNetwork = [
    featureInputLayer(ObsInfo.Dimension(1), 'Name', 'ActorInput')
    fullyConnectedLayer(24, 'Name', 'ActorL1')
    reluLayer('Name', 'ActorReLU1')
    fullyConnectedLayer(24, 'Name', 'ActorL2')
    reluLayer('Name', 'ActorReLU2')
    fullyConnectedLayer(ActInfo.Dimension(1), 'Name', 'ActorL3')
    tanhLayer('Name', 'ActorTanh1')
    scalingLayer('Name', 'ActorScaling', 'Scale', ActInfo.UpperLimit)
    ];

ActorOption = rlRepresentationOptions('LearnRate', 5e-4, 'GradientThreshold', 1, 'UseDevice', 'gpu');

Actor = rlStochasticActorRepresentation(ActorNetwork, ObsInfo, ActInfo,...
    'Observation', {'ActorInput'}, ActorOption);

%% Agent
AgentOptions = rlPPOAgentOptions(...
    'SampleTime', Env.Ts,...
    'MiniBatchSize', 256);

Agent = rlPPOAgent(Actor, Critic, AgentOptions);