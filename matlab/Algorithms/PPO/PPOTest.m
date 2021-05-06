clear; clc; close;

Env = CartPoleContinuousAction2;

ObsInfo = Env.getObservationInfo;
ActInfo = Env.getActionInfo;

rng(1);

%% Critic Network
CriticNetwork = [
    imageInputLayer([ObsInfo.Dimension 1], 'Normalization', 'none', 'Name', 'CriticInput')
    fullyConnectedLayer(24, 'Name', 'CriticL1')
    reluLayer('Name', 'CriticReLU1')
    fullyConnectedLayer(24, 'Name', 'CriticL2')
    reluLayer('Name', 'CriticReLU2')
    fullyConnectedLayer(1, 'Name', 'CriticOutput')
    ];

CriticOption = rlRepresentationOptions(...
    'LearnRate', 1e-3,...
    'GradientThreshold', 1,...
    'UseDevice', 'cpu');

Critic = rlValueRepresentation(CriticNetwork, ObsInfo, 'Observation', {'CriticInput'}, CriticOption);

%% Actor Network
InputPath = [
    imageInputLayer([ObsInfo.Dimension 1], 'Normalization', 'none', 'Name', 'ActorInput')
    fullyConnectedLayer(24, 'Name', 'ActorL1')
    reluLayer('Name', 'ActorReLU1')
    fullyConnectedLayer(24, 'Name', 'ActorL2')
    reluLayer('Name', 'ActorReLU2')
    ];

MeanPath = [
    fullyConnectedLayer(24, 'Name', 'MeanL1')
    reluLayer('Name', 'MeanReLU1')
    fullyConnectedLayer(ActInfo.Dimension(1), 'Name', 'MeanL2')
    tanhLayer('Name', 'MeanTanh1')
    scalingLayer('Name', 'MeanScaling', 'Scale', ActInfo.UpperLimit)
    ];

DevPath = [
    fullyConnectedLayer(24, 'Name', 'DevL1')
    reluLayer('Name', 'DevReLU1')
    fullyConnectedLayer(ActInfo.Dimension(1), 'Name', 'DevL2')
    softplusLayer('Name', 'DevSoftplus1')
    ];

OutputPath = concatenationLayer(3, 2, 'Name', 'ActorOutput');

ActorNetwork = layerGraph(InputPath)...
    .addLayers(MeanPath)...
    .addLayers(DevPath)...
    .addLayers(OutputPath);

ActorNetwork = ActorNetwork.connectLayers('ActorReLU2', 'MeanL1/in')...
    .connectLayers('ActorReLU2', 'DevL1/in')...
    .connectLayers('MeanScaling', 'ActorOutput/in1')...
    .connectLayers('DevSoftplus1', 'ActorOutput/in2');

ActorOption = rlRepresentationOptions('LearnRate', 5e-4, 'GradientThreshold', 1, 'UseDevice', 'cpu');

Actor = rlStochasticActorRepresentation(ActorNetwork, ObsInfo, ActInfo,...
    'Observation', {'ActorInput'}, ActorOption);

%% Agent
AgentOptions = rlPPOAgentOptions(...
    'SampleTime', Env.Ts,...
    'MiniBatchSize', 256);

Agent = rlPPOAgent(Actor, Critic, AgentOptions);

%% Train
TrainOpts = rlTrainingOptions(...
    'MaxEpisodes',10000, ...
    'MaxStepsPerEpisode',500, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',480);

plot(Env);
TrainingStats = train(Agent, Env, TrainOpts);

save('agent_saver.mat', 'Agent');
save('train_res.mat', 'TrainingStats');