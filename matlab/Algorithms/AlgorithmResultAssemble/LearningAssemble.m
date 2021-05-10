clear; clc; close;

TrainingStatDQN = load("..\DQN\LearningResult\trial5\training_res.mat");
TrainingStatDDPG = load("..\DDPG\train_res.mat");
TrainingStatPPO = load("..\PPO\Trial\Trial(Default)\train_res.mat");

ax = gca;

hold(ax, 'on');
plot(ax, TrainingStatDQN.trainingStats.AverageReward, 'Color', [0 0 1]);
plot(ax, TrainingStatDDPG.TraningStats.AverageReward, 'Color', [0 0 0], 'Marker', 'o');
plot(ax, TrainingStatPPO.TrainingStats.AverageReward, 'Color', [1 0 0], 'Marker', 'x');