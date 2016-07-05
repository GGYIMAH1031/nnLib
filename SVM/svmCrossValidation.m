clc;
clear all;

% Load data
[labelsTraining, instancesTraining] = libsvmread('proj3_train.lsv');
[labelsTesting, instancesTesting] = libsvmread('proj3_test.lsv');
numSets = 5;
% Generate cross validation training and test datasets
[trLabels, trInstances, testLabels, testInstances] = generateCrossValidationData(labelsTraining, instancesTraining, numSets);

%% Linear SVM

CExponents = (-4:8)';
CVals = 2.^(CExponents);
[numCVals, ~] = size(CVals);
linSVMAcc = zeros(size(CVals));

% Run linear SVM cross validation
for i = 1:numCVals
    for j = 1:numSets
        params = ['-t 0 -c ', num2str(CVals(i, 1))]; 
        model = svmtrain(trLabels(:, :, j), trInstances(:, :, j), params);
        [~, acc, ~] = svmpredict(testLabels(:, :, j), testInstances(:, :, j), model);
        linSVMAcc(i) = linSVMAcc(i) + acc(1);
    end
    linSVMAcc(i) = linSVMAcc(i) / numSets;
end

% Display plot
figure(1)
semilogx(CVals, linSVMAcc, 'ro-');
xlabel('C');
ylabel('Avg Accuracy');
title('Avg Accuracy Vs C Values');

% Find best C and train and test on whole datasets
maxAccuracy = max(linSVMAcc);
[idrLin, ~] = find(linSVMAcc == maxAccuracy);
idrLinMin = min(idrLin);
params = ['-t 0 -c ', num2str(CVals(idrLinMin, 1))];
bestLinearModel = svmtrain(labelsTraining, instancesTraining, params);
[~, testLinAcc, ~] = svmpredict(labelsTesting, instancesTesting, bestLinearModel);


%% RBF Kernel SVM

kernelSVMAcc = zeros(numCVals, numCVals);

% Run RBF kernel cross validation (i iterates over C), (j iterates over
% Gamma), (k iterates over cross-validation sets)
for i = 1:numCVals
    for j = 1:numCVals
        for k = 1:numSets
            params = ['-t 2 -c ', num2str(CVals(i, 1)), ' -g ', num2str(CVals(j, 1))];
            model = svmtrain(trLabels(:, :, k), trInstances(:, :, k), params);
            [~, acc, ~] = svmpredict(testLabels(:, :, k), testInstances(:, :, k), model);
            kernelSVMAcc(i, j) = kernelSVMAcc(i, j) + acc(1);
        end
        kernelSVMAcc(i, j) = kernelSVMAcc(i, j) / numSets;
    end
end

% Display RBF kernel SVM heat map
figure(2)
imagesc(CExponents, CExponents, kernelSVMAcc);
ylabel('log(C)');
xlabel('log(Gamma)');
title('Avg Accuracy Vs C and Gamma Values');

% Find best C and gamma and train and test on whole datasets
maxKernalAccuracy = max(kernelSVMAcc(:));
[idrKer, idcKer] = find(kernelSVMAcc == maxKernalAccuracy);
idrKerMin = min(idrKer);
idcKerMin = min(idcKer);
params = ['-t 2 -c ', num2str(CVals(idrKerMin, 1)), ' -g ', num2str(CVals(idcKerMin, 1))];
bestKernelModel = svmtrain(labelsTraining, instancesTraining, params);
[~, testKerAcc, ~] = svmpredict(labelsTesting, instancesTesting, bestKernelModel);