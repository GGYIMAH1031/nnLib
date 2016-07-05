%% Multi Layer Perceptron for the Parity Problem
% This code runs for learning rates 0.5 - 0.05 with momentum = 0.9
% Network structure  is 4,8,1
% Batch gradient learning

clc
clear all;

numInputs = 4;
numHiddenNeurons = 8;
numOutputNeurons = 1;
batchGradient = 1;

weightsHiddenLayer = unifrnd(-1, 1, [(numInputs + 1), numHiddenNeurons]);
weightsOutputLayer = unifrnd(-1, 1, [(numHiddenNeurons + 1), numOutputNeurons]);
batchGradientHiddenLayer = zeros((numInputs + 1), numHiddenNeurons);
batchGradientOutputLayer = zeros((numHiddenNeurons + 1), numOutputNeurons);
deltaWPrevHiddenLayer = zeros((numInputs + 1), numHiddenNeurons);
deltaWPrevOutputLayer = zeros((numHiddenNeurons + 1), numOutputNeurons);
deltaOutputLayer = zeros(numOutputNeurons,1);
deltaHiddenLayer = zeros(numHiddenNeurons,1);

activationFunctionId = 1;
load trainSetParity4;
load outSetParity4;
mu = 0.55;
alpha = 0.9;
trainingAlgorithmId = 1;
epochVec = zeros(10,1);


for iter = 1:10
    
    weightsHiddenLayer = unifrnd(-1, 1, [(numInputs + 1), numHiddenNeurons]);
    weightsOutputLayer = unifrnd(-1, 1, [(numHiddenNeurons + 1), numOutputNeurons]);
    batchGradientHiddenLayer = zeros((numInputs + 1), numHiddenNeurons);
    batchGradientOutputLayer = zeros((numHiddenNeurons + 1), numOutputNeurons);
    deltaWPrevHiddenLayer = zeros((numInputs + 1), numHiddenNeurons);
    deltaWPrevOutputLayer = zeros((numHiddenNeurons + 1), numOutputNeurons);
    deltaOutputLayer = zeros(numOutputNeurons,1);
    deltaHiddenLayer = zeros(numHiddenNeurons,1);
    stopTraining = 0;
    ctr = 1;
    minErrCtr = 0;
    errorMat = zeros(16, 1);
    epoch = 1;
    mseVec = [];
    maeVec = [];
    mu = mu - 0.05;
    jumbleVector = randperm(16);
    while stopTraining == 0
        
        %% Take Input
        inputVec = trainSetParity4(jumbleVector(ctr), :);
        desiredOutput = outSetParity4(jumbleVector(ctr), :);
        inputHidden = [1, inputVec]';
        outputHidden = zeros(numHiddenNeurons, 1);
        potentialHidden = zeros(numHiddenNeurons, 1);
        
        %% Forward Propagation
        for i = 1:numHiddenNeurons
            [outputHidden(i, 1), potentialHidden(i, 1)] = forward(weightsHiddenLayer(:, i), inputHidden);
        end
        inputOut = [1, outputHidden(:, 1)']';
        [output, potentialOut] = forward(weightsOutputLayer(:, 1), inputOut);
        e = (desiredOutput - output);
        error = abs(e);
        errorMat(ctr, 1) = error;
        
        %% Backward Propagation
        if batchGradient == 0
            [tempWeightsOutput, ~, deltaOutputLayer] = backward(weightsOutputLayer, e, potentialOut, inputOut, mu, alpha, deltaWPrevOutputLayer);
            deltaWPrevOutputLayer = tempWeightsOutput - weightsOutputLayer;
            weightsOutputLayer = tempWeightsOutput;
        else
            [~, tempGradients, deltaOutputLayer] = backward(weightsOutputLayer, e, potentialOut, inputOut, mu, alpha, deltaWPrevOutputLayer);
            batchGradientOutputLayer = batchGradientOutputLayer + tempGradients;
        end
        for i = 1:numHiddenNeurons
            err = deltaOutputLayer * weightsOutputLayer(i+1, 1);
            if batchGradient == 0
                [tempWeightsHidden, ~, delta] = backward(weightsHiddenLayer(:,i), err, potentialHidden(i,1),...
                    inputHidden, mu, alpha, deltaWPrevHiddenLayer(:, i));
                deltaWPrevHiddenLayer(:, i) = tempWeightsHidden - weightsHiddenLayer(:,i);
                weightsHiddenLayer(:,i) = tempWeightsHidden;
            else
                [~, tempGradients, delta] = backward(weightsHiddenLayer(:,i), err, potentialHidden(i,1),...
                    inputHidden, mu, alpha, deltaWPrevHiddenLayer(:, i));
                batchGradientHiddenLayer(:,i) = batchGradientHiddenLayer(:,i) + tempGradients;
            end
        end
        
        ctr = mod(ctr, 16) + 1;
        if ctr == 1
            %% Batch weights update
            if batchGradient == 1
                batchGradientOutputLayer = batchGradientOutputLayer ./ 16;
                batchGradientHiddenLayer = batchGradientHiddenLayer ./ 16;
                
                weightsOutputLayer = weightsOutputLayer + batchGradientOutputLayer + alpha * deltaWPrevOutputLayer;
                weightsHiddenLayer = weightsHiddenLayer + batchGradientHiddenLayer + alpha * deltaWPrevHiddenLayer;
                
                deltaWPrevOutputLayer = batchGradientOutputLayer + alpha * deltaWPrevOutputLayer;
                deltaWPrevHiddenLayer = batchGradientHiddenLayer + alpha * deltaWPrevHiddenLayer;
                
                batchGradientOutputLayer = batchGradientOutputLayer .* 0;
                batchGradientHiddenLayer = batchGradientHiddenLayer .* 0;
            end
            %% Error Calculations
            if batchGradient == 0
                jumbleVector = randperm(16);
            end
            mae = max(errorMat);
            mse = sqrt(sum(errorMat.*errorMat) / 16);
            mseVec = vertcat(mseVec, mse);
            maeVec = vertcat(maeVec, mae);
            errorVec = horzcat(mseVec, maeVec);
            epoch = epoch + 1;
            
            if mae < 0.05
                stopTraining = 1;
            end
            if epoch > 200000
                stopTraining = 1;
            end
            if mod(epoch, 2000) == 0
                figure(11)
                plot(errorVec);
            end
        end
    end
    epochVec(iter, 1) = epoch;
    status = iter
    figure1 = figure;
    plot(errorVec);
end