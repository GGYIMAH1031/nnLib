%% This function creates and tests an RBF NN
% Steps:
% 1. Generate Data
% 2. Find K-Means based Gaussian Basis Centers
% 3. Create Gaussian Basis neurons
% 4. Calculate the hidden to output neuron weights
% 5. Create and test network

clc;
clear all;

%% Configuration Values
centerVals = [2, 4, 7, 11, 16];
numCenterConfigs = 5;
etaVals = [0.01, 0.02];
numEtaConfigs = 2;
varianceVals = [0, 1];
numVarianceConfigs = 2;
batchLearningVals = [0, 1];
numLearningConfigs = 2;
numVals = 75;
distribution = 'uniform';
func = 'sinusoid';
numInputs = 1;
numOutput = 1;

%% Generate Data
[data, functionOutput, desiredOutput] = generateData(numVals, distribution, func);


%% Run over all number of bases, over all eta values, all variance configurations and learning types
for l = 1:numCenterConfigs
    for m = 1:numEtaConfigs
        for n = 1:numVarianceConfigs
            for o = 1:numLearningConfigs
                
                numCenters = centerVals(l);
                eta = etaVals(m);
                uniformVariance = varianceVals(n);
                numHidden = numCenters;
                batchLearning = batchLearningVals(o);
                % preallocate arrays
                inputsHidden = zeros(numInputs, 1);
                % +1 for bias
                outputHidden = zeros(numHidden + 1, 1);
                % weightsHidden = unifrnd(0, 1, [(numHidden + 1), 1]);
                weightsHidden = zeros((numHidden + 1), 1);
                outputMatrix = zeros(numVals, (numHidden + 1));
                outputs = zeros(numVals, 1);
                
                %% Find Data Centeres using K-Means
                [centers, centerVariance, dMax] = kMeansClustering(data, numCenters, 0, 0);
                status = 'centers found!';
                sigma = dMax / sqrt(2*numCenters);
                
                %% Train RBF Network
                
                epoch = 1;
                ctr = 1;
                while epoch <= 100
                    for i = 1:numVals
                        inputsHidden = data(i, 1);
                        outputHidden(1, 1) = 1;
                        for j = 1:numHidden
                            if uniformVariance == 1
                                tempSigma = sigma;
                            else
                                tempSigma = sqrt(centerVariance(j, 1));
                            end
                            outputHidden(j+1, 1) = gaussianBasisFunction(centers(j, 1), tempSigma, inputsHidden);
                        end
                        outputMatrix(i, :) = outputHidden;
                        outputs(i, 1) = sum(outputHidden .* weightsHidden);
                        if batchLearning == 0
                            deltaW = (eta * (desiredOutput(i, 1) - outputs(i, 1)) * outputHidden);
                            weightsHidden = (weightsHidden + deltaW);
                       end
                    end

                    if mod(ctr, numVals) == 0
                        epoch = epoch + 1;
                        if batchLearning == 1
                            deltaW = ((eta * (desiredOutput - outputs)' * outputMatrix));
                            weightsHidden = (weightsHidden' + deltaW ./ numVals)';
                        end
                        outputMatrix = zeros(numVals, (numHidden + 1));
                        outputs = zeros(numVals, 1);
                        inputsHidden = zeros(numInputs, 1);
                        ctr = 0;
                    end
                    ctr = ctr + 1;
                end
                
                for i = 1:numVals
                    inputsHidden = data(i);
                    outputHidden(1, 1) = 1;
                    for j = 1:numHidden
                        if uniformVariance == 1
                            tempSigma = sigma;
                        else
                            tempSigma = sqrt(centerVariance(j, 1));
                        end
                        outputHidden(j+1, 1) = gaussianBasisFunction(centers(j, 1), tempSigma, inputsHidden);
                    end
                    outputs(i, 1) = sum(outputHidden .* weightsHidden);
                end
                
                figure(1)
                clf;
                str = strcat('num basis = ', num2str(centerVals(l)), ',eta = ', num2str(eta),...
                    ', uni var(1)/non uni var(0) = ', num2str(uniformVariance), ', stoch(0)/batch(1) = ', num2str(batchLearningVals(o)));
                plot([1:numVals], desiredOutput, 'b-');
                hold on
                plot([1:numVals], outputs, 'r-');
                xlabel('index');
                ylabel('dp, yp');
                title(str);
                name1 = strcat('output_', num2str(centerVals(l)), '_', num2str(eta), '_', num2str(uniformVariance), '_batch', num2str(o), '.bmp');
                saveas(figure(1), name1);
                
                figure(2)
                clf;
                plot(data, desiredOutput, 'b.');
                hold on
                plot(data, functionOutput, 'g-');
                hold on
                plot(data, outputs, 'r-');
                pause(0.5)
                xlabel('xp');
                ylabel('dp, yp, h(xp)');
                title(str);
                name2 = strcat('plots_', num2str(centerVals(l)), '_', num2str(eta), '_', num2str(uniformVariance), '_batch', num2str(o), '.bmp');
                saveas(figure(2), name2);
            end
        end
    end
end
status = 'Completed!'