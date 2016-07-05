%% This function creates and tests an RBF NN
% Steps:
% 1. Generate Data
% 2. Find K-Means based Gaussian Basis Centers
% 3. Create Gaussian Basis neurons
% 4. Calculate the hidden to output neuron weights
% 5. Generate output and Display

clc;
clear all;

%% Configuration Values
centerVals = [2, 4, 7, 11, 16];
numCenterConfigs = 5;
etaVals = [0.01, 0.02];
numEtaConfigs = 2;
varianceVals = [0, 1];
numVarianceConfigs = 2;
numVals = 75;
distribution = 'uniform';
func = 'sinusoid';
numInputs = 1;
numOutput = 1;

%% Generate Data
[data, functionOutput, desiredOutput] = generateData(numVals, distribution, func);


%% Run over all number of bases, over all eta values, all variance configurations
for l = 1:numCenterConfigs
    numCenters = centerVals(l);
    [centers, centerVariance, dMax] = kMeansClustering(data, numCenters, 0);
    for m = 1:numEtaConfigs
        for n = 1:numVarianceConfigs
            
            eta = etaVals(m);
            uniformVariance = varianceVals(n);
            numHidden = numCenters;
            % preallocate arrays
            inputsHidden = zeros(numInputs, 1);
            % +1 for bias
            outputHidden = zeros(numHidden + 1, 1);
            weightsHidden = zeros((numHidden + 1), 1);
            outputMatrix = zeros(numVals, (numHidden + 1));
            outputs = zeros(numVals, 1);
            
            %% Find Data Centeres using K-Means
            sigma = dMax / sqrt(2*numCenters);
            
            %% Train RBF Network
            epoch = 1;
            ctr = 1;
            while epoch <= 100
                for i = 1:numVals
                    % take input
                    inputsHidden = data(i, 1);
                    outputHidden(1, 1) = 1;
                    for j = 1:numHidden
                        % for uniform variance take sigma computed using
                        % dMax
                        if uniformVariance == 1
                            tempSigma = sigma;
                        else
                            tempSigma = sqrt(centerVariance(j, 1));
                        end
                        % find the response of each gaussian basis
                        outputHidden(j+1, 1) = gaussianBasisFunction(centers(j, 1), tempSigma, inputsHidden);
                    end
                    % calculate output of the network
                    outputMatrix(i, :) = outputHidden;
                    outputs(i, 1) = sum(outputHidden .* weightsHidden);
                    % update the weights
                    deltaW = (eta * (desiredOutput(i, 1) - outputs(i, 1)) * outputHidden);
                    weightsHidden = (weightsHidden + deltaW);
                end
                
                % check for completion of an epoch
                if mod(ctr, numVals) == 0
                    epoch = epoch + 1;
                    % flush out matrices
                    outputMatrix = zeros(numVals, (numHidden + 1));
                    outputs = zeros(numVals, 1);
                    inputsHidden = zeros(numInputs, 1);
                    ctr = 0;
                end
                ctr = ctr + 1;
            end
            
            % calculate final output after learning
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
            
            % display dp and yp
            figure(1)
            clf;
            str = strcat('Stochastic: num basis = ', num2str(centerVals(l)), ',eta = ', num2str(eta),...
                ', uni var(1)/non uni var(0) = ', num2str(uniformVariance));
            plot([1:numVals], desiredOutput, 'b-');
            hold on
            plot([1:numVals], outputs, 'r-');
            xlabel('index');
            ylabel('dp (blue), yp (red)');
            title(str);
            
            % display dp, yp and h(xp0
            figure(2)
            clf;
            plot(data, desiredOutput, 'b.');
            hold on
            plot(data, functionOutput, 'g-');
            hold on
            plot(data, outputs, 'r-');
            pause(0.5)
            xlabel('xp');
            ylabel('dp (blue), yp (red), h(xp) (green)');
            title(str);
            pause(0.05);
        end
    end
end
status = 'Completed!'