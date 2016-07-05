%% Note that the inputVec would be an nX1 vector where n is the number of input elements
% and 1 is the number of testing samples.
% desiredOutputVec would be a kX1 vector with k being the output size and 1 being
% the number of testing samples which contains the desired output required
% out of the network
% trainAlgorithmId = 1 -> Backpropagation
function [error, network] = trainNetwork(network, inputVec, desiredOutputVec, trainingAlgorithmId, learningRate, momentum)

[numLayers, ~] = size(network);
input = [1, inputVec]';
for i = 1:numLayers
    layer = network{i};
    [~, numNeurons] = size(layer);
    for j = 1:numNeurons
        neuron = layer(j);
        neuron.inputs = input;
        neuron = activateNeuron(neuron, input);
        layer(j) = neuron;
    end
    input = [1, layer(:).output]'; % add 1 for bias
    network{i} = layer;
end

tempLayer = network{numLayers};
[~, numNeurons] = size(tempLayer);
error = 0.0;
for i = 1:numNeurons
    error = error + abs(desiredOutputVec(i) - tempLayer(i).output); 
end

if trainingAlgorithmId == 1
    network = backPropTraining(network, desiredOutputVec, learningRate, momentum);
end

end

function [network] = backPropTraining(network, desiredOutputVec, learningRate, momentum)

[numLayers, ~] = size(network);
desiredOutput = desiredOutputVec;

deltaMat = {};
for i = 1:numLayers
    layer = network{i};
    [~, numNeurons] = size(layer);
    deltaMat = vertcat(deltaMat, zeros(numNeurons, 1));
end
% deltaMat = fliplr(deltaMat);

for i = 0:numLayers-1
    layerIdx = numLayers - i;
    layer = network{layerIdx};
    [~, numNeurons] = size(layer);
    for j = 1:numNeurons
        input = [layer(j).inputs]';
        if layerIdx == numLayers % output layer
            e = (desiredOutput(j) - layer(j).output);
        else % hidden layer
            nextLayer = network{layerIdx + 1};
            nextLayerDeltas = deltaMat{(layerIdx + 1)};
            e = sum(nextLayerDeltas .* nextLayer(:).weights(j+1));
        end
        phiDash = activate(layer(j).potential, layer(j).activationFunctionId, 2); % derivative of activation function
        delta = e * phiDash;
        deltaMat{layerIdx}(j) = delta;
        layer(j).weights = layer(j).weights + (learningRate * delta * input');
    end
    network{layerIdx} = layer;
end
end