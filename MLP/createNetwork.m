%% This function creates the neural network described by the networkStructureMat
% networkStructureMat is an mX1 matrix with m being the number of layers
% and networkStructureMat(i, 1) is the number of neurons in each layer with
% networkStructureMat(1, 1) being the number of input elements in the network
% network is the network that is created according to networkStructureMat
function [network] = createNetwork(networkStructureMat, activationFunctionId)

network = {};
[numLayers, ~] = size(networkStructureMat);
idxCtr = 0;
for i = 2:numLayers
    layer = [];
    numNeurons = networkStructureMat(i, 1);
    % the number of inputs to each neuron of the current layer is equal to
    % the number of units in the previous layer
    numInputs = networkStructureMat(i-1, 1);
    for j = 1:numNeurons
        idxCtr = idxCtr + 1;
        neuron = createNeuron(numInputs, activationFunctionId, idxCtr);
%         a = -1;
%         b = 1;
        [numWts, ~] = size(neuron.weights);
%         neuron.weights = a + (b-a)*(rand(numWts,1));
        neuron.weights = unifrnd(-1,1,[numWts,1]);
        layer = horzcat(layer, neuron);
    end
    network = vertcat(network, layer);
end
end