%% This function creates the neural network described by the networkStructureMat
% networkStructureMat is an mX1 matrix with m being the number of layers
% and networkStructureMat(i, 1) is the number of neurons in each layer with
% networkStructureMat(1, 1) being the number of input elements in the network
% network is the network that is created according to networkStructureMat
function [network] = createLayer(networkStructureMat, activationFunctionId)

network = {};
[numLayers, val] = size(networkStructureMat);
for i = 2:numLayers
    layer = [];
    numNeurons = networkStructureMat(i, 1);
    % the number of inputs to each neuron of the current layer is equal to
    % the number of units in the previous layer
    numInputs = networkStructureMat(i-1, 1);
    for j = 1:numNeurons
        neuron = createNeuron(numInputs, activationFunctionId);
        layer = horzcat(layer, neuron);
    end
    network = vertcat(network, layer);
end
end