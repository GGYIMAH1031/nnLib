%% This function creates a neuron
% numInputs is the number of inputs to the neuron
% activationFunctionId is the index of the activation function that is to
% be used for this particular neuron.
function [neuron] = createNeuron(numInputs, activationFunctionId, idx)
inputs = (numInputs + 1); % bias is the first input
neuron = struct('weights', zeros(inputs, 1), 'inputs', zeros(inputs, 1),  'potential', 0.0,...
    'activationFunctionId', activationFunctionId, 'idx', idx, 'output', 0.0, 'epoch', 0);
end