%% This function generates the random data points, inputs them to the function 
% and generates desired outputs for training the RBF net
function [data, funcOutput, noisyOutput] = generateData(numVals, distribution, func)

if strcmp(distribution, 'uniform')
    data = unifrnd(0, 1, [numVals, 1]);
end
data = sort(data);
if strcmp(func, 'sinusoid')
    u = unifrnd(-0.1, 0.1, [numVals, 1]);
    funcOutput = 0.5 + 0.4 * sin(2 * pi * data);
    noisyOutput = funcOutput + u;
end

end