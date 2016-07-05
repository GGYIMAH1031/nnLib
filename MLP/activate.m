function [output] = activate(input, activationFunction, functionOrder)

if activationFunction == 1 % Logistic Sigmoid
    output = logisticSigmoid(input, functionOrder);
end

end

function [output] = logisticSigmoid(input, order)

phiV = 1 / (1 + exp(-1*input));

if order == 1
    output = phiV;
end
if order == 2
    output = phiV * (1 - phiV);
end
end

