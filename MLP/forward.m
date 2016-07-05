function [output, potential] = forward(weights, inputs)

potential = sum(weights .* inputs);
output = activate(potential, 1, 1);

end
