function [response] = gaussianBasisFunction(center, sigma, input)

response = exp( (-1/(2*sigma^2)) * norm(input - center)^2);

end