function [weights, gradient, delta] = backward(weights, error, potential, inputs,  mu, alpha, deltaWPrev)

e = error;
phiDash = activate(potential, 1, 2);
delta = e * phiDash;
gradient = mu * delta * inputs;
weights = weights + gradient + alpha * deltaWPrev;

end
