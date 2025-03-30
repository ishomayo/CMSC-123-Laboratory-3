function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Ensure all matrices are in double format
Theta1 = double(Theta1);
Theta2 = double(Theta2);
X = double(X);

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1'); % Line 12 - Ensure X is double
h2 = sigmoid([ones(m, 1) h1] * Theta2');

[dummy, p] = max(h2, [], 2);

end

