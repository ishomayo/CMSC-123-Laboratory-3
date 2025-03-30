function [J] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two-layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Ensure all data is double precision
Theta1 = double(Theta1);
Theta2 = double(Theta2);
X = double(X);
y = double(y);

% Setup some useful variables
m = size(X, 1);
J = 0;

% Convert labels into one-hot encoding
y_temp = eye(num_labels);
Y = double(zeros(m, num_labels)); % Ensure Y is double
for i = 1 : m
  Y(i,:) = y_temp(int32(y(i)), :);
end

% Forward propagation
X = [ones(size(X, 1), 1) X]; % Add bias
z2 = X * Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2, 1), 1) a2]; % Add bias to hidden layer
z3 = a2 * Theta2';
h = sigmoid(z3);
h = double(h); % Ensure h is double

% Compute cost (without regularization)
J = sum(sum((-Y) .* log(h) - (1 - Y) .* log(1 - h))) / m;

% Compute cost (with regularization)
J = J + ((lambda / (2 * m)) * ...
    (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))));

end

