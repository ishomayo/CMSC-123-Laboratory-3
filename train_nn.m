function [Theta1, Theta2] = train_nn()
    % Load preprocessed data
    [X_train, y_train, ~, ~, input_layer_size, num_labels] = preprocess_data();

    % Set neural network parameters
    hidden_layer_size = 50; % Adjust as needed
    lambda = 0.01; % Regularization parameter

    % Initialize weights
    Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

    % Unroll parameters
    initial_nn_params = [Theta1(:); Theta2(:)];

    % Set optimization options
    options = optimset('MaxIter', 100); % Increase for better training

    % Cost function
    costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                                       num_labels, X_train, y_train, lambda);

    % Train neural network using fmincg
    [nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

    % Reshape trained parameters
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
end

