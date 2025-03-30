function test_ga_nn()
    % Load test data
    [~, ~, X_test, y_test, ~, ~] = preprocess_data();

    % Train using GA
    [Theta1, Theta2] = ga_nn();

    % Predict on test data
    predictions = predict(Theta1, Theta2, X_test);

    % Compute accuracy
    accuracy = mean(double(predictions == y_test)) * 100;
    fprintf('Test Accuracy (GA): %.2f%%\n', accuracy);
end

