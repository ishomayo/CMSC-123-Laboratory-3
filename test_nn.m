function test_nn()
    % Load trained network and test data
    [~, ~, X_test, y_test, ~, ~] = preprocess_data();
    [Theta1, Theta2] = train_nn();

    % Predict on test data
    predictions = predict(Theta1, Theta2, X_test);

    % Compute accuracy
    accuracy = mean(double(predictions == y_test)) * 100;
    fprintf('Test Accuracy: %.2f%%\n', accuracy);
end

