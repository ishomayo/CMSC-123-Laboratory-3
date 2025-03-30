function test_ga_nn()
    % Load test data
    [~, ~, X_test, y_test, ~, ~] = preprocess_data();

    % Train using GA
    [Theta1, Theta2] = ga_nn();

    % Predict on test data
    predictions = predict(Theta1, Theta2, X_test);

    % Display predictions vs. actual results
    fprintf('Predicted vs. Actual Labels:\n');
    disp([predictions y_test]); % Show side-by-side comparison

    % Check for misclassifications
    misclassified = find(predictions != y_test);
    if isempty(misclassified)
        fprintf('No misclassified samples. Model is 100%% accurate.\n');
    else
        fprintf('Model misclassified %d samples.\n', length(misclassified));
        disp([predictions(misclassified) y_test(misclassified)]); % Show incorrect cases
    end

    % Display sample of predictions
    fprintf('Sample of Predicted vs. Actual Labels (First 10):\n');
    disp([predictions(1:10) y_test(1:10)]);

    % Compute accuracy
    accuracy = mean(double(predictions == y_test)) * 100;
    fprintf('Correctly Classified: %d / %d\n', sum(predictions == y_test), length(y_test));
    fprintf('Test Accuracy (GA): %.2f%%\n', accuracy);
end

