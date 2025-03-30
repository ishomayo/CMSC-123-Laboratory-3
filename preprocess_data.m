function [X_train, y_train, X_test, y_test, input_layer_size, num_labels] = preprocess_data()
    % Load dataset with categorical values
    fid = fopen('bank.csv', 'r');
    data = textscan(fid, '%d %s %s %s %s %d %s %s %s %d %s %d %d %d %d %s %s', 'Delimiter', ';', 'HeaderLines', 1);
    fclose(fid);

    % Encode categorical variables
    function encoded = encode_categories(data_column)
        [~, ~, encoded] = unique(data_column);
    end

    job = encode_categories(data{2});
    marital = encode_categories(data{3});
    education = encode_categories(data{4});
    default = encode_categories(data{5});
    housing = encode_categories(data{7});
    loan = encode_categories(data{8});
    contact = encode_categories(data{9});
    month = encode_categories(data{11});
    poutcome = encode_categories(data{16});
    y = double(strcmp(data{17}, 'yes')) + 1; % Convert 'yes' -> 1, 'no' -> 0 (ensure double type)


    % Construct the feature matrix
    X = [data{1}, job, marital, education, default, data{6}, housing, loan, ...
         contact, data{10}, month, data{12}, data{13}, data{14}, data{15}, ...
         poutcome];

    % Normalize numerical features
    function X_norm = normalize_features(X)
        mu = mean(X);
        sigma = std(X);
        X_norm = (X - mu) ./ sigma;
    end
    X = normalize_features(X);

    % Split into training and testing sets
    m = size(X, 1);
    rand_indices = randperm(m);
    train_size = floor(0.8 * m);

    X_train = X(rand_indices(1:train_size), :);
    y_train = y(rand_indices(1:train_size), :);
    X_test = X(rand_indices(train_size+1:end), :);
    y_test = y(rand_indices(train_size+1:end), :);

    % Define network parameters
    input_layer_size = size(X_train, 2); % Number of features
    num_labels = 2; % Binary classification (yes/no)
end

