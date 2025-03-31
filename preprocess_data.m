function [X_train, y_train, X_test, y_test, input_layer_size, num_labels] = preprocess_data()
    % Open file
    fid = fopen('bank.csv', 'r');
    if fid == -1
        error('Error: Could not open file.');
    end

    % Read header line
    header_line = fgetl(fid);
    headers = strsplit(header_line, ';');
    disp("Column Names:");
    disp(headers);

    % Read data (assume all columns are strings, we will convert later)
    data = textscan(fid, repmat('%s', 1, length(headers)), 'Delimiter', ';', 'HeaderLines', 0);
    fclose(fid);

    % Convert target variable (y) ["yes" -> 2, "no" -> 1]
    labels = strtrim(erase(data{end}, '"')); % Remove spaces & quotes
    y = double(strcmpi(labels, 'yes')) + 1; % "yes" -> 2, "no" -> 1

    fprintf('Number of "yes" (2): %d\n', sum(y == 2));
    fprintf('Number of "no" (1): %d\n', sum(y == 1));

    % Function to encode categorical variables
    function encoded = encode_categories(data_column)
        [~, ~, encoded] = unique(data_column);
    end

    % Encode categorical columns
    job = encode_categories(data{2});
    marital = encode_categories(data{3});
    education = encode_categories(data{4});
    default = encode_categories(data{5});
    housing = encode_categories(data{7});
    loan = encode_categories(data{8});
    contact = encode_categories(data{9});
    month = encode_categories(data{11});
    poutcome = encode_categories(data{16});

    % Convert numeric columns from string to double
    function num_data = convert_numeric(col)
        num_data = str2double(data{col});
        num_data(isnan(num_data)) = 0; % Replace NaNs with 0
    end

    % Construct feature matrix
    X = [convert_numeric(1), job, marital, education, default, ...
         convert_numeric(6), housing, loan, contact, ...
         convert_numeric(10), month, convert_numeric(12), ...
         convert_numeric(13), convert_numeric(14), convert_numeric(15), ...
         poutcome];

    % Normalize numerical features
    function X_norm = normalize_features(X)
        mu = mean(X);
        sigma = std(X);
        sigma(sigma == 0) = 1; % Avoid division by zero
        X_norm = (X - mu) ./ sigma;
    end
    X = normalize_features(X);

    % Split into training and testing sets (90% train, 10% test)
    m = size(X, 1);
    rand_indices = randperm(m);
    train_size = floor(0.9 * m);

    X_train = X(rand_indices(1:train_size), :);
    y_train = y(rand_indices(1:train_size), :);
    X_test = X(rand_indices(train_size+1:end), :);
    y_test = y(rand_indices(train_size+1:end), :);

    % Define network parameters
    input_layer_size = size(X_train, 2); % Number of features
    num_labels = 2; % Binary classification (yes/no)
end

