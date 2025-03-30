function [best_Theta1, best_Theta2] = ga_nn()
    % Load preprocessed data
    [X_train, y_train, ~, ~, input_layer_size, num_labels] = preprocess_data();

    % Set neural network parameters
    hidden_layer_size = 25; % Number of hidden neurons
    lambda = 1; % Regularization parameter

    % GA Parameters
    pop_size = 30;  % Population size
    generations = 10; % Number of generations
    mutation_rate = 0.1; % Mutation probability
    elite_size = 5; % Number of best individuals to keep

    % Initialize population
    population = initialize_population(pop_size, input_layer_size, hidden_layer_size, num_labels);

    % Evolve through generations
        for gen = 1:generations
        fprintf('Generation %d/%d\n', gen, generations);

        % Evaluate fitness
        fitness = evaluate_fitness(population, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);

        % Print fitness values
        fprintf('Fitness values:\n');
        disp(fitness); % Display all fitness values for the current generation

        % Print best fitness in this generation
        fprintf('Best Fitness: %f\n', min(fitness));

        % Selection
        selected_parents = select_parents(population, fitness, elite_size);

        % Crossover & Mutation
        population = crossover_mutation(selected_parents, mutation_rate, input_layer_size, hidden_layer_size, num_labels);
    end


    % Get best individual from final population
    final_fitness = evaluate_fitness(population, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);
    [~, best_idx] = min(final_fitness);
    best_nn_params = population{best_idx};

    % Reshape best weights
    best_Theta1 = reshape(best_nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                          hidden_layer_size, (input_layer_size + 1));

    best_Theta2 = reshape(best_nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                          num_labels, (hidden_layer_size + 1));
end

