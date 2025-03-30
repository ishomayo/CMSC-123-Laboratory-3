function fitness = evaluate_fitness(population, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
    pop_size = length(population);
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        fitness(i) = nnCostFunction(population{i}, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
    end
end

