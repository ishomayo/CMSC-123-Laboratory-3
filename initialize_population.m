function population = initialize_population(pop_size, input_layer_size, hidden_layer_size, num_labels)
    epsilon_init = 0.12;
    population = cell(pop_size, 1);
    for i = 1:pop_size
        Theta1 = rand(hidden_layer_size, input_layer_size + 1) * 2 * epsilon_init - epsilon_init;
        Theta2 = rand(num_labels, hidden_layer_size + 1) * 2 * epsilon_init - epsilon_init;
        population{i} = [Theta1(:); Theta2(:)];
    end
end

