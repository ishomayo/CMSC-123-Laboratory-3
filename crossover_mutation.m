function new_population = crossover_mutation(parents, mutation_rate, input_layer_size, hidden_layer_size, num_labels)
    new_population = parents; % Start with elites
    pop_size = length(parents);

    while length(new_population) < pop_size
        % Select two parents randomly
        p1 = parents{randi(length(parents))};
        p2 = parents{randi(length(parents))};

        % Single-point crossover
        split_point = randi(length(p1));
        child = [p1(1:split_point); p2(split_point+1:end)];

        % Mutation
        for j = 1:length(child)
            if rand < mutation_rate
                child(j) = child(j) + randn * 0.3; % Small perturbation
            end
        end

        % Add child to new population
        new_population{end + 1} = child;
    end
end

