function new_population = crossover_mutation(parents, mutation_rate)
    pop_size = length(parents);
    new_population = cell(1, pop_size);

    % Keep the best individuals (elitism)
    new_population{1} = parents{1};

    for i = 2:pop_size
        % Select two distinct parents
        idx = randperm(pop_size, 2);
        p1 = parents{idx(1)};
        p2 = parents{idx(2)};

        % Single-point crossover
        split_point = randi(length(p1));
        child = [p1(1:split_point); p2(split_point+1:end)];

        % Mutation with adaptive scaling
        mutation_mask = rand(size(child)) < mutation_rate;
        mutation_values = randn(size(child)) .* abs(child) * 0.3;
        child(mutation_mask) = child(mutation_mask) + mutation_values(mutation_mask);

        % Store the child
        new_population{i} = child;
    end
end

