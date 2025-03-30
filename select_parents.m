function selected_parents = select_parents(population, fitness, elite_size)
    [sorted_fitness, sorted_indices] = sort(fitness);
    selected_parents = population(sorted_indices(1:elite_size)); % Keep the elite

    % Tournament selection for the rest
    for i = (elite_size + 1):length(population)
        idx1 = randi(length(population));
        idx2 = randi(length(population));
        if fitness(idx1) < fitness(idx2)
            selected_parents{end + 1} = population{idx1};
        else
            selected_parents{end + 1} = population{idx2};
        end
    end
end

