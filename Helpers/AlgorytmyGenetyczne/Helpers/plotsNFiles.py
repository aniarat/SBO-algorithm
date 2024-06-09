from matplotlib import pyplot as plt

def make_plot(values: list, file_name: str, title: str, func_enum : str, mutation_type: str, crossover_type: str, selection_type: str, num_genes: str, num_of_dimensions: str, num_generations: str, sol_per_pop: str, num_parents_mating: str,  y_scale: str = 'linear') -> None:
    plt.figure(figsize=(10, 8))
    plt.plot(values, label=title)
    plt.yscale(y_scale)
    plt.title(f'{title}, Funkcja: {func_enum}, \nMutacja: {mutation_type}, Krzyżowanie: {crossover_type}, Selekcja: {selection_type}, \nGA Parametry: num_genes={num_genes}, num_of_dimensions={num_of_dimensions}, num_generations={num_generations}, \nsol_per_pop={sol_per_pop}, num_parents_mating={num_parents_mating}', fontsize=10)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{file_name}.png')
    plt.show()
    # plt.figure(figsize=(12, 8))
    plt.close('all')


def save_to_file(values: list, spec: list, file_name: str) -> None:
    with open(f'./{file_name}.txt', 'w') as f:
        for i in range(len(values)):
            f.write(f'epoch {i+1}: f{spec[i]} = {values[i]}\n')
            i += 1
