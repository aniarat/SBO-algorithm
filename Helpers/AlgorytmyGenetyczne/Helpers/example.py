import numpy as np

# funkcja celu
def cost_function(x):
    return x**2

# parametry
pop_size = 5
epochs = 10
alpha = 0.5
p_m = 0.1
psw = 0.1
lb, ub = -10, 10

# generowanie populacji
np.random.seed(42)
population = np.random.uniform(lb, ub, pop_size)
print("\nPOPULACJA:", population)
#obliczanie wartości funkcji celu
costs = cost_function(population)
print("FUNKCJA CELU:",costs)

# elitaryzm
elite_idx = np.argmin(costs)
elite = population[elite_idx]
print("NAJLEPSZY OSOBNIK:", elite)

#aktualizacja altanek
for epoch in range(epochs):
    print("\nPOPULACJA:", population)
    # obliczenie prawdopodobieństwa zgodnie z równaniem w artykule
    fit_list = 1 / (1 + costs)
    total_fit = np.sum(fit_list)
    probabilities = fit_list / total_fit;
    print("PRAWDOPOBIEŃSTWA DLA ALTANEK:", probabilities)

    new_population = []
    new_costs = []

    for idx in range(pop_size):
        # wybieranie altanki do modyfikacji używając metody koła ruletki

        selected_idx = np.random.choice(range(pop_size), p=probabilities)
        print("WYBRANA ALTANKA:", selected_idx)
        lamda = alpha/(1 + probabilities[idx])
        print("WARTOŚCI LAMBDA:", lamda)

        # aktualizacja altanki
        new_pos = population[idx] + lamda * (((population[selected_idx] + elite) / 2) - population[idx])
        print("NOWE ROZWIĄZANIE:", new_pos)

        # mutacja
        if np.random.rand() < p_m:
            new_pos += np.random.normal(0, psw * (ub - lb))

        # sprawdzenie czy nowe rozwiązanie jest w przedziale
        new_pos = np.clip(new_pos, lb, ub)

        # obliczenie funkcji kosztu po aktualizacji rozwiązania
        new_cost = cost_function(new_pos)

        new_population.append(new_pos)
        print("NOWA POPULACJA:", new_population)
        new_costs.append(new_cost)
        print("NOWE FUNKCJE CELU:", new_costs)

    # aktualizacja populacji i wartosci funkcji celu
    population = np.array(new_population)
    print("POPULACJA PO AKTUALIZACJI:", population)
    costs = np.array(new_costs)
    print("FUNKCJE CELU PO AKTUALIZACJI:", costs)

    # aktualizacja elity
    new_elite_idx = np.argmin(costs)
    if costs[new_elite_idx] < cost_function(elite):
        elite = population[new_elite_idx]
    print("NOWY NAJLEPSZY OSOBNIK:", elite)

    # nowe najlepsze rozwiązanie w epoce
    print(f'Epoka {epoch+1}/{epochs}, Najlepszy koszt: {cost_function(elite)}, Najlepsze rozwiązanie: {elite}')
# ostateczne najlepsze rozwiązanie
print(f'\nOstatecznie najlepsze rozwiązanie: {elite}, Koszt: {cost_function(elite)}')

