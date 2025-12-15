# Projekt nr 3

## Wykorzystanie biblioteki PyGAD do algorytmów genetycznych w języku Python

### Optymalizacja funkcji z wykorzystaniem biblioteki PyGAD

---

## 1. Wprowadzenie

Wykorzystamy szkielet projektu z przykładu z wykładu `example_02.py`.

---

## 2. Reprezentacja binarna

Musimy ustalić zakres naszych parametrów w przedziale **[0, 1]**, konfigurując odpowiednio:

* `init_range_low = 0`
* `init_range_high = 2`
* `gene_type = int`

Umożliwi nam to wygenerowanie osobnika, który będzie się składał z wartości **0 i 1**.

Musimy pamiętać w kodzie, że jeśli nasz osobnik ma np. **60 bitów**, to:

* pierwsze 20 bitów → pierwsza zmienna,
* drugie 20 bitów → druga zmienna,
* trzecie 20 bitów → trzecia zmienna.

Liczba bitów, a co za tym idzie liczba zmiennych optymalizowanej funkcji, powinna być możliwa do konfiguracji.

Liczbę bitów konfigurujemy w polu:

* `num_genes`

---

## 3. Funkcja celu

Przygotuj funkcję celu, którą będziesz optymalizować:

```python
def fitnessFunction(individual):
    # tutaj rozkoduj binarnego osobnika!
    # Napisz funkcję decodeInd
    ind = decodeInd(individual)
    result = (ind[0] + 2 * ind[1] - 7) ** 2 + (2 * ind[0] + ind[1] - 5) ** 2
    return result,
```

Niech to będą funkcje realizowane w ramach **projektu nr 1 i 2**.

---

## 4. Metody selekcji (PyGAD)

Przetestuj następujące metody selekcji:

* selekcja turniejowa (`tournament`),
* koło ruletki (`rws`),
* selekcja losowa (`random`).

---

## 5. Algorytmy krzyżowania

Wybierz i przetestuj algorytmy krzyżowania:

* jednopunktowe (`single_point`),
* dwupunktowe (`two_points`),
* jednorodne (`uniform`).

---

## 6. Algorytmy mutacji

Przetestuj dwa algorytmy mutacji:

* losową (`random`),
* zamiana indeksów (`swap`).

---

## 7. Konfiguracja eksperymentów

Wykonaj eksperymenty, odpowiednio konfigurując główną klasę PyGAD (na bazie `example_02.py`).

Przykładowa konfiguracja:

```python
ga_instance = pygad.GA(
    num_generations=num_generations,
    sol_per_pop=sol_per_pop,
    num_parents_mating=num_parents_mating,
    num_genes=num_genes,
    fitness_func=fitness_func,
    init_range_low=0,
    init_range_high=2,
    gene_type=int,
    mutation_num_genes=mutation_num_genes,
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    keep_elitism=1,
    K_tournament=3,
    random_mutation_max_val=32.768,
    random_mutation_min_val=-32.768,
    logger=logger,
    on_generation=on_generation,
    parallel_processing=['thread', 4]
)
```

---

## 8. Reprezentacja rzeczywista (wariant alternatywny)

Zmieniamy:

* `init_range_low` → dolna granica przedziału poszukiwań,
* `init_range_high` → górna granica przedziału poszukiwań,
* `gene_type = float`.

Biblioteka **PyGAD** nie posiada wbudowanych algorytmów krzyżowania dla reprezentacji rzeczywistej.

Należy wykorzystać algorytmy zaimplementowane w **projekcie nr 2** i dostosować je do PyGAD.

### Implementacja krzyżowania w PyGAD (fragment)

```python
def single_point_crossover(self, parents, offspring_size):
    """
    Applies the single-point crossover.
    """
    if self.gene_type_single:
        offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
    else:
        offspring = numpy.empty(offspring_size, dtype=object)

    for k in range(offspring_size[0]):
        crossover_point = numpy.random.randint(low=0, high=parents.shape[1], size=1)[0]

        if not (self.crossover_probability is None):
            probs = numpy.random.random(size=parents.shape[0])
            indices = numpy.where(probs <= self.crossover_probability)[0]

            if len(indices) == 0:
                offspring[k, :] = parents[k % parents.shape[0], :]
                continue
            elif len(indices) == 1:
                parent1_idx = indices[0]
                parent2_idx = parent1_idx
            else:
                indices = random.sample(list(set(indices)), 2)
                parent1_idx, parent2_idx = indices
        else:
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]

        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring
```

### Przykładowa własna implementacja krzyżowania

```python
def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        random_split_point = numpy.random.choice(range(offspring_size[1]))
        parent1[random_split_point:] = parent2[random_split_point:]
        offspring.append(parent1)
        idx += 1
    return numpy.array(offspring)
```

Funkcja krzyżująca zwraca gotową populację po procesie krzyżowania.

---

## 9. Mutacja Gaussa

Zaimplementuj dodatkowo mutację Gaussa:

```python
def mutation_func(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = numpy.random.choice(range(offspring.shape[1]))
        offspring[chromosome_idx, random_gene_idx] += numpy.random.random()
    return offspring
```

---

## 10. Zadania do wykonania

1. Dokonaj optymalizacji funkcji z **projektu nr 1 oraz 2**.

   Wykorzystaj gotowe biblioteki zawierające implementacje funkcji testowych:

   ```python
   import benchmark_functions as bf
   from opfunu import cec_based

   # benchmark_functions
   # pip install benchmark_functions
   func = bf.Hyperellipsoid(n_dimensions=10)
   print(func.suggested_bounds())
   print(func.minimum())

   # cec
   # pip install opfunu
   func = cec_based.cec2014.F32014(ndim=10)
   print(func.bounds)
   print(func.x_global)
   print(func.f_global)
   ```

2. Wykorzystaj reprezentację **binarną** oraz **rzeczywistą**.

3. Program wykonaj jako **aplikację konsolową** (menu nie jest wymagane).

4. Przygotuj **sprawozdanie** podobne do sprawozdań z projektów nr 1 oraz nr 2.
