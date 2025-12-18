import os
from pathlib import Path

import pandas as pd

# Sciezka do folderu z wynikami
# Dla struktury jak na obrazku uzyj: "wyniki/binary" (config -> funkcja -> run_* -> pliki)
extract_path = Path("wyniki/binary")

# Lista folderow funkcji (stala)
functions = ["f162014_10", "f162014_20", "f162014_30", "michalewicz2", "michalewicz5", "michalewicz10"]

# Wyniki zbiorcze
summary_data = []

# Przejscie po wszystkich konfiguracjach
for config_path in extract_path.iterdir():
    if not config_path.is_dir():
        continue

    result_row = {"config": config_path.name}

    # Dla kazdej funkcji
    for func in functions:
        func_path = config_path / func
        if not func_path.is_dir():
            continue

        best_values = []

        # Przejscie po plikach CSV w danej funkcji (rekurencyjnie, bo CSV leza np. w run_1/run_1.csv)
        for file_path in func_path.rglob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                if "best_fitness" in df.columns:
                    best_value = df["best_fitness"].min()
                    best_values.append(best_value)
            except Exception as e:
                print(f"Blad w pliku {file_path}: {e}")

        # Jesli znaleziono dane
        if best_values:
            result_row[f"{func}_min"] = min(best_values)
            result_row[f"{func}_max"] = max(best_values)
            result_row[f"{func}_avg"] = sum(best_values) / len(best_values)
        else:
            # Jesli brak danych, wstaw NaN
            result_row[f"{func}_min"] = None
            result_row[f"{func}_max"] = None
            result_row[f"{func}_avg"] = None

    summary_data.append(result_row)

# Konwersja do DataFrame
summary_df = pd.DataFrame(summary_data)

# Zapis do pliku Excel
output_path = f"podsumowanie_wynikow_{extract_path.name}.xlsx"
summary_df.to_excel(output_path, index=False)

print(f"Zapisano wyniki do pliku: {output_path}")
