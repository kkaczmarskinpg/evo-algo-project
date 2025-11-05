import pandas as pd
import os

# Sciezka do folderu z wynikami
extract_path = "wyniki_max"

# Lista folderow funkcji (stala)
functions = ["f16_2014_10", "f16_2014_20", "f16_2014_30", "michalewicz10", "michalewicz20", "michalewicz30"]

# Wyniki zbiorcze
summary_data = []

# Przejscie po wszystkich konfiguracjach
for config_name in os.listdir(extract_path):
    config_path = os.path.join(extract_path, config_name)
    if not os.path.isdir(config_path):
        continue

    result_row = {"config": config_name}

    # Dla kazdej funkcji
    for func in functions:
        func_path = os.path.join(config_path, func)
        if not os.path.isdir(func_path):
            continue

        best_values = []

        # Przejscie po plikach CSV w danej funkcji
        for file in os.listdir(func_path):
            if file.endswith(".csv"):
                file_path = os.path.join(func_path, file)
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
output_path = "podsumowanie_wynikow.xlsx"
summary_df.to_excel(output_path, index=False)

print(f"Zapisano wyniki do pliku: {output_path}")
