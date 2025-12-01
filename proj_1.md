## 1. Cel Projektu

Celem projektu jest **implementacja klasycznego algorytmu genetycznego (AG)** zdolnego do rozwiązywania problemów optymalizacyjnych, tj. poszukiwania minimum (lub maksimum) funkcji wielu zmiennych [1, 2].

Implementacja musi być elastyczna i umożliwiać **konfigurację dowolnej liczby zmiennych** (np. 5, 10, 20, 27 zmiennych) [1].

> **Rekomendowany język programowania:** Python [1].
> **Wymagania zespołowe:** Projekt realizowany jest w grupach 2-3 osobowych [1].
> **Wymagania architektoniczne:** Aplikacja musi być zamodelowana z wykorzystaniem **obiektowego paradygmatu programowania (OOP)**, z uwzględnieniem dobrych praktyk wytwarzania oprogramowania [3, 4].

## 2. Struktura i Wymagania Implementacyjne

Implementacja algorytmu genetycznego musi zawierać następujące elementy, z możliwością pełnej konfiguracji z poziomu interfejsu graficznego (GUI) [5-7]:

### 2.1. Reprezentacja i Parametry Ogólne
1.  **Reprezentacja Chromosomu:** Implementacja **binarnej reprezentacji chromosomu** [5].
2.  **Dokładność:** Konfiguracja dokładności (precyzji) rozwiązania [5, 8].
3.  **Populacja i Epoki:** Konfiguracja **wielkości populacji** oraz **liczby epok** (iteracji) [5].

### 2.2. Metody Selekcji
Należy zaimplementować następujące metody selekcji, włącznie z konfiguracją ich parametrów [5]:
1.  Selekcja **najlepszych** [5].
2.  Selekcja **kołem ruletki** [5].
3.  Selekcja **turniejowa** [5].
4.  **Strategia Elitarna:** Implementacja i konfiguracja procentowej lub liczbowej części osobników przechodzących do kolejnej populacji [6, 9].

### 2.3. Operatory Genetyczne
Należy zaimplementować i umożliwić konfigurację prawdopodobieństwa ich wystąpienia dla [5, 6]:
1.  **Krzyżowanie:**
    *   Krzyżowanie **jednopunktowe** [5].
    *   Krzyżowanie **dwupunktowe** [5].
    *   Krzyżowanie **jednorodne (uniform)** [5].
    *   Krzyżowanie **ziarniste (discrete)** [5].
2.  **Mutacja:**
    *   Mutacja **brzegowa** [5].
    *   Mutacja **jednopunktowa** [6].
    *   Mutacja **dwupunktowa** [6].
3.  **Inwersja:** Implementacja operatora inwersji [6].

## 3. Funkcje Testowe

Należy wybrać i przetestować algorytm na dwóch funkcjach, które muszą umożliwiać ustawienie różnej liczby zmiennych (wymiarów) [10]:
1.  **Funkcja "Łatwiejsza":** Jedna funkcja z pliku `FunkcjeTestowe.pdf` (biblioteka Benchmark Functions) [6, 11].
2.  **Funkcja "Trudniejsza":** Jedna funkcja z pliku `Cec2014.pdf` lub dowolna inna funkcja z konferencji CEC (np. CEC 2014 obejmuje funkcje unimodalne, multimodalne, hybrydowe i kompozycyjne, co czyni je trudnymi do optymalizacji) [6, 12, 13].

> **Uwaga:** Funkcji **nie należy implementować samodzielnie** [10]. Należy skorzystać z gotowych implementacji, na przykład z bibliotek:
> *   `https://gitlab.com/luca.baronti/python_benchmark_functions` [10, 11].
> *   `https://github.com/thieu1995/opfunu/tree/master/opfunu/cec_based` (opfunu) [10, 14].

## 4. Wymagania Aplikacyjne (GUI)

Aplikacja musi być zrealizowana w formie graficznej (GUI) i oferować następujące funkcjonalności [7, 10]:
1.  **Konfiguracja:** Wszystkie parametry algorytmu muszą być konfigurowalne z poziomu GUI [7].
2.  **Wyświetlanie Czasu:** Wyświetlanie **czasu wykonywanych obliczeń** [7].
3.  **Zapis Danych:** Zapisywanie kolejnych wyników algorytmu w poszczególnych iteracjach do **pliku lub bazy danych** (np. SQLite) [7].
4.  **Generowanie Wykresów:** Możliwość wygenerowania w aplikacji następujących wykresów [7]:
    *   Wartości funkcji celu od kolejnej iteracji.
    *   Średniej wartości funkcji celu oraz odchylenia standardowego od kolejnej iteracji.

## 5. Eksperymenty i Raportowanie

### 5.1. Eksperymenty
Ponieważ algorytmy probabilistyczne (jakim jest AG) wprowadzają losowość, konieczne jest przeprowadzenie uśrednionych eksperymentów [15, 16]:
*   Każda testowana konfiguracja algorytmu musi być **powtórzona przynajmniej 10 razy** w celu uśrednienia wyników i zniwelowania efektu losowości [16, 17].

### 5.2. Sprawozdanie (Raport)
Sprawozdanie musi zawierać kompleksową analizę wyników [3, 18]:
1.  **Technologie i Wymagania Środowiskowe:** Informacje o wykorzystanych technologiach i wymaganiach środowiska do uruchomienia aplikacji [18].
2.  **Opis Funkcji Testowych:** Wybrane funkcje wraz z rysunkiem, wartościami optimum i argumentami, dla których optimum jest osiągnięte (dla testowanych wymiarów, np. 10, 20, 30 zmiennych) [17, 18].
3.  **Wykresy:** Wykresy zależności wartości funkcji celu od iteracji oraz średniej wartości i odchylenia standardowego – **tylko z najlepszego uruchomienia** w danej konfiguracji [3, 17].
4.  **Porównanie Wyników i Czasu:** Porównanie wyników osiągniętych przy różnych konfiguracjach algorytmu. Należy zamieścić **średnie wyniki z 10 uruchomień**, **najlepszy wynik**, **najgorszy wynik** oraz porównanie czasu obliczeń [17].
5.  **Tabela Podsumowująca:** Zawierająca nazwę optymalizowanej funkcji, liczbę zmiennych, rzeczywistą wartość optimum, uzyskaną wartość optimum oraz błąd (różnicę) [3].

### 5.3. Dostarczenie Projektu
1.  **Wideo:** Nagranie **krótkiego wideo (ok. 1–2 minuty)** demonstrującego działanie projektu w praktyce [4].
2.  **Archiwum ZIP:** Kod źródłowy, sprawozdanie i nagranie wideo należy wgrać w postaci archiwum ZIP o nazwie `P1_Nazwisko1Imie1_Nazwisko2Imie2_Nazwisko3Imie3.zip` [4].
