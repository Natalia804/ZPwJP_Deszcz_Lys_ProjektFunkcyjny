import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import shap


# =============== FUNKCJE POMOCNICZE =============== #

def wczytaj_dane(path: str) -> pd.DataFrame:
    """
    Wczytuje dane z pliku CSV i usuwa kolumnę 'CDR' (jeśli istnieje).

    Parameters
    ----------
    path : str
        Ścieżka do pliku CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame zawierający wczytane dane (bez kolumny 'CDR').
    """
    data = pd.read_csv(path)
    if 'CDR' in data.columns:
        data = data.drop(columns=['CDR'])
    return data


def przygotuj_dane_kategoryczne(data: pd.DataFrame) -> pd.DataFrame:
    """
    Dokonuje mapowania kolumn kategorycznych:
    - 'M/F' -> is_male (0/1),
    - 'Group' -> is_demented, is_converted (0/1).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame z wczytanymi danymi (z kolumnami 'M/F', 'Group').

    Returns
    -------
    pd.DataFrame
        DataFrame z dodanymi kolumnami is_male, is_demented, is_converted.
    """
    if 'M/F' in data.columns:
        data['is_male'] = data['M/F'].map({'M': True, 'F': False}).astype(int)

    if 'Group' in data.columns:
        data[['is_demented', 'is_converted']] = data['Group'].apply(
            lambda x: pd.Series({
                'Demented': (1, 0),
                'Nondemented': (0, 0),
                'Converted': (0, 1)
            }.get(x, (None, None)))
        )
        # Jeśli pojawiłyby się NaN, to konwertujemy je do 0
        data['is_demented'] = data['is_demented'].fillna(0).astype(int)
        data['is_converted'] = data['is_converted'].fillna(0).astype(int)

    return data


def wykres_macierzy_konfuzji(y_true: pd.Series, 
                             y_pred: pd.Series, 
                             ax=None, 
                             labels=["Not Demented", "Demented"]) -> None:
    """
    Rysuje macierz konfuzji przy pomocy Seaborn i matplotlib.

    Parameters
    ----------
    y_true : pd.Series
        Prawdziwe etykiety (0/1).
    y_pred : pd.Series
        Przewidywane etykiety (0/1).
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Obiekt osi, na którym ma być narysowana macierz (domyślnie None).
    labels : list of str
        Etykiety osi X/Y w macierzy (np. ["Not Demented", "Demented"]).
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    if ax:
        ax.set_xlabel("Przewidywania")
        ax.set_ylabel("Rzeczywistość")
    else:
        plt.xlabel("Przewidywania")
        plt.ylabel("Rzeczywistość")


def detekcja_outlier_zscore(data: pd.DataFrame, 
                            column: str, 
                            threshold: float = 3.0) -> pd.DataFrame:
    """
    Zwraca wiersze, które są outlierami w danej kolumnie na podstawie Z-score.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame ze standaryzowanymi danymi.
    column : str
        Nazwa kolumny, w której wykrywamy wartości odstające.
    threshold : float, optional
        Próg Z-score powyżej którego uznajemy wartości za odstające (domyślnie 3).

    Returns
    -------
    pd.DataFrame
        Wiersze DataFrame, w których wartości w danej kolumnie przekraczają threshold.
    """
    if data[column].notnull().sum() > 0:
        z_scores = zscore(data[column].dropna())
        outliers_idx = np.where(np.abs(z_scores) > threshold)[0]
        return data.iloc[outliers_idx]
    return pd.DataFrame()


# =============== FUNKCJE-SEKCJE (OBSŁUGA STREAMLIT) =============== #

def wprowadzenie_section() -> None:
  
    st.title("Projekt funkcyjny: Analiza choroby Alzheimera za pomocą metod uczenia maszynowego")
    st.write("###### `Przemiot`: Zaawansowane programowanie w języku python")
    st.write("###### `Prowadząca`: dr hab. Iwona Skalna")
    st.write("###### `Autorki`: Natalia Łyś, Zuzanna Deszcz")


    # Tytuł aplikacji
    st.header("Analiza Zbioru Danych: *Alzheimer Feature*")
    st.write("#### Wprowadzenie")
    st.markdown("""
    <p>
    <strong> Choroba Alzheimera (AD)  </strong> to najbardziej powszechna odmiana demencji. 
    W Europie choroba ta jest głównym skutkiem utraty samodzielności i upośledzenia osób starszych. 
    Szacuje się ilość chorych na <strong> 10 milionów ludzi </strong>.
    Zbiór danych zawiera informacje na temat poniżej szerzej objaśnionymi medycznymi
    warunkami, ale również socjoekonomicznymi i diagnozą demencji u pacjenta. Zbiór, z którego korzystamy pochodzi ze stronny 
    <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle</a>.
    </p>
    """, unsafe_allow_html=True)


def charakterystyka_danych_section(data: pd.DataFrame) -> None:
    """
    Wyświetla sekcję charakterystyki danych:
    - Statystyki opisowe
    - Rozkład zmiennej docelowej Group
    - Przykładowy podgląd danych
    - Wykresy korelacji
    - Boxploty wybranych zmiennych
    """
    st.title("Charakterystyka Zbioru Danych")
    st.markdown("""
    Zbiór danych zawiera informacje medyczne, socjo-ekonomiczne oraz diagnozę demencji u pacjentów. 
    Dane te zostały zaczerpnięte z <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle</a>, bazując na badaniach wykorzystujących uczenie maszynowe do analizy demencji.
    """, unsafe_allow_html=True)

    st.subheader("Zmienna objaśniana (Group)")
    st.markdown("""
    <span style="color: #1f77b4; font-weight: bold;">Group</span>: Diagnoza choroby:
    <ul>
      <li><code style="color: #2ca02c;">Demented</code>: Osoby zdiagnozowane z demencją.</li>
      <li><code style="color: #ff7f0e;">Nondemented</code>: Osoby bez demencji.</li>
      <li><code style="color: #d62728;">Converted</code>: Osoby, które po pewnym czasie zostały zaklasyfikowane jako zdrowe.</li>
    </ul>
    """, unsafe_allow_html=True)
    
    
    st.subheader("Zmienne objaśniające")
    st.markdown("""
    1. <span style="color: #1f77b4; font-weight: bold;">is_male</span>: Zmienna określająca płeć badanego.
    2. <span style="color: #1f77b4; font-weight: bold;">Age</span>: Wiek osoby.
    3. <span style="color: #1f77b4; font-weight: bold;">EDUC (Years of Education)</span>: Liczba lat edukacji.
    4. <span style="color: #1f77b4; font-weight: bold;">SES (Socioeconomic Status)</span>: Status społeczno-ekonomiczny (1-5, gdzie 5 to najwyższy status).
    5. <span style="color: #1f77b4; font-weight: bold;">MMSE (Mini Mental State Examination)</span>: Skala oceniająca funkcje poznawcze (pamięć, uwaga, orientacja przestrzenna).
    6. <span style="color: #1f77b4; font-weight: bold;">CDR (Clinical Dementia Rating)</span>: Skala oceny zaawansowania demencji.
    7. <span style="color: #1f77b4; font-weight: bold;">eTIV (Estimated Total Intracranial Volume)</span>: Szacowana całkowita objętość wewnątrzczaszkowa.
    8. <span style="color: #1f77b4; font-weight: bold;">nWBV (Normalized Whole Brain Volume)</span>: Znormalizowana objętość całego mózgu.
    9. <span style="color: #1f77b4; font-weight: bold;">ASF (Atlas Scaling Factor)</span>: Czynnik skalowania atlasu używany do dopasowania obrazu mózgu do wzorca.
    """, unsafe_allow_html=True)
    
    # Źródła danych
    st.subheader("Źródła Danych")
    st.markdown("""
    1. <a href="https://cordis.europa.eu/article/id/428863-mind-reading-software-finds-hidden-signs-of-dementia/pl" target="_blank" style="color: #007BFF; font-weight: bold;">Cordis.europa.eu</a>  
    2. <a href="https://www.sciencedirect.com/science/article/pii/S2352914819300917?via%3Dihub" target="_blank" style="color: #007BFF; font-weight: bold;">ScienceDirect - *Machine learning in medicine: Performance calculation of dementia prediction by support vector machines (SVM)*</a>  
    3. <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle Dataset</a>
    """, unsafe_allow_html=True)


    st.subheader("Podgląd pierwszych wierszy zbioru danych:")
    st.dataframe(data.head())

    st.subheader("Podstawowe statystyki:")
    st.write(data.describe())

    # Brakujące dane
    st.subheader("Brakujące dane:")
    missing_values = data.isnull().sum()
    missing_percent = (missing_values / len(data)) * 100
    st.write("Braki danych w poszczególnych kolumnach (w procentach):")
    st.bar_chart(missing_percent)
    st.write("Braki widzimy na niskim poziomie.")

    # Rozkład zmiennej Group
    st.header("Dytrybucja zmiennej objaśniającej Group")
    group_distribution = data['Group'].value_counts(normalize=True)
    st.bar_chart(group_distribution)
    st.write(group_distribution)
    st.write("Będziemy w analizie głównie korzystać z Demented i Nondemented, a różnica proporcji między nimi nie jest duża.")

    # Relacja między płcią a demencją
    st.subheader("Relacja między płcią a demencją")
    if 'is_male' in data.columns and 'is_demented' in data.columns:
        matrix_demented = [
            [len(data[(data['is_male'] == 1) & (data['is_demented'] == 1)]),
             len(data[(data['is_male'] == 1) & (data['is_demented'] == 0)])],
            [len(data[(data['is_male'] == 0) & (data['is_demented'] == 1)]),
             len(data[(data['is_male'] == 0) & (data['is_demented'] == 0)])]
        ]
        matrix_demented_df = pd.DataFrame(
            matrix_demented,
            index=['Male', 'Female'],
            columns=['Demented', 'Not Demented']
        )
        st.write("Tabela relacji płci a demencja:")
        st.dataframe(matrix_demented_df)

        fig, ax = plt.subplots()
        matrix_demented_df.plot(kind='bar', ax=ax, color=['indigo', 'gold'])
        ax.set_ylabel("Liczba obserwacji")
        ax.set_title("Relacja: Płeć a Demencja")
        st.pyplot(fig)
        
    st.markdown("""
    Liczba pacjentów z demencją w danych jest podobna. Tutaj proporcja jest zupełnie inna niż we
    wcześniejszym wykresie. Dane okazują się próbką niereprezentatywną, ponieważ aż **2/3
    populacji dotkniętej chorobą Alzheimera to kobiety** (1), co nie odzwierciedla nasz zestaw danych.
    Natomiast więcej jest obserwacji pacjentek żeńskich (**58%** to kobiety). Co ciekawe, wiąże się to
    między innymi z faktem, że objawy **AD rozwijają się z wiekiem**, natomiast mężczyźni zwykle
    żyją mniej niż kobiety (2).\n
    (1) [Castro-Aldrete L., *Sex and gender considerations in Alzheimer’s disease: The Women’s Brain Project contribution*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10097993/) \n
    (2) [Sangha K., *Gender differences in risk factors for transition from mild cognitive impairment to Alzheimer’s disease: A CREDOS study*](https://doi.org/10.1016/J.COMPPSYCH.2015.07.002)
    """)

    # Macierz korelacji
    st.subheader("Macierz korelacji (zmienne numeryczne)")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if not numeric_data.empty:
        corr_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
                    mask=mask, linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Brak zmiennych numerycznych w zbiorze danych.")

    
    st.markdown("""
    Z mapy ciepła możemy wywnioskować, że interesująca nas korelacja zachodzi pomiędzy:
    - **nWBV – Age**
    - **nWBV – MMSE**
    - **nWBV – eTIV**
    - **ASF – eTIV**

    Zmienna **nWBV** jest stworzona ze zmiennej **eTIV**, także ich korelacja nie dziwi. Z literatury
    również wynika, że korelacja pomiędzy **Age** i **nWBV** wynika z procesu obumierania tkanki w
    mózgu wraz z wiekiem. Bardzo wysoka korelacja **ASF** i **eTIV** wynika z faktu, że **ASF** jest
    indeksem utworzonym z wartości **eTIV**.
    
    Jeśli natomiast weźmiemy pod lupę korelacje zmiennych objasniających z objaśnianymi, zauważamy 
    tu znacznie wyższą bezwzględną korelację zmiennej is_demented ze zmiennymi
    MMSE, nWBV, is_male oraz SES, a także, choć znacznie niższe, korelacje zmiennych Age i MMSE z
    klasyfikatorem is_converted. Na tej podstawie można stwierdzić zależność stwierdzonych
    zachorowań od poziomu umiejętności poznawczych, znormalizowanej objętości mózgu, płci oraz statusu ekonomicznego,
    natomiast wiek oraz umiejętności poznawcze opiszemy jako istotnie skorelowane ze
    zmienną is_converted.
    """)
    

    # Boxploty dla wybranych kolumn
    st.subheader("Dystrybucja zmiennych numerycznych względem grup diagnozy")
    st.write("Interaktywny wykres pozwala na eksploracje zmiennych.")
    columns_to_plot = ['Age', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    available_columns = [col for col in columns_to_plot if col in data.columns]

    if available_columns:
        selected_column = st.selectbox("Wybierz kolumnę do wizualizacji:", available_columns)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=data, x='Group', y=selected_column, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Brak dostępnych zmiennych do wizualizacji.")

         
    interpretacja = """
    Grupa Converted ma większy zakres wieku, i osoby z tej grupy diagnostycznej wydają się być starsze, ale generlanie nie widać drastycznych różnic między gruopami. Wyniki MMSE w grupie Demented są znacząco niższe, z większą zmiennością i wartościami odstającymi, podczas gdy grupy Nondemented i Converted mają zbliżone wyniki. Grupa Nondemented charakteryzuje się najwyższym medianowym eTIV, a grupa Converted najniższym, z wartościami odstającymi w grupie Demented. Najniższy mediana nWBV obserwowana jest w grupie Demented, co wskazuje na większe zmniejszenie objętości mózgu, podczas gdy grupy Nondemented i Converted są bardziej zbliżone. Mediana ASF pozostaje podobna między grupami, ale grupa Nondemented wykazuje większy rozkład i wartości odstające, z najmniejszym zakresem w grupie Converted.
    """

    st.header("Interpretacja Wykresów")
    st.write(interpretacja)
    
    # Zapisanie przetworzonych danych do session_state
    selected_columns = ['nWBV', 'MMSE', 'eTIV', 'SES', 'is_demented', 'is_male']
    

    available_columns = [col for col in selected_columns if col in data.columns]
    if available_columns:
        st.session_state.data_selected = data[available_columns]
    else:
        st.error("Brak wymaganych kolumn do zapisania przetworzonych danych.")

def braki_outliery_section() -> None:
    """
    Wyświetla sekcję dotyczącą:
    - Usuwania/wypełniania braków danych,
    - Analizy wartości odstających,
    - Zapisu przetworzonych danych do st.session_state.
    """
    st.title("Usuwanie braków danych i analiza wartości odstających")

    # Sprawdzenie, czy przetworzone dane są dostępne
    if "data_selected" in st.session_state:
        data_selected = st.session_state.data_selected
        #st.write("Dane zostały wczytane z `session_state`.")
    else:
        st.error("Dane nie zostały przetworzone w poprzednich sekcjach.")
        st.stop()

    data_numeric = data_selected.select_dtypes(include=['float64', 'int64'])
    st.subheader("Podgląd danych przed przetwarzaniem")
    st.dataframe(data_numeric.head())

    # Analiza braków
    st.subheader("Analiza braków danych")
    missing_numeric = data_numeric.isnull().mean() * 100
    st.bar_chart(missing_numeric)

    # Obsługa braków danych
    st.subheader("Obsługa braków danych")
    method_numeric = st.radio(
        "Interaktywna obsługa danych pozwala na zobaczenie najlepszej możliwości",
        ["Usuwanie wierszy", "Wypełnianie średnią", "Wypełnianie medianą", "Wypełnianie modą"]
    )
    

    if method_numeric == "Usuwanie wierszy":
        data_numeric_cleaned = data_numeric.dropna()
    elif method_numeric == "Wypełnianie średnią":
        data_numeric_cleaned = data_numeric.fillna(data_numeric.mean())
    elif method_numeric == "Wypełnianie medianą":
        data_numeric_cleaned = data_numeric.fillna(data_numeric.median())
    else:
        data_numeric_cleaned = data_numeric.fillna(data_numeric.mode().iloc[0])
    
    st.write("Dane numeryczne po czyszczeniu:")
    st.dataframe(data_numeric_cleaned.head())

    
    st.markdown("""
    Najmiej na średnią i odchylenie standardowe na zmienne miało wpływ uzupełnianie braków modą, 
    dlatego postawiono na to rozwiązanie. warto jedna zauważyć, że wybór metody w tym przypadku nie miał dużego wpływu na wyniki.
    """)
    data_numeric_cleaned = data_numeric.fillna(data_numeric.mode().iloc[0])
    data_cleaned = data_numeric_cleaned

    # Debug: Podgląd scalonych danych
    st.write("Dane po czyszczeniu:")
    st.dataframe(data_cleaned.head())


    # Zapis do session_state
    st.session_state.data_cleaned = data_cleaned

    # Analiza outlierów
    st.subheader("Analiza wartości odstających (Z-score)")
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_numeric_cleaned)
    data_standardized = pd.DataFrame(data_standardized, columns=data_numeric_cleaned.columns)

    outliers_summary = {
        "Kolumna": [],
        "Liczba wartości odstających (|z|>3)": []
    }

    for col in data_standardized.columns:
        outliers = detekcja_outlier_zscore(data_standardized, col, threshold=3)
        outliers_summary["Kolumna"].append(col)
        outliers_summary["Liczba wartości odstających (|z|>3)"].append(len(outliers))

    outliers_summary_df = pd.DataFrame(outliers_summary)
    st.write("Podsumowanie liczby wartości odstających w każdej kolumnie:")
    st.dataframe(outliers_summary_df)

    st.write("""
    Ze względu na naturę medyczną problemu, zdecydowano się nie usuwać wartości odstających,
    aby nie tracić potencjalnie ważnych obserwacji.
    """)


def dzielenie_section() -> None:
    """
    Wyświetla sekcję dotyczącą:
    - Podziału danych na zbiór uczący i testowy,
    - Ewentualnej standaryzacji (poza kolumnami binarnymi),
    - Zapisu wynikowych zbiorów do st.session_state.
    """
    st.title("Dzielenie na zbiór uczący i testowy")

    if "data_cleaned" not in st.session_state:
        st.error("Brak oczyszczonych danych. Upewnij się, że poprzednie sekcje zostały wykonane.")
        st.stop()

    data_cleaned = st.session_state["data_cleaned"]

    # Sprawdzamy, czy mamy kolumnę docelową
    if "is_demented" not in data_cleaned.columns:
        st.error("Kolumna docelowa 'is_demented' nie istnieje w zbiorze danych.")
        st.stop()

    X = data_cleaned.drop(columns=["is_demented"])
    y = data_cleaned["is_demented"]

    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    # Jeśli jest kolumna binarna is_male, wyłącz ją z standaryzacji
    if "is_male" in numeric_cols:
        numeric_cols = numeric_cols.drop("is_male")

    if not numeric_cols.empty:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.write("Podgląd X_train po standaryzacji:")
    st.dataframe(X_train.head())
    st.write("Podgląd y_train:")
    st.write(y_train.head())
    st.write(f"Liczba próbek w zbiorze uczącym: {len(X_train)}")
    st.write(f"Liczba próbek w zbiorze testowym: {len(X_test)}")

    # Zapisujemy do session_state
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test


def metody_uczenia_section() -> None:
    """
    Wyświetla sekcję dotyczącą trenowania różnych metod uczenia maszynowego (Decision Tree, SVM, Random Forest):
    - Wykonuje GridSearchCV
    - Pokazuje najlepsze parametry
    - Rysuje macierze konfuzji
    - Wyświetla i porównuje metryki (accuracy, precision, recall, F1)
    - Opcjonalnie analizuje SHAP
    """
    st.title("Metody uczenia maszynowego w identyfikacji demencji")

    st.write("Do optymalizacji hiperparametrów modeli użyto metody Grid search. Celem jest znalezienie najlepszej kombinacji wartości hiperparametrów, które maksymalizują wydajność modelu na zadanym zbiorze danych.")
    
    required_keys = ["X_train", "X_test", "y_train", "y_test"]
    if not all(k in st.session_state for k in required_keys):
        st.error("Brak danych do uczenia maszynowego. Upewnij się, że poprzednie sekcje zostały wykonane.")
        st.stop()

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    # ========== Drzewa Decyzyjne ========== #
    st.header("Metoda 1: Drzewa Decyzyjne")
    
    st.markdown("""
    Drzewa decyzyjne to intuicyjna metoda uczenia maszynowego, która pozwala na modelowanie decyzji w sposób hierarchiczny.
    Analizujemy m.in.:
    - `max_depth`: Maksymalna głębokość drzewa decyzyjnego wpływa na złożoność modelu.
    - `criterion`: Miara podziału danych (`gini` lub `entropy`). Wybór zależy od charakterystyki danych.
    """)
    
    param_grid_tree = {
        'max_depth': [3, 5, 7, 10],
        'criterion': ['gini', 'entropy']
    }
    
    # Tworzenie modelu
    tree_model = DecisionTreeClassifier(random_state=42)
    grid_search_tree = GridSearchCV(tree_model, param_grid_tree, cv=5, scoring='f1', n_jobs=-1)
    grid_search_tree.fit(X_train, y_train)

    # Najlepszy model 
    best_tree_model = grid_search_tree.best_estimator_
    y_pred_tree = best_tree_model.predict(X_test)
    
    

    st.write("Najlepsze parametry Drzewa Decyzyjnego:", grid_search_tree.best_params_)

    st.subheader("Macierz konfuzji - Drzewo Decyzyjne")
    fig, ax = plt.subplots()
    wykres_macierzy_konfuzji(y_test, y_pred_tree, ax=ax)
    st.pyplot(fig)

    tree_accuracy = accuracy_score(y_test, y_pred_tree)
    tree_precision = precision_score(y_test, y_pred_tree)
    tree_recall = recall_score(y_test, y_pred_tree)
    tree_f1 = f1_score(y_test, y_pred_tree)
    
     # Wyświetlanie wyników
    metrics_data = {
        "Metryka": ["Dokładność", "Precyzja", "Czułość", "F1-score"],
        "Wartość": [tree_accuracy, tree_precision, tree_recall, tree_f1]
    }
    metrics_df = pd.DataFrame(metrics_data)

    st.table(metrics_df)

    # ========== SVM ========== #
    st.header("Metoda 2: Support Vector Machines (SVM)")
    st.markdown("""
    Support Vector Machines (SVM) znajduje optymalną hiperpłaszczyznę do separacji klas. 
    W celu znalezienia najlepszych parametrów zastosowano Grid Search. Parametry:
    - `C`: Regularizacja (kontrola nadmiarowego dopasowania).
    - `kernel`: Typ jądra (np. 'linear', 'rbf').
    """)
    
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    svm_model = SVC(random_state=42, probability=True)
    grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='f1', n_jobs=-1)
    grid_search_svm.fit(X_train, y_train)

    best_svm_model = grid_search_svm.best_estimator_
    y_pred_svm = best_svm_model.predict(X_test)

    st.write("Najlepsze parametry SVM:", grid_search_svm.best_params_)

    st.subheader("Macierz konfuzji - SVM")
    fig, ax = plt.subplots()
    wykres_macierzy_konfuzji(y_test, y_pred_svm, ax=ax)
    st.pyplot(fig)

    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_precision = precision_score(y_test, y_pred_svm)
    svm_recall = recall_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm)
    
    # Wyświetlanie wyników
    metrics_data_svm = {
        "Metryka": ["Dokładność", "Precyzja", "Czułość", "F1-score"],
        "Wartość": [svm_accuracy, svm_precision, svm_recall, svm_f1]
    }
    metrics_df_svm = pd.DataFrame(metrics_data_svm)

    st.table(metrics_df_svm)

    # ========== Random Forest ========== #
    st.header("Metoda 3: Random Forest")
    st.markdown("""
    Random Forest to zespół drzew decyzyjnych, które tworzą silny model predykcyjny. 
    W celu znalezienia najlepszych parametrów zastosowano Grid Search:
    - `n_estimators`: Liczba drzew w lesie.
    - `max_depth`: Maksymalna głębokość drzew.
    """)
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20]
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    best_rf_model = grid_search_rf.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)

    st.write("Najlepsze parametry Random Forest:", grid_search_rf.best_params_)

    st.subheader("Macierz konfuzji - Random Forest")
    fig, ax = plt.subplots()
    wykres_macierzy_konfuzji(y_test, y_pred_rf, ax=ax)
    st.pyplot(fig)

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf)
    rf_recall = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    
    # Wyświetlanie wyników
    metrics_data_rf = {
        "Metryka": ["Dokładność", "Precyzja", "Czułość", "F1-score"],
        "Wartość": [rf_accuracy, rf_precision, rf_recall, rf_f1]
    }
    metrics_df_rf = pd.DataFrame(metrics_data_rf)

    st.table(metrics_df_rf)

    # ========== Podsumowanie wyników ========== #
    st.subheader("Porównanie wyników")
    results = [
        {
            "Model": "Decision Tree",
            "Dokładność": tree_accuracy,
            "Precyzja": tree_precision,
            "Czułość": tree_recall,
            "F1-score": tree_f1,
        },
        {
            "Model": "SVM",
            "Dokładność": svm_accuracy,
            "Precyzja": svm_precision,
            "Czułość": svm_recall,
            "F1-score": svm_f1,
        },
        {
            "Model": "Random Forest",
            "Dokładność": rf_accuracy,
            "Precyzja": rf_precision,
            "Czułość": rf_recall,
            "F1-score": rf_f1,
        }
    ]
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Wykres porównania
    fig, ax = plt.subplots(figsize=(8, 4))
    results_df.set_index("Model")[["Dokładność", "Precyzja", "Czułość", "F1-score"]].plot(kind="bar", ax=ax)
    plt.title("Porównanie metryk klasyfikacji")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    
    
    
    # Interpretacja wyników
    st.header("Interpretacja wyników")
    
    st.write("""
    Na podstawie wyników analizy trzech modeli uczenia maszynowego — Drzewa Decyzyjnego, SVM oraz Random Forest — możemy wyciągnąć wnioski dotyczące diagnozowania demencji na podstawie dostarczonych zmiennych. 
    """)


    st.subheader("1. Dokładność (Accuracy)")
    st.write("""
    Wszystkie trzy modele osiągnęły tę samą dokładność wynoszącą 85,33%. Oznacza to, że 85,33% wszystkich diagnoz (zarówno pozytywnych, jak i negatywnych) zostało prawidłowo sklasyfikowanych, co wskazuje to na ogólną solidność modeli, ale dokładność sama w sobie nie wystarcza w przypadku niezbalansowanych zbiorów danych, z którymi mamy tutaj doczynienia.
    """)

    st.subheader("2. Precyzja (Precision)")
    st.write("""
    Najwyższą precyzję uzyskano dla Drzewa Decyzyjnego i SVM (90,91%), co oznacza, że większość przypadków oznaczonych jako „Demented” faktycznie należała do tej grupy. Random Forest uzyskał precyzję 84,62%, co jest niższym wynikiem, ale nadal akceptowalnym. Może to oznaczać większą liczbę fałszywie pozytywnych diagnoz w porównaniu z innymi modelami.
    """)

    st.subheader("3. Czułość (Recall)")
    st.write("""
    Czułość Drzewa Decyzyjnego i SVM wynosi 68,97%. To stosunkowo niski wynik, co oznacza, że oba modele nie zidentyfikowały wielu przypadków rzeczywistej demencji (fałszywe negatywy). To duży problem w modelach medycznych. Wolelibyśmy, żeby ewentualnie wykryw demencji tam gdzie jej nie ma niż na odwrót. Random Forest osiągnął lepszy wynik w tej kategorii (75,86%), co wskazuje na jego większą zdolność do wykrywania rzeczywistych przypadków demencji. Jest to istotne w analizie medycznej, ponieważ błędne pominięcie diagnozy demencji może mieć poważne konsekwencje.
    """)

    st.subheader("4. F1-score")
    st.write("""
    Wskaźnik F1-score uwzględnia zarówno precyzję, jak i czułość, dzięki czemu stanowi bardziej zbalansowaną miarę wydajności modelu. Drzewo Decyzyjne i SVM osiągnęły wynik 78,43%, co wskazuje na ich równoważność w równoważeniu fałszywych pozytywów i negatywów. Random Forest uzyskał najlepszy wynik F1-score na poziomie 80%. Wskazuje to, że ten model lepiej balansuje między precyzją a czułością, co czyni go najbardziej odpowiednim wyborem w tej analizie.
    """)

    # Wnioski
    st.header("Wnioski")
    st.write("""
    1. **Random Forest** wykazał najlepszą wydajność w zakresie zrównoważenia precyzji i czułości (F1-score = 80%). Jego wyższa czułość czyni go szczególnie przydatnym, gdy kluczowe jest minimalizowanie liczby pominiętych przypadków demencji, co dla nas jest ważne w tym przypadku.
    2. **Drzewo Decyzyjne i SVM** osiągnęły bardzo podobne wyniki, szczególnie w kategoriach precyzji i F1-score, ale ich niższa czułość jest niepożądana w praktyce medycznej.
    """)

    # Analiza SHAP na przykładzie najlepszego modelu (np. Random Forest)
    st.header("Analiza interpretowalności modelu Random Forest - wartości SHAP")

    st.markdown("""
    # Praktyczny poradnik: Jak interpretować wykresy SHAP?
    
    Wykresy SHAP (SHapley Additive exPlanations) są narzędziem do zrozumienia, jak poszczególne cechy wpływają na decyzje modelu. Oto, jak je interpretować i jak korzystać z nich w praktyce:

    ---
    
    ## Jak interpretować wykres SHAP?
    - **Oś X:** Pokazuje wartości SHAP, które mierzą wpływ danej cechy na wynik modelu:
      - **Po prawej stronie (wartości dodatnie):** Cecha zwiększa wynik modelu (przyczynia się do wyższego prawdopodobieństwa wyniku) - w naszym przypadku zwiększa prawdopodobieństwo zachorowania.
      - **Po lewej stronie (wartości ujemne):** Cecha zmniejsza wynik modelu (zmniejsza prawdopodobieństwo wyniku) - zmniejsza prawdopodobieństwo zachorowania.
    - **Oś Y:** Lista cech (np. wiek, płeć, poziom edukacji), uporządkowana według ich ważności w modelu.
    - **Kolor punktów:** Wskazuje wartość cechy:
      - **Czerwony:** Wysoka wartość cechy - w naszym przypadku np. im lepsze wyniki w teście MMSE.
      - **Niebieski:** Niska wartość cechy - gorsze wyniki w teście MMSE.
    """)

    # Tworzenie obiektu SHAP Explainer
    explainer = shap.KernelExplainer(best_rf_model.predict, X_train)
    # Obliczanie wartości SHAP
    shap_values = explainer.shap_values(X_test)



    plt.clf() 
    shap.summary_plot(
        shap_values,  
        X_test,
        feature_names=X_test.columns,  
        show=False
    )
    st.pyplot(plt.gcf())
    
    st.write("""
    ### 1. `MMSE`
    - **Wysokie wyniki MMSE (czerwone kropki):** Obniżają prawdopodobieństwo demencji (SHAP < 0), co oznacza, że dobre wyniki w teście poznawczym chronią przed diagnozą demencji.
    - **Niskie wyniki MMSE (niebieskie kropki):** Zwiększają prawdopodobieństwo demencji (SHAP > 0), co sugeruje, że obniżenie funkcji poznawczych jest silnie związane z diagnozą demencji.

    ### 2. `is_male`
    - **Mężczyźni (niebieskie punkty):** mają większe prawdopodobieństwo demencji.
    - **Kobiety (czerwone punkty):** mniejsze prawdopodobieństwo demencji.
    - **Wniosek:** Płeć męska może być czynnikiem ryzyka w tym kontekście. Wiemy z badań, że ten wniosek jest  nieprawdziwy, a próbka, która została zbadana jest niereprezentatywna.

    ### 3. `nWBV`
    - **Niższa objętość mózgu (niebieskie kropki):** Zwiększa ryzyko demencji (SHAP > 0).
    - **Wyższa objętość mózgu (czerwone kropki):** Zmniejsza ryzyko demencji (SHAP < 0).
    - **Wniosek:** Utrata objętości mózgu jest istotnym wskaźnikiem ryzyka demencji.

    ### 4. `eTIV`
    - **Wpływ eTIV jest mniej wyraźny, ale ogólnie:**
      - Niższe wartości eTIV mogą nieznacznie zwiększać ryzyko demencji.
      - Wyższe wartości eTIV mają łagodny efekt ochronny.
    - **Wniosek:** Warto monitorować eTIV jako dodatkowy wskaźnik.

    ### 5. `SES`
    - **Niższy SES (niebieskie punkty):** Zwiększa ryzyko demencji (SHAP > 0).
    - **Wyższy SES (czerwone punkty):** Zmniejsza ryzyko demencji (SHAP < 0).
    - **Wniosek:** Osoby z niższym statusem społeczno-ekonomicznym są bardziej narażone na demencję. Wsparcie dla tej grupy może zmniejszyć ryzyko.

    """)


def podsumowanie_section() -> None:
    """
    Wyświetla końcową sekcję podsumowującą projekt:
    - Najważniejsze wnioski
    - Krótka interpretacja wyników
    """
    st.title("Podsumowanie i wnioski")
    st.markdown("""
    <h4>Najważniejsze obserwacje:</h4>
    <ul>
    #### **Podsumowanie**: 
    Projekt miał na celu ekspolarcja zmiennych i ustalenie, które wpływają istotnie na zachorowanie na chorobę Alzheimera. W tym celu wykorzystano modele uczenia maszynowego (SVM, Drzewa decyzyjne, Random Forest) i wybrano najlepszy, który maksymalizuje możliwości przewidywania. Udało się ustalić jakie zmienne najbardziej wpływają na zachororwania i zweryfikować je z dostępna literaturą. 
    Najsilniejsze efekty widać w przypadku MMSE i nWBV, a is_male, eTIV i SES mają zwykle mniejszy, bardziej zróżnicowany wpływ.
    - **MMSE** i **nWBV** to najważniejsze wskaźniki. Niskie wyniki w teście poznawczym i mniejsza objętość mózgu wyraźnie zwiększają ryzyko demencji.
    - **Płeć męska** i **niski SES** to dodatkowe czynniki ryzyka, które sugerują konieczność ukierunkowanego wsparcia.
    - **eTIV** i inne parametry mózgowe mają umiarkowany wpływ, ale ich monitorowanie może być pomocne w ocenie ryzyka
     #### **Rekomendacje**: 
    Z badania udało się uzyskać Wczesna diagnoza i interwencje poprawiające wyniki MMSE mogą znacząco wpłynąć na ograniczenie ryzyka. Regularne badania MRI/CT dla osób z grup ryzyka mogą pomóc w wczesnym wykryciu. Szczególna uwaga na edukację zdrowotną i profilaktykę w tych grupach.
    Model uwzględnia zarówno cechy biologiczne, jak i społeczne, co sugeruje potrzebę podejścia interdyscyplinarnego w ocenie i prewencji demencji.
    </ul>
    """, unsafe_allow_html=True)


def dokumentacja_section() -> None:
    """
    Wyświetla sekcję dokumentacji projektu:
    - Opis celu i zakresu
    - Sposób uruchomienia
    - Struktura projektu
    - Przykład docstringów
    """
    st.title("Dokumentacja projektu")
    st.markdown("""
    <h4>Cel i zakres</h4>
    <p>
      Projekt ma na celu zbudowaniu analizy danych medycznych oraz socjo-ekonomicznych dotyczących alzheimera,
      w tym przetwarzania braków danych, standaryzacji, trenowania modeli klasyfikacji
      oraz interpretacji wyników.
    </p>

    <h4>Instrukcja uruchomienia</h4>
    <ul>
      <p>Strona już działa i jest dotępna. Kod jest dostępny do wglądu w prawym górnym rogu w githubie</strong>
      </p>
    </ul>

    <h4>Struktura plików</h4>
    <p>
      <ul>
        <li><code>program.py</code> – główny plik uruchamiający Streamlit</li>
        <li><code>alzheimer_features.csv</code> – plik z danymi</li>
        <li><code>requirements.txt </code> – plik z wymaganymi bibliotekami do funkcjonowania aplikacji</li>
      </ul>
    </p>

    <h4>Podział zadań</h4>
        <ul>
            <li><strong>Zuzanna Deszcz</strong>:
                <ul>
                    <li><strong>Przetwarzanie Danych:</strong>
                        <ul>
                            <li>Wczytywanie danych z pliku CSV.</li>
                            <li>Mapowanie i przekształcanie danych kategorycznych.</li>
                        </ul>
                    </li>
                    <li><strong>Tworzenie Wizualizacji:</strong>
                        <ul>
                            <li>Generowanie wykresów statystycznych i korelacyjnych.</li>
                            <li>Tworzenie macierzy konfuzji i boxplotów.</li>
                        </ul>
                    </li>
                    <li><strong>Sekcje Wprowadzające w Streamlit:</strong>
                        <ul>
                            <li>Opracowanie sekcji wprowadzającej i charakterystyki zbioru danych w aplikacji Streamlit.</li>
                        </ul>
                    </li>
                    <li><strong>Czyszczenie Danych:</strong>
                        <ul>
                            <li>Analiza i obsługa braków danych (np. usuwanie, uzupełnianie wartości).</li>
                            <li>Wykrywanie i analiza wartości odstających (outlierów).</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li><strong>Natalia Łyś</strong>:
                <ul>
                    <li><strong>Podział Danych:</strong>
                        <ul>
                            <li>Dzielenie danych na zbiory uczący i testowy.</li>
                            <li>Standaryzacja danych numerycznych.</li>
                        </ul>
                    </li>
                    <li><strong>Trenowanie Modeli Uczenia Maszynowego:</strong>
                        <ul>
                            <li>Implementacja i optymalizacja modeli takich jak Drzewa Decyzyjne, SVM, Random Forest.</li>
                            <li>Użycie GridSearchCV do doboru najlepszych hiperparametrów.</li>
                        </ul>
                    </li>
                    <li><strong>Analiza i Prezentacja Wyników:</strong>
                        <ul>
                            <li>Ocena modeli za pomocą metryk (dokładność, precyzja, czułość, F1-score).</li>
                            <li>Tworzenie i interpretacja wykresów porównawczych.</li>
                            <li>Implementacja analizy interpretowalności modeli (np. SHAP).</li>
                        </ul>
                    </li>
                </ul>
            </li>
        </ul>

    
    <h4>Dokumentacja funkcji (docstringi)</h4>
    <p>
      Każda funkcja zawiera krótki opis, parametry i zwracane wartości, z uwagi na czytelność kodu.
    </p>
    """, unsafe_allow_html=True)


# =============== FUNKCJA GŁÓWNA APLIKACJI =============== #

def main() -> None:
    """
    Główna funkcja aplikacji Streamlit.
    Odpowiada za stworzenie bocznego menu (sidebar) i wywoływanie odpowiednich sekcji.
    """
    st.sidebar.title("Nawigacja")
    sections = [
        "Wprowadzenie",
        "Charakterystyka zbioru danych",
        "Usuwanie braków i analiza outlierów",
        "Dzielenie na zbiór uczący i testowy",
        "Metody uczenia maszynowego",
        "Podsumowanie i wnioski",
        "Dokumentacja"
    ]
    selected_section = st.sidebar.radio("Przejdź do sekcji:", sections)

    # Wczytanie danych tylko raz (jeżeli jeszcze nie ma w session_state)
    if "data" not in st.session_state:
        data_path = 'alzheimer_features.csv'  
        data = wczytaj_dane(data_path)
        data = przygotuj_dane_kategoryczne(data)
        st.session_state["data"] = data

    # Wywołanie odpowiedniej sekcji
    if selected_section == "Wprowadzenie":
        wprowadzenie_section()
    elif selected_section == "Charakterystyka zbioru danych":
        charakterystyka_danych_section(st.session_state["data"])
    elif selected_section == "Usuwanie braków i analiza outlierów":
        braki_outliery_section()
    elif selected_section == "Dzielenie na zbiór uczący i testowy":
        dzielenie_section()
    elif selected_section == "Metody uczenia maszynowego":
        metody_uczenia_section()
    elif selected_section == "Podsumowanie i wnioski":
        podsumowanie_section()
    elif selected_section == "Dokumentacja":
        dokumentacja_section()


if __name__ == "__main__":
    main()
