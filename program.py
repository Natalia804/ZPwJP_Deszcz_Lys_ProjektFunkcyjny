from sklearn.inspection import PartialDependenceDisplay
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import numpy as np
from scipy.stats import zscore, pointbiserialr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.inspection import PartialDependenceDisplay
import shap

st.sidebar.title("Nawigacja")
sections = ["Wprowadzenie", "Charakterystyka zbioru danych", "Usuwanie braków i analiza outlierów", "Dzielenie na zbiór uczący i testowy", "Metody uczenia maszynowego", "Podsumowanie i wnioski"]
selected_section = st.sidebar.radio("Przejdź do sekcji:", sections)

data_path = 'alzheimer_features.csv' 
data = pd.read_csv(data_path)
data = data.drop(columns=['CDR'])


if selected_section == "Wprowadzenie":

    st.title("Projekt: zastosowanie wybranych metod klasyfikacyjnych")
    st.write("###### `Przemiot`: Uczenie maszynowe")
    st.write("###### `Prowadząca`: Justyna Tora")
    st.write("###### `Autorki`: Natalia Łyś, Zuzanna Deszcz")


    # Tytuł aplikacji
    st.header("Analiza Zbioru Danych: *Alzheimer Feature*")
    st.write("#### Wprowadzenie")
    st.markdown("""
    *Choroba Alzheimera (AD)* to najbardziej powszechna odmiana demencji. 
    W Europie choroba ta jest głównym skutkiem utraty samodzielności i upośledzenia osób starszych. 
    Szacuje się ilość chorych na *10 milionów ludzi*.
    """)

elif selected_section == "Charakterystyka zbioru danych":
    # Charakterystyka zbioru danych
    st.title("Charakterystyka Zbioru Danych")
    st.markdown("""
    Zbiór danych zawiera informacje medyczne, socjo-ekonomiczne oraz diagnozę demencji u pacjentów. 
    Dane te zostały zaczerpnięte z <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle</a>, bazując na badaniach wykorzystujących uczenie maszynowe do analizy demencji
    """, unsafe_allow_html=True)

    st.subheader("Zmienna objaśniana")
    st.markdown("""
    <span style="color: #1f77b4; font-weight: bold;">Group</span>: Diagnoza choroby:
    - <code style="color: #2ca02c;">Demented</code>: Osoby zdiagnozowane z demencją.
    - <code style="color: #ff7f0e;">Nondemented</code>: Osoby bez demencji.
    - <code style="color: #d62728;">Converted</code>: Osoby, które po pewnym czasie zostały zaklasyfikowane jako zdrowe.
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

    # Separator wizualny
    st.markdown("---")

    # Źródła danych
    st.header("Źródła Danych")
    st.markdown("""
    1. <a href="https://cordis.europa.eu/article/id/428863-mind-reading-software-finds-hidden-signs-of-dementia/pl" target="_blank" style="color: #007BFF; font-weight: bold;">Cordis.europa.eu</a>  
    2. <a href="https://www.sciencedirect.com/science/article/pii/S2352914819300917?via%3Dihub" target="_blank" style="color: #007BFF; font-weight: bold;">ScienceDirect - *Machine learning in medicine: Performance calculation of dementia prediction by support vector machines (SVM)*</a>  
    3. <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle Dataset</a>
    """, unsafe_allow_html=True)

    # Podsumowanie
    st.markdown("""
    **Podsumowanie**  
    Aplikacja ta ma na celu analizę zbioru danych oraz przewidywanie występowania demencji u pacjentów na podstawie wyżej wymienionych zmiennych.
    """)

    # Sprawdzenie, czy dane są załadowane
    if 'data' not in locals():
        st.error("Dane nie zostały załadowane! Proszę wczytać dane przed kontynuacją.")
        st.stop()

    # Wyświetlenie podstawowych informacji o danych
    st.write("Podgląd pierwszych wierszy zbioru danych:")
    st.dataframe(data.head())

    st.subheader("Podstawowe statystyki:")
    st.write(data.describe())

    # Analiza braków danych
    st.subheader("Brakujące dane:")
    missing_values = data.isnull().sum()
    missing_percent = (missing_values / len(data)) * 100
    st.write("Braki danych w poszczególnych kolumnach (w procentach):")
    st.bar_chart(missing_percent)
    if missing_values.any():
        st.write("")
    else:
        st.success("Zbiór danych nie zawiera braków.")

    # Mapowanie zmiennych kategorycznych na zmienne binarne
    if 'M/F' not in data.columns:
        st.error("Kolumna `M/F` nie istnieje w zbiorze danych.")
        st.stop()


    if 'Group' in data.columns and 'M/F' in data.columns:
        data['is_male'] = data['M/F'].map({'M': True, 'F': False})
        data[['is_demented', 'is_converted']] = data['Group'].apply(
            lambda x: pd.Series({
                'Demented': (True, False),
                'Nondemented': (False, False),
                'Converted': (False, True)
            }.get(x, (None, None)))
        )    
    else:
        st.error("Brak wymaganych kolumn `Group` lub `M/F` w zbiorze danych.")
        
    data['is_demented'] = data['is_demented'].astype(int)
    data['is_converted'] = data['is_converted'].astype(int)
    data['is_male'] = data['is_male'].astype(int)
        
    st.header("Dytrybucja zmiennej objaśniającej Group")
    group_distribution = data['Group'].value_counts(normalize=True)
    st.bar_chart(group_distribution)
    st.write(group_distribution)

    # Relacja między płcią a demencją
    st.subheader("Relacja między płcią a demencją")
    if 'is_male' in data.columns and 'is_demented' in data.columns:
        matrix_demented = [
            [len(data[(data['is_male'] == True) & (data['is_demented'] == True)]),
             len(data[(data['is_male'] == True) & (data['is_demented'] == False)])],
            [len(data[(data['is_male'] == False) & (data['is_demented'] == True)]),
             len(data[(data['is_male'] == False) & (data['is_demented'] == False)])]
        ]
        matrix_demented_df = pd.DataFrame(
            matrix_demented, 
            index=['Male', 'Not male'], 
            columns=['Demented', 'Not demented']
        )
        st.write("Tabela relacji:")
        st.dataframe(matrix_demented_df)

        st.write("Wykres relacji:")
        fig, ax = plt.subplots()
        matrix_demented_df.plot(kind='bar', ax=ax, color=['indigo', 'gold'])
        ax.set_ylabel("Liczba obserwacji")
        ax.set_title("Relacja: Płeć a Demencja")
        st.pyplot(fig)
    else:
        st.error("Brak danych do analizy relacji między płcią a demencją.")
        
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
    st.subheader("Macierz korelacji dla zmiennych numerycznych")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if not numeric_data.empty:
        correlation_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", mask=np.triu(np.ones_like(correlation_matrix, dtype=bool)), linewidths=0.5, vmin=-1, vmax=1, ax=ax)
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
    


    # Dystrybucja zmiennych numerycznych względem grup
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

    # Wyświetlenie treści w Streamlit
    st.header("Interpretacja Wykresów")
    st.write(interpretacja)

    # Zapisanie przetworzonych danych do session_state
    selected_columns = ['nWBV', 'MMSE', 'eTIV', 'SES', 'is_demented', 'is_male']
    

    available_columns = [col for col in selected_columns if col in data.columns]
    if available_columns:
        st.session_state.data_selected = data[available_columns]
    else:
        st.error("Brak wymaganych kolumn do zapisania przetworzonych danych.")

elif selected_section == "Usuwanie braków i analiza outlierów":
    st.title("Usuwanie braków danych i analiza wartości odstających")

    # Sprawdzenie, czy przetworzone dane są dostępne
    if "data_selected" in st.session_state:
        data_selected = st.session_state.data_selected
        #st.write("Dane zostały wczytane z `session_state`.")
    else:
        st.error("Dane nie zostały przetworzone w poprzednich sekcjach.")
        st.stop()


    # Rozdzielenie danych na numeryczne i kategoryczne
    data_numeric = data_selected.select_dtypes(include=['float64', 'int64'])


    # Debug: Podgląd danych przed przetwarzaniem
    st.subheader("Podgląd danych przed przetwarzaniem")
    st.markdown("""
    Ze względu na wysoką korelację części zmiennych między sobą i zdecydowanie 
    wyższe korealcje zmiennych ze zmiennymi objaśnanymi zdecydowano się na wybranie tych zmiennych do dalszej analizy.
    Dane:
    """)
    st.dataframe(data_numeric.head())
    
    # Analiza braków danych
    st.subheader("Analiza braków danych")
    missing_numeric = data_numeric.isnull().mean() * 100


    st.write("Procent braków danych w kolumnach numerycznych:")
    st.bar_chart(missing_numeric)


    # Obsługa braków danych 
    st.subheader("Obsługa braków danych")
    method_numeric = st.radio("Wybierz metodę wypełniania braków:", 
                              ["Usuwanie wierszy", "Wypełnianie średnią", "Wypełnianie medianą", "Wypełnianie modą"])

    if method_numeric == "Usuwanie wierszy":
        data_numeric_cleaned = data_numeric.dropna()
    elif method_numeric == "Wypełnianie średnią":
        data_numeric_cleaned = data_numeric.fillna(data_numeric.mean())
    elif method_numeric == "Wypełnianie medianą":
        data_numeric_cleaned = data_numeric.fillna(data_numeric.median())
    elif method_numeric == "Wypełnianie modą":
        data_numeric_cleaned = data_numeric.fillna(data_numeric.mode().iloc[0])
        
    

    # Debug: Podgląd danych numerycznych po czyszczeniu
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
    #st.success("Dane zostały przetworzone i zapisane w `session_state`.")

    # Analiza wartości odstających za pomocą Z-score
    st.subheader("Analiza wartości odstających (Z-score)")
    
    # Standaryzacja danych
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_numeric_cleaned)
    data_standardized = pd.DataFrame(data_standardized, columns=data_numeric_cleaned.columns)

    # Funkcja do wykrywania wartości odstających
    def detect_outliers_zscore(data, column, threshold=3):
        if data[column].notnull().sum() > 0:
            z_scores = zscore(data[column].dropna())
            outliers = data.loc[data.index[np.abs(z_scores) > threshold]]
            return outliers
        return pd.DataFrame()

    # Detekcja wartości odstających
    outliers_zscore = {}
    for col in data_standardized.columns:
        outliers_zscore[col] = detect_outliers_zscore(data_standardized, col)

    # Przygotowanie danych do tabelki
    outliers_summary = {
        "Kolumna": [],
        "Liczba wartości odstających": []
    }
    for col, outliers in outliers_zscore.items():
        outliers_summary["Kolumna"].append(col)
        outliers_summary["Liczba wartości odstających"].append(len(outliers))

    # Tworzenie tabeli wynikowej
    outliers_summary_df = pd.DataFrame(outliers_summary)

    # Wyświetlanie tabeli
    st.write("Podsumowanie liczby wartości odstających w każdej kolumnie:")
    st.dataframe(outliers_summary_df)

    st.write("Ze względu na naturę medyczną problemu, zdecydowano się nie usuwać outlierów, ponieważ odzwierciedlają one ważne zjawiska dla delikatnych danych.")

elif selected_section == "Dzielenie na zbiór uczący i testowy":
    st.title("Dzielenie na zbiór uczący i testowy")

    # Sprawdzenie, czy dane przetworzone są dostępne
    if "data_cleaned" in st.session_state:
        data_cleaned = st.session_state.data_cleaned
    else:
        st.error("Zmienna 'data_cleaned' nie została zdefiniowana w poprzednich sekcjach.")
        st.stop()

    # Debug: Wyświetlenie dostępnych kolumn w danych przetworzonych
    #st.write("Kolumny w `data_cleaned`:")
    #st.write(data_cleaned.head(5))


    # Oddzielenie zmiennej docelowej od danych wejściowych
    target_column = "is_demented"
    if target_column not in data_cleaned.columns:
        st.error(f"Kolumna docelowa '{target_column}' nie istnieje w danych.")
        st.stop()

    X = data_cleaned.drop(columns=[target_column])
    y = data_cleaned[target_column]

    # Debug: Podgląd danych wejściowych i zmiennej docelowej
    #st.write("Dane wejściowe (X):")
    #st.dataframe(X.head())
    #st.write("Zmienna docelowa (y):")
    #st.write(y.head())

  

  
    # 'is_demented' osobno
    is_male = X['is_male']

    # Zmienne numeryczne, pomijając 'is_demented'
    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns.drop('is_male')

    # Sprawdź, czy są zmienne numeryczne do standaryzacji
    if not numerical_columns.empty:
        scaler = StandardScaler()
        # Standaryzuj wybrane zmienne numeryczne
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    else:
        st.warning("Brak zmiennych numerycznych do standaryzacji.")

    # Dodaj z powrotem kolumnę 'is_demented' do DataFrame
    X['is_male'] = is_male

    # Podział danych
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        st.error(f"Błąd podczas dzielenia danych: {e}")
        st.stop()

    # Debug: Podgląd zbioru uczącego i testowego
    st.write("Podgląd X_train po standryzacji:")
    st.dataframe(X_train.head())
    st.write("Podgląd y_train:")
    st.write(y_train.head())
    st.write(f"Liczba próbek w zbiorze uczącym: {len(X_train)}")
    st.write(f"Liczba próbek w zbiorze testowym: {len(X_test)}")

    # Zapisanie danych do session_state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    #st.success("Dane zostały zapisane do `session_state`.")

elif selected_section == "Metody uczenia maszynowego":
    # Nagłówek sekcji
    st.title("Metody uczenia maszynowego w identyfikacji demencji")

    if "X_train" in st.session_state and "X_test" in st.session_state and "y_train" in st.session_state and "y_test" in st.session_state:
        X_train, X_test = st.session_state.X_train, st.session_state.X_test
        y_train, y_test = st.session_state.y_train, st.session_state.y_test
    else:
        st.error("Brak danych do uczenia maszynowego. Upewnij się, że poprzednie sekcje zostały poprawnie wykonane.")
        st.stop()


    st.write("Kształt danych treningowych (X_train):", X_train.shape)
    st.write("Kształt etykiet (y_train):", y_train.shape)

    if np.any(pd.isnull(X_train)):
        st.error("Dane X_train zawierają wartości NaN. Sprawdź wcześniejsze etapy przetwarzania danych.")
        st.stop()


    ### Drzewa Decyzyjne
    st.header("Metoda 1: Drzewa Decyzyjne")
    st.markdown("""
    Drzewa decyzyjne to intuicyjna metoda uczenia maszynowego, która pozwala na modelowanie decyzji w sposób hierarchiczny.
    W celu znalezienia optymalnych parametrów zastosowano Grid Search. Analizujemy m.in.:
    - `max_depth`: Maksymalna głębokość drzewa decyzyjnego wpływa na złożoność modelu.
    - `criterion`: Miara podziału danych (`gini` lub `entropy`). Wybór zależy od charakterystyki danych.
    """)

    # Parametry do Grid Search
    param_grid_tree = {
        'max_depth': [3, 5, 7, 10],
        'criterion': ['gini', 'entropy']
    }

    # Tworzenie modelu drzewa decyzyjnego
    tree_model = DecisionTreeClassifier(random_state=42)
    grid_search_tree = GridSearchCV(tree_model, param_grid_tree, cv=5, scoring='f1', n_jobs=-1)
    grid_search_tree.fit(X_train, y_train)

    # Najlepszy model
    best_tree_model = grid_search_tree.best_estimator_
    y_pred_tree = best_tree_model.predict(X_test)

    # Metryki
    tree_accuracy = accuracy_score(y_test, y_pred_tree)
    tree_precision = precision_score(y_test, y_pred_tree)
    tree_sensitivity = recall_score(y_test, y_pred_tree)
    tree_f1 = f1_score(y_test, y_pred_tree)

    # Wyświetlanie wyników
    st.write("Najlepsze parametry dla Drzewa Decyzyjnego:")
    st.write(grid_search_tree.best_params_)

    # Macierz konfuzji
    st.subheader("Macierz konfuzji dla Drzewa Decyzyjnego")
    cm_tree = confusion_matrix(y_test, y_pred_tree)
    fig, ax = plt.subplots()
    sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Demented", "Demented"], yticklabels=["Not Demented", "Demented"], ax=ax)
    ax.set_xlabel("Przewidywania")
    ax.set_ylabel("Rzeczywistość")
    st.pyplot(fig)
    
    
    # Wyświetlanie wyników
    metrics_data = {
        "Metryka": ["Dokładność", "Precyzja", "Czułość", "F1-score"],
        "Wartość": [tree_accuracy, tree_precision, tree_sensitivity, tree_f1]
    }
    metrics_df = pd.DataFrame(metrics_data)

    st.table(metrics_df)

    ### SVM
    st.header("Metoda 2: Support Vector Machines (SVM)")
    st.markdown("""
    Support Vector Machines (SVM) znajduje optymalną hiperpłaszczyznę do separacji klas. 
    W celu znalezienia najlepszych parametrów zastosowano Grid Search. Parametry:
    - `C`: Regularizacja (kontrola nadmiarowego dopasowania).
    - `kernel`: Typ jądra (np. 'linear', 'rbf').
    """)

    # Parametry do Grid Search
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }

    # Tworzenie modelu SVM
    svm_model = SVC(random_state=42, probability=True)
    grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='f1', n_jobs=-1)
    grid_search_svm.fit(X_train, y_train)

    # Najlepszy model
    best_svm_model = grid_search_svm.best_estimator_
    y_pred_svm = best_svm_model.predict(X_test)

    # Metryki
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_precision = precision_score(y_test, y_pred_svm)
    svm_sensitivity = recall_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm)

   
    # Macierz konfuzji
    st.subheader("Macierz konfuzji dla SVM")
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    fig, ax = plt.subplots()
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Demented", "Demented"], yticklabels=["Not Demented", "Demented"], ax=ax)
    ax.set_xlabel("Przewidywania")
    ax.set_ylabel("Rzeczywistość")
    st.pyplot(fig)
    
    
      # Wyświetlanie wyników
    metrics_data_svm = {
        "Metryka": ["Dokładność", "Precyzja", "Czułość", "F1-score"],
        "Wartość": [svm_accuracy, svm_precision, svm_sensitivity, svm_f1]
    }
    metrics_df_svm = pd.DataFrame(metrics_data_svm)

    st.table(metrics_df_svm)

    ### Random Forest
    st.header("Metoda 3: Random Forest")
    st.markdown("""
    Random Forest to zespół drzew decyzyjnych, które tworzą silny model predykcyjny. 
    W celu znalezienia najlepszych parametrów zastosowano Grid Search:
    - `n_estimators`: Liczba drzew w lesie.
    - `max_depth`: Maksymalna głębokość drzew.
    """)

    # Parametry do Grid Search
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20]
    }

    # Tworzenie modelu Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    # Najlepszy model
    best_rf_model = grid_search_rf.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)

    # Metryki
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf)
    rf_sensitivity = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    
    

    # Macierz konfuzji
    st.subheader("Macierz konfuzji dla Random Forest")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Demented", "Demented"], yticklabels=["Not Demented", "Demented"], ax=ax)
    ax.set_xlabel("Przewidywania")
    ax.set_ylabel("Rzeczywistość")
    st.pyplot(fig)
    
     # Wyświetlanie wyników
    metrics_data_rf = {
        "Metryka": ["Dokładność", "Precyzja", "Czułość", "F1-score"],
        "Wartość": [rf_accuracy, rf_precision, rf_sensitivity, rf_f1]
    }
    metrics_df_rf = pd.DataFrame(metrics_data_rf)

    st.table(metrics_df_rf)
    
    st.header("Analiza wyników")

    # Funkcja do obliczania metryk
    def calculate_metrics(model, X, y_true):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    # Analiza wyników dla każdego modelu
    models = {
        "Drzewo Decyzyjne": best_tree_model,
        "SVM": best_svm_model,
        "Random Forest": best_rf_model,
    }

    results = []
    for name, model in models.items():
        accuracy, precision, recall, f1 = calculate_metrics(model, X_test, y_test)
        results.append({
            "Model": name,
            "Dokładność": accuracy,
            "Precyzja": precision,
            "Czułość": recall,
            "F1-score": f1
        })

    # Wyświetlenie wyników w tabeli
    results_df = pd.DataFrame(results)
    st.subheader("Porównanie wyników modeli")
    st.dataframe(results_df)

    # Wizualizacja wyników
    st.subheader("Wykres porównania metryk")
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.set_index("Model").plot(kind="bar", ax=ax)
    plt.title("Porównanie metryk klasyfikacji")
    plt.ylabel("Wartość metryki")
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
  
    ## SHAP

    st.header("Analiza interpretowalności modelu Random Forest - wartości SHAP")

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
    Czerwone punkty (wysokie MMSE) na lewo → obniżają przewidywanie. \n\n
    Niebieskie (niskie MMSE) na prawo → zwiększają przewidywanie.

    ### 2. `is_male`
    Cecha binarna (0/1).  Bycie mężczyzną zwiększa przewidywanie tak samo symetrycznie jak bycie kobietą je obniża.

    ### 3. `nWBV`
    Duży rozrzut wartośc, co oznacza różnorodny (dodatni bądź ujemny) wpływ w zależności od obserwacji.

    ### 4. `eTIV`
    Przewaga punktów blisko zera, często z niewielkim wpływem, czasem lekko ujemnym.

    ### 5. `SES`
    Punkty przy zerze z drobnym ujemnym efektem, więc niewielkie znaczenie w porównaniu do MMSE czy nWBV.

    #### **Podsumowanie**: 
    Najsilniejsze efekty widać w przypadku MMSE i nWBV, a is_male, eTIV i SES mają zwykle mniejszy, bardziej zróżnicowany wpływ.
    """)
   
elif selected_section == "Podsumowanie i wnioski":

    st.header("Podsumowanie i wnioski")

    st.markdown("""
    ### Kluczowe obserwacje:
    - **Najlepszy model:** Na podstawie wyników analizy, model o najwyższym F1-score to Random Forest.
    - **Wpływ cech:** Analiza SHAP pokazała, że cechy takie jak `MMSE` oraz `nWBV` miały największy wpływ na przewidywania modelu. Natomiast wpływ płci, statusu ekonomicznego również ma wpływ na ostateczne wyniki.

    ### Wnioski:
    - Modele oparte na Random Forest i SVM oraz Drzewa deycyzyjne są skuteczne w przewidywaniu demencji, ale najlepiej w tej sytuacji wypada Random Forest.
    - Cechy takie jak `MMSE`, `eTIV`, i `nWBV` są kluczowe dla diagnostyki demencji.
    """)

