**README -- Projet Stage Data Science (Développement analyse prédictive
simplifié)**

**1. Contexte du projet**

Ce projet s'inscrit dans le cadre du sujet :\
**\"Développement d'une Analyse Prédictive Simplifiée\"**

L'objectif principal est de concevoir une plateforme permettant :

-   d'importer des données depuis différentes sources (connecteurs),

-   de les préparer,

-   puis d'appliquer des modèles de Machine Learning généralisables
    (prévision, classification, régression)\
    afin de rendre l'analyse prédictive accessible et adaptable à
    différents cas d'usage métier.

**2. Architecture générale du projet**
```
Projet/
├── Connecteurs/
│   └── connecteurs.py
├── Preprocessing/
│   └── preprocessing.py
├── Modeling/
│   └── modeling.py
├── api/
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   └── forecast.html
├── data/
│   └── (datasets téléchargés et nettoyés)
├── tests/
│   ├── test_connecteurs.py
│   ├── test_preprocessing.py
│   └── test_modeling.py
├── best_churn_model.joblib
├── best_forecasting_model_SARIMA.joblib
└── .env
```
 Connecteurs/ : gestion des connecteurs (Kaggle, World Bank, MongoDB).

Preprocessing/ : nettoyage et préparation des données.

Modeling/ : création, entraînement et sauvegarde des modèles.

api/ : application Flask (backend + templates HTML).

data/ : fichiers CSV téléchargés et nettoyés.

tests/ : contient trois tests unitaires principaux :

test_connecteurs.py

test_preprocessing.py

test_modeling.py

main.py : script principal qui lance automatiquement tout le pipeline :

Téléchargement et enrichissement des données (connecteurs)

Prétraitement

Entraînement des modèles (churn + prévision)

Sauvegarde des modèles

Fichiers .joblib : modèles sauvegardés (classification churn et forecasting).

Fichier .env : variables d’environnement (accès MongoDB, clés API…).


**3. Pipeline global**

**3.1 Étape 1 -- Connecteurs de données (connecteurs.py)**

Cette phase a deux résultats principaux :

**1. Téléchargement et enrichissement :**

-   Télécharge le dataset Kaggle **mashlyn/online-retail-ii-uci**.

-   Enrichit chaque ligne avec des indicateurs économiques de la
    **Banque Mondiale** :

    -   PIB (GDP)

    -   Inflation

    -   Consommation

**2. Double sauvegarde des données :**

-   **Dans une base MongoDB** (collection spécifiée dans .env).

-   **Dans un fichier CSV local** (data/online_retail.csv) pour une
    exploitation hors ligne.

Ces deux actions sont réalisées par les fonctions :

-   store_in_mongo(df) → insertion dans MongoDB

-   save_clean_csv(df, output_path) → sauvegarde locale

**Scripts impliqués :**

-   connecteurs/connecteurs.py

**3.2 Étape 2 -- Prétraitement (preprocessing.py)**

**Objectif :**\
Préparer des données propres et cohérentes issues de MongoDB avant la
modélisation.

**Pipeline détaillé :**

**1. Chargement des données**

-   Connexion à MongoDB via les informations stockées dans .env.

-   Extraction complète de la collection sous forme de DataFrame pandas.

-   Suppression de la colonne \_id (inutile pour la modélisation).

**2. Nettoyage initial**

1.  Suppression des lignes sans Customer_ID

    -   *Raison : ces données ne permettent pas d\'identifier un
        client.*

2.  Affichage du nombre de valeurs manquantes par colonne

    -   *Permet de savoir quelles colonnes nécessitent une imputation.*

3.  Suppression des pays sans Country_Code

    -   *Si un pays n'a pas pu être enrichi avec son code ISO (donc pas
        d'indicateurs économiques disponibles), la ligne est supprimée.*

**3. Imputation des indicateurs économiques**

-   Colonnes concernées : GDP, Inflation, Consumption.

-   Méthode : imputation par la médiane, regroupée par Year.

    -   Exemple : toutes les valeurs manquantes pour l'année 2010 sont
        remplacées par la médiane des valeurs disponibles de la même
        année.

    -   **Pourquoi pas par pays ?**\
        Parce que certains couples (Country_Code, Year) ont très peu de
        données. L'utilisation par année uniquement assure plus de
        robustesse.

**4. Résultat final**

-   Jeu de données nettoyé et imputé.

-   Plus aucune valeur manquante dans les variables clés.

-   Prêt pour la phase de modélisation.

**3.3 Étape 3 -- Modélisation (modeling.py)**

Deux grandes familles de modèles sont développées :

1.  **Classification** (prédire le churn client)

2.  **Prévision** (prévoir les ventes futures)

**2.1 Modélisation du Churn**

**Étapes :**

1.  Création de la variable cible **Churn**

    -   Calcul :\
        days_since_last_purchase = date max des factures -- date
        d'achat.

    -   Règle : si un client n'a pas acheté depuis 180 jours → Churn =
        1, sinon 0.

2.  Encodage du pays

    -   Country est transformé en entier avec LabelEncoder.

3.  Sélection des variables explicatives (features)

    -   Quantity, Price, Year, Customer_ID, Country_encoded.

4.  Division du jeu de données

    -   80% pour l'entraînement, 20% pour le test.

5.  Modèles testés :

    -   Régression Logistique

    -   Random Forest

    -   KNN

6.  Évaluation

    -   Métrique : **Accuracy (précision)**

    -   Visualisations pour Random Forest :

        -   Matrice de confusion (avec Plotly)

        -   Importance des variables

7.  Sélection et sauvegarde

    -   Le meilleur modèle est choisi selon la meilleure accuracy.

    -   Sauvegarde du modèle dans **best_churn_model.joblib**.

**2.2 Prévision des ventes**

**Préparation des données :**

-   Agrégation des ventes par mois :

    -   sales = Quantity × Price

    -   Groupement par mois (pd.Grouper(freq=\'M\'))

**Modèles testés :**

1.  ARIMA

2.  Prophet

3.  SARIMA (Saisonnière)

**Validation :**

-   Découpage du jeu de données : 80% train / 20% test

-   Métriques calculées :

    -   RMSE (Root Mean Squared Error)

    -   MAE (Mean Absolute Error)

    -   MAPE (% d'erreur moyenne)

    -   R² (coefficient de détermination)

**Visualisation :**

-   Pour SARIMA :

    -   Affichage interactif (Plotly) : série historique + prévisions
        futures (12 mois)

**Choix du meilleur modèle :**

-   Le modèle avec le meilleur **R²** est sélectionné.

-   Sauvegarde sous :\
    best_forecasting_model\_\<nom_du_modele\>.joblib

**4. Points importants**

**Gestion des valeurs manquantes :**

-   Colonnes numériques : imputation par médiane.

-   Colonnes catégorielles : suppression des lignes trop incomplètes.

**Gestion des anomalies :**

-   Suppression des clients sans identifiant.

-   Suppression des pays non enrichis.

-   Anomalies temporelles prises en compte par les modèles.

**Critères d'évaluation :**

-   Churn : Accuracy

-   Prévision : RMSE, MAE, MAPE, R² (critère principal)

**5. Sorties générées**

1.  Modèle churn :

    -   best_churn_model.joblib

2.  Modèle prévision :

    -   best_forecasting_model\_\<ARIMA\|Prophet\|SARIMA\>.joblib

3.  Visualisations interactives :

    -   matrice de confusion

    -   forecast (prévisions)

**6. Application Web (api/app.py)**

**Fonctionnalités principales :**

1.  **Prédiction du churn** (/predict_churn_form)

    -   Formulaire pour saisir :

        -   Quantity, Price, Year, Customer_ID, Country

    -   Affichage du résultat et de la probabilité associée.

2.  **Prévisions des ventes** (/forecast)

    -   Graphique interactif Plotly avec :

        -   Historique des ventes.

        -   Prévisions sur 12 mois.

    -   Tableau des prévisions.

3.  **Téléchargement Excel** (/download_forecast_excel)

    -   Téléchargement d'un fichier .xlsx contenant les prévisions
        mensuelles.

**7. Technologies utilisées**

-   Python

-   Pandas, NumPy, Scikit-learn, Statsmodels, Prophet

-   Plotly

-   MongoDB

-   Flask, HTML

-   Kaggle API, World Bank API

**10. Livrables finaux**

-   Connecteur de données (Kaggle + Banque Mondiale → MongoDB)

-   Données nettoyées et enrichies

-   Modèle prédictif du churn

-   Modèle de prévision des ventes

-   Application web Flask interactive
