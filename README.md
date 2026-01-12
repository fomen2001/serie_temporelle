# serie_temporelle
📊 Analyse et Modélisation de Séries Temporelles

Ce dépôt regroupe trois travaux pratiques (TP) consacrés à l’analyse, la transformation et la modélisation de séries temporelles, à l’aide de bibliothèques Python telles que pandas, matplotlib et statsmodels. L’objectif est de comprendre le comportement temporel des données, d’évaluer leur stationnarité et d’appliquer des modèles statistiques adaptés.

🔹 TP 1 – Analyse et décomposition d’une série temporelle
Objectif

Analyser une série temporelle réelle (Daily Minimum Temperature) afin d’identifier ses composantes fondamentales.

Étapes principales

Chargement et vérification du dataset

Exploration des données (types de colonnes, valeurs manquantes, anomalies)

Visualisation de la série temporelle complète

Décomposition de la série à l’aide de la méthode STL (Seasonal-Trend decomposition using Loess)

Analyse des composantes :

Tendance : évolution globale de la série

Saisonnalité : motifs périodiques récurrents

Résidus : bruit et fluctuations aléatoires

Résultat

Une meilleure compréhension de la structure interne de la série et de l’influence de chaque composante sur son comportement global.

🔹 TP 2 – Analyse de la série temporelle de production électrique
Objectif

Étudier la stationnarité d’une série temporelle de production électrique et appliquer les transformations nécessaires.

Étapes principales

Chargement et exploration du dataset

Visualisation de la série temporelle

Test de stationnarité avec le test augmenté de Dickey-Fuller (ADF)

Application de transformations (différenciation, logarithme) si la série n’est pas stationnaire

Nouvelle vérification de la stationnarité après transformation

Résultat

Mise en évidence de l’importance de la stationnarité pour l’analyse et la modélisation des séries temporelles.

🔹 TP 3 – Analyse et modélisation ARMA
Objectif

Modéliser une série temporelle de ventes à l’aide d’un modèle ARMA (AutoRegressive Moving Average).

Étapes principales

Chargement du dataset 5_1_retails.csv

Préparation des données (conversion des dates, index temporel)

Visualisation et analyse exploratoire

Vérification et correction de la stationnarité (test ADF)

Analyse des fonctions ACF et PACF pour déterminer les paramètres (p, q)

Séparation des données en ensembles d’entraînement et de test

Ajustement du modèle ARMA

Réalisation de prévisions

Évaluation des performances du modèle (RMSE ou MAPE)

Résultat

Un modèle ARMA capable de capturer la dynamique temporelle des ventes et d’effectuer des prévisions fiables.

🛠️ Technologies utilisées

Python

pandas

matplotlib

statsmodels

numpy

🎯 Compétences développées

Analyse exploratoire de séries temporelles

Décomposition STL

Tests de stationnarité (ADF)

Transformation de séries temporelles

Modélisation statistique (ARMA)

Évaluation de modèles de prévision