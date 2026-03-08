# False Alarm Prediction – London Fire Brigade

Python-based EDA and machine learning project analysing 236,763 London Fire Brigade incident records to identify and predict false alarm callouts.

## Overview

Around 40% of all London Fire Brigade callouts are false alarms, diverting crews from genuine emergencies and costing millions annually. This project investigates the temporal, geographic, and property-level factors that drive false alarms — and builds classification models to predict them.

**Research Question:** What factors drive false alarm callouts, and can they be predicted using temporal, geographic, and property-related characteristics?

## Methodology

- **Data Preparation:** Loaded and cleaned incident records from the London Datastore, handling missing values, converting data types, and selecting relevant features
- **Feature Engineering:** Datetime decomposition (hour, day of week), borough-level aggregation, property category encoding
- **Exploratory Analysis:** False alarm rate patterns across time of day, day of week, London boroughs, and property types
- **Modelling:** Decision Tree and Random Forest classifiers trained and evaluated to identify key predictors

## Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Decision Tree, Random Forest)
- Jupyter Notebook

## Dataset

London Fire Brigade incident data from 2024 onwards — sourced from the London Datastore.
