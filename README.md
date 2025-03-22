# FilmModelComparison

## Description
This project focuses on comparing the performance of various machine learning models for two tasks: regression of movie ratings (predicting a numerical score) and multilabel classification of movie genres. The goal is to identify the most effective models for each task based on key performance metrics such as MAE, RMSE, and R² for regression, and ROC-AUC, Precision, Recall, and F1 Score for classification.

The data analysis and preprocessing are conducted in the `films_analysis.ipynb` notebook, preparing features for model training. Potential applications include audience preference analysis, movie recommendation systems, or automated content tagging.

## Tools and Technologies
- **Language**: Python
- **Libraries**: scikit-learn, XGBoost, pandas, numpy, matplotlib
- **Data**: [48,000+ movies dataset](https://www.kaggle.com/datasets/yashgupta24/48000-movies-dataset)
- **Environment**: Jupyter Notebook 

## Movie Rating Regression

### Target Variable
Movie Rating — a numerical score ranging from 0 to 10.

### Models
The following machine learning models were evaluated:
- **Const(mean)**: Baseline model predicting the mean rating.
- **Ridge (L2-reg)**: Linear regression with L2 regularization.
- **Decision Tree**: Single decision tree.
- **Random Forest**: Ensemble of decision trees.
- **XGBoost**: Gradient boosting.

### Results
| Model          | MAE    | RMSE   | R² Score |
|----------------|--------|--------|----------|
| Const(mean)    | 0.873  | 1.130  | 0.000   |
| Ridge (L2-reg) | 0.712  | 0.936  | 0.313    |
| Decision Tree  | 0.649  | 0.883  | 0.388    |
| Random Forest  | 0.605  | 0.833  | 0.457    |
| XGBoost        | 0.594  | 0.815  | 0.480    |

**Insights**: XGBoost outperformed other models, achieving the lowest error (MAE and RMSE) and the highest R² score (0.480).

## Genre Multilabel Classification

### Target Variable
Movie Genres — a multilabel classification task where a movie can belong to multiple genres simultaneously.

### Approach
A separate binary classifier was trained for each genre due to overlapping labels.

### Models
- **Logistic Regression**: Logistic regression classifier.
- **SVM**: Support Vector Machine.
- **Decision Tree**: Single decision tree.
- **Random Forest**: Ensemble of decision trees.
- **XGBoost**: Gradient boosting.

### Results
Best-performing models and metrics for each genre:

| Genre       | Best Model    | ROC-AUC | Precision | Recall | PR-AUC | F1 Score |
|-------------|---------------|---------|-----------|--------|--------|----------|
| Action      | Random Forest | 0.721   | 0.222     | 0.815  | 0.335  | 0.484    |
| Adventure   | XGBoost       | 0.708   | 0.208     | 0.839  | 0.322  | 0.472    |
| Animation   | XGBoost       | 0.727   | 0.226     | 0.817  | 0.338  | 0.489    |
| Biography   | XGBoost       | 0.711   | 0.208     | 0.842  | 0.324  | 0.473    |
| Comedy      | Random Forest | 0.642   | 0.352     | 0.978  | 0.479  | 0.671    |
| Crime       | XGBoost       | 0.707   | 0.213     | 0.800  | 0.322  | 0.468    |
| Documentary | Random Forest | 0.935   | 0.539     | 0.712  | 0.631  | 0.658    |
| Drama       | Random Forest | 0.724   | 0.558     | 0.972  | 0.716  | 0.816    |
| Family      | XGBoost       | 0.708   | 0.223     | 0.761  | 0.323  | 0.470    |
| Fantasy     | XGBoost       | 0.721   | 0.225     | 0.798  | 0.336  | 0.483    |
| Horror      | Random Forest | 0.848   | 0.343     | 0.737  | 0.476  | 0.569    |
| Mystery     | XGBoost       | 0.724   | 0.216     | 0.845  | 0.334  | 0.484    |
| Romance     | Random Forest | 0.679   | 0.250     | 0.856  | 0.332  | 0.528    |
| Sci-Fi      | XGBoost       | 0.717   | 0.227     | 0.768  | 0.335  | 0.476    |
| Thriller    | Random Forest | 0.726   | 0.299     | 0.823  | 0.363  | 0.568    |

**Insights**: Random Forest and XGBoost were the top performers across most genres. The documentary and the drama received the highest ratings, probably because of the greater amount of data.
