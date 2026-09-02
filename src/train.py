"""
┌─────────────────────────────────────────────────────────────────────┐
│  CHECKLIST — update before running this script for a new experiment │
└─────────────────────────────────────────────────────────────────────┘

  RUN_NUMBER / RUN_NAME
      Used for both the mlflow run name and the confusion matrix filename
      (conf_matrix/confusion_matrix_fold_1_run{RUN_NUMBER}_{RUN_NAME}.png).
      Bump RUN_NUMBER and give RUN_NAME a new, accurate description every
      run (and add an entry to training_diary.md under the same run number),
      or it'll overwrite the previous run's confusion matrix file and the
      names will end up describing the wrong model.

  model / fold_model
      Built separately (fold_model is a duplicate used just for the single
      held-out fold check) — if you change the model type or its params,
      change both, not just one.

  mlflow.log_param(...) calls
      Keep these matching whatever model/params you're actually using
      (feature_set, model_type, class_weight, etc.) — a stale logged param
      is misleading later when comparing runs in MLflow.

  FEATURE_COLUMNS
      Update if switching to a different feature set (e.g. Set B), and
      update the "feature_set" param to match.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("data/processed/spurs_matches_feature_set_b.csv")

FEATURE_COLUMNS = ["rolling_points_5", "rolling_goals_scored_5", "rolling_goals_conceded_5", "rolling_goal_diff_5", "h2h_win_rate", "h2h_goal_diff", "h2h_prior_matches_played"]
TARGET_COLUMN = "result"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# the highest proportion here is the accuracy you'd get by always predicting the majority class
print("Class balance (baseline-beating threshold):")
print(df["result"].value_counts(normalize=True))

mlflow.set_experiment("spurs-match-predictor")

RUN_NUMBER = 10
RUN_NAME = "set_B_xgboost_base"

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("feature_set", "B_rolling+h2h")
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("random_state", 42)
    # mlflow.log_param("class_weight", "balanced")

    """not used for random forest; determines how many iterations to run for logreg"""
    # mlflow.log_param("scaler", "StandardScaler")
    
    """not used for logreg; determines how many trees in the forest for random forest or xgboost"""
    mlflow.log_param("n_estimators", 100)

    """XGBoost"""
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
    """Random Forest"""
    #model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    """Logistic Regression"""
    #model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))

    folds = (StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

    # cross_validate handles the train/test split and scoring per fold for us
    # scores = cross_validate(model, X, y, cv=folds, scoring=["accuracy", "f1_macro"])
    scores = cross_validate(model, X, y_encoded, cv=folds, scoring=["accuracy", "f1_macro"])

    mean_accuracy = scores["test_accuracy"].mean()
    mean_f1 = scores["test_f1_macro"].mean()

    mlflow.log_metric("mean_cv_accuracy", mean_accuracy)
    mlflow.log_metric("std_cv_accuracy", scores["test_accuracy"].std())
    mlflow.log_metric("mean_cv_f1_macro", mean_f1)

    # log a final model trained on ALL data for the artifact itself
    model.fit(X, y_encoded)
    mlflow.xgboost.log_model(model, "model")
    # model.fit(X, y)
    # mlflow.sklearn.log_model(model, "model")

    print(f"Mean CV accuracy: {mean_accuracy:.3f}")
    print(f"Mean CV F1 (macro): {mean_f1:.3f}")

    # confusion matrix for just the first fold, to check the low F1 isn't the model
    # collapsing to predicting "win" every time
    # train_idx, test_idx = next(folds.split(X, y))
    # X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_idx, test_idx = next(folds.split(X, y_encoded))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_encoded, y_test_encoded = y_encoded[train_idx], y_encoded[test_idx]
    y_test = y.iloc[test_idx]

    """XGBoost"""
    fold_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
    fold_model.fit(X_train, y_train_encoded)
    fold_preds_encoded = fold_model.predict(X_test)
    fold_preds = label_encoder.inverse_transform(fold_preds_encoded)

    """RandomForest"""
    # fold_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    """LogReg"""
    #fold_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    # fold_model.fit(X_train, y_train)
    # fold_preds = fold_model.predict(X_test)

    labels = ["win", "draw", "loss"]
    cm = confusion_matrix(y_test, fold_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"predicted_{l}" for l in labels])

    print("Confusion matrix (fold 1):")
    print(cm_df)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title("Confusion matrix (fold 1)")
    cm_path = f"conf_matrix/confusion_matrix_fold_1_run{RUN_NUMBER}_{RUN_NAME}.png"
    plt.savefig(cm_path)
    plt.close()

    report = classification_report(y_test, fold_preds, labels=labels, output_dict=True)
    for label in labels:
        mlflow.log_metric(f"f1_{label}", report[label]["f1-score"])
        mlflow.log_metric(f"precision_{label}", report[label]["precision"])
        mlflow.log_metric(f"recall_{label}", report[label]["recall"])

    mlflow.log_artifact(cm_path)