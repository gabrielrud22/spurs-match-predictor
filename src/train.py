import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/processed/spurs_matches_model_ready.csv")

FEATURE_COLUMNS = ["rolling_points_5", "rolling_goals_scored_5", "rolling_goals_conceded_5", "rolling_goal_diff_5"]
TARGET_COLUMN = "result"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# the highest proportion here is the accuracy you'd get by always predicting the majority class
print("Class balance (baseline-beating threshold):")
print(df["result"].value_counts(normalize=True))

mlflow.set_experiment("spurs-match-predictor")

RUN_NAME = "rolling_form_randomforest_set_A_balanced_class_weight"

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("feature_set", "A_rolling_form")
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    # mlflow.log_param("scaler", "StandardScaler")

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    #model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    folds = (StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

    # cross_validate handles the train/test split and scoring per fold for us
    scores = cross_validate(model, X, y, cv=folds, scoring=["accuracy", "f1_macro"])

    mean_accuracy = scores["test_accuracy"].mean()
    mean_f1 = scores["test_f1_macro"].mean()

    mlflow.log_metric("mean_cv_accuracy", mean_accuracy)
    mlflow.log_metric("std_cv_accuracy", scores["test_accuracy"].std())
    mlflow.log_metric("mean_cv_f1_macro", mean_f1)

    # log a final model trained on ALL data for the artifact itself
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "model")

    print(f"Mean CV accuracy: {mean_accuracy:.3f}")
    print(f"Mean CV F1 (macro): {mean_f1:.3f}")

    # confusion matrix for just the first fold, to check the low F1 isn't the model
    # collapsing to predicting "win" every time
    train_idx, test_idx = next(folds.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    fold_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    #fold_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    fold_model.fit(X_train, y_train)
    fold_preds = fold_model.predict(X_test)

    labels = ["win", "draw", "loss"]
    cm = confusion_matrix(y_test, fold_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"predicted_{l}" for l in labels])

    print("Confusion matrix (fold 1):")
    print(cm_df)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title("Confusion matrix (fold 1)")
    cm_path = f"conf_matrix/confusion_matrix_fold_1_{RUN_NAME}.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)