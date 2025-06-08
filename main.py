# adaptive_bci_ml_pipeline/main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Integer

# Load your data
# X = pd.read_csv('eeg_features.csv')
# y_freq = pd.read_csv('labels_freq.csv')
# y_score = pd.read_csv('target_scores.csv')
# groups = pd.read_csv('subject_ids.csv')

# Example mock input
np.random.seed(42)
X = pd.DataFrame(np.random.randn(500, 50), columns=[f"feat_{i}" for i in range(50)])
y_freq = np.random.choice([7, 9, 11, 13], size=500)
y_score = np.random.rand(500)
groups = np.repeat(np.arange(50), 10)

# Preprocess features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Classifier
clf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
clf.fit(X_scaled, y_freq)
print(f"Classifier Accuracy: {clf.score(X_scaled, y_freq):.2f}")

# Train Regressor
reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
reg.fit(X_scaled, y_score)
y_pred = reg.predict(X_scaled)
print(f"Regressor R^2: {r2_score(y_score, y_pred):.2f}")
print(f"Regressor RMSE: {mean_squared_error(y_score, y_pred, squared=False):.3f}")

# Feature Importance
importances = reg.feature_importances_
top_indices = np.argsort(importances)[-10:]
plt.barh(range(len(top_indices)), importances[top_indices])
plt.yticks(range(len(top_indices)), [X.columns[i] for i in top_indices])
plt.xlabel("Feature Importance")
plt.title("Top EEG Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Bayesian Optimization
def simulate_features_for_freq(freq):
    idx = y_freq == freq
    return X_scaled[idx].mean(axis=0)

def objective(freq_val):
    freq = freq_val[0]
    eeg_features = simulate_features_for_freq(freq)
    return -reg.predict([eeg_features])[0]

search_space = [Integer(7, 13, name='frequency')]
result = gp_minimize(objective, search_space, acq_func="EI", n_calls=10)
print(f"Optimal Frequency: {result.x[0]} Hz")

# LOSO-CV Evaluation
logo = LeaveOneGroupOut()
r2_scores = []
for train_idx, test_idx in logo.split(X_scaled, y_score, groups):
    reg.fit(X_scaled[train_idx], y_score[train_idx])
    y_pred = reg.predict(X_scaled[test_idx])
    r2_scores.append(r2_score(y_score[test_idx], y_pred))

print(f"LOSO-CV Mean R^2: {np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}")
