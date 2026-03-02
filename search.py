from db import get_top_experiments, log_experiment
from train_model import train_xgboost

learning_rates = [0.01, 0.1, 0.2]
max_depths = [2, 3, 4]
n_estimators_list = [50, 80, 100]

for lr in learning_rates:
    for depth in max_depths:
        for n in n_estimators_list:
            hyperparams = {
                "learning_rate": lr,
                "max_depth": depth,
                "n_estimators": n
            }
            metrics = train_xgboost(hyperparams)
            log_experiment("xgboost", hyperparams, metrics)
            print(f"Experiment done: depth={depth} lr={lr} logloss={metrics['val_logloss']}")

top = get_top_experiments(limit=5)

print("\nTop Experiments (lowest logloss):")
for row in top:
    print(row)