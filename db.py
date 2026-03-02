import psycopg2
import json

def log_experiments(model_name, hyperparams, metrics):
    conn = psycopg2.connect(
        host="localhost",
        database="ml_hyperopt",
        user="postgres",
        password="kiit1280",
        port="5432",
    )

    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    hyperparameters JSONB NOT NULL,
    
    val_accuracy FLOAT,
    val_precision FLOAT,
    val_recall FLOAT,
    val_f1 FLOAT,
    val_roc_auc FLOAT,
    val_logloss FLOAT,
    
    best_iteration INT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""")

    cur.execute(
        """
        INSERT INTO experiments (model_name, hyperparameters, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_logloss)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """,
        (
            model_name,
            json.dumps(hyperparams),
            metrics.get("accuracy"),
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1"),
            metrics.get("roc_auc"),
            metrics.get("logloss"),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()