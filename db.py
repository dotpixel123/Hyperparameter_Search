import psycopg2
import json

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="ml_hyperopt",
        user="postgres",
        password="kiit1280",
        port="5432",
    )


def log_experiment(model_name, hyperparams, metrics):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO experiments 
        (model_name, hyperparameters, val_accuracy, val_precision, 
         val_recall, val_f1, val_roc_auc, val_logloss, best_iteration)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            model_name,
            json.dumps(hyperparams),
            metrics.get("val_accuracy"),
            metrics.get("val_precision"),
            metrics.get("val_recall"),
            metrics.get("val_f1"),
            metrics.get("val_roc_auc"),
            metrics.get("val_logloss"),
            metrics.get("best_iteration"),
        ),
    )

    conn.commit()
    cur.close()
    conn.close()


def get_top_experiments(limit=5):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT * FROM experiments
        ORDER BY val_logloss ASC
        LIMIT %s
        """,
        (limit,),
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows