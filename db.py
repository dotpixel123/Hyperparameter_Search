import psycopg2
import json
import os 

host = os.getenv("DB_HOST", "localhost")

def get_connection():
    conn = psycopg2.connect(
        host=host,
        database="ml_hyperopt",
        user="postgres",
        password="kiit1280",
        port="5432",
    )
    return conn

def create_table():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255),
            hyperparameters JSONB,
            val_accuracy FLOAT,
            val_precision FLOAT,
            val_recall FLOAT,
            val_f1 FLOAT,
            val_roc_auc FLOAT,
            val_logloss FLOAT,
            best_iteration INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    cur.close()
    conn.close()

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


def get_top_experiments(limit):

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

def get_experiment_by_id(exp_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT * FROM experiments
        WHERE id = %s
        """,
        (exp_id,),
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    return row

def get_experiments():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT * FROM experiments
        ORDER BY id DESC
        """,
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows