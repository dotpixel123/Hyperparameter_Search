import fastapi 
from db import create_table, get_top_experiments, get_experiment_by_id, get_experiments

app = fastapi.FastAPI()

@app.on_event("startup")
def startup_event():
    create_table()

@app.get("/top-experiments")
def top_experiment(limit: int = 1):     
    return get_top_experiments(limit=limit)

@app.get("/experiment/{exp_id}")
def get_experiment(exp_id: int):
    return get_experiment_by_id(exp_id)

@app.get("/experiments")
def experiments():
    return get_experiments()