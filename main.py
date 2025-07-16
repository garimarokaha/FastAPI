from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('Saved/DecisionTreePrediction.pkl')

templates = Jinja2Templates(directory="templates")

class Patient(BaseModel):
    gulcose: float
    bloodpressure: int
    dfp: float
    age: int

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about")
def about():
    return {"Display": "Broad AI is the educational sector"}

@app.post("/predict")
async def predict(data: Patient):
    input_data = np.array([[data.gulcose, data.bloodpressure, data.dfp, data.age]])
    prediction = model.predict(input_data)[0]
    return {"prediction": int(prediction)}