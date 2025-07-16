from fastapi import FastAPI
from pydantic import BaseModel
import joblib #model lai load garna ko lagi joblib
import numpy as np

app = FastAPI()  #app bhane object banayeko, yehi ibject bata route banaune ho
Model = joblib.load('saved/DecisionTreePrediction.pkl')

#Glucose
#BloodPressure 
#DiabetesPedigreeFunction (dfp)
#Age

class Patient(BaseModel): #data banako ra data validation ko lagi pydentic use gareko ho
    Glucose: float
    BloodPressure: int
    dfp : float
    Age : int 
    #yo data laii model ma pass garnu parchha aaba. 

@app.get('/') #@ ley route banaune
def show():
    return {'message':'Hello Guys!!'}


@app.get('/about')
def about():
    return {'Display':'Broad AI is a education sector'}


@app.post('/predict')
async def predict(data: Patient):
    input_data = np.array([[
        data.Glucose, data.BloodPressure, data.dfp, data.Age
    ]])

    pred = await Model.predict(input_data)[0]
    return {'prediction':int(pred)}

    