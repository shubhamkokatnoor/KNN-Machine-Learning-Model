from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import uvicorn

app = FastAPI()

MyAi = joblib.load("./models/diebetes.pkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/knn/ai")

def diebetes_prediction(data:dict):
    param = [[
        data["Glucose"],
        data["BMI"],
        data["Age"]
    ]]

    res = MyAi.predict(param)[0]
    print(res)

    return{
        "Prediction":f"{res}"
    }