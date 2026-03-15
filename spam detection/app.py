from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
preprocess = joblib.load("preprocess.pkl")

@app.get("/")
def home():
    return {"message": "Spam Detection API is running"}

@app.post("/predict")
def predict(text: str):

    clean_text = preprocess(text)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)

    return {"prediction": prediction[0]}