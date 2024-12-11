from deploy.fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load

app = FastAPI()

try:
    loaded_model = load('model.joblib')
    vectorizer = load('vectorizer.joblib')
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {e}")

class PredictionRequest(BaseModel):
    texts: list[str]

class PredictionResponse(BaseModel):
    predictions: list[int]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        text_vec = vectorizer.transform(request.texts)
        raw_predictions = loaded_model.predict(text_vec).tolist()
        value = 0
        for prediction in raw_predictions:
            if prediction == "Positive":
                value += 1 
            elif prediction == "Neutral":
                value += 0
            elif prediction == "Negative":
                value -= 1
            else:
                raise HTTPException(status_code=400, detail="Unknown sentiment label")
        
        value /= len(request.texts)

        return PredictionResponse(value=value)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health():
    return {"status": "ok"}
