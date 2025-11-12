
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
# import tensorflow as tf
# import numpy as np
# from typing import List

# #Initialize FastAPI app

# app = FastAPI(title="Hug Sentiment Analysis API", version="1.0")

# #Configure CORS

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "https://davinciai-sable.vercel.app",  # Update after deploying frontend
#        # "*"  # Remove in production
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# #Load model and tokenizer at startup
# MODEL_PATH = "./my_sentiment_model"
# model = None
# tokenizer = None

# @app.on_event("startup")
# async def load_model():
#     global model, tokenizer
#     try:
#         print("Loading model and tokenizer...")
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#         model = TFAutoModelForSequenceClassification.from_pretrained(
#             MODEL_PATH,
#             use_safetensors=False
#         )
#         print("Model loaded successfully!")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         raise

# #Request/Response models
# class TextInput(BaseModel):
#     text: str

# class PredictionResponse(BaseModel):
#     text: str
#     sentiment: str
#     confidence: float
#     probabilities: dict

# class BatchTextInput(BaseModel):
#     texts: List[str]

# class BatchPredictionResponse(BaseModel):
#     predictions: List[PredictionResponse]

# #Health check endpoint
# @app.get("/")
# async def root():
#     return {
#         "message": "Sentiment Analysis API ",
#         "status": "running",
#         "model_loaded": model is not None
#     }

# #Single prediction endpoint
# @app.post("/predict", response_model=PredictionResponse)
# async def predict_sentiment(input_data: TextInput):
#     if model is None or tokenizer is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")
    
#     try:
#         #Tokenize input
#         inputs = tokenizer(
#             input_data.text,
#             return_tensors="tf",
#             padding=True,
#             truncation=True,
#             max_length=128
#         )

#         #Get predictions
#         outputs = model(inputs)
#         probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

#         #Get predicted class and confidence
#         predicted_class = int(np.argmax(probabilities))
#         confidence = float(probabilities[predicted_class])

#         sentiment = "Positive" if predicted_class == 1 else "Negative"

#         return PredictionResponse(
#             text=input_data.text,
#             sentiment=sentiment,
#             confidence=confidence,
#             probabilities={
#                 "negative": float(probabilities[0]),
#                 "positive": float(probabilities[1])
#             }
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
# #Batch prediction endpoint
# @app.post("/predict-batch", response_model=BatchPredictionResponse)
# async def predict_batch(input_data: BatchTextInput):
#     if model is None or tokenizer is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")
    
#     try:
#         predictions = []

#         for text in input_data.texts:
#             #Tokenize input
#             inputs = tokenizer(
#                 text,
#                 return_tensors="tf",
#                 padding=True,
#                 truncation=True,
#                 max_length=128
#             )
#             #Get predictions
#             outputs = model(inputs)
#             probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

#             #Get predicted class and confidence
#             predicted_class = int(np.argmax(probabilities))
#             confidence = float(probabilities[predicted_class])

#             sentiment = "Positive" if predicted_class == 1 else "Negative"

#             predictions.append(
#                 PredictionResponse(
#                     text=text,
#                     sentiment=sentiment,
#                     confidence=confidence,
#                     probabilities={
#                         "negative": float(probabilities[0]),
#                         "positive": float(probabilities[1])
#                     }
#                 )
#             )
#         return BatchPredictionResponse(predictions=predictions)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
    
# #Model info endpoint
# @app.get("/model-info")
# async def model_info():
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")
    
#     return {
#         "model_path": MODEL_PATH,
#         "num_labels": model.config.num_labels,
#         "max_length": 128,
#         "model_type": model.config.model_type
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np
from typing import List
from pathlib import Path
from huggingface_hub import snapshot_download

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API")

#Configure CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://davinciai-sable.vercel.app",  # Update after deploying frontend
       # "*"  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = "./my_sentiment_model"
HF_MODEL_REPO = "Leonardo-0021/sentiment-model"  # CHANGE THIS

model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        print("Checking for local model...")
        
        # Check if model exists locally
        if not Path(MODEL_PATH).exists() or not any(Path(MODEL_PATH).iterdir()):
            print(f"Model not found locally. Downloading from Hugging Face: {HF_MODEL_REPO}")
            
            # Download model from Hugging Face
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                local_dir=MODEL_PATH,
                local_dir_use_symlinks=False
            )
            print("Model downloaded successfully!")
        else:
            print("Model found locally.")
        
        # Load model and tokenizer
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            use_safetensors=False
        )
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# Request/Response models
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict

class BatchTextInput(BaseModel):
    texts: List[str]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API",
        "status": "running",
        "model_loaded": model is not None,
        "model_source": HF_MODEL_REPO
    }

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(
            input_data.text,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        outputs = model(inputs)
        probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
        
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        
        return PredictionResponse(
            text=input_data.text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities={
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for text in input_data.texts:
            inputs = tokenizer(
                text,
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            outputs = model(inputs)
            probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
            
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            
            sentiment = "Positive" if predicted_class == 1 else "Negative"
            
            predictions.append(PredictionResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                probabilities={
                    "negative": float(probabilities[0]),
                    "positive": float(probabilities[1])
                }
            ))
        
        return BatchPredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Model info endpoint
@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": MODEL_PATH,
        "huggingface_repo": HF_MODEL_REPO,
        "num_labels": model.config.num_labels,
        "max_length": 128,
        "model_type": model.config.model_type
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


