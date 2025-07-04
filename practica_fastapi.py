from fastapi import FastAPI
import pandas as pd
import time 
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

app = FastAPI()

@app.get('/Saluda')
def saluda(name: str): 
    return {'message': f'Hola {name}, encantado de conocerte!'}

@app.get('/Sumar')
def suma(a: int, b: int):
    return f"La suma de {a} más {b} da {a+b}"

@app.get('/Par')
def par(a: int):
    if a % 2 == 0:
        result = "par"
    else:
        result = "impar"
    return f" el número {a} es {result}"

@app.get("/Resumir")
def summarize_text(text: str):
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarization_pipeline(text, max_length=45, min_length=5, do_sample=False)
    return {"input": text, "summary": summary}

@app.get("/Sentimiento")
def sentiment_analysis(text: str):
    sentiment_pipeline = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    return {"input": text, "analisis": sentiment_pipeline(text)}
