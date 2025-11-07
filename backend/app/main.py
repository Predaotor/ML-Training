from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import JSONResponse 
import pandas as pd 
from app.models.rnn_model import rnn_model 
from app.preprocess import preprocess_data , sc
from io import StringIO
app=FastAPI() 

@app.post("/predict/rnn") 
async def precit_stock(file: UploadFile=File(None), text_data:str=Form(None)):
    try:
        if file:
            content=await file.read() 
            df=pd.read_csv(StringIO(content.decode("utf-8")), skiprows=[1,2])
            X_test, sc=preprocess_data(df)
            
        elif text_data:
            X_test=preprocess_data(text_data) 
        else:
            return JSONResponse(status_code=400, content={"error":"No input data provided"})
        
        
        # Make predictions 
        preds=rnn_model.predict(X_test) 
        preds=sc.inverse_transform(preds).flatten().tolist() 
        
        return {"predicted_prices":preds[-10:]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":str(e)})
    
@app.get("/")
def root():
    return {"message":"Welcome to the Google stock price Prediction API"}

