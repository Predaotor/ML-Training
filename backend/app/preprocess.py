import pandas as pd 
import joblib 
import numpy as np 
import os 
from io import StringIO
path=r"C:\Users\ladom\Desktop\LLm\Tensorflow\scaler.pk1"
sc=joblib.load(path)
# Define functions that handles data preprocessing 
def preprocess_data(input_data, lookback=60): 
    # Preprocess new stock  data for RNN model 
    
    if isinstance(input_data,str):
        if input_data.endswith(".csv"):
            df=pd.read_csv(input_data)
        elif input_data.strip().startswith("{") or  input_data.strip().startswith("["):
            df=pd.read_json(StringIO(input_data))
        else:
            # plain text 
            prices=[float(x) for x in input_data.replace("\n", ",").split(",")]
            df=pd.DataFrame(prices, columns=["Open"])
    elif isinstance(input_data, pd.DataFrame):
        df=input_data.copy() 
    else:
        raise ValueError ("Unsupported input type. Must be CSV path, JSON String, or text list ")
    
    # Try to ifnd the Open column 
    if "Open" not in df.columns:
        possible_columns=["Price", "price", "open", "close", "Close", "value", "Value"]
        found_col=next((col for col in possible_columns if col in df.columns), None )
        if not found_col:
            raise ValueError("No recognizable price column found in data")
        df["Open"]=df[found_col]
        
    # Get numpy array of open prices 
    
    inputs=df["Open"].values.reshape(-1,1) 
    inputs=sc.transform(inputs)
    
    # Prepare X_test for RNN 
    X_test=[] 
    for i in range(lookback, len(inputs)):
        X_test.append(inputs[i-lookback:i,0])
    X_test=np.array(X_test) 
    X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    
    return X_test, sc