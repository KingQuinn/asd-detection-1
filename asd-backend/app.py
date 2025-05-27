from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)  


SUPABASE_URL = "https://uunvtyrecichdaotqlkt.supabase.co"  #Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV1bnZ0eXJlY2ljaGRhb3RxbGt0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDczMzI2MTMsImV4cCI6MjA2MjkwODYxM30.gJ1WNO0U15m9BAVqAcTrTlchR7gNo_3VZAib2gA2-SQ" #Supabase anon Key  
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

with open("asd_detection2.pkl", "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoders = model_data["encoders"]
    svm_model = model_data["svm_model"]
    knn_model = model_data["knn_model"]
    dt_model = model_data["dt_model"]

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the ASD Detection APP!"})
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_data = {
        "A1": data["A1"], "A2": data["A2"], "A3": data["A3"], "A4": data["A4"], "A5": data["A5"],
        "A6": data["A6"], "A7": data["A7"], "A8": data["A8"], "A9": data["A9"], "A10": data["A10"],
        "Age": data["Age"], "Sex": data["Sex"], "Jauundice": data["Jaundice"], "Family_ASD": data["Family_ASD"]
    }
    new_df = pd.DataFrame([new_data])
    for col in ["Sex", "Jauundice", "Family_ASD"]:
        new_df[col] = label_encoders[col].transform(new_df[col].str.lower())
    new_df["Age"] = scaler.transform(new_df[["Age"]])
    svm_proba = svm_model.predict_proba(new_df)[:, 1]
    knn_proba = knn_model.predict_proba(new_df)[:, 1]
    dt_proba = dt_model.predict_proba(new_df)[:, 1]
    stacked_input = pd.DataFrame({'SVM': svm_proba, 'KNN': knn_proba, 'DecisionTree': dt_proba})
    prediction = model.predict(stacked_input)[0]
    result = "ASD Positive" if prediction == 1 else "ASD Negative"

    
    data_to_insert = {
        "A1": new_data["A1"], "A2": new_data["A2"], "A3": new_data["A3"], "A4": new_data["A4"],
        "A5": new_data["A5"], "A6": new_data["A6"], "A7": new_data["A7"], "A8": new_data["A8"],
        "A9": new_data["A9"], "A10": new_data["A10"], "Age": new_data["Age"], "Sex": new_data["Sex"],
        "Jaundice": new_data["Jauundice"], "Family_ASD": new_data["Family_ASD"], "Prediction": result
    }
    supabase.table('predictions').insert(data_to_insert).execute()

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)