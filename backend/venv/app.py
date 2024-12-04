import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from modeldef import LogisticRegression
import pickle
import logging
import os
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

modelPATH = os.path.join(BASE_DIR, "models", "model_params.pkl")
originalDS = os.path.join(BASE_DIR, "DSLabeled.xls")
corrected_preds = os.path.join(BASE_DIR, "correctedPreds.xls")
wrongpreds = os.path.join(BASE_DIR, "WrongPreds.xls")
scalerPATH = os.path.join(BASE_DIR, "models", "scaler (1).pkl")

try:
    model = LogisticRegression(
        learning_rate=0.975,
        num_iter=6500,
        add_bias=True,
        verbose=False,
        reg_strength=0.1,
        class_weight={0: 1, 1: 2},
    )

    
    with open(modelPATH, "rb") as f:
        model_params = pickle.load(f)

    model.theta = model_params["theta"]
    model.learning_rate = model_params["learning_rate"]
    model.num_iter = model_params["num_iter"]
    model.add_bias = model_params["add_bias"]
    model.reg_strength = model_params["reg_strength"]
    model.class_weight = model_params["class_weight"]

    logging.info("Model initialized and parameters loaded successfully!")
except Exception as e:
    logging.error(f"Error loading the model: {str(e)}")
    model = None

feedbackCounter = 0

def val_input(data):
    required_fields = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise KeyError(f"Missing fields: {', '.join(missing)}")
    return True


#preprocess input data
def preprocess_input(data):

    scaler = joblib.load(scalerPATH)
    
    oneHotCat = {
        "GenHlth": [1, 2, 3, 4, 5],  # Example categories
        "MentHlth": list(range(0, 31)),  # Days from 1 to 30
        "PhysHlth": list(range(0, 31)),  # Days from 1 to 30
        "Age": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # Age groups
        "Education": [1, 2, 3, 4, 5, 6],  # Education levels
        "Income": [1, 2, 3, 4, 5, 6, 7, 8],  # Income levels
    }

    features = {
        "HighBP": int(data['HighBP']),
        "HighChol": int(data['HighChol']),
        "CholCheck": int(data['CholCheck']),
        "BMI": float(data['BMI']),
        "Smoker": int(data['Smoker']),
        "Stroke": int(data['Stroke']),
        "HeartDiseaseorAttack": int(data['HeartDiseaseorAttack']),
        "PhysActivity": int(data['PhysActivity']),
        "Fruits": int(data['Fruits']),
        "Veggies": int(data['Veggies']),
        "HvyAlcoholConsump": int(data['HvyAlcoholConsump']),
        "AnyHealthcare": int(data['AnyHealthcare']),
        "NoDocbcCost": int(data['NoDocbcCost']),
        "GenHlth": int(data['GenHlth']),
        "MentHlth": int(data['MentHlth']),
        "PhysHlth": int(data['PhysHlth']),
        "DiffWalk": int(data['DiffWalk']),
        "Sex": int(data['Sex']),
        "Age": int(data['Age']),
        "Education": int(data['Education']),
        "Income": int(data['Income']),
    }

    allFeat = pd.DataFrame([features])
    logging.info(f"Raw features DataFrame:\n{allFeat}")

    # Normalize BMI
    cols_to_normalize = ['BMI']  
    allFeat[cols_to_normalize] = scaler.transform(allFeat[cols_to_normalize])
    logging.info(f"After BMI normalization:\n{allFeat[['BMI']]}")

    # OHE
    for col, categories in oneHotCat.items():
        for category in categories[1:]:
            col_name = f"{col}_{category}"
            allFeat[col_name] = (allFeat[col] == category).astype(int)

    allFeat = allFeat.drop(columns=oneHotCat.keys(), errors='ignore')

    logging.info(f"One-hot encoded DataFrame:\n{allFeat}")
 
    processed_features = allFeat.values
    logging.info(f"Processed features shape: {processed_features.shape}")
    return processed_features

def retrainModel(df):
    """
    Retrain the model using the updated dataset.
    """ 
    try:
        X = df.drop('Diabetes_binary', axis=1).values
        y = df['Diabetes_binary'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training retraining
        logging.info("Model just started retraining!!...")
        model.train(X_train, y_train)
        
        model_params = {
                "theta": model.theta,
                "learning_rate": model.learning_rate,
                "num_iter": model.num_iter,
                "add_bias": model.add_bias,
                "reg_strength": model.reg_strength,
                "class_weight": model.class_weight,
            }
        
        y_pred = model.predict(X_test)
        precision, recall, accuracy, f1 = model.evaluate(y_test, y_pred)
        
        print(f"Precision = {precision:.2f}")
        print(f"Recall = {recall:.2f}")
        print(f"Accuracy = {accuracy:.2f}")
        print(f"F1 = {f1:.2f}")
        
        with open(modelPATH, "wb") as f:
                pickle.dump(model_params, f)

        logging.info("Model retrained and parameters saved successfully!")
        

    except Exception as e:
            logging.error(f"Error during retraining: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the model file path and reload the app.'}), 500

    try:
        data = request.json
        logging.info(f"Data received: {data}")

        val_input(data)

        features_preprocessed = preprocess_input(data)
        logging.info(f"Preprocessed features:\n{features_preprocessed}")

        prediction = model.predict(features_preprocessed)[0]
        is_diabetic = int(prediction)
        prediction_label = "Prediabetes or Diabetes" if is_diabetic == 1 else "No Diabetes"

        logging.info(f"Prediction result: {prediction_label}")

        return jsonify({'prediction': is_diabetic, 'label': prediction_label})

    except KeyError as e:
        logging.error(f"KeyError: {str(e)}")
        return jsonify({'error': f'Missing or invalid field: {str(e)}'}), 400
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


#Feedback IMPLMENT 
@app.route('/feedback', methods=['POST'])
def feedback():
    global feedbackCounter
    try:
        feedbackCounter += 1
        
        logging.info("Feedback recieved")
           
        data = request.json
        features = data['features']
        falseLabel = int(data['Falselabel'])  #  incorrect label
        trueLabel = int(data['TrueLabel'])    #  true label
        isCorrect = data['correct']
        
        processed_features = preprocess_input(features)
        
        originaldf = pd.read_csv(originalDS)
        originaldf['Diabetes_binary'] = originaldf['Diabetes_binary'].astype(int)
        originaldf.to_csv(originalDS, index=False)
 
        featureCols = pd.read_csv(originalDS, nrows=1).columns.tolist() 
        
        feedbackdf = pd.DataFrame(processed_features, columns=featureCols[1:])
                
        for col in feedbackdf.columns:
            if col != 'BMI':  
                feedbackdf[col] = feedbackdf[col].astype(int)
     
        if isCorrect:
            feedbackdf.insert(0, 'Diabetes_binary', trueLabel) 
            feedbackdf.to_csv(originalDS, mode='a', header=False, index=False)
            logging.info("Correct feedback saved to the dataset.")
        else:
            wrong_pred_df = feedbackdf.copy()
            wrong_pred_df.insert(0, 'Diabetes_binary', falseLabel) 
            wrong_pred_df.to_csv(wrongpreds, mode='a', header=False, index=False)
            logging.info("Original incorrect prediction logged.")

            corrected_pred_df = feedbackdf.copy()
            corrected_pred_df.insert(0, 'Diabetes_binary', trueLabel)
            corrected_pred_df.to_csv(corrected_preds, mode='a', header=False, index=False)
            feedbackdf.insert(0, 'Diabetes_binary', trueLabel)
            feedbackdf.to_csv(originalDS, mode='a', header=False, index=False)
            logging.info("Corrected prediction saved.")
            
        updateddf = pd.concat([originaldf, feedbackdf], ignore_index=True)
        
        # print("\nUpdated DataFrame:")
        # print(updated_df)
        print(f"\nFeedback count: {feedbackCounter}")

        if feedbackCounter == 10:
            logging.info("Retraining triggered after receiving 10 feedback entries.")
            retrainModel(updateddf)
            feedbackCounter = 0 

        return jsonify({'message': 'Feedback processed successfully!'}), 200

    except Exception as e:
        logging.error(f"Error saving feedback: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
