from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
from car_data_prep import prepare_data

app = Flask(__name__)

# Load the trained model
en_model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_dict = {
        'manufactor': request.form.get('manufactor', ''),
        'Year': request.form.get('year', ''),
        'model': request.form.get('model', ''),
        'Hand': request.form.get('hand', ''),
        'Gear': request.form.get('gear', ''),
        'capacity_Engine': request.form.get('capacity_engine', ''),
        'Engine_type': request.form.get('engine_type', ''),
        'Prev_ownership': request.form.get('prev_ownership', ''),
        'Curr_ownership': request.form.get('curr_ownership', ''),
        'City': request.form.get('city', ''),
        'Color': request.form.get('color', ''),
        'Km': request.form.get('km', ''),
        'Test': request.form.get('test', ''),
        'Area': pd.NA,
        'Pic_num': pd.NA,
        'Cre_date': pd.NA,
        'Repub_date': pd.NA,
        'Description': 'Description',
        'Supply_score': 0
    }

    # Debugging to check the received data
    print("Received form data:")
    for key, value in feature_dict.items():
        print(f'{key}: {value}')
    
    final_features = pd.DataFrame([feature_dict])
    
    # Debugging to check the DataFrame before processing
    print("Final Features DataFrame:")
    print(final_features)
    
    try:
        processed_data = prepare_data(final_features, encoder_path='encoder.pkl', scaler_path='scaler.pkl', fit=False)
        # Debugging to check the processed data
        print("Processed Data DataFrame:")
        print(processed_data)
        print(processed_data['Year'])
        prediction = en_model.predict(processed_data)
        # Debugging to check the prediction
        print("Prediction:")
        print(prediction)
    except Exception as e:
        print(f"Error during data preparation or prediction: {e}")
        return render_template('index.html', prediction_text='An error occurred during prediction.')

    output_text = f'המחיר לרכב הוא: {int(prediction[0])} ש"ח'
    
    return render_template('index.html', prediction_text=output_text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
