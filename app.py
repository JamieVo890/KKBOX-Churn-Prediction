from flask import Flask, request, render_template, redirect, url_for, send_file
from src.pipeline.prediction_pipeline import PredictPipeline
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():

   
    users = request.files.get('file1')
    members = request.files.get('file2')
    transactions = request.files.get('file3')

    # Check if all files are present and have a filename
    if (users is None or users.filename == '' or 
        members is None or members.filename == '' or 
        transactions is None or transactions.filename == ''):
        return redirect(request.url)
    
    prediction_pipeline = PredictPipeline(users, members, transactions)
    predictions = prediction_pipeline.predict()
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    predictions_df.to_csv("predictions.csv", index=False)
    return render_template('results.html', download_link=url_for('download_file', filename="predictions.csv"))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)