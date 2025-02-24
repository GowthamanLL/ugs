from flask import Flask, request, jsonify, send_file, render_template,send_from_directory, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import joblib
import io
import os
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




df = None

def process_data(file_path):
    global df
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)
    aqi_features = ['CO AQI', 'NO2 AQI', 'O3 AQI', 'PM10 AQI', 'PM2.5 AQI', 'SO2 AQI']
    df[aqi_features] = df[aqi_features].apply(pd.to_numeric, errors='coerce')
    
    def categorize_risk(aqi):
        if aqi <= 100:
            return "Low Risk"
        elif aqi <= 200:
            return "Moderate Risk"
        elif aqi <= 300:
            return "High Risk"
        else:
            return "Critical Risk"
    
    df['Risk_Level'] = df['AQI (Index)'].apply(categorize_risk)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        process_data(file_path)
        return redirect(url_for('cluster_analysis'))

@app.route('/cluster_analysis')
def cluster_analysis():
    return render_template('cluster_analysis.html')

@app.route('/cluster_map')
def cluster_map():
    global df
    df = process_data("C:\\Users\\Gokulnath\\Desktop\\flask\\uploaded_files\\coimbatore_pollution_no_aqi1.csv")
    if df is None:
        return "No data uploaded. Please upload a dataset first."
    
    X_cluster = df[['CO AQI', 'NO2 AQI', 'O3 AQI', 'PM10 AQI', 'PM2.5 AQI', 'SO2 AQI']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Pollution_Cluster'] = kmeans.fit_predict(X_cluster)
    
    cluster_map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    for _, row in df.iterrows():
        color = "green" if row['Pollution_Cluster'] == 2 else ("orange" if row['Pollution_Cluster'] == 1 else "red")
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            color=color,
            radius=10,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(cluster_map)
    
    cluster_map.save("static/cluster_map.html")
    return render_template('cluster_map.html')

@app.route('/high_risk_areas')
def high_risk_areas():
    global df
    if df is None:
        return "No data uploaded. Please upload a dataset first."
    
    high_risk_areas = df[df['Risk_Level'].isin(["High Risk", "Critical Risk"])]
    risk_map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    
    for _, row in high_risk_areas.iterrows():
        color = "red" if row['Risk_Level'] == "Critical Risk" else "orange"
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"AQI: {row['AQI (Index)']} | Risk: {row['Risk_Level']}",
            icon=folium.Icon(color=color)
        ).add_to(risk_map)
    
    risk_map.save("static/high_risk_areas.html")
    return render_template('high_risk_areas.html')

@app.route('/box_plot')
def box_plot():
    global df
    if df is None:
        return "No data uploaded. Please upload a dataset first."
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Pollution_Cluster'], y=df['AQI (Index)'], palette='coolwarm')
    plt.title("Comparing AQI Levels in Different Pollution Clusters")
    plt.xlabel("Pollution Cluster (0=High Pollution, 1=Medium, 2=Low Pollution)")
    plt.ylabel("AQI Index")
    plot_path = "static/box_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    return render_template('box_plot.html', plot_path=plot_path)


# Upload AQI Data
@app.route('/upload_aqi_data', methods=['POST'])
def upload_aqi_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    return jsonify({'message': 'AQI data uploaded successfully!', 'filename': filename})

# Alias route to support '/upload'
@app.route('/upload', methods=['POST'])
def upload():
    return upload_aqi_data()

# Predict AQI
@app.route('/predict_aqi', methods=['POST'])
def predict_aqi():
    try:
        data = request.json
        new_data = data['values']

        model = joblib.load("models/aqi_rf_best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        features = ['CO AQI', 'NO2 AQI', 'O3 AQI', 'PM10 AQI', 'PM2.5 AQI', 'SO2 AQI']
        labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Very Severe', 'Severe']

        new_data_df = pd.DataFrame(new_data, columns=features)
        new_data_scaled = scaler.transform(new_data_df)
        prediction = model.predict(new_data_scaled)

        category_mapping = {i: label for i, label in enumerate(labels)}
        predicted_category = [category_mapping[p] for p in prediction]

        return jsonify({'Predicted AQI Category': predicted_category})
    except Exception as e:
        print(f"Error in predict_aqi function: {e}")
        return jsonify({'error': f'Failed to predict AQI: {str(e)}'}), 500

# Function to ensure AQI_Category is computed
def process_aqi_categories(df):
    bins = [0, 50, 100, 200, 300, 400, 450, float('inf')]
    labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Very Severe', 'Severe']
    df['AQI_Category'] = pd.cut(df['AQI (Index)'], bins=bins, labels=labels)
    return df

@app.route('/all_aqi_visualization', methods=['GET'])
def all_aqi_visualization():
    try:
        latest_file = max([os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)], key=os.path.getctime)
        df = pd.read_csv(latest_file)
        print("Available columns:", df.columns)
        print(df.head())  # Debugging: Check if 'AQI (Index)' exists and has valid values

        if 'AQI_Category' not in df:
            df = process_aqi_categories(df)

        plt.figure(figsize=(10, 6))
        sns.countplot(x='AQI_Category', data=df, palette='coolwarm', 
                      order=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Very Severe', 'Severe'])
        plt.title("Distribution of All AQI Categories")
        plt.xlabel("AQI Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        image_path = os.path.join("static", "all_aqi_visualization.png")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        return render_template("visualization.html", img_file=image_path)
    except Exception as e:
        print(f"Error in all_aqi_visualization: {e}")
        return jsonify({'error': f'Failed to generate visualization: {str(e)}'}), 500

@app.route('/high_pollution_visualization', methods=['GET'])
def high_pollution_visualization():
    try:
        latest_file = max([os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)], key=os.path.getctime)
        df = pd.read_csv(latest_file)

        if 'AQI_Category' not in df:
            df = process_aqi_categories(df)

        high_pollution = df[df['AQI_Category'].isin(['Poor', 'Very Poor', 'Very Severe', 'Severe'])]

        plt.figure(figsize=(10, 6))
        sns.countplot(x='AQI_Category', data=high_pollution, palette='Reds')
        plt.title("Distribution of High Pollution Areas")
        plt.xlabel("AQI Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        image_path = os.path.join("static", "high_pollution_visualization.png")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        return render_template("visualization.html", img_file=image_path)
    except Exception as e:
        print(f"Error in high_pollution_visualization: {e}")
        return jsonify({'error': f'Failed to generate visualization: {str(e)}'}), 500
    
# Heatmap Visualization
@app.route('/heatmap_visualization', methods=['GET'])
def heatmap_visualization():
    try:
        latest_file = max(
            [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)],
            key=os.path.getctime
        )
        df = pd.read_csv(latest_file)
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
        heat_data = [[row['latitude'], row['longitude'], row['AQI (Index)']] for index, row in df.iterrows()]
        HeatMap(heat_data).add_to(m)

        heatmap_path = 'static/high_pollution_heatmap.html'
        m.save(heatmap_path)

        return send_file(heatmap_path)
    except Exception as e:
        print(f"Error in heatmap_visualization: {e}")
        return jsonify({'error': f'Failed to generate heatmap: {str(e)}'}), 500

@app.route('/download_heatmap')
def download_heatmap():
    return send_from_directory(directory="static", path="high_pollution_heatmap.html", as_attachment=True)

from flask import send_from_directory

@app.route('/download_cluster_map')
def download_cluster_map():
    return send_from_directory(directory="static", path="cluster_map.html", as_attachment=True)

@app.route('/download_high_risk')
def download_high_risk():
    return send_from_directory(directory="static", path="high_risk_areas.html", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)