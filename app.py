from flask import Flask, request, jsonify, send_file, render_template,send_from_directory, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import joblib
import io
import os
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import io
import base64
import geopandas as gpd
import libpysal
import esda
import splot.esda
import matplotlib.pyplot as plt
import contextily as ctx
import libpysal
from libpysal.weights import Queen
from esda.moran import Moran
from libpysal.weights import Queen, KNN
from esda.moran import Moran
from esda.getisord import G_Local
from sklearn.cluster import KMeans
from shapely.geometry import Point
import os
from splot.esda import plot_moran
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW


app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_files'
RESULT_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
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
    df = process_data("uploaded_files\coimbatore_pollution_no_aqi1.csv")
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

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

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
        df = pd.read_csv("uploaded_files\coimbatore_pollution_no_aqi1.csv")
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
        df = pd.read_csv("uploaded_files\coimbatore_pollution_no_aqi1.csv")

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
        df = pd.read_csv("uploaded_files\coimbatore_pollution_no_aqi1.csv")
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

# Load the pre-trained model
model = tf.keras.models.load_model('models/aqi_model.h5')

# Features and targets used in training
FEATURES = ['pm2.5 (µg/m³)', 'pm10 (µg/m³)', 'Temperature (°C)', 'Humidity (%)', 'Wind_Speed (km/h)', 'NO2 (µg/m³)', 'SO2 (µg/m³)', 'CO (mg/m³)', 'O3 (µg/m³)']
TARGETS = ['AQI', 'pm2.5 (µg/m³)', 'pm10 (µg/m³)', 'NO2 (µg/m³)', 'SO2 (µg/m³)', 'CO (mg/m³)', 'O3 (µg/m³)']

def preprocess_data(data):
    """
    Prepares data for LSTM model by scaling it and structuring it into sequences.
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    data.fillna(method='ffill', inplace=True)  # Forward fill missing values

    scaled_features = feature_scaler.fit_transform(data[FEATURES])
    scaled_targets = target_scaler.fit_transform(data[TARGETS])

    X = []
    for i in range(60, len(scaled_features)):
        X.append(scaled_features[i-60:i, :])

    return np.array(X), feature_scaler, target_scaler

def predict_future(data, feature_scaler, target_scaler, days=5):
    """
    Uses the trained LSTM model to predict AQI for the next 'days' days.
    """
    last_60_days = data[-60:]
    predictions = []

    for _ in range(days):
        last_60_scaled = feature_scaler.transform(last_60_days)
        last_60_scaled = np.reshape(last_60_scaled, (1, 60, last_60_scaled.shape[1]))

        predicted_scaled = model.predict(last_60_scaled)
        predicted_values = target_scaler.inverse_transform(predicted_scaled)

        predictions.append(predicted_values[0])
        next_day_features = np.hstack([predicted_values[0], last_60_days[-1, 3:]])
        last_60_days = np.vstack([last_60_days[1:], next_day_features[:9]])

    return np.array(predictions)

@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    data = pd.read_csv(file)
    return jsonify({"message": "File processed successfully", "data_preview": data.head().to_dict()}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Read uploaded file
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'])

    # Extract last 3 months for prediction
    last_3_months = data[data['Date'] >= data['Date'].max() - pd.DateOffset(months=3)]

    # Preprocess data
    X, feature_scaler, target_scaler = preprocess_data(last_3_months)

    # Predict next 5 days
    predictions = predict_future(last_3_months[FEATURES].values, feature_scaler, target_scaler, days=5)

    # Generate prediction dates
    prediction_dates = pd.date_range(last_3_months['Date'].max() + pd.Timedelta(days=1), periods=5).strftime('%Y-%m-%d')

    # Create structured prediction results
    prediction_results = [{"date": date, "aqi": round(pred[0], 2)} for date, pred in zip(prediction_dates, predictions)]

    # Save actual data
    data[['Date', 'AQI']].to_csv("latest_aqi_data.csv", index=False)

    # Save predictions
    prediction_df = pd.DataFrame(prediction_results)
    prediction_df.to_csv("latest_predictions.csv", index=False)

    return render_template('predictions.html', predictions=prediction_results)




@app.route('/aqi-trends')
def aqi_trends():
    # Load AQI data
    data = pd.read_csv("latest_aqi_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter last 3 months
    last_3_months = data[data['Date'] >= data['Date'].max() - pd.DateOffset(months=3)]

    # Create a visually appealing Seaborn style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Customizing the AQI trend line
    plt.plot(last_3_months['Date'], last_3_months['AQI'], marker='o', linestyle='-', color="#ff5733", markersize=6, linewidth=2, label="AQI")
    
    # Formatting the graph
    plt.xlabel("Date", fontsize=13, fontweight='bold', color="#333")
    plt.ylabel("AQI Level", fontsize=13, fontweight='bold', color="#333")
    plt.title("AQI Trend Over Last 3 Months", fontsize=16, fontweight='bold', color="#222")
    plt.xticks(rotation=45, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Adding a shaded background for better readability
    plt.gca().set_facecolor("#f4f4f4")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend with custom style
    plt.legend(loc="upper right", fontsize=12, frameon=True, shadow=True, facecolor="white", edgecolor="black")

    # Save the plot
    img_path = "static/aqi_trend.png"
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    plt.close()

    return render_template("aqi_trends.html", img_path=img_path)


@app.route('/aqi-predictions')
def aqi_prediction():
    # Load actual AQI data (past 3 months)
    aqi_data = pd.read_csv("latest_aqi_data.csv")
    aqi_data.columns = aqi_data.columns.str.strip()  # Remove any unwanted spaces
    aqi_data['Date'] = pd.to_datetime(aqi_data['Date'])

    # Filter last 3 months of data
    last_3_months = aqi_data[aqi_data['Date'] >= aqi_data['Date'].max() - pd.DateOffset(months=3)]

    # Load predicted AQI data (next 5 days)
    predicted_data = pd.read_csv("latest_predictions.csv")
    predicted_data.columns = predicted_data.columns.str.strip()  # Remove spaces
    predicted_data.rename(columns={"date": "Date", "aqi": "AQI"}, inplace=True)
    predicted_data['Date'] = pd.to_datetime(predicted_data['Date'])

    # Combine actual and predicted AQI data
    full_data = pd.concat([last_3_months, predicted_data], ignore_index=True)

    # Apply Seaborn style
    sns.set_style("darkgrid")
    plt.figure(figsize=(14, 6))

    # Plot actual AQI values
    plt.plot(last_3_months['Date'], last_3_months['AQI'], marker='o', linestyle='-', color="#2a9df4",
             markersize=5, linewidth=2, label="Actual AQI")

    # Plot predicted AQI values
    plt.plot(predicted_data['Date'], predicted_data['AQI'], marker='x', linestyle='--', color="#ff3333",
             markersize=8, linewidth=2, label="Predicted AQI")

    # Customize the plot
    plt.xlabel("Date", fontsize=14, fontweight='bold', color="#222")
    plt.ylabel("AQI Level", fontsize=14, fontweight='bold', color="#222")
    plt.title("AQI Trends (Past 3 Months) & Predictions (Next 5 Days)", fontsize=16, fontweight='bold', color="#111")

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Add legend
    plt.legend(loc="upper right", fontsize=12, frameon=True, shadow=True, facecolor="white", edgecolor="black")

    # Save the visualization
    img_path = "static/aqi_prediction.png"
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    plt.close()

    return render_template("aqi_predictions.html", img_path=img_path)




from flask import Flask, render_template, request
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from shapely.geometry import Point
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
from geneticalgorithm import geneticalgorithm as ga
from sklearn.cluster import KMeans



# Load Pollution Data
pollution_data = pd.read_csv("./uploaded_files/coimbatore_pollution_with_regions_updated.csv")
pollution_data["geometry"] = pollution_data.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
pollution_data = gpd.GeoDataFrame(pollution_data, geometry="geometry")
# Check for required columns
required_cols = {'pollution_level', 'latitude', 'longitude'}
if not required_cols.issubset(pollution_data.columns):
    raise KeyError(f"Missing columns: {required_cols - set(pollution_data.columns)}")

coords = np.array(list(zip(pollution_data.geometry.x, pollution_data.geometry.y)))

factories = gpd.read_file("uploaded_files/coimbatore_factories.geojson")
roads = gpd.read_file("uploaded_files/coimbatore_roads.geojson")

# Compute Distance to Nearest Factory
# Ensure factory geometries are points (take centroids if necessary)
if not all(factories.geometry.type == "Point"):
    factories["geometry"] = factories.geometry.centroid

dist_matrix_factories = distance_matrix(coords, np.array(list(zip(factories.geometry.x, factories.geometry.y))))
pollution_data["nearest_factory_distance"] = np.min(dist_matrix_factories, axis=1)

# Ensure road geometries are points
if not all(roads.geometry.type == "Point"):
    roads["geometry"] = roads.geometry.centroid

# Compute Distance to Nearest Road
dist_matrix_roads = distance_matrix(coords, np.array(list(zip(roads.geometry.x, roads.geometry.y))))
pollution_data["nearest_road_distance"] = np.min(dist_matrix_roads, axis=1)

# -------------------------------
# Step 6: Geographically Weighted Regression (GWR)
# -------------------------------
X = pollution_data[['nearest_factory_distance', 'nearest_road_distance']].values
y = pollution_data[['pollution_level']].values

# Handle Zero Variance Columns
zero_var_cols = [col for col in pollution_data[['nearest_factory_distance', 'nearest_road_distance']].columns if pollution_data[col].nunique() == 1]
if zero_var_cols:
    print(f"Dropping zero-variance columns: {zero_var_cols}")
    pollution_data.drop(columns=zero_var_cols, inplace=True)

w = KNN.from_dataframe(pollution_data, k=5)
w.transform = 'r'

# Run GWR
bw = Sel_BW(coords, y, X).search()
gwr_model = GWR(coords, y, X, bw)
gwr_results = gwr_model.fit()
print(gwr_results.summary())

# Load Green Space Suitability Data
green_space_data = pd.read_csv("./uploaded_files/coimbatore_green_space_suitability_scores - coimbatore_green_space_suitability_scores.csv")
scaler = MinMaxScaler()
green_space_data[['green_space_suitability_score']] = scaler.fit_transform(green_space_data[['green_space_suitability_score']])

# Optimization function
def find_optimal_sites():
    best_score = -1
    best_k = 3

    for k in range(3, 10):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(green_space_data[['latitude', 'longitude']])
        score = kmeans.inertia_
        if score < best_score or best_score == -1:
            best_score = score
            best_k = k

    green_space_data['cluster'] = KMeans(n_clusters=best_k, random_state=42).fit_predict(green_space_data[['latitude', 'longitude']])

    def fitness_function(X):
        selected_sites = green_space_data.iloc[X.astype(int)].drop_duplicates()
        coords = np.array(list(zip(selected_sites['latitude'], selected_sites['longitude'])))
        dist_penalty = np.mean(distance_matrix(coords, coords)) if len(coords) > 1 else 0
        return -selected_sites['green_space_suitability_score'].sum() + 0.1 * dist_penalty

    varbound = np.array([[0, len(green_space_data)-1]] * 10)

    model = ga(
        function=fitness_function,
        dimension=10,
        variable_type='int',
        variable_boundaries=varbound,
        algorithm_parameters={
            'max_num_iteration': 500,
            'population_size': 100,
            'mutation_probability': 0.1,
            'elit_ratio': 0.1,
            'crossover_probability': 0.8,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': 50
        }
    )
    model.run()
    best_solution = model.output_dict['variable']
    return green_space_data.iloc[best_solution.astype(int)]

@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/find_optimal_locations', methods=['POST'])
def find_optimal_locations():
    optimal_sites = find_optimal_sites()
    optimal_sites.to_csv(os.path.join(UPLOAD_FOLDER, "optimal_green_space_sites.csv"), index=False)
    # Create Folium Map
    map_center = [pollution_data["latitude"].mean(), pollution_data["longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB positron")

    # Add Pollution Hotspots
    for _, row in pollution_data.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color="red" if row["pollution_level"] > 50 else "blue",
            fill=True,
            fill_opacity=0.6,
            popup=f"Pollution Level: {row['pollution_level']}",
        ).add_to(m)

    # Add Optimal Green Space Locations
    for _, row in optimal_sites.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
            popup=f"Suitability Score: {row['green_space_suitability_score']}",
        ).add_to(m)

    # Save map
    map_path = "static/optimal_green_space_map.html"
    m.save(map_path)

    return jsonify({"map_path": map_path})  # Send JSON response to frontend

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    pollution_data = pd.read_csv(file_path)
    pollution_data["geometry"] = pollution_data.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    pollution_data = gpd.GeoDataFrame(pollution_data, geometry="geometry")
    
    required_cols = {'pollution_level', 'latitude', 'longitude'}
    if not required_cols.issubset(pollution_data.columns):
        return jsonify({"error": f"Missing columns: {required_cols - set(pollution_data.columns)}"}), 400
    
    coords = np.array(list(zip(pollution_data.geometry.x, pollution_data.geometry.y)))
    
    # Moran's I Test
    w = Queen.from_dataframe(pollution_data)
    w.transform = 'r'
    y = pollution_data["pollution_level"].values
    moran = Moran(y, w)
    moran_path = os.path.join(RESULT_FOLDER, "morans_i.png")
    plot_moran(moran, zstandard=True, figsize=(8, 4))
    plt.title("Moran’s I Test for Spatial Autocorrelation")
    plt.savefig(moran_path)
    plt.close()
    
    # Getis-Ord Gi* Hotspot Analysis
    knn = KNN.from_dataframe(pollution_data, k=5)
    g = G_Local(y, knn)
    pollution_data["hotspot"] = g.Zs
    hotspot_path = os.path.join(RESULT_FOLDER, "hotspot.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    pollution_data.plot(column="hotspot", cmap="coolwarm", legend=True, ax=ax)
    ctx.add_basemap(ax, crs=pollution_data.crs, source=ctx.providers.CartoDB.Positron)
    plt.title("Getis-Ord Gi* Hotspot Analysis")
    plt.savefig(hotspot_path)
    plt.close()
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    pollution_data["cluster"] = kmeans.fit_predict(coords)
    cluster_path = os.path.join(RESULT_FOLDER, "clusters.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    pollution_data.plot(column="cluster", cmap="viridis", legend=True, ax=ax)
    ctx.add_basemap(ax, crs=pollution_data.crs, source=ctx.providers.CartoDB.Positron)


    plt.title("Pollution Clusters using K-Means")
    plt.savefig(cluster_path)
    plt.close()
    
    # return jsonify({
    #     "moran_path": moran_path,
    #     "hotspot_path": hotspot_path,
    #     "cluster_path": cluster_path
    # })
    pollution_data.to_csv(os.path.join(UPLOAD_FOLDER, "pollution_sources_identified.csv"), index=False)

    # optimal_sites = find_optimal_sites()  
    # optimal_sites.to_csv(os.path.join(UPLOAD_FOLDER, "optimal_green_space_sites.csv"), index=False)
    return redirect(url_for("results"))

@app.route("/results")
def results():
    return render_template("results.html")


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)