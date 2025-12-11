# Predicting Car Accident Severity
AI project for Hanyang ITE3051

Gavin Pryor, Department of Financial Management, gavin.anaiah@gmail.com <br>
Randy Fu, Department of Computer Science, randyfu333@hanyang.ac.kr <br>
Fatima Zahra El Bajdali,Department of Computer Science, bfz2005@hanyang.ac.kr <br>
Marwa Errahmani Department of Computer Science, marwaerrahmani111@gmail.com <br>

Our project intends to investigate the factors in a car accident and predict the severity of the accident using real world data and deep learning techniques. Specifically, we want to highlight which factors have the greatest impact on severity and have an AI model predict how severe an accident is based on those factors. This can later be integrated into a type of alert system, where alerts can be sent to would-be drivers warning them of the road conditions and how severe an accident could be if they get into one. We intend to train a neural network model capable of classifying accident severity into multiple levels (minor, moderate, or severe).  We will use a dataset from Kaggle, which contains detailed records of traffic incidents across the United States. The dataset includes factors such as weather conditions, temperature, visibility, time of day, and road type all of which may influence accident outcomes.

[**Link to Project Demo**](https://drive.google.com/drive/folders/1_I9qA-8oQ6jerc1wdJB3SyRG2FbpFLt3?usp=sharing)

# Introduction

Road traffic accidents remain a major public-safety concern worldwide. Beyond the human cost, they create economic losses, traffic congestion, and significant pressure on emergency-response systems. Importantly, not all accidents have the same impact: some are minor incidents, while others result in serious injury or even fatalities. Being able to predict the likely severity of an accident using contextual information, such as weather, time of day, road type, visibility conditions, and local accident density, can support faster response prioritization, smarter warning systems, and more informed planning decisions for both drivers and infrastructure managers.

This project aims to develop and evaluate a machine-learning pipeline capable of predicting accident severity (minor, moderate, or severe) from real-world traffic incident data by learning patterns from environmental conditions, roadway characteristics, spatial-temporal context, and surrounding accident history. The system is designed not only to classify the severity level but also to understand how these diverse factors interact, allowing it to generalize to new situations and provide meaningful insights about the conditions under which severe accidents are most likely to occur. We also create neighborhood level descriptors such as kernel density estimates (KDE) and grid-cell accident statistics to capture the influence of local accident patterns. These features are then used to train deep learning classifiers alongside baseline models, allowing us to compare performance across architectures.

An additional objective of this project is to understand which factors contribute most to accident severity. After training, we conduct interpretability analyses, such as feature importance evaluations and model behavior visualizations, to identify the signals that most strongly shape the model’s predictions. Finally, the project includes a short demonstration video and a clear, structured technical blog that explain the model, methods, and findings, as well as a simple concept for how such a predictive system could be integrated into real-world notifications to drivers, predictive rerouting for GPS systems, or dynamic traffic-management dashboards for city planners.

# Dataset
The dataset we are using for this project is the US Accidents (2016 - 2023) dataset found on Kaggle. This dataset contains around 7.7 million accident records across 49 US states from February 2016 to March 2023, using multiple APIs that provide streaming traffic incident data. These APIs broadcast traffic data captured by various entities, including the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road networks.

Our target feature and what we are predicting is Severity, which is a categorical variable with possible values of 1 (least severe), 2, 3, or 4 (most severe). In this case a "severe" accident is one that causes a longer traffic delay, with heavy congestion and long backups.

Following our processing the data, we use four broad groups of features to predict severity:
1. Time and location features that capture when and where the accident occurred. Examples include time of the accident, geographic coordinates of the crash location, and a spatial grid cells representing the area surrounding the crash.

2. Weather and environmental features, which describe the driving conditions at the time of the incident. Examples are:
+ Temperature, humidity, and air pressure
+ Visibility and wind conditions
+ Rain, snow, or other forms of precipitation
+ Indicators for clean weather, windy weather, thunderstorms, or weather intensity

3. Roadway and infrastructure features, representing characteristics of the road and nearby traffic elements, such as the presence of a crossing, junction, bump, or roundabout or, whether there is a stop sign or traffic signal

4. Spatial density and contextual features, summarizing patterns of nearby accidents and the severity of past incidents in the same area, such as local accident density within small spatial grids

These groups together give the model information about time, weather, road design, and spatial context that help it predict how severe each accident is likely to be. The original dataset, along with the full list of original features and their descriptions, can be found [here](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).

# Methodology
### Data Processing
To process the data, we had to drop unnecessary columns, including leaky variables and redundant time or location fields, so we could focus on the timestamp and our feature-engineered spatial variables derived from latitude and longitude. 
```
# drop unnessecary columns
cleaned = data.drop(columns={
    'Source',
    'End_Lat',
    'End_Lng',
    'End_Time',
    'Distance(mi)',
    'Description',
    'Street',
    'City',
    'Country',
    'County',
    'State',
    'Zipcode',
    'Country',
    'Timezone',
    'Airport_Code',
    'Weather_Timestamp',
    'Amenity',
    'Civil_Twilight',
    'Nautical_Twilight',
    'Astronomical_Twilight'
})
```
We thought that location could be a strong indicator of how severe an accident may occur. However, while the dataset provided raw latitude and longitude, it is difficult for an AI model to utilize raw location data to identify patterns. Instead of using the raw lat/lng values, we projected coordinates into a metric CRS, created point geometries, generated multi-scale grid cells, and computed KDE-based density features to capture both local and regional accident patterns. We created these additional features as follows and appended it to our dataset:

```
# create GeoDataFrame from latitude and longitude for start locations
gdf = gpd.GeoDataFrame(cleaned, geometry=gpd.points_from_xy(cleaned.Start_Lng, cleaned.Start_Lat),crs="EPSG:4326")
gdf = gdf.to_crs(epsg=5070)
gdf["x"] = gdf.geometry.x
gdf["y"] = gdf.geometry.y

print("Data frame created")

#Multi-scale grid / tile IDs
'''
grid tiles produce compact neighborhood identifiers that capture local context and 
let you compute cell-level aggregates (counts, mean severity). Multi-scale tiles 
(fine + coarse) allow the model to see both micro and macro spatial structure
'''
def tile_id(x, y, scale):
    tile_x = (x // scale).astype(int)
    tile_y = (y // scale).astype(int)
    return tile_x.astype(str) + "_" + tile_y.astype(str)

gdf["cell_1km"] = tile_id(gdf["x"], gdf["y"], 1000)
gdf["cell_5km"] = tile_id(gdf["x"], gdf["y"], 5000)

print("Grid cells computed")

#KDE density (projected coords)
'''Gives measure of local accident concentration or density'''
import scipy.ndimage as ndi

# Fast gridded KDE (recommended)
bandwidth = 1000.0      # same units as projected coords (meters)
grid_size = 500.0       # cell size for histogram in meters (tweak for resolution/perf)

xs = gdf["x"].values
ys = gdf["y"].values

# define grid bounds with padding to avoid edge effects
pad = bandwidth * 3
xmin, xmax = xs.min() - pad, xs.max() + pad
ymin, ymax = ys.min() - pad, ys.max() + pad

nx = int(np.ceil((xmax - xmin) / grid_size))
ny = int(np.ceil((ymax - ymin) / grid_size))

H, xedges, yedges = np.histogram2d(xs, ys, bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])

# smooth counts with gaussian filter (sigma in pixels = bandwidth / grid_size)
sigma = bandwidth / grid_size
H_smooth = ndi.gaussian_filter(H, sigma=sigma, mode="constant")

# map each point to its grid cell and assign smoothed density
ix = np.minimum(np.maximum(((xs - xmin) / grid_size).astype(int), 0), H_smooth.shape[0]-1)
iy = np.minimum(np.maximum(((ys - ymin) / grid_size).astype(int), 0), H_smooth.shape[1]-1)

# add per-point grid indices and an id for the KDE grid cell
gdf["kde_ix"] = ix
gdf["kde_iy"] = iy
gdf["kde_cell_kdegrid"] = [f"{i}_{j}" for i, j in zip(ix, iy)]

kde_vals = H_smooth[ix, iy]

# optional: convert counts -> density per square meter (makes values comparable)
cell_area = grid_size * grid_size
kde_density = kde_vals / (cell_area)

gdf["kde_1km"] = kde_density
gdf["kde_density_m2"] = kde_density

# build a DataFrame with one row per KDE-grid cell (counts, density, center coords)
gi, gj = np.indices(H_smooth.shape)       # gi.shape == gj.shape == H_smooth.shape
gi_f = gi.ravel()
gj_f = gj.ravel()
counts_f = H_smooth.ravel()
x_centers = xmin + (gi_f + 0.5) * grid_size
y_centers = ymin + (gj_f + 0.5) * grid_size

kde_grid_df = pd.DataFrame({
    "kde_cell_kdegrid": [f"{i}_{j}" for i, j in zip(gi_f, gj_f)],
    "kde_grid_count": counts_f,
    "kde_grid_density_m2": counts_f / (cell_area),
    "kde_grid_x": x_centers,
    "kde_grid_y": y_centers
})
```

We also handled missing values in weather and environmental columns, removed invalid or extreme outliers, and standardized continuous fields where needed. For the weather-related variables, our original dataset had multiple subcategories of weather events. For example, scattered clouds vs. mostly cloudy vs. partly cloudy etc. We condensed dozens of noisy text categories into cleaner groups like weather type, intensity, and flags for windy or thunder conditions, which made the variables much easier for the model to learn from. We cleaned the data as follows: 
```
    # reduce the number of categories
    def to_category(txt: str) -> str:
        if txt in ('unknown', ''):
            return 'Unknown'
        if re.search(r'\b(wintry mix|rain and snow|snow and rain|snow and sleet|sleet and snow|freezing drizzle|freezing rain|freezing fog|ice pellets)\b', txt):
            if re.search(r'\b(freezing rain|freezing drizzle|freezing fog|ice pellets)\b', txt):
                return 'Freezing / Ice'
            return 'Wintry Mix'
        if re.search(r'\b(t-?storm|thunderstorm|thunder)\b', txt):
            return 'Thunderstorm'
        if re.search(r'\b(snow grains|snow showers?|snow)\b', txt):
            return 'Snow'
        if re.search(r'\bsleet\b', txt):
            return 'Sleet'
        if re.search(r'\bhail\b', txt):
            return 'Hail'
        if re.search(r'\b(drizzle)\b', txt):
            return 'Drizzle'
        if re.search(r'\b(rain showers?|showers?)\b', txt):
            return 'Rain'
        if re.search(r'\brain\b', txt):
            return 'Rain'
        if re.search(r'\b(fog|mist)\b', txt):
            return 'Fog / Mist'
        if re.search(r'\bhaze\b', txt):
            return 'Haze'
        if re.search(r'\b(smoke)\b', txt):
            return 'Smoke'
        if re.search(r'\b(dust(storm)?|dust whirls?)\b', txt):
            return 'Dust'
        if re.search(r'\b(sand)\b', txt):
            return 'Sand'
        if re.search(r'\bsqualls?\b', txt):
            return 'Squalls'
        if re.search(r'\b(tornado|funnel cloud)\b', txt):
            return 'Tornado'
        if re.search(r'\bovercast\b', txt):
            return 'Overcast'
        if re.search(r'\b(scattered clouds|mostly cloudy|partly cloudy|cloudy)\b', txt):
            return 'Cloudy'
        if re.search(r'\b(clear|fair)\b', txt):
            return 'Clear'
        if re.search(r'\b(volcanic ash)\b', txt):
            return 'Other'
        return 'Other'
```

Finally, since less severe accidents were far more common than severe accidents in the dataset, we utilized random undersampling to get an even number of each Severity class. This ensured that the model could not simply learn to favor the majority classes, which would have produced deceptively high accuracy while failing to recognize the rarer, high-severity cases. By training on a balanced dataset, each severity level contributed equally to the learning process, improving the model’s ability to generalize and leading to a more meaningful and fair comparison across classes.

All of this gave us a cleaner, more meaningful version of the dataset that better reflects real accident conditions and sets the groundwork for stronger modeling. Our dataset ended up with the following table headers:

	ID	Severity	Start_Time	Start_Lat	Start_Lng	Temperature(F)	Wind_Chill(F)	Humidity(%)	Pressure(in)	Visibility(mi)	Wind_Direction	Wind_Speed(mph)	Precipitation(in)	Weather_Condition	Bump	Crossing	Give_Way	Junction	No_Exit	Railway	Roundabout	Station	Stop	Traffic_Calming	Traffic_Signal	Turning_Loop	Sunrise_Sunset	Weather_Clean	Weather_Windy	Weather_Thunder	Weather_Intensity	geometry	x	y	cell_1km	cell_5km	kde_ix	kde_iy	kde_cell_kdegrid	kde_1km	kde_density_m2	kde_grid_count	kde_grid_density_m2	kde_grid_x	kde_grid_y	cell1_count	cell1_mean_sev

### Modeling
To predict Severity, we tested three different models and compared performance for each: Random Forest, XGBoost, and an attention based neural network. These models represent three increasing levels of complexity. Random Forest is a well established ensemble method that is often used as a starting point for tabular prediction tasks. It provides fast training times and interpretable feature splits, so it served as a strong baseline for us. XGBoost is a gradient boosted decision tree model that is well known for state of the art performance on structured datasets. It builds trees sequentially and corrects previous mistakes at each iteration, which typically results in higher accuracy than Random Forest when the underlying relationships are non linear. Finally, after covering neural networks in class, we implemented an attention based neural network as our deep learning alternative. Attention architectures have recently become popular across many domains because they can learn how to focus on the most informative parts of the input. For a task like accident severity prediction, the ability to assign higher weight to certain weather, spatial, or roadway conditions has the potential to reveal patterns that models based on fixed tree splits may miss. This network allowed us to explore whether a more flexible model could provide further gains and whether attention weights could provide useful insight into which features mattered most during prediction.

Our goal in comparing these three models was to find the ideal balance between predictive performance, runtime feasibility on our laptops without access to high end GPUs, and interpretability. Each model gave us different strengths. Random Forest trained quickly and served as a reference point for improvement. XGBoost offered strong performance with moderate training time and produced clear feature importance values (as shown in the Evaluation section). The attention based network required much longer for training and more tuning, but it allowed us to examine model behaviour through the lens of learned attention weights. By choosing these three models, we were able to evaluate traditional ensemble methods alongside a modern deep learning architecture and identify which approach was most practical and effective for our data.

To train our models, we utilized Sci-kit learn for random forest and xgboost, and tensorflow/keras for the neural network. Our train-test split was a stratified 70/30 split.

#### Random Forest
We started by first preprocessing the data and encoding our categorical variables
```
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric),
    ('bool', 'passthrough', boolean),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical)
])
```
After a few tests, we settled on the following parameters by balancing training time and accuracy
```
rf = RandomForestClassifier(
    # parameters chosen to minimize runtime
    n_estimators=200,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=1, 
)

model = Pipeline([
    ('preprocess', preprocessor),
    ('rf', rf)
])
```
#### XGBoost
Similar to our RandomForest model, we started with encoding our data. We had to make small adjustments to our classifications as our categories have ranges from 1-4 while XGBoost is 0-indexed. Our initial parameters were set based off a general baseline. 
```
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Encode target labels as XGBoost is 0-indexed while our data is 1-4
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=1, stratify=y_enc)

xgb_model = Pipeline([
    ('preprocess', preprocessor),
    ('xgb', xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1, # perform hyper-parameter tuning to find optimal learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        
        objective='multi:softmax',
        num_class=4,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ))
])
```
#### Attention-based Neural Network
For our neural network, we defined the following architecture:
```
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class FeatureAttentionLayer(layers.Layer):
    def __init__(self, attention_dim=64, **kwargs):
        super(FeatureAttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='W_attention'
        )
        self.b = self.add_weight(
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True,
            name='b_attention'
        )
        self.u = self.add_weight(
            shape=(self.attention_dim, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='u_attention'
        )
        super(FeatureAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute attention scores
        v = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        vu = tf.matmul(v, self.u)
        alphas = tf.nn.softmax(vu, axis=1)
        # Weighted sum of inputs
        output = inputs * alphas
        return output

# Build attention model
def build_attention_model(input_dim, num_classes=3, attention_dim=64):
    
    # Input
    inputs = keras.Input(shape=(input_dim,), name='input_features')
    
    # Attention
    attended = FeatureAttentionLayer(attention_dim=attention_dim, name='feature_attention')(inputs)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', name='dense1')(attended)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.3, name='dropout1')(x)
    
    x = layers.Dense(64, activation='relu', name='dense2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    x = layers.Dense(32, activation='relu', name='dense3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Dropout(0.2, name='dropout3')(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='AttentionAccidentModel')
    return model
```
We also decided to use the following model parameters: 
```
model = build_attention_model(
    input_dim=X_train_scaled.shape[1],
    num_classes=3,
    attention_dim=64
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    callbacks=training_callbacks,
    verbose=1
)
```
# Evaluation & Analysis: 
To evaluate the models, we focused on macro-F1 score rather than accuracy. Accuracy treats all errors the same and can be misleading when the distribution of predicted classes is unbalanced or when certain types of mistakes are more harmful. In our context, failing to correctly identify a moderate or severe accident is much more costly than misclassifying a minor one. The macro-F1 score averages precision and recall for each class equally, so performance on the high-impact cases contributes just as much as the common ones. This provides a more honest assessment of real-world usefulness.

Below is a summary of each of the models and their performance: 

| Model                    | Macro-F1  | Accuracy | Training Time                                      |
| ------------------------ | --------- | -------- | ---------------------------------------------------|
| **XGBoost**              | **0.71**  | 0.71     | ~16 seconds                                        |
| **Attention Neural Net** | 0.71–0.72 | 0.72     | several min to several hours (depending on laptop) |
| **Random Forest**        | 0.52      | 0.55     | ~2 seconds                                         |

Both XGBoost and the Attention-based Neural Network achieved similar macro-F1 values around 0.71, while Random Forest lagged significantly at 0.52. The key difference was runtime: the neural network required an order of magnitude more computation to reach the same level of performance, making XGBoost a much more attractive option for development, iteration, and deployment on limited hardware. 

Each model learned somewhat different patterns about what drives accident severity. Below, we list the top five features from each model:
| Rank | Random Forest  | XGBoost              | Attention Network    |
| ---- | -------------- | -------------------- | -------------------- |
| 1    | cell1_mean_sev | num__cell1_mean_sev  | Stop                 |
| 2    | cell1_count    | num__cell1_count     | Humidity (%)         |
| 3    | kde_grid_count | num__Wind_Chill (F)  | Wind Direction Calm  |
| 4    | kde_density_m2 | bool__Traffic_Signal | Wind Direction South |
| 5    | kde_1km        | num__kde_density_m2  | Weather Clean Rain   |

Several patterns stand out across models:

1. Spatial accident history is the strongest signal. Both gradient-boosted trees placed cell-level metrics at the top:
    + `cell1_mean_sev` (average severity of nearby accidents)
    + `cell1_count` (volume of nearby accidents)

This suggests accident severity is highly localized. Places where severe crashes have happened before tend to produce severe crashes again, likely due to infrastructure design, traffic flow, or visibility constraints.

2. Local density features matter. KDE-based features, such as `kde_density_m2` and `kde_grid_count`, were repeatedly ranked in the top positions. These variables summarize how concentrated recent crashes are, which again reflects the importance of spatial clustering.

3. Weather effects are meaningful across models. The attention network emphasized atmospheric conditions (Humidity, Wind direction, Weather Clean Rain). Weather does not appear uniformly as strong as spatial context in trees, but the neural network clearly learned that micro-environmental conditions shape risk.

4. Traffic signs and intersection structure elevate risk. Two indicators emerged strongly: Stop sign present and traffic signal present. These variables may capture busy intersections, decision points, and conflict zones, where driver error is more likely.

Below is a summary of each of the models and their performance: 

| Model          | Macro-F1 |
|----------------|----------|
| XGBoost        | 0.71     |
| Attention NN   | 0.75     |
| Random Forest  | 0.52     |

1. XGBoost Results:
```
XGBoost Macro f1: 0.7078

XGBoost Classification Report:
               precision    recall  f1-score   support

           0       0.78      0.80      0.79     20209
           1       0.70      0.59      0.64     20209
           2       0.65      0.77      0.70     20209
           3       0.72      0.68      0.70     20208

    accuracy                           0.71     80835
    macro avg       0.71      0.71      0.71     80835
    weighted avg       0.71      0.71      0.71     80835


XGBoost Confusion Matrix:
 [[16116  1859  1411   823]
 [ 2836 11830  3153  2390]
 [  879  1533 15598  2199]
 [  779  1699  3931 13799]]
Training took 16.32 seconds
   ```
2. Attention Neural Network Results
```
MODEL EVALUATION

1. Making predictions on test set...

OVERALL METRICS:
   Accuracy:  0.7864
   Precision: 0.7766 (macro)
   Recall:    0.7446 (macro)
   F1-Score:  0.7543 (macro)

PER-CLASS METRICS:

   Minor:
      Precision: 0.7933
      Recall:    0.7805
      F1-Score:  0.7868
      Support:   20,209

   Moderate:
      Precision: 0.7374
      Recall:    0.5414
      F1-Score:  0.6244
      Support:   20,209

   Severe:
      Precision: 0.7991
      Recall:    0.9118
      F1-Score:  0.8518
      Support:   40,417

CONFUSION MATRIX:
   (Rows = True, Columns = Predicted)

                     Minor   Moderate     Severe
   Minor            15,773      1,683      2,753
   Moderate          2,758     10,942      6,509
   Severe            1,353      2,213     36,851
```
3. Random Forest Results:
``` 
Macro f1: 0.5174

Classification Report:
               precision    recall  f1-score   support

           1       0.56      0.67      0.61     20209
           2       0.65      0.17      0.27     20209
           3       0.52      0.81      0.63     20209
           4       0.56      0.55      0.56     20208

    accuracy                           0.55     80835
    macro avg       0.57      0.55      0.52     80835
    weighted avg       0.57      0.55      0.52     80835


Confusion Matrix:
 [[13555  1014  2641  2999]
 [ 6525  3425  6261  3998]
 [ 1888   283 16460  1578]
 [ 2108   569  6448 11083]]
Training took 2.29 seconds
```
Also, the attention model required roughly 20–30 times more computation than XGBoost for a nearly identical macro-F1 score. On consumer hardware, such differences matter. Parameter sweeps, re-training, and real-time deployment all become more feasible when runtime is measured in seconds rather than minutes. For this reason, we chose XGBoost as the final model.

The model predictions could be calibrated by adjusting thresholds instead of using the default like 0.5. In a real deployment, the objective is not simply to label severity correctly, but to prioritize safety. For example, if the model predicts high probability of severity, even if not the top prediction, a system could trigger driver warnings, traffic rerouting, or dynamic signage. From a practical perspective, false negatives on severe accidents are more costly than false positives. In that scenario, it would be entirely reasonable to shift the threshold to make the model more sensitive to severe cases, even if that slightly reduces precision on minor cases. This also depends on the availability of resources or the cost of an action like deploying traffic personnel. If these personnel are extremely limited for example, deploying to areas where it's not needed can be much more costly. More analysis on this is required before deployment.

Given these results, we decided to focus on improving our XGBoost model's performance as described in the modeling section. To do so, we created a parameter sweep using Wandb for analysis to try and find the optimal parameters for our XGBoost model given our dataset. However, after running through the sweep the accuracy of the model did not dramatically increase, instead hovering around 0.72. The results of our sweeps are shown in the following graphs:

![f1-weighted graph](https://github.com/randyf333/AI-ApplicationsProject/blob/main/visualizations/Screenshot%202025-12-09%20154023.png)
![f1-macro graph](https://github.com/randyf333/AI-ApplicationsProject/blob/main/visualizations/Screenshot%202025-12-09%20154119.png)
![raw f1-score graph](https://github.com/randyf333/AI-ApplicationsProject/blob/main/visualizations/Screenshot%202025-12-09%20154058.png)

The parameter importance for f1-macro is show below. The most important parameters are listed at the top, and the correlation is listed as either positive(green) or negative(red) respectively:

![f1-macro parameter importance](https://github.com/randyf333/AI-ApplicationsProject/blob/main/visualizations/Screenshot%202025-12-09%20154949.png)

For the final model, we used 5-fold cross-validation to test different hyperparameter values to identify the most optimal values of the top 3 most impactful. 
```
import warnings

warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in",
)

xgb_clf = XGBClassifier(
    objective="multi:softprob",
    num_class=4,              # severity 1 to 4
    eval_metric="mlogloss",
    random_state=1,
    n_estimators=300,
    tree_method="hist"        # good default on modern CPUs
)

xgb_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("xgb", xgb_clf)
])

# top three most impactful parameters according to sweep: max depth, gamma, and subsample
param_grid = {
    "xgb__max_depth":   [3, 5, 7],
    "xgb__gamma":       [0, 0.5, 1.0],
    "xgb__subsample":   [0.7, 0.85, 1.0],
}

# set up stratified K fold CV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

# grid search with K fold CV
grid_search = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid,
    scoring="f1_macro",   # main metric
    cv=cv,
    n_jobs=-1,
    verbose=2
)
```
Combining a GridSearch and sweep, we ended up with the final model results:\
Best params: {'xgb__gamma': 0.5, 'xgb__max_depth': 7, 'xgb__subsample': 0.7}\
Best CV macro F1: 0.7132117427623111\
Test macro F1: 0.7146090213946625
Classification report:
               precision    recall  f1-score   support

           0       0.79      0.82      0.80     20209
           1       0.70      0.59      0.64     20209
           2       0.66      0.76      0.71     20209
           3       0.72      0.70      0.71     20208

    accuracy                           0.72     80835
    macro avg       0.72      0.72      0.71     80835
    weighted avg       0.72      0.72      0.71     80835

<img width="619" height="598" alt="Screenshot 2025-12-11 at 9 57 54 AM" src="https://github.com/user-attachments/assets/1cd7adde-11e1-42b1-9380-34335048481f" />

Even with parameter sweeps and cross-validation to find the best parameters, we saw little to no improvement in our accuracy or f1 score. These results indicate to us that our current features may not carry enough predictive power to increase our prediction accuracy. More feature engineering or more data may be required to improve the performance of our model. 

<img width="819" height="590" alt="Screenshot 2025-12-11 at 10 01 05 AM" src="https://github.com/user-attachments/assets/b6ace7e2-f7cc-4999-b5d6-0f160df6340b" />

For feature importance, the big spatial signals stayed the same, but the important change in this final model is that it now relies on specific high risk grid cells rather than just broad spatial features. Encoded cell IDs like cat__cell_1km_-2053_1728 and cat__cell_5km_-153_374 jumped into the top ranks, showing the model has learned precise geographic hotspots instead of general patterns. Weather features dropped in importance and became more selective. Only a few strong indicators remain, meaning the model filtered out noisy categories and kept the ones that consistently correlate with severity. Intersection indicators like traffic signals and crossings are still helpful but got pushed lower because the detailed spatial encodings explain risk even better on their own. Overall, the final model shifted from general trends to high resolution spatial detail, which is a sign of a more mature feature pipeline and a more confident model.

# Related Work
We referenced the following sources:

Datasets & Documentation:
+ US Accidents (2016–2023) — Kaggle
  
Methods & Technical Resources:
+ XGBoost official documentation
+ Scikit-learn documentation
+ TensorFlow/Keras guides for custom layers
+ Blog posts/tutorials on: Attention mechanisms for tabular data

Tools Utilized
+ Python 3.10
+ Pandas, NumPy, Matplotlib, Seaborn
+ scikit-learn
+ TensorFlow / Keras
+ XGBoost
+ Weights & Biases
+ PyTorch
+ GeoPandas

# Code Citation & Implementation Details

Below are sources used in `Nwprocessing-and-modeling.ipynb`.
- Attention concept: Vaswani et al. (2017), Bahdanau et al. (2014)
- Implementation: [TensorFlow Custom Layers Guide](https://www.tensorflow.org/guide/keras/custom_layers_and_models)

- Architecture pattern: Goodfellow et al. (2016) - Deep Feedforward Networks
- BatchNorm: Ioffe & Szegedy (2015)
- Dropout: Srivastava et al. (2014) 
- [Keras Functional API](https://keras.io/guides/functional_api/)

- Frequency encoding: Micci-Barreca (2001) 
- One-hot: [Pandas get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

- [Scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [train_test_split docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

- Early stopping: Prechelt (1998) 
- [Keras Callbacks API](https://keras.io/api/callbacks/)




# Conclusion
Through this project, we were able to explore how different environmental, temporal, and spatial factors contribute to the severity of car accidents. By testing multiple machine learning models, we gained a better understanding of what types of algorithms work best for this kind of structured, real world data. The consistent finding across all models is that severity is strongly tied to location based patterns: areas with dense accident histories or specific high risk grid cells produce more severe outcomes. Weather and intersection structure provide additional signals as well.

XGBoost offered the best overall balance of accuracy, interpretability, and computational efficiency, reaching a macro F1 of ~0.71 while training in seconds. The attention-based neural network achieved our highest performance at 0.75 macro F1 and revealed interesting patterns through its learned attention weights, such as prioritizing traffic control features and weather conditions differently than tree-based models. However, the 20-30x increase in training time for a marginal performance gain made XGBoost more suitable for large scale deployment without requiring highly advanced resources. The true cap performance we observed suggests that additional predictive power will likely come from new data sources rather than more complex models. Examples of data that could improve results include traffic flow data, lighting conditions, or various types of driver behavior data.

In conclusion, our results can be used both as a predictive engine, able to be used by city governments or navigation apps, as well as a data analysis project that can be examined for what features likely correlate the highest to severe accidents, to ultimately save drivers time from sitting in traffic or taking long detours, or potentially saving their lives.
