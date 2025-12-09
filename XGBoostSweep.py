import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import re
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

data = pd.read_csv("C:/Users/Randy.RF-VICTUS/.cache/kagglehub/datasets/sobhanmoosavi/us-accidents/versions/13/US_Accidents_March23.csv")
print("Data loaded")
# drop unnessecary columns
cleaned = data.drop(columns={
    #'ID',
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
numeric = [
    'Severity',
    'Start_Lat',
    'Start_Lng',
    'Temperature(F)',
    'Wind_Chill(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)'
]

categorical = [
    'Wind_Direction',
    'Weather_Condition',
    'Bump',
    'Crossing',
    'Give_Way',
    'Junction',
    'No_Exit',
    'Railway',
    'Roundabout',
    'Station',
    'Stop',
    'Traffic_Calming',
    'Traffic_Signal',
    'Turning_Loop',
    'Sunrise_Sunset'
]

datetime = [
    'Start_Time'
]
print("Columns dropped")
# handle nulls
for col in numeric:
    cleaned[col] = cleaned[col].fillna(cleaned[col].median())

for col in categorical:
    cleaned[col] = cleaned[col].fillna('Unknown')

# only keep lat and lng within the usa
cleaned = cleaned[(cleaned['Start_Lat'].between(20, 50)) & (cleaned['Start_Lng'].between(-130, -60))]

# keep possible weather values
cleaned = cleaned[(cleaned['Temperature(F)'] > -50) & (cleaned['Temperature(F)'] < 130)]
cleaned = cleaned[(cleaned['Visibility(mi)'] <= 50)]
cleaned = cleaned[(cleaned['Wind_Speed(mph)'] < 120)]
# handle duplicates
cleaned = cleaned.drop_duplicates()
def clean_weather_condition(data, col='Weather_Condition'):
    s = (
        data[col].fillna('Unknown')
        .astype(str)
        .str.strip()
    )
    s = s.str.replace(r'\s+', ' ', regex=True)
    s = s.str.replace(r'\s*/\s*', ' / ', regex=True)  # normalize " / "
    s_lower = s.str.lower()

    # separating out weather conditions
    windy_flag    = s_lower.str.contains(r'\bwindy\b')
    heavy_flag    = s_lower.str.contains(r'\bheavy\b')
    light_flag    = s_lower.str.contains(r'\blight\b')
    thunder_flag = s_lower.str.contains(r'\b(?:t-?storm|thunder)\b', regex=True)


    base = s_lower
    base = base.str.replace(r'\bwindy\b', '', regex=True)
    base = base.str.replace(r'\bheavy\b', '', regex=True)
    base = base.str.replace(r'\blight\b', '', regex=True)
    base = base.str.replace(r'\bpatch(es)? of\b', '', regex=True)
    base = base.str.replace(r'\bshallow\b', '', regex=True)
    base = base.str.replace(r'\bblowing\b', '', regex=True)
    base = base.str.replace(r'\bdrifting\b', '', regex=True)
    base = base.str.replace(r'\blow\b', '', regex=True)
    base = base.str.replace(r'\bnearby\b', '', regex=True)
    base = base.str.replace(r'\bin the vicinity\b', '', regex=True)
    base = base.str.replace(r'\bn/a precipitation\b', 'unknown', regex=True)

    # collapse slashes to spaces after removals
    base = base.str.replace(r'\s*/\s*', ' ', regex=True)
    base = base.str.replace(r'\s+', ' ', regex=True).str.strip()

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

    cleaned_cat = base.apply(to_category)

    # intensity of weather
    intensity = np.where(heavy_flag, 'Heavy',
                  np.where(light_flag, 'Light', 'Normal'))

    out = data.copy()
    out['Weather_Clean'] = cleaned_cat
    out['Weather_Windy'] = windy_flag.astype(int)
    out['Weather_Thunder'] = thunder_flag.astype(int)
    out['Weather_Intensity'] = intensity.astype(str)
    return out

cleaned = clean_weather_condition(cleaned, 'Weather_Condition')
print("Weather conditions cleaned")
# change to datetime
cleaned['Start_Time'] = pd.to_datetime(cleaned['Start_Time'], format='mixed', errors='coerce')
#Create GeoDataFrame from latitude and longitude for start locations
gdf = gpd.GeoDataFrame(cleaned, geometry=gpd.points_from_xy(cleaned.Start_Lng, cleaned.Start_Lat),crs="EPSG:4326")
gdf = gdf.to_crs(epsg=5070) # Convert to a projected coordinate system for accurate distance calculations
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

# merge grid-level statistics back to points (so each point gets its grid's aggregate values)
gdf = gdf.merge(
    kde_grid_df[["kde_cell_kdegrid", "kde_grid_count", "kde_grid_density_m2", "kde_grid_x", "kde_grid_y"]],
    on="kde_cell_kdegrid",
    how="left"
)

print("Gridded KDE computed (fast) and KDE-grid attributes merged into gdf")

#Simple cell-level aggregates. Compute count and mean severity per 1km cell
cell_stats = gdf.groupby("cell_1km").agg(cell1_count=("ID","count"), 
                                         cell1_mean_sev=("Severity","mean")).reset_index()
gdf = gdf.merge(cell_stats, on="cell_1km", how="left")

print("Cell-level aggregates computed")
'''
geometry: shapely Point for each accident (original lat/lng converted to projected coords).
x, y: projected coordinates (meters) extracted from geometry.
cell_1km / cell_5km: coarse grid tile IDs (string "tilex_tiley") at 1 km and 5 km scales.
kde_ix / kde_iy: integer array indices of the KDE histogram cell (grid column/row) each point falls in.
kde_cell_kdegrid: string id for the KDE cell ("ix_iy") — links points to the gridded KDE row.
kde_1km / kde_density_m2: per-point smoothed KDE density (converted to density per m^2) assigned from the gridded KDE.
kde_grid_count: count (smoothed) for each KDE grid cell (one row per cell in kde_grid_df).
kde_grid_density_m2: per-cell density (counts / cell_area) for each KDE grid cell.
kde_grid_x / kde_grid_y: center coordinates (projected, meters) of each KDE grid cell.
cell1_count: per-1km-cell raw count (groupby on cell_1km).
cell1_mean_sev: per-1km-cell mean Severity.

'''
# drop ID and lat/lng columns
model_data = gdf.drop(columns={'ID', 'Start_Lat', 'Start_Lng', 'geometry', 'x', 'y'})
# balance classes (right before modeling)
min_count = model_data['Severity'].value_counts().min()

model_data = model_data.groupby('Severity', group_keys=False).apply(lambda x: x.sample(min_count, random_state=1))

numeric = [
    'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
    'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'Precipitation(in)',
    'kde_ix', 'kde_iy',
    'kde_cell_kdegrid', 'kde_1km', 'kde_density_m2',
    'kde_grid_count', 'kde_grid_density_m2', 'kde_grid_x', 'kde_grid_y',
    'cell1_count', 'cell1_mean_sev'
]

categorical = [
    'Wind_Direction',
    'Weather_Condition',
    'Sunrise_Sunset',
    'Weather_Clean',
    'Weather_Intensity',
    'cell_1km',
    'cell_5km'
]

boolean = [
    'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
    'Railway', 'Roundabout', 'Station', 'Stop',
    'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
    'Weather_Windy', 'Weather_Thunder'
]

X = model_data.drop(columns={'Severity'})
y = model_data['Severity']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric),
    ('bool', 'passthrough', boolean),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical)
])
# Model 2 - XG Boost:

# Encode target labels as XGBoost is 0-indexed while our data is 1-4
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=1, stratify=y_enc)

import wandb
wandb.login("b48322a7f66eeb2bdc68eabd3bf355f8884ed184")
wandb.init(project="xgboost-accident-severity", entity="randyf333-hanyang")
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'f1-score', 'goal': 'maximize' #better than accuracy for imbalanced classes
    },

    'parameters':{
        'learning_rate': {'min': 0.01, 'max':0.3},
        'max_depth': {'min': 3, 'max':12},
        'subsample': {'min': 0.5, 'max':1.0},
        'colsample_bytree': {'min': 0.5, 'max':1.0},
        'n_estimators': {'values': [100, 200, 300, 400, 500]}, 
        'gamma': {'min': 0, 'max':5},
        'reg_alpha': {'min': 0, 'max':5},
        'reg_lambda': {'min': 0, 'max':10},
        'min_child_weight': {'min': 1, 'max':10}
    }
}
from sklearn.metrics import f1_score
def train(config = None):
    with wandb.init(config=config):
        config = wandb.config

        model = xgb.XGBClassifier(
            learning_rate=config.learning_rate,
            max_depth=int(config.max_depth),
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            n_estimators=int(config.n_estimators),
            gamma=config.gamma,
            objective='multi:softprob',
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            min_child_weight=config.min_child_weight,

            num_class=4,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,

        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        pipeline.fit(
            X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        wandb.log({
            'accuracy': accuracy,
            'f1-macro': f1_macro,
            'f1-weighted': f1_weighted
        })

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="xgboost-accident-severity", entity="randyf333-hanyang")
    wandb.agent(sweep_id, function=train, count=30)