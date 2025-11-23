import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from category_encoders import TargetEncoder

# Load Data
df = pd.read_csv('data/clean/housing_cleaned-geo.csv')

# Preprocessing
df['transfer_date'] = pd.to_datetime(df['transfer_date'])
df = df[df['transfer_date'].dt.year >= 2024]
df = df.sort_values(by='transfer_date').reset_index(drop=True)

# Log Price
df['log_price'] = np.log1p(df['price'])

# Distance to Centres
CENTRES_OSGB = {
    'bhm': (406000, 286000), 'cov': (433000, 279000), 'lei': (458500, 306000),
    'not': (457119, 340206), 'der': (435187, 336492), 'sto': (488000, 347000),
    'wol': (391500, 298500), 'sol': (415000, 279000)
}
for key, (E0, N0) in CENTRES_OSGB.items():
    df[f'dist_{key}_km'] = np.hypot(df['oseast1m'] - E0, df['osnrth1m'] - N0) / 1000.0
df['min_dist_to_retail_centre_km'] = df[[f'dist_{key}_km' for key in CENTRES_OSGB]].min(axis=1)
df = df.drop(columns=[f'dist_{key}_km' for key in CENTRES_OSGB])

# Location Clustering
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
df['location_cluster'] = kmeans.fit_predict(df[['oseast1m', 'osnrth1m']])
df['location_cluster'] = df['location_cluster'].astype(str)

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

y_train = train_df['log_price']
y_test = test_df['log_price']
X_train = train_df.drop(columns=['log_price', 'price', 'transfer_date'])
X_test = test_df.drop(columns=['log_price', 'price', 'transfer_date'])

# Feature columns
num_feats = ['log_total_floor_area', 'IMD_Rank', 'oseast1m', 'osnrth1m', 'min_dist_to_retail_centre_km', 'energy_band_num']
cat_feats = ['property_type', 'ruc21', 'location_cluster']
bin_feats = ['new_build', 'is_leasehold']

# Standard Preprocessor (for Tree models)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats),
        ('bin', 'passthrough', bin_feats)
    ],
    remainder='drop'
)

# Neural Network Preprocessor (with Target Encoding)
nn_target_feats = ['town_city', 'district']
nn_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats),
        ('target', TargetEncoder(), nn_target_feats),
        ('bin', 'passthrough', bin_feats)
    ],
    remainder='drop'
)

# Models
models = {
    'XGBoost': (XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, gamma=0.1, random_state=42, n_jobs=-1), preprocessor),
    'LightGBM': (LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1), preprocessor),
    'CatBoost': (CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=0, allow_writing_files=False), preprocessor),
    'MLP': (MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=500, random_state=42, early_stopping=True), nn_preprocessor)
}

results = []
print("=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

for name, (model, prep) in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    pipeline = Pipeline([('pre', prep), ('model', model)])
    pipeline.fit(X_train, y_train)
    
    # Test predictions
    y_pred_test = pipeline.predict(X_test)
    y_test_gbp = np.expm1(y_test)
    y_pred_test_gbp = np.expm1(y_pred_test)
    
    r2 = r2_score(y_test, y_pred_test)
    mae_gbp = mean_absolute_error(y_test_gbp, y_pred_test_gbp)
    
    print(f"Test R2 Score: {r2:.4f}")
    print(f"Test MAE (£): {mae_gbp:,.0f}")
    
    results.append({'Model': name, 'Test R2': r2, 'Test MAE (£)': mae_gbp})

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
results_df = pd.DataFrame(results).sort_values(by='Test R2', ascending=False)
print(results_df.to_string(index=False))
print(f"{'='*60}")
