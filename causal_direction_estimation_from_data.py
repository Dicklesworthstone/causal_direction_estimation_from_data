from itertools import combinations
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kurtosis, skew
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.exceptions import DataConversionWarning
from skfda import FDataGrid
from skfda.exploratory.stats import mean, var
from skfda.exploratory.depth import ModifiedBandDepth
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore', category=DataConversionWarning) # Suppress DataConversionWarnings
use_debug_shapes = False  # Set to True to enable additional debugging prints
use_normalize_input_data = True  # Set to True to normalize input data

def preprocess_data(data):
    data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if use_normalize_input_data:
        data = normalize_data(data)
    return data

def check_sufficient_variation(data):
    if data.var().min() < 1e-8:
        raise ValueError("Data variation is too low for meaningful analysis.")

def get_models():
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=0.001),  # Reduced regularization
        'BayesianRidge': BayesianRidge(alpha_1=0.001, alpha_2=0.001, lambda_1=0.001, lambda_2=0.001),  # Reduced regularization
        'Lasso': Lasso(alpha=0.001),  # Reduced regularization
        'ElasticNet': ElasticNet(alpha=0.001),  # Reduced regularization
        'GradientBoosting': GradientBoostingRegressor(),
        'KNeighbors': KNeighborsRegressor(),
        'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)  # Reduced regularization
    }
    
def fit_models_and_analyze_errors(X, y, models, column_names):
    check_sufficient_variation(pd.DataFrame(X))  # Check variation in X
    check_sufficient_variation(pd.DataFrame(y))  # Check variation in y
    if X.size < 2 or y.size < 2:
        raise ValueError("Insufficient data for model fitting.")
    threshold = 0.00001
    print(f"Analyzing errors for columns: {column_names[0]} as independent and {column_names[1]} as dependent")
    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    errors = {}
    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in models.items():
        print(f"    Fitting model: {name}")
        error_list = []
        mse_scores = []
        r2_scores = []
        low_variance_flag = False
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if use_debug_shapes:
                print(f"      Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test).reshape(-1, 1)
            if np.var(predictions) < threshold:
                print(f"      Low variance in predictions for model {name}. Flagging for review.")
                low_variance_flag = True
            error = y_test - predictions
            error_list.append(error)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mse_scores.append(mse)
            r2_scores.append(r2)
        aggregated_errors = np.vstack(error_list)
        if aggregated_errors.size == 0 or np.var(aggregated_errors) < 1e-8:
            print(f"      Aggregated errors for model {name} have near-zero variance. Skipping analysis for this model.")
            continue  # Skip the current model if errors have near-zero variance
        mean_mse = np.mean(mse_scores)
        mean_r2 = np.mean(r2_scores)
        errors[name] = {
            'errors': aggregated_errors if not low_variance_flag else np.array([]),
            'cv_score': mean_mse,
            'r2_score': mean_r2,
            'low_variance': low_variance_flag
        }
    return errors

def transform_to_fdata(errors):
    errors = np.array(errors).reshape(-1)
    if errors.size == 0 or np.var(errors) == 0:
        print("Error array is empty or has zero variance. Skipping this analysis.")
        return None  # Return None instead of raising an error    
    n_points = len(errors)
    grid_points = np.linspace(0, 1, n_points)
    data_matrix = errors.reshape((1, -1))
    fdata = FDataGrid(data_matrix=data_matrix, grid_points=[grid_points])
    return fdata

def add_jitter(data, jitter_amount=1e-6):
    return data + np.random.normal(0, jitter_amount, data.shape)

def synchronize_errors(errors_X1, errors_X2):
    # Convert errors to numpy arrays
    errors_X1 = np.array(errors_X1)
    errors_X2 = np.array(errors_X2)
    # Handle cases where either array is empty
    if errors_X1.size == 0 or errors_X2.size == 0:
        print("One of the error arrays is empty. Skipping synchronization.")
        return np.array([]), np.array([])  # Return empty arrays to avoid further processing
    # Find indices where either array contains NaN or inf
    invalid_indices = np.where(np.isnan(errors_X1) | np.isinf(errors_X1) | np.isnan(errors_X2) | np.isinf(errors_X2))[0]
    # Remove invalid indices from both arrays
    clean_errors_X1 = np.delete(errors_X1, invalid_indices)
    clean_errors_X2 = np.delete(errors_X2, invalid_indices)
    return clean_errors_X1, clean_errors_X2

def fda_analysis(errors_X1, errors_X2):
    # Clean and synchronize errors
    clean_errors_X1, clean_errors_X2 = synchronize_errors(errors_X1, errors_X2)
    # Add a check after cleaning and synchronizing errors
    if clean_errors_X1.size == 0 or clean_errors_X2.size == 0:
        print("Insufficient data after cleaning for FDA analysis.")
        return None, None, float('nan'), float('nan')  # Return None and NaNs
    errors_X1_jittered = add_jitter(clean_errors_X1)
    errors_X2_jittered = add_jitter(clean_errors_X2)
    fdata_x1 = transform_to_fdata(errors_X1_jittered)
    fdata_x2 = transform_to_fdata(errors_X2_jittered)
    if fdata_x1 is None or fdata_x2 is None:
        print("Functional data analysis skipped due to insufficient data.")
        return None, None, float('nan'), float('nan')  # Return None and NaNs
    # Check for zero variance
    min_variance_threshold = 1e-8  # Adjust based on your data scale
    if np.var(fdata_x1.data_matrix) < min_variance_threshold or np.var(fdata_x2.data_matrix) < min_variance_threshold:
        raise ValueError("Functional data has near-zero variance.")
    # Compute the variance of the functional data
    variance_x1 = var(fdata_x1).data_matrix[0, 0]  # Extract first element
    variance_x2 = var(fdata_x2).data_matrix[0, 0]  # Extract first element
    mean_x1 = mean(fdata_x1).data_matrix[0]  # Extract array
    mean_x2 = mean(fdata_x2).data_matrix[0]  # Extract array
    try:
        depth_x1 = ModifiedBandDepth().fit_transform(fdata_x1)
        depth_x2 = ModifiedBandDepth().fit_transform(fdata_x2)
        depth_comparison = depth_x1.mean() - depth_x2.mean()
    except Exception as e:
        depth_comparison = float('nan')
        print(f"Depth analysis failed: {e}")
    mean_difference = np.linalg.norm(mean_x1 - mean_x2)
    return variance_x1, variance_x2, depth_comparison, mean_difference

def fit_distributions(errors):
    # Check if errors array is empty
    if errors.size == 0:
        print("Error array is empty. Skipping distribution fitting.")
        return None, float('inf')
    if np.any(np.isinf(errors)) or np.any(np.isnan(errors)):
        print("Skipping distribution fitting due to extreme values in errors.")
        return None, float('inf')
    # List of distributions to try
    distributions = [stats.norm, stats.gamma, stats.beta, stats.expon]
    best_fit, best_aic = None, float('inf')
    # Normalize data for distributions like Beta
    min_error, max_error = errors.min(), errors.max()
    if min_error == max_error:  # This prevents division by zero
        print("No variation in errors. Skipping distribution fitting.")
        return None, float('inf')
    errors_normalized = (errors - min_error) / (max_error - min_error)
    for distribution in distributions:
        # Fit distribution to the data
        try:
            if distribution in [stats.beta]:
                params = distribution.fit(errors_normalized)
            else:
                params = distribution.fit(errors)
            # Calculate AIC
            aic = 2 * len(params) - 2 * distribution.logpdf(errors, *params).sum()
            if aic < best_aic:
                best_aic = aic
                best_fit = distribution
        except Exception as e:
            print(f"Error fitting {distribution.name}: {e}")
    return best_fit, best_aic

def prediction_error_analysis(errors_X1, errors_X2, threshold=3):
    if not isinstance(errors_X1, dict) or not isinstance(errors_X2, dict):
        raise ValueError("errors_X1 and errors_X2 should be dictionaries")
    scores_X1_to_X2 = 0
    scores_X2_to_X1 = 0
    common_models = set(errors_X1.keys()).intersection(errors_X2.keys())
    for model in common_models:
        errors_x1 = errors_X1[model]['errors'].ravel()
        errors_x2 = errors_X2[model]['errors'].ravel()
        # Statistical Tests
        shapiro_weight = 0.5
        shape_weight = 0.5
        autocorr_weight = 1.0
        fda_weight = 1.5  # Functional Data Analysis
        aic_weight = 2.0  # Distribution Fitting
        # Scoring for X1 -> X2 direction
        if len(errors_x1) >= 3:
            if shapiro(errors_x1)[1] > 0.05: 
                scores_X1_to_X2 += shapiro_weight
        if len(errors_x1) > 0:
            if abs(kurtosis(errors_x1, fisher=True)) < 3 and abs(skew(errors_x1)) < 0.5:
                scores_X1_to_X2 += shape_weight
            if len(errors_x1) > 1:  # More than 1 data point required for ACF with nlags=1
                if abs(acf(errors_x1, nlags=1)[1]) < 0.1: 
                    scores_X1_to_X2 += autocorr_weight
        # Scoring for X2 -> X1 direction
        if len(errors_x2) >= 3:
            if shapiro(errors_x2)[1] > 0.05: 
                scores_X2_to_X1 += shapiro_weight
        if len(errors_x2) > 0:
            if abs(kurtosis(errors_x2, fisher=True)) < 3 and abs(skew(errors_x2)) < 0.5:
                scores_X2_to_X1 += shape_weight
            if len(errors_x2) > 1:
                if abs(acf(errors_x2, nlags=1)[1]) < 0.1: 
                    scores_X2_to_X1 += autocorr_weight
        # FDA Analysis
        var_x1, var_x2, depth_comparison, mean_difference = fda_analysis(errors_x1, errors_x2)
        if var_x1 is None or var_x2 is None:
            print("Skipping FDA scoring due to earlier skip.")
            fda_weight = 0  # Set fda_weight to 0 to skip FDA scoring
        # Scoring based on FDA results
        if var_x1 is not None and var_x2 is not None:
            if var_x1 > var_x2: 
                scores_X1_to_X2 += fda_weight
            elif var_x2 > var_x1: 
                scores_X2_to_X1 += fda_weight
        # Scoring based on AIC
        aic_x1 = fit_distributions(errors_x1)[1]
        aic_x2 = fit_distributions(errors_x2)[1]
        if aic_x1 < aic_x2: 
            scores_X1_to_X2 += aic_weight
        else: 
            scores_X2_to_X1 += aic_weight
    # Calculate ratio and determine direction
    ratio = scores_X1_to_X2 / scores_X2_to_X1 if scores_X2_to_X1 != 0 else float('inf')
    if ratio >= threshold: 
        causal_direction = f"{list(errors_X1.keys())[0]} -> {list(errors_X2.keys())[0]}"
    elif ratio <= 1 / threshold: 
        causal_direction = f"{list(errors_X2.keys())[0]} -> {list(errors_X1.keys())[0]}"
    else: 
        causal_direction = "Inconclusive"
    return ratio, causal_direction

def analyze_causal_direction(data, column1, column2, models, threshold=3):
    # Ensure column1 and column2 are valid data columns
    if column1 not in data.columns or column2 not in data.columns:
        raise ValueError(f"Invalid columns: {column1}, {column2}")
    print(f"Analyzing causal direction between '{column1}' and '{column2}'")    
    models = get_models()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[[column1, column2]])
    column_names = (column1, column2)
    X1 = data_scaled[:, [0]]
    X2 = data_scaled[:, [1]]
    errors_X1 = fit_models_and_analyze_errors(X1, X2, models, column_names)
    errors_X2 = fit_models_and_analyze_errors(X2, X1, models, column_names)
    ratio, causal_direction = prediction_error_analysis(errors_X1, errors_X2, threshold)
    # Print the result with the threshold consideration
    if causal_direction != "Inconclusive":
        print(f"    Calculated Score Ratio: {ratio} (Direction: {causal_direction})")
    else:
        print(f"    Analysis inconclusive. Ratio of {ratio} does not meet the threshold of {threshold}")
    return column1, column2, ratio, causal_direction, errors_X1['LinearRegression']['errors'], errors_X2['LinearRegression']['errors']

def create_heatmap(matrix, labels, title):
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, mask=mask, xticklabels=labels, yticklabels=labels, cmap='viridis', annot=True)
    plt.title(title)
    plt.show()
    
def compute_summary_statistics(data):
    print("Computing summary statistics...")
    stats = data.describe().T
    correlation_matrix = data.corr()
    return stats, correlation_matrix

def normalize_data(data):
    return (data.sub(data.mean(axis=1), axis=0)).div(data.std(axis=1), axis=0)

# Main Script
def process_data(file_path):
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
    data = preprocess_data(data)    
    if use_normalize_input_data:
        data_normalized = normalize_data(data)
    else:
        data_normalized = data
    # Exclude date index from column names
    column_names = data_normalized.columns.tolist()
    num_vars = len(column_names)
    ratio_matrix = np.zeros((num_vars, num_vars))
    total_combinations = len(list(combinations(column_names, 2)))
    processed_combinations = 0    
    for i, (col1, col2) in enumerate(combinations(column_names, 2)):
        _, _, ratio, _, errors_col1, errors_col2 = analyze_causal_direction(data_normalized, col1, col2, get_models())
        index1, index2 = column_names.index(col1), column_names.index(col2)
        ratio_matrix[index1, index2] = ratio
        ratio_matrix[index2, index1] = 1 / ratio  # Inverse for the opposite direction
        processed_combinations += 1
        print(f"Processed {processed_combinations} of {total_combinations} column combinations")
    return ratio_matrix, column_names

# Example Usage
file_path = 'chems_macro_market_signals_with_consumer__partial.csv'  # Replace with your CSV file path
print("Starting causal relationship analysis...")
ratio_matrix, labels = process_data(file_path)

print("Generating heatmaps...")
create_heatmap(ratio_matrix, labels, "Causal Relationship Strength Heatmap")
print("Analysis complete.")
