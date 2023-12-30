import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.utils import resample
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress DataConversionWarnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def get_models():
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'BayesianRidge': BayesianRidge(),
        'Lasso': Lasso(alpha=0.01),  # Reduced regularization
        'ElasticNet': ElasticNet(alpha=0.01),  # Reduced regularization
        'GradientBoosting': GradientBoostingRegressor(),
        'KNeighbors': KNeighborsRegressor(),
        'SVR': SVR(),
    }

def fit_models_and_analyze_errors(X, y, models, column_names):
    threshold = 0.001
    print(f"Analyzing errors for columns: {column_names[0]} as independent and {column_names[1]} as dependent")
    # Convert X to NumPy array and reshape if it's a pandas Series or 1D NumPy array
    if isinstance(X, pd.Series) or (isinstance(X, np.ndarray) and X.ndim == 1):
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        X = X.reshape(-1, 1)
    if isinstance(y, pd.Series) or (isinstance(y, np.ndarray) and y.ndim == 1):
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        y = y.reshape(-1, 1)
    errors = {}
    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in models.items():
        print(f"    Fitting model: {name}")
        error_list = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Reshape X_train and X_test if they are 1D
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            # Reshape y_train and y_test if they are 1D
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            # After model.predict(X_test)
            if np.var(predictions) < threshold:
                print(f"Low variance in predictions for model {name}. Might indicate model underperformance.")
            error_list.append(y_test - predictions)
        # Aggregate errors across all folds
        aggregated_errors = np.concatenate(error_list)
        # After computing aggregated_errors
        if np.var(aggregated_errors) < threshold:
            print(f"Low variance in error for model {name}. Might indicate model underperformance.")
        # Flatten y for cross_val_score
        y_flattened = np.ravel(y)
        # Compute a cross-validated score for each model (e.g., MSE)
        mse_scorer = make_scorer(mean_squared_error)
        cv_score = cross_val_score(model, X, y_flattened, cv=tscv, scoring=mse_scorer)
        # Store errors, CV score, and R² score
        r2_scorer = make_scorer(r2_score)
        r2_score_result = cross_val_score(model, X, y_flattened, cv=tscv, scoring=r2_scorer)
        errors[name] = {'errors': aggregated_errors, 'cv_score': np.mean(cv_score), 'r2_score': np.mean(r2_score_result)}
    return errors

def calculate_mahalanobis_distance(errors):
    if errors.ndim < 2:
        errors = errors[:, np.newaxis]
    print("Errors shape:", errors.shape)  # Diagnostic print
    if errors.size == 0 or (errors.size == 1 and errors.ndim == 1):
        print("Insufficient data for Mahalanobis calculation")
        return np.nan
    if np.isnan(errors).any():
        print("NaN values found in errors")
        return np.nan
    mean = np.mean(errors, axis=0)
    cov = np.cov(errors.T)
    if cov.ndim < 2 or np.linalg.det(cov) == 0:
        print("Covariance matrix is singular or not 2D")
        return np.nan
    # Additional diagnostic checks
    print("Error term sample:", errors[:5])  # Print a sample of the error terms
    print("Error term variance:", np.var(errors))  # Print variance of the errors
    try:
        inv_covmat = np.linalg.inv(cov)
        distances = [mahalanobis(err, mean, inv_covmat) for err in errors]
        return np.mean(chi2.cdf(np.square(distances), df=errors.shape[1]))
    except np.linalg.LinAlgError:
        print("Error in calculating Mahalanobis distance")
        return np.nan

def calculate_mahalanobis_ratio(data1, data2, models, column_names):
    errors_X1 = fit_models_and_analyze_errors(data1, data2, models, column_names)
    errors_X2 = fit_models_and_analyze_errors(data2, data1, models, column_names)
    mahalanobis_X1 = np.mean([calculate_mahalanobis_distance(e['errors']) for e in errors_X1.values()])
    mahalanobis_X2 = np.mean([calculate_mahalanobis_distance(e['errors']) for e in errors_X2.values()])
    return mahalanobis_X1 / mahalanobis_X2

def permutation_test_mahalanobis(data1, data2, models, column_names, n_iterations=1000):
    print("    Performing permutation test...")    
    observed_ratio = calculate_mahalanobis_ratio(data1, data2, models, column_names)
    count = 0
    for _ in range(n_iterations):
        permuted_data1, permuted_data2 = np.random.permutation(data1), np.random.permutation(data2)
        permuted_ratio = calculate_mahalanobis_ratio(permuted_data1, permuted_data2, models, column_names)
        count += permuted_ratio > observed_ratio
    p_value = count / n_iterations
    return p_value

def bootstrap_test(data1, data2, n_iterations=1000, size=None):
    print("    Performing bootstrap test...")
    if size is None:
        size = max(len(data1), len(data2))
    observed_diff = np.mean(data1) - np.mean(data2)
    count = 0
    for _ in range(n_iterations):
        sample1 = resample(data1, n_samples=size)
        sample2 = resample(data2, n_samples=size)
        diff = np.mean(sample1) - np.mean(sample2)
        count += diff > observed_diff
    p_value = count / n_iterations
    return p_value

def analyze_causal_direction(data, column1, column2, models):
    print(f"Analyzing causal direction between '{column1}' and '{column2}'")    
    models = get_models()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[[column1, column2]])
    column_names = (column1, column2)  # Tuple containing the names of the columns
    X1 = data_scaled[:, [0]]
    X2 = data_scaled[:, [1]]
    errors_X1 = fit_models_and_analyze_errors(X1, X2, models, column_names)
    errors_X2 = fit_models_and_analyze_errors(X2, X1, models, column_names)
    mahalanobis_X1 = np.mean([calculate_mahalanobis_distance(e['errors']) for e in errors_X1.values()])
    mahalanobis_X2 = np.mean([calculate_mahalanobis_distance(e['errors']) for e in errors_X2.values()])
    ratio = mahalanobis_X1 / mahalanobis_X2
    # Determine the direction and construct the arrow string
    if ratio > 1:
        causal_direction = f"{column1} ⮕ {column2}"
    else:
        causal_direction = f"{column2} ⮕ {column1}"
    print(f"    Calculated Mahalanobis ratio: {ratio} (Direction: {causal_direction})")
    return column1, column2, ratio, causal_direction, errors_X1['LinearRegression']['errors'], errors_X2['LinearRegression']['errors']

def normalize_data(data):
    return (data.sub(data.mean(axis=1), axis=0)).div(data.std(axis=1), axis=0)

def create_heatmap(matrix, labels, title):
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, mask=mask, xticklabels=labels, yticklabels=labels, cmap='viridis', annot=True)
    plt.title(title)
    plt.show()

# Main Script
def process_data(file_path):
    print(f"Loading data from {file_path}")    
    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
    data = data.replace([np.inf, -np.inf], np.nan).ffill()
    data_normalized = normalize_data(data)
    num_vars = len(data_normalized.columns)
    mahalanobis_matrix = np.zeros((num_vars, num_vars))
    p_value_matrix__permutation = np.zeros((num_vars, num_vars))  # Matrix to store p-values (permutation test)
    p_value_matrix__bootstrap = np.zeros((num_vars, num_vars))  # Matrix to store p-values (bootstrap test)
    total_combinations = len(list(combinations(data_normalized.columns, 2)))
    processed_combinations = 0    
    for i, (col1, col2) in enumerate(combinations(data_normalized.columns, 2)):
        data1, data2 = data_normalized[col1], data_normalized[col2]        
        _, _, ratio, _, errors_col1, errors_col2 = analyze_causal_direction(data_normalized, col1, col2, get_models())
        p_value__permutation = permutation_test_mahalanobis(data1, data2, get_models(), (col1, col2))
        p_value_bootstrap = bootstrap_test(errors_col1, errors_col2)
        index1, index2 = data_normalized.columns.get_loc(col1), data_normalized.columns.get_loc(col2)
        mahalanobis_matrix[index1, index2] = ratio
        mahalanobis_matrix[index2, index1] = 1 / ratio  # Inverse for the opposite direction
        p_value_matrix__permutation[index1, index2] = p_value__permutation
        p_value_matrix__permutation[index2, index1] = p_value__permutation  # Same p-value for both directions
        p_value_matrix__bootstrap[index1, index2] = p_value_bootstrap
        p_value_matrix__bootstrap[index2, index1] = p_value_bootstrap  # Same p-value for both directions
        processed_combinations += 1
        print(f"Processed {processed_combinations} of {total_combinations} column combinations")
    return mahalanobis_matrix, p_value_matrix__permutation, p_value_matrix__bootstrap, data_normalized.columns

# Example Usage
file_path = 'chems_macro_market_signals_with_consumer.csv'  # Replace with your CSV file path
print("Starting causal relationship analysis...")
mahalanobis_matrix, p_value_matrix__permutation, p_value_matrix__bootstrap, labels = process_data(file_path)

print("Generating heatmaps...")
create_heatmap(mahalanobis_matrix, labels, "Causal Relationship Strength Heatmap")
create_heatmap(p_value_matrix__permutation, labels, "P-Value (Permutation Test) Heatmap for Causal Relationships")
create_heatmap(p_value_matrix__bootstrap, labels, "P-Value (Bootstrap Test) Heatmap for Causal Relationships")
print("Analysis complete.")