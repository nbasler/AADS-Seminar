# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
    train_test_split
)
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
    precision_recall_curve, accuracy_score, auc,
    roc_curve, RocCurveDisplay,
    make_scorer, fbeta_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler,OneHotEncoder
from sklearn.pipeline import Pipeline

# For sampling techniques
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.utils.class_weight import compute_class_weight

# For CNN
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, Concatenate, BatchNormalization
    )
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasClassifier

# $$
# Config class
class Config:
    RANDOM_STATE = 42
    test_size = 0.1
    
config = Config()
#%%
# Function to create a simple classification
def create_optimized_parameter_grids():
    """
    Create optimized parameter grids for classical supervised-learning models.
    Retunrs:
        - nested Dictionary with model names as keys and dictionaries containing the parameters.
    """
    return {
        "SGD": {
            'model': Pipeline([
                ('scaler', StandardScaler()),  # Could also apply MinMaxScaler or RobustScaler
                ('classifier', SGDClassifier(random_state=42, n_jobs=-1))
            ]),
            'param_grid': {
                'classifier__loss': ['hinge', 'log_loss'],
                'classifier__max_iter': [100, 10000],
                'classifier__tol': [0.00001, 0.01],
                'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                'classifier__alpha': [0.00001, 0.01],
                'classifier__learning_rate': ['constant', 'optimal'],
                'classifier__eta0': [0.001, 0.1],
                'classifier__class_weight': [None, 'balanced'],
                }
        },
        # SVC takes exponentially longer to train than SGD, so it is commented out by default.
        # Uncomment to use SVC, but be aware of the long training time.
        # "SVC": {
        #     'model': Pipeline([
        #         ('scaler', StandardScaler()),
        #         ('classifier', SVC(probability=True, random_state=42))
        #     ]),
        #     'param_grid': {
        #         'classifier__C': [0.1, 10],
        #         'classifier__kernel': ['linear', 'rbf', 'poly'],
        #         'classifier__gamma': ['scale', 'auto'],
        #         'classifier__class_weight': [None, 'balanced'],
        #         'classifier__shrinking': [True, False],
        #         'classifier__probability': [True],
        #         'classifier__tol': [0.0001, 0.01],
        #         'classifier__max_iter': [100, 1000]
        #     }
        # },
        "Logistic": {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42))
            ]),
            'param_grid': {
                'classifier__solver': ['saga', 'lbfgs'],
                'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__class_weight': [None, 'balanced'],
                'classifier__l1_ratio': [None, 1],
                'classifier__tol': [0.00001, 0.01],
                'classifier__fit_intercept': [True, False],
                'classifier__max_iter': [100, 1000],
                'classifier__l1_ratio': [None, 0.5, 1]  # Only for 'elasticnet' penalty 
            }
        },
        "RandomForest": {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(bootstrap=True, oob_score=True, random_state=42, n_jobs=-1))
            ]),
            'param_grid': {
                'classifier__n_estimators': [10, 100],
                'classifier__max_depth': [None, 25, 50],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__max_features': ['sqrt', 'log2'],
                'classifier__class_weight': ['balanced', None],
                'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            }
        },
        "KNN": {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', KNeighborsClassifier(n_jobs=-1))
            ]),
            'param_grid': {
                'classifier__n_neighbors': [2, 10],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan'],
                'classifier__p': [1, 2]
            }
        },
        "GradientBoosting": {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ]),
            'param_grid': {
                'classifier__n_estimators': [100, 1000],
                'classifier__learning_rate': [0.001, 0.1],
                'classifier__max_depth': [2, 10],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__subsample': [0.2, 1.0],
                'classifier__max_features': ['sqrt', 'log2']
            }
        }
    }

def create_custom_scorer():
    """
    Custom scorer for model evaluation.
    
    Returns:
        - Scorer function that uses Fbeta-Score.
    """
    # F2-Score: F2-Score emphasizes recall more than precision.
    # Fbeta-Score: Beta > 1 weights recall(FNR) more than precision.
    return make_scorer(fbeta_score, beta=2) 

# Applying sample weights to avoid class imbalance issues at the 
# model itself, thus not relying on resampling techniques with artifical data,
# which could impair the models variance and generalization capabilities.
# def apply_sampling_strategy(X_train, y_train, strategy='smote'):
#     """
#     Applies different sampling strategies to the dataset.
#     Args:
#         - X_train: Features.
#         - y_trainy: Class Labels
#         - strategy: Sampling strategy to apply ('smote', 'smoteenn', 'undersample', 'none').
#     Returns:
#         - X_train_resampled: Resampled features.
#         - y_train_resampled: Resampled class labels.
#     """
    
#     if strategy == 'smote':
#         sampler = SMOTE(random_state=42)
#     elif strategy == 'smoteenn':
#         sampler = SMOTEENN(random_state=42)
#     elif strategy == 'undersample':
#         sampler = RandomUnderSampler(random_state=42)
#     else:
#         return X_train, y_train
    
#     X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
#     return X_train_resampled, y_train_resampled

def train_and_evaluate_models_random(X_train, y_train, param_grids, n_iter=20, scoring=None, cv_folds=5, random_state=42):
    """
    Trains and tunes models using RandomizedSearchCV and cross_val_predict.
    Returns a results dict and a summary DataFrame.
    """
    results = {}
    summary_rows = []
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    y_train = np.asarray(y_train).squeeze()
    if y_train.ndim > 1:
        y_train = y_train.ravel()
        
    for name, grid_info in param_grids.items():
        print(f"\nOptimizing {name}...")
        start_time = time.time()
        search = RandomizedSearchCV(
            estimator=grid_info['model'],
            param_distributions=grid_info['param_grid'],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=random_state
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Cross-validated predictions for metrics
        y_pred = cross_val_predict(best_model, X_train, y_train, cv=cv, n_jobs=-1)
        if hasattr(best_model, "predict_proba"):
            y_scores = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        else:
            y_scores = cross_val_predict(best_model, X_train, y_train, cv=cv, method="decision_function", n_jobs=-1)
        cm = confusion_matrix(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        recall = recall_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)
        f2 = fbeta_score(y_train, y_pred, beta=2)
        acc = accuracy_score(y_train, y_pred)
        fpr, tpr, _ = roc_curve(y_train, y_scores)
        auc_roc = auc(fpr, tpr)
        elapsed = time.time() - start_time
        
        results[name] = {
            "best_model": best_model,
            "best_params": search.best_params_,
            "cv_results": search.cv_results_,
            "y_pred": y_pred,
            "y_scores": y_scores,
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "accuracy": acc,
            "roc_curve": (fpr, tpr, auc_roc),
            "train_time": elapsed
        }
        summary_rows.append({
            "Model": name,
            "Accuracy": f"{acc:.4f}",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
            "F1": f"{f1:.4f}",
            "F2": f"{f2:.4f}",
            "AUC-ROC": f"{auc_roc:.4f}",
            "Train Time (s)": f"{elapsed:.2f}",
            "Best Params": search.best_params_
        })
    summary_df = pd.DataFrame(summary_rows)
    return results, summary_df

def train_best_model_from_summary(X_train, y_train, summary_df, param_grids, model_name=None, random_state=42):
    """
    Trains classifier(s) using the best parameters from summary_df.
    If model_name is specified, trains only that model. If not, trains all models in summary_df.
    Returns the trained classifier if model_name is given, or a dict of trained classifiers if not.
    """
    import ast
    y_train = np.asarray(y_train).squeeze()
    if y_train.ndim > 1:
        y_train = y_train.ravel()
    if model_name is not None:
        # Train only the specified model
        row = summary_df[summary_df['Model'] == model_name]
        if row.empty:
            raise ValueError(f"Model '{model_name}' not found in summary_df.")
        best_params = row['Best Params'].values[0]
        if isinstance(best_params, str):
            best_params = ast.literal_eval(best_params)
        model_pipeline = param_grids[model_name]['model']
        model_pipeline.set_params(**best_params)
        model_pipeline.fit(X_train, y_train)
        return model_pipeline
    else:
        # Train all models in summary_df
        trained_models = {}
        for model_name in summary_df['Model']:
            row = summary_df[summary_df['Model'] == model_name]
            best_params = row['Best Params'].values[0]
            if isinstance(best_params, str):
                best_params = ast.literal_eval(best_params)
            model_pipeline = param_grids[model_name]['model']
            model_pipeline.set_params(**best_params)
            model_pipeline.fit(X_train, y_train)
            trained_models[model_name] = model_pipeline
        return trained_models

def evaluate_models_on_test(trained_models, X_test, y_test):
    """
    Evaluates all trained models on the test data and stores results in the same format as train_and_evaluate_models_random.
    Args:
        - trained_models: dict of trained model pipelines (as from train_best_model_from_summary)
        - X_test: Test features
        - y_test: Test labels
    Returns:
        - results: dict with evaluation results for each model
    """
    
    results_test = {}
    y_test = np.asarray(y_test).squeeze()
    if y_test.ndim > 1:
        y_test = y_test.ravel()
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.decision_function(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc_roc = auc(fpr, tpr)
        
        results_test[name] = {
            "y_pred": y_pred,
            "y_scores": y_scores,
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "accuracy": acc,
            "roc_curve": (fpr, tpr, auc_roc)
        }
        
    return results_test

#%%
# Convolutional Neural Network (CNN) for image classification

def create_cnn_model(learning_rate=0.001, dropout_rate=0.3, dense_units=64, input_shape=None, **kwargs):
    """
    Creates an optimised CNN model with BatchNormalisation and extended layers.
    Handles both grayscale and RGB images.
    """
    if input_shape is None:
        input_shape = (64, 64, 1)  # Default to grayscale
    # If input_shape is (H, W), add channel dim for grayscale
    if len(input_shape) == 2:
        input_shape = (*input_shape, 1)
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate/2),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate/2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        Flatten(),
        Dense(dense_units, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(dense_units//2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate/2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def train_cnn(input_df, feature_column='pixel_array', target_column='classification'):
    """
    Trains a CNN model on image data from a DataFrame.
    The function also further downscales the images compared to the already
    downsized images used in the previous version.
    """
    # Ensure input_df is a DataFrame
    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("input_df must be a pandas DataFrame.")
    
    # Retrieve image shape from the first entry in the feature column
    img_shape = input_df[feature_column].iloc[0].shape
    
    # Convert feature and target columns to numpy arrays
    # Stack all images into a 4D numpy array (n_samples, H, W, C)
    X = np.stack(input_df[feature_column].values)
    if X.ndim == 3:
        # If grayscale images without channel dim, add channel dim
        X = X[..., np.newaxis]
    elif X.ndim != 4:
        raise ValueError(f"Images must be 3D (H,W) or 4D (H,W,C). Got shape {X.shape}")
    
    # Check if the input data has both classes
    n_classes = input_df[target_column].nunique()
    if n_classes < 2:
        raise ValueError("Need at least two classes in 'classification' column for classification. Found only one.")

    # Encode the target variable into integers
    cat_encoder = OneHotEncoder(drop='first')  # Use drop='first' to avoid dummy variable trap
    y = cat_encoder.fit_transform(input_df[[target_column]]).toarray()
    y = np.asarray(y).squeeze()
    if y.ndim > 1:
        y = y.ravel()
    
    # Ensure y is not float but integer type
    if y.dtype == 'float64':
        y = y.astype('int32')

    # Store labels of class in case needed
    classification_labels = input_df[target_column].astype('category')
       
    # Get train test split
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        test_size = config.test_size,
                                                        random_state = config.RANDOM_STATE,
                                                        shuffle = True,
                                                        stratify = y
                                                        )    
    
    # Calculate class weights for imbalanced classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Create KerasClassifier for GridSearch
    cnn_model = KerasClassifier(
        model=create_cnn_model,
        model__input_shape=img_shape if len(img_shape) == 3 else (*img_shape, 1),
        verbose=0
    )
    
    # Define Parameter-Grid
    param_grid = {
        'model__learning_rate': [0.0001, 0.001],
        'model__dropout_rate': [0.3, 0.4, 0.5],
        'model__dense_units': [64, 128, 256],
        'batch_size': [16, 32],
        'epochs': [20, 30]
    }
    
    # Callbacks for training improvements
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # GridSearch mit Zeitmessung
    print("\n Optimizing CNN...")
    train_start_time = time.time()
    
    grid_search = RandomizedSearchCV(
        estimator=cnn_model,
        param_distributions=param_grid,
        cv=3,
        scoring=create_custom_scorer(),
        n_jobs=-1,
        verbose=0
    )
    
    # Fit without SMOTE (implemented in previous verison),
    # just use class weights
    grid_search.fit(
        X_train,
        y_train,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        validation_split=0.2
    )
    
    train_time = time.time() - train_start_time
    
    # Test using best parameter
    test_start_time = time.time()
    best_model = grid_search.best_estimator_.model
    y_pred = best_model.predict(X_test).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)
    test_time = time.time() - test_start_time
    
    # Konfusionsmatrix berechnen
    cm = confusion_matrix(y_test, y_pred_binary)
    
    results = {
        'cnn': {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'y_pred': y_pred_binary,
            'y_scores': y_pred,
            'precision': precision_score(y_test, y_pred_binary),
            'recall': recall_score(y_test, y_pred_binary),
            'f1': f1_score(y_test, y_pred_binary),
            'f2': fbeta_score(y_test, y_pred_binary, beta=2),
            'confusion_matrix': cm,
            'train_time': train_time,
            'test_time': test_time,
            'cv_results': grid_search.cv_results_
        }
    }
    
    print(f"\nCNN Ergebnisse:")
    print(f"  F2-Score:  {results['cnn']['f2']:.4f}")
    print(f"  Precision: {results['cnn']['precision']:.4f}")
    print(f"  Recall:    {results['cnn']['recall']:.4f}")
    print(f"  F1-Score:  {results['cnn']['f1']:.4f}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  Trainingszeit: {train_time:.2f} Sekunden")
    print(f"  Testzeit: {test_time:.2f} Sekunden")
    
    return results

def plot_cnn_special_visualizations(cnn_results, y_test):
    """Spezielle Visualisierungen für CNN-Modelle."""
    # 1. AUC-PR und AUC-ROC für CNN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for name, res in cnn_results.items():
        # PR Kurve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, res['y_scores'])
        auc_pr = auc(recall_curve, precision_curve)
        ax1.plot(recall_curve, precision_curve, marker='.', 
                label=f'{name} (AUC-PR = {auc_pr:.3f})', linewidth=2)
        
        # ROC Kurve
        fpr, tpr, _ = roc_curve(y_test, res['y_scores'])
        auc_roc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'{name} (AUC-ROC = {auc_roc:.3f})', linewidth=2)
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('CNN Precision-Recall Kurven', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot([0, 1], [0, 1], 'k--', label='Zufällig', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('CNN ROC Kurven', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Threshold-Vergleichstabelle für Multi-Input CNN
    if 'multi_cnn' in cnn_results:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_data = []
        
        for threshold in thresholds:
            y_pred_threshold = (cnn_results['multi_cnn']['y_scores'] > threshold).astype(int)
            threshold_data.append({
                'Threshold': f'{threshold:.1f}',
                'Precision': f"{precision_score(y_test, y_pred_threshold):.4f}",
                'Recall': f"{recall_score(y_test, y_pred_threshold):.4f}",
                'F1-Score': f"{f1_score(y_test, y_pred_threshold):.4f}"
            })
        
        df_threshold = pd.DataFrame(threshold_data)
        
        # Erstelle Tabelle
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df_threshold.values, colLabels=df_threshold.columns, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Färbe Header
        for i in range(len(df_threshold.columns)):
            table[(0, i)].set_facecolor('#FF6B6B')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Multi-Input CNN: Threshold-Vergleich', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

# %%
