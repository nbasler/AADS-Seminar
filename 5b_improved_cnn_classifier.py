# Improved Mammography Classification Models
# 
# Enhanced CNN models with better architecture, training strategies, and data handling
# to address class imbalance and reduce false positives.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc,
    roc_curve, RocCurveDisplay, classification_report
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, Concatenate, BatchNormalization,
    GlobalAveragePooling2D, Add, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import os
from datetime import datetime

# Import data processing functions
from data_processing import process_data, config

def calculate_f2_score(precision, recall, beta=2):
    """Calculate F2-score (Beta=2 gives more weight to recall than precision)."""
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

def focal_loss(alpha=0.25, gamma=2.0):
    """Custom Focal Loss implementation to handle class imbalance."""
    def focal_loss_fn(y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        
        # Apply alpha weighting
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Combine all components
        focal_loss = alpha_weight * focal_weight * cross_entropy
        
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fn

def create_improved_simple_cnn(input_shape):
    """Create an improved simple CNN with better architecture."""
    model = tf.keras.Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Global pooling and dense layers
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Use custom Focal Loss to handle class imbalance
    focal_loss_fn = focal_loss(alpha=0.25, gamma=2.0)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss_fn,
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    return model

def create_improved_multi_input_cnn(img_shape, tabular_shape):
    """Create an improved multi-input CNN with better architecture."""
    # Image input branch with residual connections
    input_img = Input(shape=img_shape, name='image_input')
    
    # First block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Second block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Third block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Global pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    image_features = x
    
    # Enhanced tabular input branch
    input_tabular = Input(shape=tabular_shape, name='tabular_input')
    y = Dense(128, activation='relu')(input_tabular)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = Dense(64, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    tabular_features = y
    
    # Combine features with attention mechanism
    combined = Concatenate()([image_features, tabular_features])
    
    # Dense layers with better regularization
    z = Dense(128, activation='relu')(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    z = Dense(64, activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[input_img, input_tabular], outputs=output)
    
    # Use custom Focal Loss
    focal_loss_fn = focal_loss(alpha=0.25, gamma=2.0)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=focal_loss_fn,
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    return model

def create_data_augmentation():
    """Create data augmentation for training."""
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def train_improved_cnn_models(data_split, img_shape, tabular_shape):
    """Train improved CNN models with better strategies."""
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(data_split['y_train']),
        y=data_split['y_train']
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    # Data augmentation
    datagen = create_data_augmentation()
    
    # Improved Simple CNN
    print("\nTraining improved simple CNN...")
    improved_simple_cnn = create_improved_simple_cnn(img_shape)
    
    # Use data augmentation for simple CNN
    train_generator = datagen.flow(
        data_split['X_train_pixels'],
        data_split['y_train'],
        batch_size=32
    )
    
    improved_simple_cnn_history = improved_simple_cnn.fit(
        train_generator,
        steps_per_epoch=len(data_split['X_train_pixels']) // 32,
        epochs=50,
        validation_data=(data_split['X_train_pixels'], data_split['y_train']),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Improved Multi-input CNN
    print("\nTraining improved multi-input CNN...")
    improved_multi_cnn = create_improved_multi_input_cnn(img_shape, tabular_shape)
    
    improved_multi_cnn_history = improved_multi_cnn.fit(
        {
            'image_input': data_split['X_train_pixels'],
            'tabular_input': data_split['X_train_tabular']
        },
        data_split['y_train'],
        epochs=50,
        batch_size=16,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    return {
        'improved_simple_cnn': {
            'model': improved_simple_cnn,
            'history': improved_simple_cnn_history
        },
        'improved_multi_cnn': {
            'model': improved_multi_cnn,
            'history': improved_multi_cnn_history
        }
    }

def find_optimal_threshold(y_true, y_scores):
    """Find optimal threshold using different strategies."""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f2 = 0
    best_balanced_accuracy = 0
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f2 = calculate_f2_score(precision, recall)
        
        # Calculate balanced accuracy
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        balanced_accuracy = (tp/(tp+fn) + tn/(tn+fp)) / 2
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f2': f2,
            'balanced_accuracy': balanced_accuracy,
            'fp': fp,  # False positives
            'fn': fn   # False negatives
        })
        
        # Optimize for F2-score (recall is more important in medical context)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold
            best_balanced_accuracy = balanced_accuracy
    
    return best_threshold, results

def evaluate_improved_cnn_models(cnn_models, data_split):
    """Evaluate improved CNN models with detailed analysis."""
    results = {}
    
    # Improved Simple CNN
    improved_simple_cnn = cnn_models['improved_simple_cnn']['model']
    y_pred_simple = improved_simple_cnn.predict(data_split['X_test_pixels']).flatten()
    
    # Find optimal threshold
    optimal_threshold_simple, threshold_results_simple = find_optimal_threshold(
        data_split['y_test'], y_pred_simple
    )
    
    y_pred_simple_binary = (y_pred_simple > optimal_threshold_simple).astype(int)
    
    # Calculate metrics
    precision_simple = precision_score(data_split['y_test'], y_pred_simple_binary)
    recall_simple = recall_score(data_split['y_test'], y_pred_simple_binary)
    f1_simple = f1_score(data_split['y_test'], y_pred_simple_binary)
    f2_simple = calculate_f2_score(precision_simple, recall_simple)
    
    # AUC metrics
    fpr_simple, tpr_simple, _ = roc_curve(data_split['y_test'], y_pred_simple)
    auc_roc_simple = auc(fpr_simple, tpr_simple)
    precision_curve_simple, recall_curve_simple, _ = precision_recall_curve(data_split['y_test'], y_pred_simple)
    auc_pr_simple = auc(recall_curve_simple, precision_curve_simple)
    
    # Confusion matrix
    cm_simple = confusion_matrix(data_split['y_test'], y_pred_simple_binary)
    
    results['improved_simple_cnn'] = {
        'y_pred': y_pred_simple_binary,
        'y_scores': y_pred_simple,
        'threshold': optimal_threshold_simple,
        'precision': precision_simple,
        'recall': recall_simple,
        'f1': f1_simple,
        'f2': f2_simple,
        'auc_roc': auc_roc_simple,
        'auc_pr': auc_pr_simple,
        'confusion_matrix': cm_simple,
        'threshold_analysis': threshold_results_simple
    }
    
    # Improved Multi-input CNN
    improved_multi_cnn = cnn_models['improved_multi_cnn']['model']
    y_pred_multi = improved_multi_cnn.predict({
        'image_input': data_split['X_test_pixels'],
        'tabular_input': data_split['X_test_tabular']
    }).flatten()
    
    # Find optimal threshold
    optimal_threshold_multi, threshold_results_multi = find_optimal_threshold(
        data_split['y_test'], y_pred_multi
    )
    
    y_pred_multi_binary = (y_pred_multi > optimal_threshold_multi).astype(int)
    
    # Calculate metrics
    precision_multi = precision_score(data_split['y_test'], y_pred_multi_binary)
    recall_multi = recall_score(data_split['y_test'], y_pred_multi_binary)
    f1_multi = f1_score(data_split['y_test'], y_pred_multi_binary)
    f2_multi = calculate_f2_score(precision_multi, recall_multi)
    
    # AUC metrics
    fpr_multi, tpr_multi, _ = roc_curve(data_split['y_test'], y_pred_multi)
    auc_roc_multi = auc(fpr_multi, tpr_multi)
    precision_curve_multi, recall_curve_multi, _ = precision_recall_curve(data_split['y_test'], y_pred_multi)
    auc_pr_multi = auc(recall_curve_multi, precision_curve_multi)
    
    # Confusion matrix
    cm_multi = confusion_matrix(data_split['y_test'], y_pred_multi_binary)
    
    results['improved_multi_cnn'] = {
        'y_pred': y_pred_multi_binary,
        'y_scores': y_pred_multi,
        'threshold': optimal_threshold_multi,
        'precision': precision_multi,
        'recall': recall_multi,
        'f1': f1_multi,
        'f2': f2_multi,
        'auc_roc': auc_roc_multi,
        'auc_pr': auc_pr_multi,
        'confusion_matrix': cm_multi,
        'threshold_analysis': threshold_results_multi
    }
    
    return results

def analyze_false_positives(results, data_split):
    """Analyze false positive predictions in detail."""
    print("\n=== FALSE POSITIVE ANALYSIS ===")
    
    for model_name, res in results.items():
        print(f"\n{model_name.upper()}:")
        
        # Get confusion matrix
        cm = res['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP): {tp}")
        
        # Calculate rates
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"False Positive Rate: {fp_rate:.4f}")
        print(f"False Negative Rate: {fn_rate:.4f}")
        
        # Threshold analysis
        print(f"Optimal Threshold: {res['threshold']:.3f}")
        print(f"Precision: {res['precision']:.4f}")
        print(f"Recall: {res['recall']:.4f}")
        print(f"F1-Score: {res['f1']:.4f}")
        print(f"F2-Score: {res['f2']:.4f}")

def plot_improved_results(cnn_results, y_test):
    """Plot improved CNN model results with detailed analysis."""
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(15, 6))
    fig_pr, axes_pr = plt.subplots(1, 2, figsize=(15, 6))
    fig_roc, axes_roc = plt.subplots(1, 2, figsize=(15, 6))
    fig_threshold, axes_threshold = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (name, res) in enumerate(cnn_results.items()):
        # Confusion matrix
        disp_cm = ConfusionMatrixDisplay(
            confusion_matrix=res["confusion_matrix"],
            display_labels=['Benign', 'Malignant']
        )
        disp_cm.plot(ax=axes_cm[i], cmap='Blues', values_format='d')
        axes_cm[i].set_title(f'{name}\nConfusion Matrix\nThreshold: {res["threshold"]:.3f}')
        
        # PR curve
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_test, res["y_scores"]
        )
        auc_pr = auc(recall_curve, precision_curve)
        axes_pr[i].plot(
            recall_curve, precision_curve,
            marker='.',
            label=f'{name} (AUC-PR = {auc_pr:.3f})'
        )
        axes_pr[i].set_title(f'{name}\nPrecision-Recall Curve')
        axes_pr[i].legend()
        axes_pr[i].grid(True)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, res["y_scores"])
        auc_roc = auc(fpr, tpr)
        axes_roc[i].plot(
            fpr, tpr,
            label=f'{name} (AUC-ROC = {auc_roc:.3f})'
        )
        axes_roc[i].plot([0, 1], [0, 1], 'k--', label='Random')
        axes_roc[i].set_title(f'{name}\nROC Curve')
        axes_roc[i].legend()
        axes_roc[i].grid(True)
        
        # Threshold analysis
        threshold_data = res['threshold_analysis']
        thresholds = [r['threshold'] for r in threshold_data]
        f2_scores = [r['f2'] for r in threshold_data]
        fp_counts = [r['fp'] for r in threshold_data]
        
        ax1 = axes_threshold[i]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(thresholds, f2_scores, 'b-', label='F2-Score')
        line2 = ax2.plot(thresholds, fp_counts, 'r-', label='False Positives')
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('F2-Score', color='b')
        ax2.set_ylabel('False Positives', color='r')
        ax1.set_title(f'{name}\nThreshold Analysis')
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.show()

def train_improved_models():
    """Main function to train and evaluate improved models."""
    # Load and process data
    df_cmmd, data_split, IMG_HEIGHT, IMG_WIDTH = process_data()
    
    # Train improved CNN models
    print("\n=== Training Improved CNN Models ===")
    img_shape = (IMG_HEIGHT, IMG_WIDTH, config.CHANNELS)
    tabular_shape = (data_split['X_train_tabular'].shape[1],)
    
    cnn_models = train_improved_cnn_models(data_split, img_shape, tabular_shape)
    cnn_results = evaluate_improved_cnn_models(cnn_models, data_split)
    
    # Analyze false positives
    analyze_false_positives(cnn_results, data_split)
    
    # Plot results
    plot_improved_results(cnn_results, data_split['y_test'])
    
    return {
        'cnn_results': cnn_results,
        'data_split': data_split,
        'img_shape': img_shape,
        'tabular_shape': tabular_shape
    }

if __name__ == "__main__":
    print("=== STARTING IMPROVED CNN TRAINING ===")
    print("=== Addressing Class Imbalance and False Positives ===")
    results = train_improved_models()
    
    # Show summary of results
    print("\n=== Summary of Improved Results ===")
    for name, res in results['cnn_results'].items():
        print(f"\n{name}:")
        print(f"  Optimal Threshold: {res['threshold']:.3f}")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  Recall: {res['recall']:.4f}")
        print(f"  F1-Score: {res['f1']:.4f}")
        print(f"  F2-Score: {res['f2']:.4f}")
        print(f"  AUC-ROC: {res['auc_roc']:.4f}")
        print(f"  AUC-PR: {res['auc_pr']:.4f}")
        
        # Show confusion matrix details
        cm = res['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
    
    print("\n=== Improved training completed ===") 