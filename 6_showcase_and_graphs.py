# Imports
import random as rd
import cv2
import importlib
import feature_engineering
import random as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import file_extraction as fe
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score,
    accuracy_score, f1_score, fbeta_score, precision_recall_curve, roc_curve,
    average_precision_score,auc
) 


# Showcase image preprocessing and bounding box extraction functions_________________________________________
def plot_contours_results(target_df, img_index=None, column='pixel_array'):
    """
    Visualize calcification and mass contours on the original image and optionally show enhanced/binary images.

    Args:
        - target_df: DataFrame containing the images and metadata.
        - img_index: Index of the image in the target DataFrame to visualize.
                     If None, a random index is selected.
        - column: Column name in the DataFrame that contains the image data.
    Returns:
        None: Displays the plots.
    """
    importlib.reload(feature_engineering)
    
    # Set random image index if not provided
    if img_index is None:
        img_index = rd.randint(0,len(target_df)-1)
    else:
        img_index = int(img_index)
    
    # Get contours, enhanced images and thresholded images (binary) 
    orig_img = target_df.loc[img_index, column]
    
    # Apply CLAHE to the original image to show the change of the histogram
    # Note: this step is applied when 'feature_extractor' is used, but no for single images in extract_contours.
    # Thus this step is necessary here.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    orig_img_clahe = clahe.apply(orig_img.astype(np.uint8))
    
    # Get contours and enahnced images to visualize
    contours_calci, contours_mass, enhanced_img_calci, enhanced_img_mass, thresh_img_calci, thresh_img_mass = feature_engineering.extract_contours(orig_img_clahe)

    # Initiate the subplots for visualization
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    fig = plt.gcf()
    fig.suptitle(
        f'Patient ID: {target_df["ID1"].iloc[img_index]} (row: {img_index} | Abnormality: {target_df["abnormality"].iloc[img_index]} | Classification: {target_df["classification"].iloc[img_index]})',
        fontsize=14
        )

    # Calcification row
    
    # Normal un-enahnced image
    axes[0, 0].imshow(orig_img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    
    # Enhanced image for mass (GLAHE Histogram qualization, gaussian blur, laplacian filter, and gamma correction)   
    if enhanced_img_calci is not None:
        axes[0, 1].imshow(enhanced_img_calci, cmap='gray')
        axes[0, 1].set_title('Enhanced for Calcification')
    else:
        axes[0, 1].axis('off')
    
    # Binary image for calci (black and white)
    if thresh_img_calci is not None:
        axes[0, 2].imshow(thresh_img_calci, cmap='gray')
        axes[0, 2].set_title('Binary for Calcifications')
    else:
        axes[0, 2].axis('off')
    
    # Found ROI patches on original image
    axes[0, 3].imshow(orig_img, cmap='gray')
    axes[0, 3].set_title(f'{len(contours_calci)} bounding boxes for Calcification found')
    for cc in contours_calci:
        x, y, w, h = cv2.boundingRect(cc)
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='purple', facecolor='none')
        axes[0, 3].add_patch(rect)

    # Mass row
    
    # Normal un-enahnced image
    axes[1, 0].imshow(orig_img, cmap='gray')
    axes[1, 0].set_title('Original Image')
    
    # Enhanced image for mass (GLAHE Histogram equalization, gaussian blur, and gamma correction)
    if enhanced_img_mass is not None:
        axes[1, 1].imshow(enhanced_img_mass, cmap='gray')
        axes[1, 1].set_title('Enhanced for Mass')
    else:
        axes[1, 1].axis('off')
    
    # Binary image for mass (black and whte)
    if thresh_img_mass is not None:
        axes[1, 2].imshow(thresh_img_mass, cmap='gray')
        axes[1, 2].set_title('Binary for Mass')
    else:
        axes[1, 2].axis('off')
    
    # Found ROI patches for mass on original image
    axes[1, 3].imshow(orig_img, cmap='gray')
    axes[1, 3].set_title(f'{len(contours_mass)} bounding boxes for Mass found')
    for cm in contours_mass:
        x, y, w, h = cv2.boundingRect(cm)
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        axes[1, 3].add_patch(rect)

    plt.tight_layout()
    plt.show()
    
# Show different histogram equalization methods and their effects on the image
def plot_histogram_equalization(image_array=None):
    if image_array is None:
        # Load the DataFrame with images
        df_cmmd, _ = fe.extract_cmmd_dicom(limit=1000)
        if df_cmmd is None:
            print("No data available in the DataFrame.")
            return
        test_idx = rd.randint(0, len(df_cmmd) - 1)
        image_array = df_cmmd['pixel_array'].iloc[test_idx]
    
    image_array_equal = cv2.equalizeHist(image_array.astype(np.uint8))
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

    clahe = clahe.apply(image_array.astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    if image_array is None:
        fig.suptitle(f'Histogram Equalization Methods Comparison for Patient ID: {df_cmmd["ID1"].iloc[test_idx]} (Row: {test_idx} | Abnormality: {df_cmmd["abnormality"].iloc[test_idx]} | Classification: {df_cmmd["classification"].iloc[test_idx]})', fontsize=14)
    else:
        fig.suptitle(f'Histogram Equalization Methods Comparison for given image', fontsize=14)
    axes[0].imshow(image_array, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(image_array_equal, cmap='gray')
    axes[1].set_title('Image with Equalized Histogram ')
    axes[2].imshow(clahe, cmap='gray')
    axes[2].set_title('Image with CLAHE equalization')
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust rect to lower suptitle distance
    plt.show()
    
    return
    
# %%
# Performance Plots_______________________________________________________________________________________

def plot_combined_roc_curves(results, y_train):
    """
    Compares ROC_AUC for differend models based on the training results
    Args:
        - results: Dictionary containing model names and their results.
        - y_train: True labels for the training set.    
    """
    plt.figure(figsize=(12, 10))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, (name, res) in enumerate(results.items()):
        if res['y_scores'] is not None:
            fpr, tpr, _ = roc_curve(y_train, res['y_scores'])
            auc_roc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC-ROC = {auc_roc:.3f})', 
                    color=colors[i % len(colors)], linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC-ROC = 0.500)', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Comparison (In-sample)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_auc_curves(results, y_train, in_sample=True):
    """
    Plots ROC curves (left) and Precision-Recall curves (right) for all models in results.
    Shows AUC-ROC and AUC-PRC in the legends. Includes DummyClassifier as baseline in PRC plot.
    Args:
        - results: Dictionary containing model names and their results (must have 'y_scores').
        - y_train: True labels for the training set.
    """
    if in_sample:
        in_sample = 'In-sample'
    else:
        in_sample = 'Out-of-sample'
        
    # Create subplots for ROC and Precision-Recall curves 
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'{in_sample} ROC and Precision-Recall Curves', fontsize=16, fontweight='bold')

    # ROC curves (left)
    for i, (name, res) in enumerate(results.items()):
        if res['y_scores'] is not None:
            fpr, tpr, _ = roc_curve(y_train, res['y_scores'])
            auc_roc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, label=f'{name} (AUC-ROC = {auc_roc:.3f})', color=f'C{i}', linewidth=2)
    axes[0].plot([0, 1], [0, 1], color='k', linestyle='--', label='Dummy (AUC-ROC = 0.500)', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # Precision-Recall curves (right)
    for i, (name, res) in enumerate(results.items()):
        if res['y_scores'] is not None:
            precision, recall, _ = precision_recall_curve(y_train, res['y_scores'])
            auc_prc = average_precision_score(y_train, res['y_scores'])
            axes[1].plot(recall, precision, label=f'{name} (AUC-PRC = {auc_prc:.3f})', color=f'C{i}', linewidth=2)
    # Add DummyClassifier PRC
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(np.zeros_like(y_train).reshape(-1, 1), y_train)
    if hasattr(dummy, 'predict_proba'):
        y_dummy_score = dummy.predict_proba(np.zeros_like(y_train).reshape(-1, 1))[:, 1]
    else:
        y_dummy_score = dummy.decision_function(np.zeros_like(y_train).reshape(-1, 1))
    precision_dummy, recall_dummy, _ = precision_recall_curve(y_train, y_dummy_score)
    auc_prc_dummy = average_precision_score(y_train, y_dummy_score)
    
    axes[1].plot(recall_dummy,
                 precision_dummy,
                 color='k',
                 linestyle='--',
                 label=f'Dummy (AUC-PRC = {auc_prc_dummy:.3f})',
                 linewidth=2,
                 alpha=0.5)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
    return

def compare_model_metrics(results, y_true, metrics=None):
    """
    Creates and prints a formatted table comparing metrics for all models in results.

    Args:
        results: dict, keys are model names, values are dicts with 'y_pred' and 'y_scores'.
        y_true: array-like, true labels.
        metrics: list of metric names to include (default: common classification metrics).
    Returns:
        pandas.DataFrame with metrics for each model.
    """

    if metrics is None:
        metrics = [
            'Accuracy', 'Precision', 'Recall', 'F1', 'F2', 'AUC-ROC', 'AUC-PRC'
        ]

    rows = []
    for name, res in results.items():
        y_pred = res.get('y_pred')
        y_scores = res.get('y_scores')
        row = {'Model': name}
        if y_pred is not None:
            row['Accuracy'] = accuracy_score(y_true, y_pred)
            row['Precision'] = precision_score(y_true, y_pred, zero_division=0)
            row['Recall'] = recall_score(y_true, y_pred, zero_division=0)
            row['F1'] = f1_score(y_true, y_pred, zero_division=0)
            row['F2'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        else:
            row['Accuracy'] = row['Precision'] = row['Recall'] = row['F1'] = row['F2'] = None
        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            row['AUC-ROC'] = auc(fpr, tpr)
            row['AUC-PRC'] = average_precision_score(y_true, y_scores)
        else:
            row['AUC-ROC'] = row['AUC-PRC'] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[['Model'] + metrics]
    print(df.to_string(index=False, float_format="%.3f"))
    return df

def evaluate_and_plot_classifier(model, X, y, classification_labels=None, in_sample=True, plot_title_prefix='Classifier'):
    """
    Evaluates a trained model on X, y and plots confusion matrix, ROC, and precision-recall curves.
    Trains and compares to a DummyClassifier internally. Both ROC and PRC plots include both model and dummy curves with AUC values in legends.
    """
    
    # Predict class and compute scores
    y_pred = model.predict(X)
    if hasattr(model, 'predict_proba'):
        y_pred_score = model.predict_proba(X)[:, 1]
    else:
        y_pred_score = model.decision_function(X)

    # Train DummyClassifier on the same data
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X, y)
    y_pred_dummy = dummy.predict(X)
    if hasattr(dummy, 'predict_proba'):
        y_pred_score_dummy = dummy.predict_proba(X)[:, 1]
    else:
        y_pred_score_dummy = dummy.decision_function(X)

    # Compute confusion matrices
    cm = confusion_matrix(y, y_pred, normalize='true')
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=classification_labels.cat.categories if classification_labels is not None else None)
    cm_dummy = confusion_matrix(y, y_pred_dummy, normalize='true')
    disp_cm_dummy = ConfusionMatrixDisplay(confusion_matrix=cm_dummy,
                                           display_labels=classification_labels.cat.categories if classification_labels is not None else None)

    # Compute ROC and PRC curves
    fpr, tpr, _ = roc_curve(y, y_pred_score)
    fpr_dummy, tpr_dummy, _ = roc_curve(y, y_pred_score_dummy)
    auc_roc = auc(fpr, tpr)
    auc_roc_dummy = auc(fpr_dummy, tpr_dummy)

    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_score)
    precision_curve_dummy, recall_curve_dummy, _ = precision_recall_curve(y, y_pred_score_dummy)
    auc_prc = average_precision_score(y, y_pred_score)
    auc_prc_dummy = average_precision_score(y, y_pred_score_dummy)

    # Compute perfomrance metrics
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    f2 = fbeta_score(y, y_pred, beta=2)
    print(f'{plot_title_prefix} - Precision: {precision:.2f},'
          f' Recall: {recall:.2f}, Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}, F2 Score: {f2:.2f}')

    precision_dummy = precision_score(y, y_pred_dummy)
    recall_dummy = recall_score(y, y_pred_dummy)
    accuracy_dummy = accuracy_score(y, y_pred_dummy)
    f1_dummy = f1_score(y, y_pred_dummy)
    f2_dummy = (5 * precision_dummy * recall_dummy) / (4 * precision_dummy + recall_dummy) if (4 * precision_dummy + recall_dummy) != 0 else 0
    print(f'Dummy Classifier - Precision: {precision_dummy:.2f},'
          f' Recall: {recall_dummy:.2f}, Accuracy: {accuracy_dummy:.2f}, F1 Score: {f1_dummy:.2f}, F2 Score: {f2_dummy:.2f}')

    # Store performance metrics in a dictionary
    performance ={
        'y_pred': y_pred,
        'y_pred_score': y_pred_score,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'f2': f2,
        'confusion_matrix': cm,
        'prc': (precision_curve, recall_curve),
        'roc': (fpr, tpr)
    }
    
    # Determine if in-sample or out-of-sample for title:
    if in_sample:
        in_sample = 'In-sample'
    else:
        in_sample = 'Out-of-sample'
        
    # Set Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(2, 2)
    fig.suptitle(f'{in_sample} Confusion Matrix and AUC-ROC and AUC-PRC', fontsize=16, fontweight='bold')

    # Confusion matrix
    disp_cm.plot(ax=axes[0, 0], cmap='Blues', values_format='.2f')
    axes[0, 0].set_title(f'{plot_title_prefix}')
    disp_cm_dummy.plot(ax=axes[0, 1], cmap='Blues', values_format='.2f')
    axes[0, 1].set_title('Dummy Classifier')

    # ROC curve (bottom left)
    axes[1, 0].plot(fpr, tpr, color='C0', linestyle='-', marker='.', label=f'{plot_title_prefix} (AUC = {auc_roc:.3f})')
    axes[1, 0].plot(fpr_dummy, tpr_dummy, color='k', linestyle='--', marker='.', label=f'Dummy (AUC = {auc_roc_dummy:.3f})')
    axes[1, 0].set_title(f'ROC Curve')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()

    # Precision-recall curve (bottom right)
    axes[1, 1].plot(recall_curve, precision_curve, color='C0', linestyle='-', marker='.',
                    label=f'{plot_title_prefix} (AUC = {auc_prc:.3f})')
    axes[1, 1].plot(recall_curve_dummy, precision_curve_dummy, color='k', linestyle='--',
                    marker='.', label=f'Dummy (AUC = {auc_prc_dummy:.3f})')
    axes[1, 1].set_title('Precision-Recall Curve')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
    return performance