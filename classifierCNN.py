import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_auc_score
from CNN_Attention import CNN

def classify_model_with_preprocessed_data_CNN(batch_size=64):
    """
    Load a pre-trained CNN model and evaluate it on preprocessed positive/negative feature datasets.
    Computes and saves prediction results and evaluation metrics.

    Returns:
        result_df (DataFrame): Actual labels, predicted labels, and probabilities.
        metrics_df (DataFrame): Evaluation metrics including Sn, Sp, Acc, MCC, and AUC.
    """

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained CNN model
    model = CNN().to(device)
    model.load_state_dict(torch.load('CNN_ATT-esm+anova.pth', map_location=device))
    model.eval()

    # Load preprocessed positive and negative feature files
    positive_data = pd.read_csv('GUI/pos_combined_features.csv')
    negative_data = pd.read_csv('GUI/neg_combined_features.csv')

    # Convert features to tensors
    X_positive = torch.tensor(positive_data.values, dtype=torch.float32)
    X_negative = torch.tensor(negative_data.values, dtype=torch.float32)

    # Assign labels: 1 for positive, 0 for negative
    y_positive = torch.ones(X_positive.size(0), dtype=torch.float32)
    y_negative = torch.zeros(X_negative.size(0), dtype=torch.float32)

    # Concatenate data and labels
    X_test = torch.cat((X_positive, X_negative), dim=0)
    y_test = torch.cat((y_positive, y_negative), dim=0)

    # Prepare DataLoader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true = []
    y_pred = []
    y_pred_proba = []

    # Inference
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch).squeeze()
            preds = torch.round(outputs)
            y_true.extend(y_batch.tolist())
            y_pred.extend(preds.tolist())
            y_pred_proba.extend(outputs.tolist())

    # Save prediction results
    result_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Predicted_Probability': y_pred_proba
    })
    result_df.to_csv('GUI/CNN-ATT-esm+anova_result.csv', index=False)

    # Compute evaluation metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tn_t, fp_t, fn_t, tp_t = torch.tensor(tn), torch.tensor(fp), torch.tensor(fn), torch.tensor(tp)
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = ((tp_t * tn_t) - (fp_t * fn_t)) / torch.sqrt((tp_t + fp_t) * (tp_t + fn_t) * (tn_t + fp_t) * (tn_t + fn_t))
    mcc = mcc.item() if mcc.numel() == 1 else mcc
    auc = roc_auc_score(y_true, y_pred_proba)

    # Create metrics DataFrame
    metrics_dict = {
        'Sensitivity (Sn)': [sn],
        'Specificity (Sp)': [sp],
        'Accuracy (Acc)': [acc],
        'Matthews Correlation Coefficient (MCC)': [mcc],
        'Area Under the ROC Curve (AUC)': [auc]
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv('GUI/CNN-ATT-esm+anova_result_target.csv', index=False)

    return result_df, metrics_df
