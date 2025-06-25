import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, accuracy_score


def voting_ensemble(dnn_file, cnn_file, mlp_file):
    dnn_df = pd.read_csv(dnn_file)
    cnn_df = pd.read_csv(cnn_file)
    mlp_df = pd.read_csv(mlp_file)

    # Ensure all files have the same number of samples
    assert len(dnn_df) == len(cnn_df) == len(mlp_df), "Mismatch in sample count across files"

    dnn_pred = dnn_df['Predicted']
    cnn_pred = cnn_df['Predicted']
    mlp_pred = mlp_df['Predicted']
    final_predictions = []
    # Perform majority voting for each sample
    for i in range(len(dnn_pred)):
        positive_count = dnn_pred[i] + cnn_pred[i] + mlp_pred[i]
        if positive_count >= 2:
            final_predictions.append(1)
        else:
            final_predictions.append(0)

    # Combine all relevant data into a single DataFrame
    result_df = pd.DataFrame({
        'Actual': dnn_df['Actual'],
        'DNN_Predicted': dnn_df['Predicted'],
        'DNN_Predicted_Probability': dnn_df['Predicted_Probability'],
        'CNN_Predicted': cnn_df['Predicted'],
        'CNN_Predicted_Probability': cnn_df['Predicted_Probability'],
        'MLP_Predicted': mlp_df['Predicted'],
        'MLP_Predicted_Probability': mlp_df['Predicted_Probability'],
        'Final_Predicted': final_predictions
    })

    return result_df
def calculate_metrics(actual, predicted):
    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    mcc = matthews_corrcoef(actual, predicted)
    acc = accuracy_score(actual, predicted)

    return sn, sp, mcc, acc


def evaluate_voting_results(csv_file):
    df = pd.read_csv(csv_file)
    actual = df['Actual']
    final_predicted = df['Final_Predicted']
    sn, sp, mcc, acc = calculate_metrics(actual, final_predicted)

    # Attempt to compute AUC using average predicted probabilities
    try:
        if all(col in df.columns for col in [
            'DNN_Predicted_Probability',
            'CNN_Predicted_Probability',
            'MLP_Predicted_Probability'
        ]):
            mean_probability = (
                df['DNN_Predicted_Probability'] +
                df['CNN_Predicted_Probability'] +
                df['MLP_Predicted_Probability']
            ) / 3
            auc = roc_auc_score(actual, mean_probability)
        else:
            auc = None
    except ValueError:
        auc = None
    print(f"Sensitivity (Sn): {sn:.4f}")
    print(f"Specificity (Sp): {sp:.4f}")
    print(f"Accuracy (ACC): {acc:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"Area Under the Curve (AUC): {auc:.4f}" if auc is not None else "AUC cannot be calculated.")

    # Save metrics to CSV
    metrics_dict = {
        'Sn': [sn],
        'Sp': [sp],
        'Acc': [acc],
        'Mcc': [mcc],
        'AUC': [auc if auc is not None else 'Unavailable']
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv('CSV/Voting_Evaluation_Metrics.csv', index=False, encoding='utf-8-sig')

    return sn, sp, mcc, auc, acc

dnn_file = 'CSV/DNN-ATT-esm+anova.csv'
cnn_file = 'CSV/CNN-ATT-esm+anova.csv'
mlp_file = 'CSV/MLP-ATT-esm+anova.csv'

result_df = voting_ensemble(dnn_file, cnn_file, mlp_file)
result_df.to_csv('ONEHOT+esm2+anova.csv', index=False)

csv_file = 'CSV/Voting_Ensemble_Results.csv'
evaluate_voting_results(csv_file)