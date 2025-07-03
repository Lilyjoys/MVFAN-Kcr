import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import pandas as pd
import numpy as np
from imblearn.metrics import specificity_score

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from esm2_extractor2 import ESM2FeatureExtractor
from onehot_encoder import OneHotEncoderProtein
from ANOVA import FeatureReducer
from classifierDNN import classify_model_with_preprocessed_data_DNN
from classifierCNN import classify_model_with_preprocessed_data_CNN
from classifierMLP import classify_model_with_preprocessed_data_MLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Support for displaying minus signs and non-ASCII characters
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Protein Classification Predictor")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.primary_color = "#2563eb"
        self.secondary_color = "#f97316"
        self.tertiary_color = "#10b981"
        self.neutral_color = "#64748b"

        self.positive_file = ""
        self.negative_file = ""
        self.output_filename = tk.StringVar(value="prediction_results.csv")
        self.results_df = None
        self.feature_importance = None

        self.create_widgets()

    def create_widgets(self):
        # Top navigation bar
        navbar = tk.Frame(self.root, bg=self.primary_color, height=50)
        navbar.pack(fill="x")
        tk.Label(navbar, text="Protein Classification Prediction System", font=("Arial", 16, "bold"),
                 bg=self.primary_color, fg="white").pack(pady=10, padx=20, side="left")

        content_frame = tk.Frame(self.root, bg="#f0f0f0")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel
        left_frame = tk.Frame(content_frame, bg="#ffffff", relief="solid", bd=1)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Dataset selection
        file_frame = tk.LabelFrame(left_frame, text="Dataset Selection", font=("Arial", 12, "bold"),
                                   padx=20, pady=15, bg="#ffffff")
        file_frame.pack(fill="x", pady=(10, 20), padx=15)

        tk.Label(file_frame, text="Positive Samples:", font=("Arial", 10), bg="#ffffff").grid(row=0, column=0, sticky="w", pady=5)
        self.positive_entry = tk.Entry(file_frame, width=40, font=("Arial", 10))
        self.positive_entry.grid(row=0, column=1, padx=5, pady=5)
        browse_btn1 = tk.Button(file_frame, text="Browse", command=self.browse_positive,
                                bg=self.primary_color, fg="white", font=("Arial", 10),
                                relief="flat", padx=10)
        browse_btn1.grid(row=0, column=2, padx=5, pady=5)

        tk.Label(file_frame, text="Negative Samples:", font=("Arial", 10), bg="#ffffff").grid(row=1, column=0, sticky="w", pady=5)
        self.negative_entry = tk.Entry(file_frame, width=40, font=("Arial", 10))
        self.negative_entry.grid(row=1, column=1, padx=5, pady=5)
        browse_btn2 = tk.Button(file_frame, text="Browse", command=self.browse_negative,
                                bg=self.primary_color, fg="white", font=("Arial", 10),
                                relief="flat", padx=10)
        browse_btn2.grid(row=1, column=2, padx=5, pady=5)

        # Output settings
        output_frame = tk.LabelFrame(left_frame, text="Output Settings", font=("Arial", 12, "bold"),
                                     padx=20, pady=15, bg="#ffffff")
        output_frame.pack(fill="x", pady=(0, 20), padx=15)

        tk.Label(output_frame, text="Output Filename:", font=("Arial", 10), bg="#ffffff").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(output_frame, textvariable=self.output_filename, width=40, font=("Arial", 10)).grid(row=0, column=1, padx=5, pady=5)

        # Progress bar
        progress_frame = tk.LabelFrame(left_frame, text="Processing Progress", font=("Arial", 12, "bold"),
                                       padx=20, pady=15, bg="#ffffff")
        progress_frame.pack(fill="x", pady=(0, 20), padx=15)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=100, mode='determinate')
        self.progress_bar.pack(fill="x", pady=10)

        self.status_label = tk.Label(progress_frame, text="Ready", font=("Arial", 10), bg="#ffffff", fg=self.neutral_color)
        self.status_label.pack(anchor="w")

        # Run button
        process_frame = tk.Frame(left_frame, bg="#ffffff")
        process_frame.pack(pady=10, padx=15)

        run_btn = tk.Button(process_frame, text="Run Prediction", command=self.run_prediction,
                            bg=self.tertiary_color, fg="white", font=("Arial", 12, "bold"),
                            relief="flat", padx=20, pady=8, cursor="hand2")
        run_btn.pack()

        # Right panel
        right_frame = tk.Frame(content_frame, bg="#ffffff", relief="solid", bd=1)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill="both", expand=True)

        # Tab: Results Report
        results_text_frame = tk.Frame(notebook, bg="#ffffff")
        notebook.add(results_text_frame, text="Results Report")

        self.results_text = tk.Text(results_text_frame, height=15, font=("Arial", 10), wrap=tk.WORD)
        self.results_text.pack(fill="both", expand=True, pady=15, padx=15)
        scrollbar = tk.Scrollbar(self.results_text, command=self.results_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Tab: Prediction Details
        results_table_frame = tk.Frame(notebook, bg="#ffffff")
        notebook.add(results_table_frame, text="Prediction Details")

        columns = ["Sample ID", "Actual", "Final_Predicted",
                   "DNN_Predicted", "DNN_Predicted_Probability",
                   "CNN_Predicted", "CNN_Predicted_Probability",
                   "MLP_Predicted", "MLP_Predicted_Probability"]

        self.results_tree = ttk.Treeview(results_table_frame, columns=columns, show="headings", height=10)

        for col in columns:
            if col == "Actual":
                self.results_tree.heading(col, text="True Label")
            elif col.endswith("_Predicted"):
                self.results_tree.heading(col, text=f"{col.split('_')[0]} Prediction")
            elif col.endswith("_Probability"):
                self.results_tree.heading(col, text=f"{col.split('_')[0]} Probability")
            else:
                self.results_tree.heading(col, text=col)

            width = 120 if "Probability" in col else 100
            self.results_tree.column(col, width=width, anchor="center")

        self.results_tree.pack(fill="both", expand=True, pady=15, padx=15)

        tree_scroll_y = tk.Scrollbar(results_table_frame, orient="vertical", command=self.results_tree.yview)
        tree_scroll_y.pack(side="right", fill="y")
        self.results_tree.configure(yscrollcommand=tree_scroll_y.set)

        tree_scroll_x = tk.Scrollbar(results_table_frame, orient="horizontal", command=self.results_tree.xview)
        tree_scroll_x.pack(side="bottom", fill="x")
        self.results_tree.configure(xscrollcommand=tree_scroll_x.set)

        # Tab: Feature Importance
        feature_frame = tk.Frame(notebook, bg="#ffffff")
        notebook.add(feature_frame, text="Feature Importance")

        self.feature_fig = Figure(figsize=(6, 4), dpi=100)
        self.feature_canvas = FigureCanvasTkAgg(self.feature_fig, master=feature_frame)
        self.feature_canvas.get_tk_widget().pack(fill="both", expand=True, pady=15, padx=15)

        # Bottom save button
        bottom_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        bottom_frame.pack(fill="x", side="bottom")

        save_btn = tk.Button(bottom_frame, text="Save Results", command=self.save_results,
                             bg=self.primary_color, fg="white", font=("Arial", 10),
                             relief="flat", padx=15, pady=5, cursor="hand2")
        save_btn.pack(side="right", padx=20)

    def browse_positive(self):
        filename = filedialog.askopenfilename(title="Select Positive Dataset",
                                              filetypes=(("TXT files", "*.txt"), ("FASTA files", "*.fasta"), ("All files", "*.*")))
        if filename:
            self.positive_file = filename
            self.positive_entry.delete(0, tk.END)
            self.positive_entry.insert(0, filename)

    def browse_negative(self):
        filename = filedialog.askopenfilename(title="Select Negative Dataset",
                                              filetypes=(("TXT files", "*.txt"), ("FASTA files", "*.fasta"), ("All files", "*.*")))
        if filename:
            self.negative_file = filename
            self.negative_entry.delete(0, tk.END)
            self.negative_entry.insert(0, filename)

    def update_progress(self, value, status):
        self.progress_var.set(value)
        self.status_label.config(text=status)
        self.root.update_idletasks()



    def run_prediction(self):
        if not self.positive_file or not self.negative_file:
            messagebox.showerror("Error", "Please select both positive and negative sample files.")
            return

        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        try:
            # Load dataset
            self.update_progress(10, "Loading datasets...")
            self.results_text.insert(tk.END, "Loading datasets...\n")

            # ========== 1. ESM2 Feature Extraction ==========
            self.update_progress(20, "Extracting ESM2 features...")
            self.results_text.insert(tk.END, "Extracting ESM2 features...\n")

            esm_extractor = ESM2FeatureExtractor()
            pos_esm, pos_ids = esm_extractor.extract_features(self.positive_file)
            neg_esm, neg_ids = esm_extractor.extract_features(self.negative_file)

            pd.DataFrame(pos_esm).to_csv("GUI/pos_esm_features.csv", index=False)
            pd.DataFrame(neg_esm).to_csv("GUI/neg_esm_features.csv", index=False)

            X_esm = np.vstack([pos_esm, neg_esm])
            y = np.array([1] * len(pos_esm) + [0] * len(neg_esm))

            # ========== 2. ESM2 Feature Reduction ==========
            self.update_progress(35, "Reducing ESM2 feature dimensions...")
            self.results_text.insert(tk.END, "Reducing ESM2 feature dimensions...\n")

            pos_esm_reduced, neg_esm_reduced = FeatureReducer()  # Reduce to 300 dims

            pos_esm_reduced.to_csv("GUI/pos_esm_ANOVA_features2.csv", index=False)
            neg_esm_reduced.to_csv("GUI/neg_esm_ANOVA_features2.csv", index=False)

            # ========== 3. One-Hot Feature Extraction ==========
            self.update_progress(50, "Extracting One-Hot features...")
            self.results_text.insert(tk.END, "Extracting One-Hot features...\n")

            onehot_encoder = OneHotEncoderProtein()
            pos_onehot, _ = onehot_encoder.process_file(self.positive_file)
            neg_onehot, _ = onehot_encoder.process_file(self.negative_file)

            pd.DataFrame(pos_onehot).to_csv("GUI/pos_onehot_features.csv", index=False)
            pd.DataFrame(neg_onehot).to_csv("GUI/neg_onehot_features.csv", index=False)

            # ========== 4. Feature Merging ==========
            self.update_progress(60, "Merging features...")
            self.results_text.insert(tk.END, "Merging features...\n")

            assert len(pos_esm_reduced) == len(pos_onehot)
            assert len(neg_esm_reduced) == len(neg_onehot)

            positive_data = pd.read_csv('GUI/pos_esm_ANOVA_features2.csv')
            negative_data = pd.read_csv('GUI/neg_esm_ANOVA_features2.csv')

            pos_features = np.hstack([pos_onehot, positive_data])
            neg_features = np.hstack([neg_onehot, negative_data])

            pd.DataFrame(pos_features).to_csv("GUI/pos_combined_features.csv", index=False)
            pd.DataFrame(neg_features).to_csv("GUI/neg_combined_features.csv", index=False)

            # ========== 5. Prepare Final Dataset ==========
            X = np.vstack([pos_features, neg_features])
            y = np.array([1] * len(pos_features) + [0] * len(neg_features))

            # ========== 6. Classification ==========
            self.update_progress(85, "Running classifiers...")
            self.results_text.insert(tk.END, "Running classifiers...\n")

            result_df1, metrics_df1 = classify_model_with_preprocessed_data_DNN()
            result_df2, metrics_df2 = classify_model_with_preprocessed_data_CNN()
            result_df3, metrics_df3 = classify_model_with_preprocessed_data_MLP()

            # ========== Display Individual Classifier Metrics ==========
            self.results_text.insert(tk.END, "=" * 50 + "\n")
            self.results_text.insert(tk.END, "Classifier Performance:\n\n")

            self.results_text.insert(tk.END, "DNN-Attention Model:\n")
            self.display_single_classifier_metrics(metrics_df1)

            self.results_text.insert(tk.END, "\nCNN-Attention Model:\n")
            self.display_single_classifier_metrics(metrics_df2)

            self.results_text.insert(tk.END, "\nMLP-Attention Model:\n")
            self.display_single_classifier_metrics(metrics_df3)

            # ========== Majority Voting ==========
            all_predictions = [result_df1['Predicted'], result_df2['Predicted'], result_df3['Predicted']]
            all_probs = [result_df1['Predicted_Probability'], result_df2['Predicted_Probability'],
                         result_df3['Predicted_Probability']]

            vote_results = []
            for i in range(len(y)):
                positive_votes = sum([p[i] for p in all_predictions])
                final_prediction = 1 if positive_votes >= 2 else 0
                vote_results.append(final_prediction)

            from sklearn.metrics import recall_score, accuracy_score, matthews_corrcoef, roc_auc_score

            vote_sn = recall_score(y, vote_results)
            vote_sp = specificity_score(y, vote_results)
            vote_acc = accuracy_score(y, vote_results)
            vote_mcc = matthews_corrcoef(y, vote_results)
            vote_auc = roc_auc_score(y, vote_results)

            self.results_text.insert(tk.END, "=" * 50 + "\n")
            self.results_text.insert(tk.END, "Majority Voting Ensemble Results:\n")
            self.results_text.insert(tk.END, f"Sensitivity (Sn): {vote_sn:.4f}\n")
            self.results_text.insert(tk.END, f"Specificity (Sp): {vote_sp:.4f}\n")
            self.results_text.insert(tk.END, f"Accuracy (ACC): {vote_acc:.4f}\n")
            self.results_text.insert(tk.END, f"Matthews Correlation Coefficient (MCC): {vote_mcc:.4f}\n")
            self.results_text.insert(tk.END, f"Area Under Curve (AUC): {vote_auc:.4f}\n\n")

            # ========== Display Prediction Results ==========
            self.display_results_in_table(
                sample_ids=pos_ids + neg_ids,
                Actual=y,
                Final_Predicted=vote_results,
                DNN_Predicted=all_predictions[0],
                DNN_Predicted_Probability=all_probs[0],
                CNN_Predicted=all_predictions[1],
                CNN_Predicted_Probability=all_probs[1],
                MLP_Predicted=all_predictions[2],
                MLP_Predicted_Probability=all_probs[2]
            )

            # Save final prediction results
            self.results_df = pd.DataFrame({
                'Actual': y,
                'Final_Predicted': vote_results,
                'DNN_Predicted': all_predictions[0],
                'DNN_Predicted_Probability': all_probs[0],
                'CNN_Predicted': all_predictions[1],
                'CNN_Predicted_Probability': all_probs[1],
                'MLP_Predicted': all_predictions[2],
                'MLP_Predicted_Probability': all_probs[2]
            })

            self.results_df.to_csv("final_predictions.csv", index=False)

            self.results_text.insert(tk.END, f"Total {len(self.results_df)} predictions generated.\n")
            self.results_text.insert(tk.END, "Click 'Save Results' at the bottom to export data.\n\n")

            self.update_progress(100, "Prediction completed")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.results_text.insert(tk.END, f"\nError: {str(e)}\n")
            self.update_progress(0, "Failed")

            # Additional methods like run_prediction, save_results, display_results_in_table, etc. remain to be translated

    def save_results(self):
        if not hasattr(self, 'results_df') or self.results_df is None:
            messagebox.showerror("Error", "No results to save. Please run prediction first.")
            return

        filename = self.output_filename.get()
        if not filename.endswith('.csv'):
            filename += '.csv'

        try:
            save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     initialfile=filename,
                                                     filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
            if save_path:
                self.results_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Results saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def display_results_in_table(self, sample_ids, Actual,
                                 Final_Predicted, DNN_Predicted, DNN_Predicted_Probability,
                                 CNN_Predicted, CNN_Predicted_Probability,
                                 MLP_Predicted, MLP_Predicted_Probability):
        # Clear the table
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Insert data row by row
        for i in range(len(Actual)):
            self.results_tree.insert("", tk.END, values=(
                sample_ids[i],
                Actual[i],
                Final_Predicted[i],
                DNN_Predicted[i],
                f"{DNN_Predicted_Probability[i]:.4f}",
                CNN_Predicted[i],
                f"{CNN_Predicted_Probability[i]:.4f}",
                MLP_Predicted[i],
                f"{MLP_Predicted_Probability[i]:.4f}"
            ))

    def display_single_classifier_metrics(self, metrics_df):
        """Display evaluation metrics of a single classifier."""
        sn = metrics_df['Sensitivity (Sn)'].values[0]
        sp = metrics_df['Specificity (Sp)'].values[0]
        acc = metrics_df['Accuracy (Acc)'].values[0]
        mcc = metrics_df['Matthews Correlation Coefficient (MCC)'].values[0]
        auc = metrics_df['Area Under the ROC Curve (AUC)'].values[0]

        self.results_text.insert(tk.END, f"Sensitivity (Sn): {sn:.4f}\n")
        self.results_text.insert(tk.END, f"Specificity (Sp): {sp:.4f}\n")
        self.results_text.insert(tk.END, f"Accuracy (Acc): {acc:.4f}\n")
        self.results_text.insert(tk.END, f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
        self.results_text.insert(tk.END, f"Area Under the ROC Curve (AUC): {auc:.4f}\n")



if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()