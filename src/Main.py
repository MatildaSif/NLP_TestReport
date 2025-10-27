"""
Sarcasm Detection and Text pattern investigation

This script looks at the Reddit Sarcasm data and loads a Hugging Face sarcasm classification model (helinivan/english-sarcasm-detector) .
This script evaluates how well a Hugging Face sarcasm classification model 
(helinivan/english-sarcasm-detector) performs on a novel dataset of informally written Reddit comments.


Goals:
- Assess how well a sarcasm detection model trained on formal news headlines generalizes to informal Reddit comments.
- Identify if certain text features are more prone to sarcasm misclassification.
"""

# ------------------------ Imports ------------------------
import os
os.environ["HF_HOME"] = "/work/tf_cache"
import transformers
import pandas as pd
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import classification_report

# ------------------------ Functions ------------------------

def get_classifier(model_type, model_name):
    """
    Initialize a Hugging Face pipeline for classification.

    Parameters:
        model_type (str): The type of classification task (e.g., "text-classification").
        model_name (str): The name or path of the pretrained model.

    Returns:
        classifier (transformers.Pipeline): A configured Hugging Face classification pipeline.
    """
    classifier = pipeline(model_type, model=model_name, return_all_scores=True, truncation=True)
    return classifier


def load_data(files_path, data_file):
    """
    Load and preprocess a labeled sarcasm dataset from a CSV file.

    Parameters:
        files_path (str): The directory where the file is stored.
        data_file (str): Filename of the CSV containing labeled comments.

    Returns:
        df (pd.DataFrame): DataFrame with 'label' and 'comment' columns.
    """
    data_path = os.path.join(files_path, data_file)
    df = pd.read_csv(data_path, quotechar='"')
    df = df[["label", "comment"]]
    print(f"Original data size: {len(df)}")
    return df


def balance_data(df, target_size=5000):
    """
    Ensure the dataset is balanced between sarcastic and non-sarcastic comments,
    and limit the total number of samples to target_size.

    Assumes the original dataset is already balanced and shuffled.

    Parameters:
        df (pd.DataFrame): DataFrame with 'label' and 'comment'.
        target_size (int): Desired final size of the balanced dataset (must be even).

    Returns:
        balanced_df (pd.DataFrame): Balanced and shuffled DataFrame.
    """
    if target_size % 2 != 0:
        raise ValueError("target_size must be an even number to ensure class balance.")

    half_size = target_size // 2
    sarcastic = df[df['label'] == 1].head(half_size)
    non_sarcastic = df[df['label'] == 0].head(half_size)

    balanced_df = pd.concat([sarcastic, non_sarcastic])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced data size: {len(balanced_df)}")
    return balanced_df



def get_numeric_label(text, classifier):
    """
    Use classifier to predict sarcasm label (0 or 1) for a single text sample.

    Parameters:
        text (str): The input comment to classify.

    Returns:
        int: Predicted label (0 = not sarcastic, 1 = sarcastic).
    """
    result = sarcasm_classifier(text)
    predictions = result[0]
    top_label = max(predictions, key=lambda x: x['score'])['label']
    return int(top_label.split('_')[-1])


def create_sarcasm_label(df, classifier, batch_size=32):
    """
    Add classifier-generated sarcasm labels using batched inference.

    Parameters:
        df (pd.DataFrame): DataFrame with 'comment'.
        classifier (pipeline): Hugging Face text classifier.
        batch_size (int): Number of comments to classify at once.

    Returns:
        pd.DataFrame: DataFrame with 'classifier_label' column.
    """
    df = df.dropna(subset=['comment']).copy()
    df['comment'] = df['comment'].astype(str)

    all_predictions = []
    for i in range(0, len(df), batch_size):
        batch = df['comment'].iloc[i:i+batch_size].tolist()
        results = classifier(batch)
        for res in results:
            top_label = max(res, key=lambda x: x['score'])['label']
            all_predictions.append(int(top_label.split('_')[-1]))

    df['classifier_label'] = all_predictions
    print("New Sarcasm labels created (batched)")
    return df


def calculate_mismatches(df_balanced, output_file):
    """
    Compare predicted sarcasm labels to ground truth, and log mismatch statistics.

    Parameters:
        df_balanced (pd.DataFrame): Balanced DataFrame with true and predicted labels.
        output_file (str): Path to file where results will be written.

    Returns:
        None
    """
    with open(output_file, 'w') as f:
        type1 = df_balanced[(df_balanced['label'] == 1) & (df_balanced['classifier_label'] == 0)]
        type2 = df_balanced[(df_balanced['label'] == 0) & (df_balanced['classifier_label'] == 1)]

        f.write(f"label1=1, label2=0: {len(type1)} rows\n")
        f.write(f"label1=0, label2=1: {len(type2)} rows\n")

        type3 = df_balanced[(df_balanced['label'] == 1) & (df_balanced['classifier_label'] == 1)]
        type4 = df_balanced[(df_balanced['label'] == 0) & (df_balanced['classifier_label'] == 0)]

        f.write(f"label1=1, label2=1: {len(type3)} rows\n")
        f.write(f"label1=0, label2=0: {len(type4)} rows\n")

        total_rows = len(df_balanced)
        match_count = (df_balanced['label'] == df_balanced['classifier_label']).sum()
        mismatch_count = total_rows - match_count

        match_percent = (match_count / total_rows) * 100
        mismatch_percent = (mismatch_count / total_rows) * 100

        f.write(f"Matching rows: {match_count} ({match_percent:.2f}%)\n")
        f.write(f"Mismatching rows: {mismatch_count} ({mismatch_percent:.2f}%)\n")

        report = classification_report(df_balanced['label'], df_balanced['classifier_label'], target_names=["Not Sarcastic", "Sarcastic"])
        f.write(report)

        print("Txt file with stats created")




# ------------------------ Definitions ------------------------
files_path = "data/"
data_file = "train-balanced-sarcasm.csv"
model_type = "text-classification"
model_name = "helinivan/english-sarcasm-detector"
output_file = "output/mismatch_results.txt"



# ------------------------ Main ------------------------
# Load the dataset
df = load_data(files_path, data_file)

# Balance the sampled dataset to get approximately 5000 comments
df_balanced = balance_data(df, target_size = 10000)

# Get the classifier pipeline
sarcasm_classifier = get_classifier(model_type, model_name)

# Create sarcasm labels and add them to the dataframe
df_balanced = create_sarcasm_label(df_balanced, sarcasm_classifier)

# save the results
calculate_mismatches(df_balanced, output_file)

# Plot results in accuracy piechart
match_count = (df_balanced['label'] == df_balanced['classifier_label']).sum()
mismatch_count = len(df_balanced) - match_count

plt.figure(figsize=(5, 5))
plt.pie([match_count, mismatch_count],
        labels=['Correct', 'Incorrect'],
        autopct='%1.1f%%',
        colors=['#66b3ff', '#ff9999'],
        startangle=90)
plt.title("Overall Model Accuracy")
plt.tight_layout()
plt.show()
