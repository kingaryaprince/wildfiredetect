import os
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, \
    roc_curve
from sklearn.preprocessing import StandardScaler
from PIL import Image
from joblib import dump, load

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Constants for image dimensions
IMAGE_WIDTH, IMAGE_HEIGHT = 350, 350
EXT = '.jpg'


def fetch_images(base_path, sample_fraction=1.0):
    fire_images = [f for f in os.listdir(os.path.join(base_path, 'fire')) if f.endswith(EXT)]
    nofire_images = [f for f in os.listdir(os.path.join(base_path, 'nofire')) if f.endswith(EXT)]

    fire_sample = np.random.choice(fire_images, int(sample_fraction * len(fire_images)), replace=False)
    nofire_sample = np.random.choice(nofire_images, int(sample_fraction * len(nofire_images)), replace=False)

    return fire_sample, nofire_sample


def split_data(base_path, sample_fraction=1.0):
    fire_images, nofire_images = fetch_images(base_path, sample_fraction=sample_fraction)

    fire_train, fire_temp = train_test_split(fire_images, test_size=0.4, random_state=42)
    nofire_train, nofire_temp = train_test_split(nofire_images, test_size=0.4, random_state=42)
    fire_val, fire_test = train_test_split(fire_temp, test_size=0.5, random_state=42)
    nofire_val, nofire_test = train_test_split(nofire_temp, test_size=0.5, random_state=42)

    # Prefixing the filenames with their respective directories
    fire_train = ['fire/' + name for name in fire_train]
    nofire_train = ['nofire/' + name for name in nofire_train]
    fire_val = ['fire/' + name for name in fire_val]
    nofire_val = ['nofire/' + name for name in nofire_val]
    fire_test = ['fire/' + name for name in fire_test]
    nofire_test = ['nofire/' + name for name in nofire_test]

    # Convert to DataFrame
    train_df = pd.DataFrame({
        'filename': list(fire_train) + list(nofire_train),
        'label': ['fire'] * len(fire_train) + ['nofire'] * len(nofire_train)
    })

    val_df = pd.DataFrame({
        'filename': list(fire_val) + list(nofire_val),
        'label': ['fire'] * len(fire_val) + ['nofire'] * len(nofire_val)
    })

    test_df = pd.DataFrame({
        'filename': list(fire_test) + list(nofire_test),
        'label': ['fire'] * len(fire_test) + ['nofire'] * len(nofire_test)
    })

    return train_df, val_df, test_df


def read_and_process_images(filepaths, base_path):
    imgs = [Image.open(os.path.join(base_path, f)) for f in filepaths]
    imgs = [img.resize((IMAGE_WIDTH, IMAGE_HEIGHT)) for img in imgs]

    # Convert PIL Images to numpy arrays
    imgs = [np.array(img) for img in imgs]

    return np.array(imgs)


def prepare_data(base_path, sample_fraction=1.0):
    train_df, val_df, test_df = split_data(base_path=base_path, sample_fraction=sample_fraction)

    X_train = read_and_process_images(train_df['filename'], base_path)
    y_train = (train_df['label'] == 'fire').astype(int).values

    X_val = read_and_process_images(val_df['filename'], base_path)
    y_val = (val_df['label'] == 'fire').astype(int).values

    X_test = read_and_process_images(test_df['filename'], base_path)
    y_test = (test_df['label'] == 'fire').astype(int).values

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def train_svm(X_train, y_train, X_val, y_val, kernel='linear', C=1):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = svm.SVC(kernel=kernel, C=C, probability=True)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_val, y_val)

    LOGGER.info(f"Validation accuracy with {kernel}-kernel SVM: {val_accuracy:.4f}")
    return clf, scaler


def evaluate_model_on_test(clf, scaler, X_test, y_test):
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test = scaler.transform(X_test)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tnr = 1 - fpr
    fnr = 1 - tpr

    print(f"Accuracy on test set: {acc:.3f}")
    print(f"F1 Score on test set: {f1:.3f}")
    print(f"Precision on test set: {precision:.3f}")
    print(f"Recall on test set: {recall:.3f}")
    print(f"Area Under ROC Curve: {auc:.3f}")
    print(f"True Positive Rate: {tpr[1]:.3f}")
    print(f"False Positive Rate: {fpr[1]:.3f}")
    print(f"True Negative Rate: {tnr[1]:.3f}")
    print(f"False Negative Rate: {fnr[1]:.3f}")
    print(f"Confusion Matrix:\n{conf_matrix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate wildfire model.')
    parser.add_argument('--path', default=r'C:/wildfire/data/images/sent', help='Base path for data')
    parser.add_argument('--train', action='store_true', help='Boolean flag indicating if training should be done')
    parser.add_argument('--model_path', default=None, help='Path to the model to load for evaluation')

    args = parser.parse_args()

    if not args.train and args.model_path is None:
        parser.error("--model_path is required when not training!")

    if args.model_path is not None:
        model_save_path = args.model_path
    else:
        dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        model_save_path = rf'C:/wildfire/models/svm_model_{dt_now}.joblib'

    LOGGER.info(f"Preparing data...")
    base_path = args.path
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(base_path)

    if args.train:
        LOGGER.info(f"Training SVM...")
        clf, scaler = train_svm(X_train, y_train, X_val, y_val, kernel='linear', C=1)
        # Save the model and scaler using joblib
        dump((clf, scaler), model_save_path)
        LOGGER.info(f"Training ended!")

    clf, scaler = load(model_save_path)

    LOGGER.info(f"Now evaluating the model ...")
    evaluate_model_on_test(clf, scaler, X_test, y_test)
    LOGGER.info(f"Model evaluation completed!")
