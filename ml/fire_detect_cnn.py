import os
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, roc_curve
from keras.metrics import Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Constants for image dimensions and extension
IMAGE_WIDTH, IMAGE_HEIGHT = 350, 350
EXT = '.jpg'


def fetch_images(base_path, sample_fraction=1.0):
    # This function retrieves the images with extension EXT (.jpg in this case) from "fire" and "nofire" directories under the specified location (base_path) 
    fire_images = [f for f in os.listdir(os.path.join(base_path, 'fire')) if f.endswith(EXT)]
    nofire_images = [f for f in os.listdir(os.path.join(base_path, 'nofire')) if f.endswith(EXT)]

    fire_sample = np.random.choice(fire_images, int(sample_fraction * len(fire_images)), replace=False)
    nofire_sample = np.random.choice(nofire_images, int(sample_fraction * len(nofire_images)), replace=False)

    return fire_sample, nofire_sample


def split_data(base_path, sample_fraction=1.0):
    # This function splits the images between test and training in the 40:60 ratio with the test data further split 50:50 between the test and validation
    # Sample fraction indicates how much fraction of the data in the base_path should be considered. Default on 1 indicates 100%

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


def plot_model_performance(model, file_name=r"C:/wildfire/logs/perf.png"):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    # plt.show()
    plt.savefig(file_name)


def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy',
                           Precision(name='precision'),
                           Recall(name='recall'),
                           TruePositives(name='true_positives'),
                           TrueNegatives(name='true_negatives'),
                           FalsePositives(name='false_positives'),
                           FalseNegatives(name='false_negatives')
                           ])
    return model


def train_model(base_path, model_save_path, optimizer, batch_size=50, epochs=10, sample_fraction=1.0):
    train_df, val_df, _ = split_data(base_path=base_path, sample_fraction=sample_fraction)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=base_path,
        x_col="filename",
        y_col="label",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=batch_size,
        class_mode='binary'
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=base_path,
        x_col="filename",
        y_col="label",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=batch_size,
        class_mode='binary'
    )

    LOGGER.info(f"creating model with optimizer {optimizer}")
    model = create_model(optimizer, )

    # Define the ModelCheckpoint callback:
    checkpoint = ModelCheckpoint(
        model_save_path,  # path to where you want to save the model
        monitor='val_loss',  # we're monitoring validation loss
        verbose=1,  # log when a new best is saved
        save_best_only=True,  # only save the best model (based on monitored value, here, val_loss)
        mode='min'  # for val_loss, "min" mode will save models with decreasing loss
    )

    model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[checkpoint])

    base, _ = os.path.splitext(model_save_path)
    plt_file_name = f"{base}.png"
    plot_model_performance(model, file_name=plt_file_name)

    return model


def predict_on_test(model, base_path, sample_fraction=1.0, batch_size=50, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    _, _, test_df = split_data(base_path=base_path, sample_fraction=sample_fraction)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=base_path,
        x_col="filename",
        y_col="label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    predictions = model.predict(test_generator)
    y_pred = [int(p[0] > 0.5) for p in predictions]
    y_true = test_generator.classes
    return y_pred, y_true


def evaluate_model_on_test(model, base_path, batch_size=50):
    y_pred, y_true = predict_on_test(model, base_path, batch_size=batch_size)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)  # compute precision
    conf_matrix = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
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
    parser.add_argument('--path', default=r'C:/wildfire/data/images/all', help='Base path for data')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Size of batch')
    parser.add_argument('--optimizer', default='adam', choices=['rmsprop', 'adam'], help='Optimizer to use')
    parser.add_argument('--train', action='store_true', help='Boolean flag indicating if training should be done')
    parser.add_argument('--model_path', default=None, help='Path to the model to load for evaluation')

    args = parser.parse_args()

    if not args.train and args.model_path is None:
        parser.error("--model_path is required when not training!")

    base_path = args.path
    optimizer = args.optimizer
    epochs = args.epochs
    batch_size = args.batch

    if args.model_path is not None:
        model_save_path = args.model_path
    else:
        dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        model_save_path = rf'C:/wildfire/models/best_model_{optimizer}_{epochs}_{dt_now}.keras'

    LOGGER.info(f"Running with optimizer: {optimizer} and epochs: {epochs}")

    if args.train:
        LOGGER.info(f"Training started")
        train_model(base_path=base_path, model_save_path=model_save_path, batch_size=batch_size, epochs=epochs, optimizer=optimizer,
                    sample_fraction=1.0)
        LOGGER.info(f"Training ended!")

    LOGGER.info(f"Now loading the model {model_save_path} ...")
    model = load_model(model_save_path)

    LOGGER.info(f"Now evaluating the model ...")
    evaluate_model_on_test(model=model, base_path=base_path, batch_size=batch_size)
    LOGGER.info(f"Model evaluation completed!")
