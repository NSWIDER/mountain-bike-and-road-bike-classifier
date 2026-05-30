"""
evaluate_tensorflow.py

Loads a saved Keras model and evaluates images from test directories,
outputting a JSON report to json/tensorflow/<modelname>.json.

Usage:
    python evaluate_tensorflow.py --model bike_classifier.keras \
        --test-dir images/mountain --test-dir images/road \
        --train-dir images/trainset

Exposes:
    run_evaluation(model_path, test_dirs, train_dir, threshold=0.5)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


def get_class_names(train_dir):
    """
    Get class names from training directory subdirectories, sorted
    alphabetically (mirrors Keras ImageDataGenerator.flow_from_directory ordering).
    """
    subdirs = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    return subdirs


def _train_model(model_path, train_dir):
    """
    Train a new Keras model using the same architecture and pipeline as
    train.py, then save it to model_path.
    """
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
    from keras.callbacks import TensorBoard

    print(f"Training new model from '{train_dir}' → saving to '{model_path}' ...")

    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    tensorboard = TensorBoard(log_dir='./logs_train', histogram_freq=0,
                              write_graph=True, write_images=False)

    classifier.fit(
        training_set,
        steps_per_epoch=5,
        epochs=10,
        validation_steps=20,
        callbacks=[tensorboard]
    )

    # Ensure the directory for the model file exists
    model_dir = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(model_dir, exist_ok=True)

    classifier.save(model_path)
    print(f"Model saved to '{model_path}'.")
    return classifier


def _preprocess_image(filepath):
    """
    Apply the same preprocessing pipeline used during training:
      - Load image, resize to 64×64
      - Rescale pixel values to [0, 1]  (identical to ImageDataGenerator rescale=1./255)
    Returns a numpy array of shape (1, 64, 64, 3).
    """
    from keras.preprocessing import image as keras_image

    img = keras_image.load_img(filepath, target_size=(64, 64))
    arr = keras_image.img_to_array(img)          # shape (64, 64, 3), dtype float32
    arr = arr / 255.0                             # rescale to [0, 1]
    arr = np.expand_dims(arr, axis=0)             # shape (1, 64, 64, 3)
    return arr


def run_evaluation(model_path, test_dirs, train_dir, threshold=0.5):
    """
    Evaluate images in *test_dirs* using the model at *model_path*.

    Parameters
    ----------
    model_path : str
        Path to the .keras model file.  If the file does not exist the model
        is trained from scratch using *train_dir* and saved to *model_path*.
    test_dirs  : list[str]
        Directories whose images will be evaluated.  The directory *name*
        (basename) is used as the ground-truth label.
    train_dir  : str
        Directory that was (or will be) used for training.  Its subdirectory
        names define the ordered class list.
    threshold  : float, optional
        Decision threshold for the sigmoid output.  Defaults to 0.5.

    Returns
    -------
    list[dict]
        Each dict contains: filename, predicted_class, confidence_score,
        ground_truth.  The list is also written to
        json/tensorflow/<model_stem>.json.
    """
    from tensorflow.keras.models import load_model

    # ── Load or train model ──────────────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"[TF] Model file '{model_path}' not found – training a new one.")
        classifier = _train_model(model_path, train_dir)
    else:
        print(f"[TF] Loading model from '{model_path}' ...")
        classifier = load_model(model_path)

    # ── Determine class names from training directory ────────────────────────
    class_names = get_class_names(train_dir)
    if len(class_names) < 2:
        raise ValueError(
            f"Expected at least 2 class subdirectories in '{train_dir}', "
            f"found: {class_names}"
        )

    results = []

    for test_dir in test_dirs:
        # The folder name is the ground-truth label for every image inside it
        ground_truth = os.path.basename(os.path.normpath(test_dir))

        files = sorted([
            f for f in os.listdir(test_dir)
            if os.path.isfile(os.path.join(test_dir, f))
        ])

        for filename in files:
            filepath = os.path.join(test_dir, filename)

            img_array = _preprocess_image(filepath)
            raw_output = classifier.predict(img_array, verbose=0)
            confidence = float(raw_output[0][0])   # sigmoid probability for class 1

            predicted_class = class_names[1] if confidence >= threshold else class_names[0]

            results.append({
                'filename': filename,
                'predicted_class': predicted_class,
                'confidence_score': round(confidence, 6),
                'ground_truth': ground_truth,
            })

            print(f"  {filename:30s}  pred={predicted_class:20s}  "
                  f"conf={confidence:.4f}  gt={ground_truth}")

    # ── Write JSON report ────────────────────────────────────────────────────
    model_stem = Path(model_path).stem
    output_dir = os.path.join('json', 'tensorflow')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_stem}.json')

    with open(output_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[TF] Report written to '{output_path}'  ({len(results)} records).")
    return results


# ── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a Keras image-classification model and output a JSON report.'
    )
    parser.add_argument(
        '--model', default='bike_classifier.keras',
        help='Path to the .keras model file (default: bike_classifier.keras)'
    )
    parser.add_argument(
        '--test-dir', action='append', dest='test_dirs',
        metavar='DIR',
        help='Directory of test images (may be given multiple times). '
             'Defaults to images/mountain and images/road.'
    )
    parser.add_argument(
        '--train-dir', default='images/trainset',
        help='Training-set root directory (default: images/trainset)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Sigmoid decision threshold (default: 0.5)'
    )

    args = parser.parse_args()

    if args.test_dirs is None:
        args.test_dirs = ['images/mountain', 'images/road']

    run_evaluation(
        model_path=args.model,
        test_dirs=args.test_dirs,
        train_dir=args.train_dir,
        threshold=args.threshold,
    )
    sys.exit(0)
