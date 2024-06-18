from keras.src.saving import load_model as keras_load_model
from sklearn.metrics import roc_curve, auc
from keras.src.models import Model
import matplotlib.pyplot as plt
from tkinter import filedialog
from pathlib import Path
from PIL import Image
import tkinter as tk
import numpy as np
import config
import random
import re
import os


def get_random_element_excluding_index(array, exclude_index):
    while True:
        random_index = random.randint(0, len(array) - 1)
        if random_index != exclude_index:
            return random_index


def load_image_as_array(img_path):
    img = Image.open(img_path)
    return np.array(img)


def numerical_sort(value):
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def load_images_from_folder(folder, start_index=0, end_index=200):
    files = sorted(os.listdir(folder), key=numerical_sort)
    files = files[start_index:end_index]
    (loaded, total) = (1, len(files))
    images = []

    for filename in files:
        img_path = os.path.join(folder, filename)

        if os.path.isfile(img_path):
            print(f"Loaded {(((loaded)/total)*100):.2f}% ({loaded} - {filename})")

            loaded += 1

            images.append(load_image_as_array(img_path))

    return images


def load_pairs_and_labels(start_index=0, end_index=200):
    pairs = []
    labels = []
    sharp_images = load_images_from_folder(config.SHARP_PATH, start_index, end_index)
    blur_images = load_images_from_folder(config.BLUR_PATH, start_index, end_index)

    for index, value in enumerate(sharp_images):
        pairs.append([sharp_images[index], blur_images[index]])
        labels.append([1])

        negIndex = get_random_element_excluding_index(blur_images, index)
        pairs.append([sharp_images[index], blur_images[negIndex]])
        labels.append([0])

    return (np.array(pairs), np.array(labels))


def normalize_pairs(pairs, batch_size=1000):
    num_batches = len(pairs) // batch_size + int(len(pairs) % batch_size != 0)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(pairs))
        pairs[start:end] = pairs[start:end] / 255.0
        print(f"Processed batch {i + 1}/{num_batches}")

    return pairs


def load_model() -> Model:
    return keras_load_model(f"{config.MODEL_PATH}.keras")


def save_model(model: Model):
    print("[INFO] saving siamese network...")
    Path(f"{config.BASE_OUTPUT}").mkdir(parents=True, exist_ok=True)
    model.save(f"{config.MODEL_PATH}.keras")


def create_test_pairs(sharp_images, blur_images, correct_percentage):
    n_samples = len(sharp_images)
    assert n_samples == len(
        blur_images
    ), "[ERROR] Sharp and blur arrays must have same length"

    n_correct = int(n_samples * correct_percentage)
    n_incorrect = n_samples - n_correct

    correct_indices = np.arange(n_samples)
    np.random.shuffle(correct_indices)
    correct_pairs = [
        (sharp_images[i], blur_images[i]) for i in correct_indices[:n_correct]
    ]

    incorrect_pairs = []
    incorrect_indices = np.arange(n_samples)
    for i in correct_indices[:n_incorrect]:
        incorrect_index = np.random.choice(incorrect_indices[incorrect_indices != i])
        incorrect_pairs.append((sharp_images[i], blur_images[incorrect_index]))

    return normalize_pairs(np.array(correct_pairs)), normalize_pairs(
        np.array(incorrect_pairs)
    )


def label_and_shuffle_pairs(correct_pairs, incorrect_pairs):
    labeled_correct_pairs = [(pair, [1]) for pair in correct_pairs]
    labeled_incorrect_pairs = [(pair, [0]) for pair in incorrect_pairs]

    labeled_pairs = labeled_correct_pairs + labeled_incorrect_pairs

    np.random.shuffle(labeled_pairs)

    pairs, labels = zip(*labeled_pairs)

    return np.array(pairs), np.array(labels)


def plot_training(H, plotPath):
    print("[INFO] plotting training history...")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


def plot_roc(model: Model, test_data, test_labels, folder_name):
    y_pred = model.predict(test_data).ravel()

    fpr, tpr, threshold = roc_curve(test_labels, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="Curva ROC (área = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")

    save_plot_path = (
        Path(__file__).parent.joinpath(folder_name).joinpath("plot_roc.png")
    )
    plt.savefig(save_plot_path.as_posix())

    plt.show()


def select_images(
    sharp_p="/var/www/facul/ic/__old__/images/test/sharp",
    blur_p="/var/www/facul/ic/__old__/images/test/blur",
):
    root = tk.Tk()
    root.withdraw()

    sharp_path = filedialog.askopenfilename(
        initialdir=sharp_p,
        title="Select sharp image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")],
    )

    blur_path = filedialog.askopenfilename(
        initialdir=blur_p,
        title="Select blur image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")],
    )

    return [load_image_as_array(sharp_path), load_image_as_array(blur_path)]
