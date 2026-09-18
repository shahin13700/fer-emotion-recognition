"""
Explainable AI (XAI) - Grad-CAM Visualizer for MiniXception
Computes Gradient-weighted Class Activation Mapping to visualize
which facial regions triggered each emotion prediction.
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def find_last_conv_layer(model):
    """
    Dynamically discovers the last residual (Add) or convolution layer in the model.
    Prevents crashing if retrained with different layer names or module counts.
    """
    for layer in reversed(model.layers):
        name = layer.name.lower()
        if 'add' in name or 'separableconv' in name or 'conv2d' in name:
            return layer.name
    return "add_7"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None, conv_model=None, subsequent_layers=None):
    """
    Generates Grad-CAM heatmap for a given input image array and target class.
    Uses direct 2-stage differentiation for robust, crash-free execution in Keras 3.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    if conv_model is None:
        conv_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.get_layer(last_conv_layer_name).output
        )

    if subsequent_layers is None:
        subsequent_layers = []
        found = False
        for layer in model.layers:
            if found:
                subsequent_layers.append(layer)
            elif layer.name == last_conv_layer_name:
                found = True

    # 1. Forward pass through convolutional feature extractor
    conv_output = conv_model(img_array)

    # 2. Differentiate class activation directly with respect to conv_output
    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        x = conv_output
        for layer in subsequent_layers:
            x = layer(x, training=False)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of target class with respect to the feature map activations
    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        grads = tf.ones_like(conv_output)

    # Pool gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map channel by its gradient importance
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU on heatmap to only consider positive influences
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.45):
    """
    Overlays Grad-CAM heatmap onto a grayscale/RGB original image.
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    if len(original_img.shape) == 2 or original_img.shape[2] == 1:
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    superimposed = np.uint8(heatmap_colored * alpha + original_bgr * (1.0 - alpha))
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB), cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)


def main(output_dir='outputs/gradcam_samples', model_path='model/emotion_model.keras'):
    """
    Renders a Grad-CAM gallery. Note the spatial resolution: the target layer's feature map
    is 3x3 for this architecture, so each heatmap is a coarse 3x3 attribution grid upsampled
    to 48x48. It shows which region of the face contributed most, not fine detail.
    """
    print("Initializing Grad-CAM Interpretability Engine...")

    os.makedirs(output_dir, exist_ok=True)

    with open('outputs/class_indices.json') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k.capitalize() for k, v in class_indices.items()}

    model = load_model(model_path)
    target_layer = find_last_conv_layer(model)
    fmap_shape = model.get_layer(target_layer).output.shape
    print(f"Targeting interpretability feature layer: {target_layer} (feature map {fmap_shape[1]}x{fmap_shape[2]})")

    test_dir = 'dataset/test'
    samples_dir = 'assets/samples'

    # Support out-of-the-box execution on clean clones without full dataset
    sample_images = {}
    if os.path.exists(test_dir):
        for emotion in sorted(os.listdir(test_dir)):
            folder = os.path.join(test_dir, emotion)
            if os.path.isdir(folder):
                files = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
                if files:
                    sample_images[emotion.lower()] = os.path.join(folder, files[0])

    # Fallback to packaged assets/samples/ for any missing categories
    if os.path.exists(samples_dir):
        for fname in sorted(os.listdir(samples_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                emo_name = os.path.splitext(fname)[0].lower()
                if emo_name not in sample_images:
                    sample_images[emo_name] = os.path.join(samples_dir, fname)

    if not sample_images:
        raise FileNotFoundError(
            "No test images found! Neither 'dataset/test' nor 'assets/samples/' contains valid images."
        )

    emotions = sorted(sample_images.keys())
    print(f"Generating Grad-CAM explanations for {len(emotions)} emotion categories...")

    conv_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer(target_layer).output
    )
    subsequent_layers = []
    found = False
    for layer in model.layers:
        if found:
            subsequent_layers.append(layer)
        elif layer.name == target_layer:
            found = True

    fig, axes = plt.subplots(len(emotions), 3, figsize=(9, 2.5 * len(emotions)), squeeze=False)
    n_correct = 0

    for i, emotion in enumerate(emotions):
        img_path = sample_images[emotion]

        # Read grayscale 48x48
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(gray_img, (48, 48))
        norm_input = resized.astype('float32') / 255.0
        inp_tensor = np.expand_dims(np.expand_dims(norm_input, axis=0), axis=-1)

        # Predict
        preds = model(inp_tensor, training=False).numpy()[0]
        top_idx = int(np.argmax(preds))
        pred_label = idx_to_class.get(top_idx, f"Class {top_idx}")
        conf = preds[top_idx] * 100

        # Compute Grad-CAM using direct 2-stage submodel
        heatmap = make_gradcam_heatmap(
            inp_tensor,
            model,
            conv_model=conv_model,
            subsequent_layers=subsequent_layers,
            pred_index=top_idx
        )
        overlay, colored_hm = overlay_heatmap(heatmap, resized)

        # Plot Original
        axes[i, 0].imshow(resized, cmap='gray')
        axes[i, 0].set_title(f"True: {emotion.capitalize()}", fontsize=10)
        axes[i, 0].axis('off')

        # Plot Heatmap
        axes[i, 1].imshow(colored_hm)
        axes[i, 1].set_title("Activation Heatmap", fontsize=10)
        axes[i, 1].axis('off')

        # Plot Overlay (flag misclassified samples honestly)
        correct = pred_label.lower() == emotion.lower()
        n_correct += int(correct)
        mark = "✓" if correct else "✗"
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"{mark} Pred: {pred_label} ({conf:.1f}%)", fontsize=10,
                             color=("black" if correct else "firebrick"))
        axes[i, 2].axis('off')

    fig.suptitle(f"MiniXception Grad-CAM ({fmap_shape[1]}x{fmap_shape[2]} attribution grid) — "
                 f"{n_correct}/{len(emotions)} samples classified correctly", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_path = os.path.join(output_dir, 'gradcam_gallery.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    print(f"Grad-CAM analysis complete! {n_correct}/{len(emotions)} samples correct. Gallery saved to {save_path}")
    return save_path

if __name__ == '__main__':
    main()
