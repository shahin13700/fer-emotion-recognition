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
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="add_7", pred_index=None):
    """
    Generates Grad-CAM heatmap for a given input image array and target class.
    """
    # Create sub-model mapping input -> (last conv output, final output)
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of target class with respect to the last feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector where each entry is the mean intensity of gradient over a feature channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in feature map by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
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


def main():
    print("Initializing Grad-CAM Interpretability Engine...")
    
    output_dir = 'outputs/gradcam_samples'
    os.makedirs(output_dir, exist_ok=True)

    with open('outputs/class_indices.json') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k.capitalize() for k, v in class_indices.items()}

    model = load_model('model/emotion_model.keras')
    test_dir = 'dataset/test'
    
    emotions = sorted(os.listdir(test_dir))
    print(f"Generating Grad-CAM explanations for {len(emotions)} emotion categories...")

    fig, axes = plt.subplots(len(emotions), 3, figsize=(9, 2.5 * len(emotions)))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i, emotion in enumerate(emotions):
        folder = os.path.join(test_dir, emotion)
        if not os.path.isdir(folder):
            continue
            
        sample_img_name = os.listdir(folder)[0]
        img_path = os.path.join(folder, sample_img_name)
        
        # Read grayscale 48x48
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(gray_img, (48, 48))
        norm_input = resized.astype('float32') / 255.0
        inp_tensor = np.expand_dims(np.expand_dims(norm_input, axis=0), axis=-1)

        # Predict
        preds = model(inp_tensor, training=False).numpy()[0]
        top_idx = int(np.argmax(preds))
        pred_label = idx_to_class[top_idx]
        conf = preds[top_idx] * 100

        # Compute Grad-CAM
        heatmap = make_gradcam_heatmap(inp_tensor, model, last_conv_layer_name="add_7", pred_index=top_idx)
        overlay, colored_hm = overlay_heatmap(heatmap, resized)

        # Plot Original
        axes[i, 0].imshow(resized, cmap='gray')
        axes[i, 0].set_title(f"True: {emotion.capitalize()}", fontsize=10)
        axes[i, 0].axis('off')

        # Plot Heatmap
        axes[i, 1].imshow(colored_hm)
        axes[i, 1].set_title("Activation Heatmap", fontsize=10)
        axes[i, 1].axis('off')

        # Plot Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Grad-CAM ({pred_label}: {conf:.1f}%)", fontsize=10)
        axes[i, 2].axis('off')

    plt.suptitle("Explainable AI: MiniXception Grad-CAM Feature Attributions", fontsize=13, y=1.002)
    save_path = os.path.join(output_dir, 'gradcam_gallery.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM analysis complete! Visual gallery saved to {save_path}")

if __name__ == '__main__':
    main()
