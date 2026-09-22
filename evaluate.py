import os
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

def main():
    os.makedirs('outputs', exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Plot Training vs Validation Curves
    # ---------------------------------------------------------
    history_path = 'outputs/history.pkl'
    if os.path.exists(history_path):
        print(f"Loading training history from {history_path}...")
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
            
        epochs = range(1, len(history['accuracy']) + 1)
        
        # Accuracy plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
        plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('outputs/accuracy_curve.png', bbox_inches='tight')
        plt.close()
        
        # Loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('outputs/loss_curve.png', bbox_inches='tight')
        plt.close()
        print("Training curves successfully saved to outputs/ (accuracy_curve.png, loss_curve.png).")
    else:
        print(f"Warning: {history_path} not found. Skipping curve plotting.")

    # ---------------------------------------------------------
    # 2. Evaluate Model on Test Dataset (Unseen Data)
    # ---------------------------------------------------------
    model_path = 'model/emotion_model.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
        
    print("\nLoading test dataset for evaluation... (using dataset/test/)")
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Critical: shuffle=False ensures prediction order matches y_true order for confusion matrix
    test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\nLoading trained model from {model_path}...")
    model = load_model(model_path)
    
    print("Running predictions on test set... This may take a moment.")
    predictions = model.predict(test_generator, verbose=1)
    
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Extract labels sorted by index to properly label our confusion matrix
    class_labels = list(test_generator.class_indices.keys())
    
    # ---------------------------------------------------------
    # 3. Save Classification Report
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)
    
    report_path = 'outputs/classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Model Evaluation - Classification Report\n")
        f.write("=========================================\n\n")
        f.write(report)
    print(f"Classification report successfully saved to {report_path}.")
    
    # ---------------------------------------------------------
    # 4. Save Confusion Matrix Heatmap
    # ---------------------------------------------------------
    print("\nGenerating Confusion Matrix heatmap...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix - FER2013')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    
    cm_path = 'outputs/confusion_matrix.png'
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix heatmap successfully saved to {cm_path}.")

if __name__ == '__main__':
    main()
