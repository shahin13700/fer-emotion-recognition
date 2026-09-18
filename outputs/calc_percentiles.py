import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('model/emotion_model.keras')

with open('outputs/class_indices.json') as f:
    class_indices = json.load(f)
idx_to_class = {v: k.capitalize() for k, v in class_indices.items()}

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    'dataset/test',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

preds = model.predict(generator, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = generator.classes

results = {}
for i in range(len(class_indices)):
    class_name = idx_to_class[i]
    correct_mask = (y_true == i) & (y_pred == i)
    correct_confs = preds[correct_mask, i]
    
    total_samples = int(np.sum(y_true == i))
    correct_count = int(len(correct_confs))
    
    p25 = float(np.percentile(correct_confs, 25))
    p50 = float(np.percentile(correct_confs, 50))
    p75 = float(np.percentile(correct_confs, 75))
    
    results[class_name] = {
        'total': total_samples,
        'correct': correct_count,
        'recall': float(correct_count / total_samples),
        'p25': p25,
        'p50': p50,
        'p75': p75
    }

print("\n" + "="*75)
print(f"{'Emotion':<10} | {'25th %ile':<10} | {'Median':<10} | {'75th %ile':<10} | {'Recall':<10} | {'Support'}")
print("="*75)
for k, v in results.items():
    print(f"{k:<10} | {v['p25']*100:>8.1f}% | {v['p50']*100:>8.1f}% | {v['p75']*100:>8.1f}% | {v['recall']*100:>8.1f}% | {v['correct']}/{v['total']}")
print("="*75)

with open('outputs/empirical_thresholds.json', 'w') as f:
    json.dump(results, f, indent=2)
