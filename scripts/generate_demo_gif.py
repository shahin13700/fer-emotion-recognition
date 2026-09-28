"""
Utility script to generate an animated demo GIF (assets/demo.gif)
simulating real-time emotion tracking, photo booth progression,
and drowsiness detection using the project's MiniXception model.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

def build_demo_gif(output_path="assets/demo.gif"):
    print("Generating animated demo GIF...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Dimensions for hero GIF
    w, h = 640, 400
    frames = []
    
    # Try loading a system font
    try:
        font_large = ImageFont.truetype("arialbd.ttf", 20)
        font_main = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except IOError:
        font_large = ImageFont.load_default()
        font_main = ImageFont.load_default()
        font_small = ImageFont.load_default()

    sample_emotions = [
        ("happy.jpg", "Happy", 0.94, (46, 204, 113)),
        ("surprise.jpg", "Surprise", 0.88, (52, 152, 219)),
        ("neutral.jpg", "Neutral", 0.72, (149, 165, 166)),
        ("angry.jpg", "Angry", 0.79, (231, 76, 60)),
        ("fear.jpg", "Fear", 0.65, (155, 89, 182)),
        ("sad.jpg", "Sad", 0.61, (41, 128, 185)),
        ("disgust.jpg", "Disgust", 0.75, (230, 126, 34)),
    ]

    # Part 1: Simulate Photo Booth Emotion Progression (15 frames)
    unlocked = []
    for step, (img_name, emotion, conf, color) in enumerate(sample_emotions, 1):
        unlocked.append(emotion)
        img_path = os.path.join("assets", "samples", img_name)
        if os.path.exists(img_path):
            face_img = Image.open(img_path).convert("RGB").resize((180, 180), Image.Resampling.BILINEAR)
        else:
            face_img = Image.new("RGB", (180, 180), color=(40, 45, 55))

        # Repeat each emotion 2 frames for smooth viewing
        for sub in range(2):
            img = Image.new("RGB", (w, h), color=(15, 18, 24))
            draw = ImageDraw.Draw(img)

            # Header HUD
            draw.rectangle([0, 0, w, 40], fill=(22, 27, 36))
            draw.text((16, 10), "🎭 EdgeVision • 817 KB MiniXception (CPU ~2ms)", fill=(0, 230, 255), font=font_large)
            draw.text((w - 180, 12), "Webcam Stream: 30 FPS", fill=(100, 200, 120), font=font_small)

            # Left: Simulated Video Feed Box
            draw.rectangle([16, 52, 340, 340], fill=(26, 32, 42), outline=(50, 60, 75), width=2)
            # Paste face centered in video feed
            img.paste(face_img, (88, 100))

            # Bounding box with emotion color
            draw.rectangle([84, 96, 272, 284], outline=color, width=2)
            # Label banner
            draw.rectangle([84, 68, 272, 96], fill=color)
            draw.text((92, 72), f"{emotion.upper()} {int(conf * 100)}%", fill=(255, 255, 255), font=font_large)

            # Right: Live Side Panel & Challenge Tracker
            draw.rectangle([356, 52, w - 16, 340], fill=(20, 24, 32), outline=(40, 48, 62), width=1)
            draw.text((370, 64), "🎮 7-EMOTION PHOTO BOOTH", fill=(255, 255, 255), font=font_large)
            
            badge_text = f"Unlocked: {len(unlocked)} / 7 Emotions"
            draw.rectangle([370, 94, 520, 120], fill=(30, 80, 50) if len(unlocked) == 7 else (35, 45, 60))
            draw.text((380, 98), badge_text, fill=(46, 204, 113) if len(unlocked) == 7 else (0, 230, 255), font=font_main)

            # Emotion Progress Bars
            bar_y = 135
            for e_name, _, e_conf, e_col in sample_emotions:
                is_curr = (e_name == emotion)
                is_unlocked = (e_name in unlocked)
                draw.text((370, bar_y), e_name, fill=(255, 255, 255) if is_curr else (140, 150, 165), font=font_small)
                # Background track
                draw.rectangle([440, bar_y + 2, 590, bar_y + 12], fill=(30, 36, 46))
                if is_curr:
                    draw.rectangle([440, bar_y + 2, 440 + int(150 * conf), bar_y + 12], fill=color)
                elif is_unlocked:
                    draw.rectangle([440, bar_y + 2, 440 + int(150 * e_conf), bar_y + 12], fill=(60, 75, 95))
                bar_y += 24

            # Bottom info
            draw.text((16, 365), "⚡ EMA Temporal Smoothing Active  •  Zero Frame-Flicker", fill=(120, 140, 165), font=font_small)
            draw.text((w - 220, 365), "github.com/shahin13700/fer-emotion-recognition", fill=(80, 100, 120), font=font_small)

            frames.append(img)

    # Part 2: Simulate Driver Fatigue Guard (4 frames)
    for alert_cycle in range(3):
        for is_alert in [True, False]:
            img = Image.new("RGB", (w, h), color=(15, 18, 24))
            draw = ImageDraw.Draw(img)

            # Header
            draw.rectangle([0, 0, w, 40], fill=(35, 20, 20) if is_alert else (22, 27, 36))
            draw.text((16, 10), "👁️ Driver Fatigue & Drowsiness Guard (MediaPipe EAR)", fill=(255, 90, 90) if is_alert else (0, 230, 255), font=font_large)

            # Left Video feed with Face Landmark simulation
            draw.rectangle([16, 52, 340, 340], fill=(26, 32, 42), outline=(200, 50, 50) if is_alert else (50, 60, 75), width=2)
            if os.path.exists("assets/samples/neutral.jpg"):
                neut = Image.open("assets/samples/neutral.jpg").convert("RGB").resize((180, 180))
                img.paste(neut, (88, 100))

            # Eye landmark overlay indicators
            draw.ellipse([135, 160, 155, 165], fill=(255, 50, 50) if is_alert else (0, 230, 255))
            draw.ellipse([205, 160, 225, 165], fill=(255, 50, 50) if is_alert else (0, 230, 255))

            if is_alert:
                # Big flashing warning
                draw.rectangle([20, 150, 336, 210], fill=(200, 20, 20))
                draw.text((45, 168), "⚠️ DROWSINESS DETECTED!", fill=(255, 255, 255), font=font_large)

            # Right Telemetry
            draw.rectangle([356, 52, w - 16, 340], fill=(20, 24, 32), outline=(40, 48, 62), width=1)
            draw.text((370, 70), "DRIVER TELEMETRY", fill=(255, 255, 255), font=font_large)
            draw.text((370, 110), "Eye Aspect Ratio (EAR):", fill=(180, 190, 205), font=font_main)
            draw.text((370, 135), "0.14" if is_alert else "0.31", fill=(255, 75, 75) if is_alert else (46, 204, 113), font=font_large)
            draw.text((430, 140), "(Threshold: < 0.22)", fill=(120, 130, 145), font=font_small)

            draw.text((370, 180), "Closed Frame Counter:", fill=(180, 190, 205), font=font_main)
            draw.text((370, 205), "24 frames (ALERT)" if is_alert else "0 frames (NORMAL)", fill=(255, 75, 75) if is_alert else (46, 204, 113), font=font_large)

            draw.text((370, 260), "MediaPipe Face Landmarker", fill=(0, 230, 255), font=font_small)
            draw.text((370, 280), "478 3D Landmarks • Real-Time", fill=(120, 130, 145), font=font_small)

            draw.text((16, 365), "⚡ CPU-only monitoring with zero cloud dependencies", fill=(120, 140, 165), font=font_small)
            frames.append(img)

    # Save as animated GIF (duration=350ms per frame, loop indefinitely)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=380,
        loop=0,
        optimize=True
    )
    print(f"Animated demo GIF successfully created at {output_path} ({os.path.getsize(output_path) // 1024} KB)")

if __name__ == "__main__":
    build_demo_gif()
