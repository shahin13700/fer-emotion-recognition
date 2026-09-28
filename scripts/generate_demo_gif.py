"""
Utility script to generate an animated demo GIF (assets/demo.gif)
simulating real-time emotion tracking and the 7-emotion photo booth challenge.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

    # Part 1: Simulate Photo Booth Emotion Progression (14 frames)
    unlocked = []
    for step, (img_name, emotion, conf, color) in enumerate(sample_emotions, 1):
        unlocked.append(emotion)
        img_path = os.path.join("assets", "samples", img_name)
        if os.path.exists(img_path):
            face_img = Image.open(img_path).convert("RGB").resize((180, 180), Image.Resampling.BILINEAR)
        else:
            face_img = Image.new("RGB", (180, 180), color=(40, 45, 55))

        # 2 frames per emotion
        for sub in range(2):
            img = Image.new("RGB", (w, h), color=(15, 18, 24))
            draw = ImageDraw.Draw(img)

            # Header HUD
            draw.rectangle([0, 0, w, 40], fill=(22, 27, 36))
            draw.text((16, 10), "🎭 EdgeVision • 817 KB MiniXception (CPU ~2ms)", fill=(0, 230, 255), font=font_large)
            draw.text((w - 180, 12), "Webcam Stream: 30 FPS", fill=(100, 200, 120), font=font_small)

            # Left: Simulated Video Feed Box
            draw.rectangle([16, 52, 340, 340], fill=(26, 32, 42), outline=(50, 60, 75), width=2)
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

    # Part 2: Challenge Complete & Generated Photo Strip Showcase (4 frames)
    strip_path = os.path.join("assets", "sample_photo_strip.png")
    strip_img = None
    if os.path.exists(strip_path):
        strip_raw = Image.open(strip_path).convert("RGB")
        strip_img = strip_raw.resize((560, 260), Image.Resampling.LANCZOS)

    for i in range(4):
        img = Image.new("RGB", (w, h), color=(15, 18, 24))
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, w, 40], fill=(22, 40, 32))
        draw.text((16, 10), "🎉 CHALLENGE COMPLETED! 7 OF 7 EMOTIONS UNLOCKED", fill=(46, 204, 113), font=font_large)
        draw.text((w - 180, 12), "Grade: PERFECT! 🏆", fill=(255, 215, 0), font=font_main)

        # Showcase the Photo Strip in the center
        if strip_img is not None:
            img.paste(strip_img, (40, 60))
            draw.rectangle([38, 58, 602, 322], outline=(0, 230, 255), width=2)
        else:
            draw.rectangle([40, 60, 600, 320], fill=(30, 36, 46))
            draw.text((180, 180), "📸 Downloadable Emotion Photo Strip Ready", fill=(0, 230, 255), font=font_large)

        # Bottom Bar
        draw.rectangle([0, 345, w, h], fill=(20, 24, 32))
        draw.text((40, 360), "📸 1-Click Exportable Retro Photo Strip  •  100% In-Memory Privacy", fill=(180, 190, 205), font=font_main)
        draw.text((w - 180, 360), "⭐ Star on GitHub", fill=(0, 230, 255), font=font_main)

        frames.append(img)

    # Save as animated GIF (duration=400ms per frame, loop indefinitely)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=420,
        loop=0,
        optimize=True
    )
    print(f"Animated demo GIF successfully created at {output_path} ({os.path.getsize(output_path) // 1024} KB)")

if __name__ == "__main__":
    build_demo_gif()
