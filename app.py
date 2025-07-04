
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

def get_mix_percentages(target_rgb, reference_rgbs):
    target_color = sRGBColor(*target_rgb, is_upscaled=True)
    target_lab = convert_color(target_color, LabColor)

    weights = []
    for rgb in reference_rgbs:
        color = sRGBColor(*rgb, is_upscaled=True)
        lab = convert_color(color, LabColor)
        delta_e = delta_e_cie2000(target_lab, lab)
        weights.append(1 / (delta_e.item() + 1e-6))  # Fixed here
    total = sum(weights)
    percentages = [(w / total) * 100 for w in weights]
    return [round(p, 1) for p in percentages]

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def load_image(image_file):
    image = Image.open(image_file)
    return image

def extract_colors(img, n_colors):
    img = img.convert('RGB')
    img_np = np.array(img)
    img_np = img_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(img_np)

    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Streamlit app UI
st.set_page_config(layout="wide")
st.title("🎨 Tut Color Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    num_colors = st.slider("Number of tones to extract", min_value=2, max_value=48, value=6)
    colors = extract_colors(image, num_colors)

    # Tut reference palette (example)
    reference_rgbs = [
        (255, 255, 255),  # White
        (255, 255, 0),    # Yellow
        (255, 0, 0),      # Red
        (0, 0, 255)       # Blue
    ]
    reference_labels = ["White", "Yellow", "Red", "Blue"]

    mix_results = [get_mix_percentages(color, reference_rgbs) for color in colors]

    fig, ax = plt.subplots(figsize=(12, len(colors) * 1.2))
    ax.axis("off")

    for i, (color, mix) in enumerate(zip(colors, mix_results)):
        y = len(colors) - i - 1
        hex_color = rgb_to_hex(color)
        ax.add_patch(patches.Rectangle((0, y), 1, 1, color=hex_color))
        ax.text(1.1, y + 0.5, f"{hex_color}", va='center', fontsize=10)

        for j, percent in enumerate(mix):
            ax.text(2 + j * 1.5, y + 0.5, f"{reference_labels[j]}: {percent:.1f}%", va='center', fontsize=10)

    output_path = "/mnt/data/app_output.png"
    plt.savefig(output_path, bbox_inches="tight", format="png")
    st.image(output_path, caption="Mixing Table", use_container_width=True)
