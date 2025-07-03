
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tut Colors Analyzer", layout="wide")
st.title("🎨 Tut Colors - Mixing Ratio Analyzer")

uploaded_file = st.file_uploader("Upload Image of Pastels", type=["jpg", "jpeg", "png"])
num_colors = st.slider("How many colors to extract?", min_value=5, max_value=25, value=18)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    crop = img_array[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    small = cv2.resize(crop, (200, 200))
    pixels = small.reshape((-1, 3))
    km = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    colors = np.round(km.cluster_centers_).astype(int)

    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def estimate_mix(rgb):
        r, g, b = rgb
        brightness = np.mean([r, g, b]) / 255
        chroma = max(rgb) - min(rgb)
        white = round((brightness ** 2) * (1 - chroma / 255) * 100, 1)
        rest = 100 - white
        total = r + g + b
        blue = b / total
        red = r / total
        yellow = g / total
        sum_parts = blue + red + yellow
        blue_pct = round(blue / sum_parts * rest, 1)
        red_pct = round(red / sum_parts * rest, 1)
        yellow_pct = round(yellow / sum_parts * rest, 1)
        return {
            "Blue": blue_pct,
            "Red": red_pct,
            "Yellow": yellow_pct,
            "White": white
        }

    hexes = [rgb_to_hex(c) for c in colors]
    brightness_order = [np.mean(c) for c in colors]
    order = np.argsort(brightness_order)[::-1]
    colors = [colors[i] for i in order]
    hexes = [hexes[i] for i in order]
    mix_ratios = [estimate_mix(c) for c in colors]

    fig, ax = plt.subplots(figsize=(9, 0.5 + num_colors * 0.35))
    ax.axis("off")

    col_labels = ["Color", "Code", "Name", "Mix Ratio (Y/R/B/W)"]
    table_data = []

    for i, color in enumerate(colors):
        mix = mix_ratios[i]
        table_data.append([
            "", f"BG{i+1}", f"Tone {i+1}",
            f"B:{mix['Blue']}% R:{mix['Red']}% Y:{mix['Yellow']}% W:{mix['White']}%"
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='left',
        colLoc='center',
        loc='center',
        colWidths=[0.1, 0.1, 0.2, 0.6]
    )

    for i, color in enumerate(colors):
        hex_color = rgb_to_hex(color)
        table[(i+1, 0)].set_facecolor(hex_color)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)

    st.pyplot(fig)
