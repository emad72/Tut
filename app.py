
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
    target_color = sRGBColor(*[c / 255 for c in target_rgb])
    target_lab = convert_color(target_color, LabColor)

    distances = []
    for ref_rgb in reference_rgbs:
        ref_color = sRGBColor(*[c / 255 for c in ref_rgb])
        lab = convert_color(ref_color, LabColor)
        delta_e = delta_e_cie2000(target_lab, lab)
        distances.append(delta_e.item())  # Fixed .asscalar()

    total = sum(distances)
    if total == 0:
        percentages = [0 for _ in distances]
    else:
        percentages = [round(100 * (1 - d / total), 1) for d in distances]

    return percentages

st.set_page_config(layout="wide")
st.title("🎨 Tut Color Analyzer")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    num_colors = st.slider("Number of tones to extract", 2, 30, 7)

    img_np = np.array(image)
    img_np = img_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, n_init=10).fit(img_np)
    colors = kmeans.cluster_centers_.astype(int)

    reference_rgbs = [
        (255, 255, 255),  # White
        (255, 255, 0),    # Yellow
        (255, 0, 0),      # Red
        (0, 0, 255),      # Blue
    ]

    mix_results = [get_mix_percentages(color, reference_rgbs) for color in colors]

    fig, ax = plt.subplots(figsize=(14, len(colors) * 1.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, len(colors))
    ax.axis("off")

    for idx, (color, mix) in enumerate(zip(colors, mix_results)):
        y = len(colors) - idx - 1
        rect = patches.Rectangle((0, y), 10, 1, linewidth=1, edgecolor='none', facecolor=np.array(color)/255)
        ax.add_patch(rect)
        txt = f"RGB: {tuple(color)} | Y: {mix[1]}% R: {mix[2]}% B: {mix[3]}% W: {mix[0]}%"
        ax.text(12, y + 0.3, txt, fontsize=12, va="center")

    output_path = "color_mix_guide.png"
    plt.savefig(output_path, bbox_inches="tight", format="png")
    st.image(output_path, caption="Mixing Guide", use_container_width=True)
