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

# Reference primary colors
reference_colors = {
    "Yellow": (255, 255, 0),
    "Red": (255, 0, 0),
    "Blue": (0, 0, 255),
    "White": (255, 255, 255)
}

def rgb_to_lab(color):
    srgb = sRGBColor(color[0]/255, color[1]/255, color[2]/255)
    return convert_color(srgb, LabColor)

def get_mix_percentages(target_rgb, reference_rgbs):
    target_lab = rgb_to_lab(target_rgb)
    distances = []
    for ref_name, ref_rgb in reference_rgbs.items():
        lab = rgb_to_lab(ref_rgb)
        delta_e = delta_e_cie2000(target_lab, lab)
        distances.append((ref_name, delta_e))

    # Convert distances to similarity (inverse)
    similarities = [(name, 1/(dist + 1e-6)) for name, dist in distances]
    total = sum(sim for _, sim in similarities)
    percentages = [(name, sim / total * 100) for name, sim in similarities]
    return dict(percentages)

def extract_colors(image, num_colors):
    image = image.resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init='auto')
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int)

def display_color_guide(colors, mix_results):
    fig, ax = plt.subplots(figsize=(10, len(colors) * 1.2))
    for i, (color, mix) in enumerate(zip(colors, mix_results)):
        y = len(colors) - i - 1
        rect = patches.Rectangle((0, y), 1, 1, linewidth=1, edgecolor='black',
                                 facecolor=np.array(color)/255)
        ax.add_patch(rect)
        text = ", ".join(f"{k}: {v:.1f}%" for k, v in mix.items())
        ax.text(1.1, y + 0.35, text, verticalalignment='center', fontsize=10)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(colors))
    ax.axis('off')
    return fig

st.set_page_config(layout="wide")
st.title("🎨 Tut Color Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    num_colors = st.slider("Number of tones to extract", min_value=2, max_value=30, value=5)
    colors = extract_colors(image, num_colors)
    mix_results = [get_mix_percentages(color, reference_colors) for color in colors]
    fig = display_color_guide(colors, mix_results)

    output_path = "/tmp/color_mix_guide_output.png"
    plt.savefig(output_path, bbox_inches="tight", format="png")
    st.pyplot(fig)

    with open(output_path, "rb") as f:
        st.download_button("Download Color Guide Image", f, file_name="Tut_Mix_Guide.png", mime="image/png")