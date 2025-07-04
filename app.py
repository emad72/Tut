
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

# Set page config
st.set_page_config(page_title="🎨 Tut Color Analyzer", layout="centered")

# Reference palette (Y, R, B, W)
reference_colors = {
    "Yellow": (255, 255, 0),
    "Red": (255, 0, 0),
    "Blue": (0, 0, 255),
    "White": (255, 255, 255)
}

def rgb_to_lab(color):
    rgb = sRGBColor(color[0]/255.0, color[1]/255.0, color[2]/255.0)
    lab = convert_color(rgb, LabColor)
    return lab

def get_mix_percentages(color_rgb, reference_palette):
    target_lab = rgb_to_lab(color_rgb)
    ref_labs = {name: rgb_to_lab(rgb) for name, rgb in reference_palette.items()}

    distances = {}
    for name, lab in ref_labs.items():
        delta_e = delta_e_cie2000(target_lab, lab)
        distances[name] = delta_e.item() if hasattr(delta_e, 'item') else float(delta_e)

    # Inverse distances (closer color -> higher weight)
    inverse = {k: 1 / (v + 1e-6) for k, v in distances.items()}
    total = sum(inverse.values())
    percentages = {k: round((v / total) * 100) for k, v in inverse.items()}

    return percentages

def plot_results(colors, mix_data):
    fig, ax = plt.subplots(figsize=(10, len(colors) * 0.6))

    for i, (color, mix) in enumerate(zip(colors, mix_data)):
        y = len(colors) - i - 1
        # Color swatch
        ax.add_patch(patches.Rectangle((0, y), 1, 1, color=np.array(color)/255))
        # Percentages
        text = ', '.join([f"{k}: {v}%" for k, v in mix.items()])
        ax.text(1.1, y + 0.5, text, va='center', ha='left', fontsize=9)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(colors))
    ax.axis('off')
    return fig

st.title("🎨 Tut Color Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    num_colors = st.slider("Number of tones to extract", 2, 48, 8)

    img_array = np.array(image)
    img_reshaped = img_array.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(img_reshaped)
    colors = kmeans.cluster_centers_.astype(int).tolist()

    mix_results = [get_mix_percentages(color, reference_colors) for color in colors]

    fig = plot_results(colors, mix_results)

    output_path = "color_mix_output.png"
    plt.savefig(output_path, bbox_inches="tight", format="png")
    st.pyplot(fig)
    with open(output_path, "rb") as file:
        st.download_button("Download Result", file, file_name="Tut_Color_Mix_Guide.png")
