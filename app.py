
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

# ---------------- Color Mixing ---------------- #
def get_mix_percentages(target_rgb, reference_rgbs):
    target_color = sRGBColor(*[v / 255 for v in target_rgb])
    target_lab = convert_color(target_color, LabColor)

    similarities = []
    for ref_rgb in reference_rgbs:
        ref_color = sRGBColor(*[v / 255 for v in ref_rgb])
        lab = convert_color(ref_color, LabColor)
        delta_e = delta_e_cie2000(target_lab, lab)
        similarities.append(delta_e)

    similarities = np.array(similarities)
    weights = np.maximum(1e-10, 1 / (similarities + 1e-6))  # Avoid divide by zero
    weights /= weights.sum()

    percentages = (weights * 100).round(1)
    return percentages

# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="Tut Analyzer", layout="centered")

st.title("🎨 Tut Color Analyzer")
st.markdown("Upload an image with your pastel tones. The app will extract the dominant colors and calculate the mix (Y/R/B/W).")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    num_colors = st.slider("Number of tones to extract", 2, 48, 6)

    img_data = np.array(img)
    img_data = img_data.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(img_data)
    colors = kmeans.cluster_centers_.astype(int)

    # Reference primary components
    reference_colors = {
        "Yellow": (255, 255, 0),
        "Red": (255, 0, 0),
        "Blue": (0, 0, 255),
        "White": (255, 255, 255),
    }
    reference_names = list(reference_colors.keys())
    reference_rgbs = list(reference_colors.values())

    mix_results = [get_mix_percentages(color, reference_rgbs) for color in colors]

    # Display result as table image
    fig, ax = plt.subplots(figsize=(8, len(colors) * 0.6))
    ax.axis("off")

    cell_height = 1
    for i, (color, mix) in enumerate(zip(colors, mix_results)):
        y = -i * cell_height

        # Color box
        rect = patches.Rectangle((0, y), 1, cell_height, color=np.array(color) / 255)
        ax.add_patch(rect)

        # Text
        text = f"{int(color[0])}, {int(color[1])}, {int(color[2])} → "
        text += " | ".join(f"{name}: {pct:.1f}%" for name, pct in zip(reference_names, mix))
        ax.text(1.1, y + 0.3, text, va="center", fontsize=10)

    ax.set_xlim(0, 10)
    ax.set_ylim(-len(colors), 1)

    output_path = "/mount/src/tut/color_mix_result.png" if os.path.exists("/mount/src/tut") else "color_mix_result.png"
    plt.savefig(output_path, bbox_inches="tight", format="png")
    st.image(output_path, caption="Mix Guide", use_container_width=True)
    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download Result Image", f, file_name="Tut_Mix_Guide.png", mime="image/png")
