import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

st.set_page_config(layout="wide")

st.title("🎨 Tut Color Mixer Analyzer")

uploaded_file = st.file_uploader("Upload Image of Pastels", type=["png", "jpg", "jpeg"])

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def estimate_mix(rgb):
    r, g, b = rgb
    w = (255 - max(rgb)) / 255
    total = r + g + b
    if total == 0:
        return {"Y": 0, "R": 0, "B": 0, "W": 100}
    y = r / total
    r_ = g / total
    b_ = b / total
    base = y + r_ + b_ + w
    return {
        "Y": round(y / base * 100, 1),
        "R": round(r_ / base * 100, 1),
        "B": round(b_ / base * 100, 1),
        "W": round(w / base * 100, 1)
    }

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    crop = img_np[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    small = np.array(Image.fromarray(crop).resize((200, 200)))
    pixels = small.reshape((-1, 3))

    n_colors = 18
    km = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)
    centers = np.round(km.cluster_centers_).astype(int)
    brightness = [np.mean(c) for c in centers]
    order = np.argsort(brightness)[::-1]
    sorted_colors = [centers[i] for i in order]
    hexes = [rgb_to_hex(c) for c in sorted_colors]
    mixes = [estimate_mix(c) for c in sorted_colors]

    fig, ax = plt.subplots(figsize=(9, len(sorted_colors) * 0.6))
    ax.axis("off")

    table_data = []
    for i, mix in enumerate(mixes):
        row = [
            f"BG{i+1}",
            f"Tone {i+1}",
            f"B:{mix['B']}% R:{mix['R']}% Y:{mix['Y']}% W:{mix['W']}%"
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=["Code", "Name", "Mix Ratio (Y/R/B/W)"],
        colLoc="left", cellLoc="left",
        loc="center", colWidths=[0.15, 0.2, 0.65]
    )

    for i, color in enumerate(sorted_colors):
        table[(i + 1, 0)].set_facecolor(rgb_to_hex(color))

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    st.pyplot(fig)
