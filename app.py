
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="Tut Color Extractor", layout="wide")

st.title("Tut Color Mixing Guide 🎨")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
num_colors = st.slider("How many tones to extract?", 3, 30, 7)

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def estimate_mix(rgb):
    r, g, b = rgb
    norm = np.array(rgb) / 255.0
    lightness = np.mean(norm)
    chroma = max(norm) - min(norm)
    grayness = 1 - chroma
    w = round(lightness * grayness * 100, 1)

    if r + g + b == 0:
        return {"Y": 0, "R": 0, "B": 0, "W": 100}

    total = r + g + b
    Y = r / total
    R = g / total
    B = b / total
    base = Y + R + B
    return {
        "Y": round(Y / base * (100 - w), 1),
        "R": round(R / base * (100 - w), 1),
        "B": round(B / base * (100 - w), 1),
        "W": w
    }

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    crop = img_np[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    small = cv2.resize(crop, (200, 200))
    pixels = small.reshape((-1, 3))

    km = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    colors = np.round(km.cluster_centers_).astype(int)

    hexes = [rgb_to_hex(c) for c in colors]
    br = [np.mean(c) for c in colors]
    order = np.argsort(br)[::-1]
    colors = [colors[i] for i in order]
    hexes = [hexes[i] for i in order]
    mix = [estimate_mix(c) for c in colors]

    fig, ax = plt.subplots(figsize=(9, 1.5 + 0.8 * len(colors)))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"Tone {i+1}", hexes[i],
                   f"B:{mix[i]['B']}% R:{mix[i]['R']}% Y:{mix[i]['Y']}% W:{mix[i]['W']}%"]
                  for i in range(len(colors))],
        colLabels=["Name", "Hex", "Mix Ratio (Y/R/B/W)"],
        loc="center", cellLoc="left", colLoc="center",
        colWidths=[0.15, 0.2, 0.65]
    )
    for i, c in enumerate(colors):
        table[(i+1, 1)].set_facecolor(rgb_to_hex(c))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    st.image(buf, caption="Color Mixing Guide", use_column_width=True)

    # زر التحميل
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="mix_guide.png">📥 Download Image</a>'
    st.markdown(href, unsafe_allow_html=True)
