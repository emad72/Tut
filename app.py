
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🎨 Tut Color Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def perceptual_mix(rgb):
    r, g, b = rgb
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = h * 360

    white = round((v ** 2) * 100, 1) if v > 0.7 and s < 0.3 else 0
    remaining = 100 - white if white < 100 else 0.0001

    ratios = {"Y": 0, "R": 0, "B": 0}
    if s < 0.1:
        pass
    elif h_deg >= 0 and h_deg < 60:
        ratios["R"] = 1 - h_deg / 60
        ratios["Y"] = h_deg / 60
    elif h_deg >= 60 and h_deg < 180:
        ratios["Y"] = max(0, 1 - abs(h_deg - 90) / 90)
        ratios["B"] = max(0, (h_deg - 120) / 60) if h_deg >= 120 else 0
    elif h_deg >= 180 and h_deg < 300:
        ratios["B"] = 1 - abs(h_deg - 240) / 60
        ratios["R"] = max(0, (h_deg - 240) / 60)
    elif h_deg >= 300 and h_deg <= 360:
        ratios["R"] = 1 - abs(h_deg - 360) / 60
        ratios["B"] = (h_deg - 300) / 60

    total = sum(ratios.values())
    if total == 0:
        return {"Y": 0, "R": 0, "B": 0, "W": 100}
    for k in ratios:
        ratios[k] = round(ratios[k] / total * remaining, 1)

    ratios["W"] = round(white, 1)
    return ratios

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    num_colors = st.slider("Number of tones to extract", min_value=2, max_value=30, value=12)
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    crop = img_np[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    small = np.array(Image.fromarray(crop).resize((200, 200)))
    pixels = small.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    colors = np.round(kmeans.cluster_centers_).astype(int)

    brightness = [np.mean(c) for c in colors]
    order = np.argsort(brightness)[::-1]
    colors = [colors[i] for i in order]

    hexes = [rgb_to_hex(c) for c in colors]
    mixes = [perceptual_mix(c) for c in colors]

    fig, ax = plt.subplots(figsize=(9, len(colors) * 0.6))
    ax.axis("off")
    table_data = [
        [f"TC{i+1}", hexes[i],
         f"Y:{mixes[i]['Y']}%  R:{mixes[i]['R']}%  B:{mixes[i]['B']}%  W:{mixes[i]['W']}%"]
        for i in range(len(colors))
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Code", "Color", "Mix (Y/R/B/W)"],
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.15, 0.2, 0.65]
    )
    for i, c in enumerate(colors):
        table[(i+1, 1)].set_facecolor(rgb_to_hex(c))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    st.pyplot(fig)

    st.download_button(
        label="📥 Download Final Mix Guide",
        data=buf.getvalue(),
        file_name="Tut_Mix_Guide_Final.png",
        mime="image/png"
    )
