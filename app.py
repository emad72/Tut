
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tut Color Analyzer", layout="wide")
st.title("🎨 Tut Color Analyzer")
st.markdown("Upload an image and get the closest artistic color mixes (Y, R, B, W)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
n_colors = st.slider("عدد درجات الألوان المراد استخراجها:", min_value=6, max_value=48, value=18)

reference_colors = {
    "Cadmium Yellow Light": ((255, 255, 153), {"Y": 80, "R": 0, "B": 0, "W": 20}),
    "Cadmium Red": ((227, 38, 54), {"Y": 0, "R": 90, "B": 0, "W": 10}),
    "Ultramarine Blue": ((18, 10, 143), {"Y": 0, "R": 0, "B": 95, "W": 5}),
    "Titanium White": ((255, 255, 255), {"Y": 0, "R": 0, "B": 0, "W": 100}),
    "Burnt Sienna": ((138, 54, 15), {"Y": 10, "R": 60, "B": 10, "W": 20}),
    "Turquoise": ((64, 224, 208), {"Y": 10, "R": 0, "B": 60, "W": 30}),
    "Violet": ((127, 0, 255), {"Y": 0, "R": 30, "B": 60, "W": 10}),
    "Orange": ((255, 165, 0), {"Y": 50, "R": 40, "B": 0, "W": 10}),
    "Pink": ((255, 182, 193), {"Y": 0, "R": 30, "B": 0, "W": 70}),
    "Olive Green": ((128, 128, 0), {"Y": 40, "R": 10, "B": 10, "W": 40}),
    "Payne's Grey": ((83, 104, 120), {"Y": 0, "R": 0, "B": 60, "W": 40}),
    "Raw Umber": ((115, 74, 18), {"Y": 20, "R": 50, "B": 10, "W": 20})
}

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img = Image.open(uploaded_file).convert("RGB")
    img_small = img.resize((200, 200))
    pixels = np.array(img_small).reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    results = []
    for i, color in enumerate(dominant_colors):
        best_match = None
        best_score = float("inf")
        for name, (ref_rgb, mix) in reference_colors.items():
            score = mean_squared_error(color, ref_rgb)
            if score < best_score:
                best_score = score
                best_match = (name, ref_rgb, mix)
        results.append((f"C{i+1}", color, *best_match))

    rows = []
    for code, rgb, name, ref_rgb, mix in results:
        mix_str = " ".join([f"{k}:{v}%" for k, v in mix.items()])
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        rows.append([code, name, hex_color, mix_str])

    df = pd.DataFrame(rows, columns=["Code", "Closest Color", "Hex", "Mix Ratio"])
    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
        colLoc="center",
        colColours=["#f5f5f5"] * 4
    )
    for i, row in df.iterrows():
        table[(i + 1, 2)].set_facecolor(row["Hex"])
        table[(i + 1, 2)].get_text().set_text("")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    output_path = "tut_mix_table.png"
    plt.savefig(output_path, bbox_inches="tight", format="png")
    with open(output_path, "rb") as f:
        st.download_button("Download Table as Image", f, "tut_mix_table.png")
