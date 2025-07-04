
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# قاعدة بيانات ألوان محددة جاهزًا بالـ Y/R/B/W
reference_colors = {
    "Cadmium Yellow": (255, 246, 0),
    "Cadmium Red": (227, 0, 34),
    "Ultramarine Blue": (18, 10, 143),
    "Titanium White": (255, 255, 255),
    "Ivory Black": (20, 20, 20)
}

mix_bases = ["Cadmium Yellow", "Cadmium Red", "Ultramarine Blue", "Titanium White"]

def rgb_to_lab(color):
    return convert_color(sRGBColor(*[v / 255.0 for v in color], is_upscaled=False), LabColor)

def get_mix_percentages(target_rgb, ref_colors):
    target_lab = rgb_to_lab(target_rgb)
    distances = []
    for name in mix_bases:
        lab = rgb_to_lab(ref_colors[name])
        d = delta_e_cie2000(target_lab, lab)
        distances.append((name, d))
    distances.sort(key=lambda x: x[1])

    # نقرب الخليط باستخدام مسافات الأقرب
    inv_d = [1 / max(d, 1e-6) for _, d in distances[:4]]
    total = sum(inv_d)
    percentages = [round((v / total) * 100) for v in inv_d]

    result = {mix_bases[i]: percentages[i] for i in range(len(percentages))}
    return result

def extract_colors(image, num_colors):
    image = image.resize((200, 200))
    data = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(data)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def draw_color_table(colors, mix_results):
    fig, ax = plt.subplots(figsize=(10, len(colors) * 0.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, len(colors))
    ax.axis("off")

    for i, (color, mix) in enumerate(zip(colors, mix_results)):
        rect = patches.Rectangle((0, i), 10, 1, facecolor=np.array(color) / 255.0)
        ax.add_patch(rect)
        text = f"RGB: {tuple(color)}"
        ax.text(12, i + 0.35, text, va='center', fontsize=9)

        mix_text = " | ".join([f"{k}: {v}%" for k, v in mix.items()])
        ax.text(30, i + 0.35, mix_text, va='center', fontsize=9)

    return fig

# واجهة Streamlit
st.title("🎨 Tut Color Analyzer")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    num_tones = st.slider("Number of tones to extract", 2, 48, 12)
    colors = extract_colors(image, num_tones)

    mix_results = [get_mix_percentages(color, reference_colors) for color in colors]

    fig = draw_color_table(colors, mix_results)
    st.pyplot(fig)
