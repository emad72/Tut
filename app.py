import streamlit as st import matplotlib.pyplot as plt import matplotlib.patches as patches import numpy as np from sklearn.cluster import KMeans from PIL import Image import os from colormath.color_objects import sRGBColor, LabColor from colormath.color_conversions import convert_color from colormath.color_diff import delta_e_cie2000

----------------- إعداد قائمة الألوان المرجعية ------------------- 

PALETTE = { "Cadmium Yellow": [255, 246, 0], "Cadmium Red": [255, 38, 0], "Ultramarine Blue": [18, 10, 143], "Titanium White": [255, 255, 255], "Raw Umber": [110, 75, 38], "Yellow Ochre": [225, 173, 1], "Alizarin Crimson": [149, 0, 28], "Cobalt Blue": [0, 71, 171], "Viridian": [64, 130, 109], "Burnt Sienna": [138, 54, 15], "Ivory Black": [41, 36, 33], }

----------------- دالة التحويل إلى LAB ------------------- 

def rgb_to_lab(rgb): srgb = sRGBColor(rgb[0]/255, rgb[1]/255, rgb[2]/255) return convert_color(srgb, LabColor)

----------------- دالة حساب النسب بناءً على CIEDE2000 ------------------- 

def get_mix_ratios(input_rgb): input_lab = rgb_to_lab(input_rgb) similarities = [] for name, rgb in PALETTE.items(): color_lab = rgb_to_lab(rgb) distance = delta_e_cie2000(input_lab, color_lab) similarities.append((name, 1 / (distance + 1e-5))) # أضف قيمة صغيرة لتجنب القسمة على صفر

total_similarity = sum([s for _, s in similarities]) ratios = [(name, round(sim / total_similarity * 100)) for name, sim in similarities] return sorted(ratios, key=lambda x: -x[1])[:4] # أعلى 4 فقط ----------------- كود التطبيق ------------------- 

st.set_page_config(page_title="🎨 Tut Analyzer", layout="centered") st.title("🎨 Tut Color Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file: image = Image.open(uploaded_file).convert("RGB") st.image(image, caption="Uploaded Image", use_column_width=True)

num_colors = st.slider("Number of tones to extract", 2, 48, 12) img_array = np.array(image) img_array = img_array.reshape(-1, 3) kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(img_array) centers = np.round(kmeans.cluster_centers_).astype(int) # جدول النتائج fig, ax = plt.subplots(figsize=(10, num_colors * 0.6)) ax.axis("off") for i, color in enumerate(centers): y_pos = num_colors - i - 1 hex_color = '#%02x%02x%02x' % tuple(color) ratios = get_mix_ratios(color) # شكل اللون ax.add_patch(patches.Rectangle((0, y_pos), 1, 1, color=hex_color)) # الاسم و النسب ax.text(1.2, y_pos + 0.3, hex_color.upper(), fontsize=10, fontweight='bold') for j, (name, percent) in enumerate(ratios): ax.text(3 + j * 2.8, y_pos + 0.3, f"{name}: {percent}%", fontsize=10) plt.xlim(0, 15) plt.ylim(0, num_colors) st.pyplot(fig) 
