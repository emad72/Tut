import streamlit as st import matplotlib.pyplot as plt import matplotlib.patches as patches import numpy as np from sklearn.cluster import KMeans from PIL import Image import os from colormath.color_objects import sRGBColor, LabColor from colormath.color_conversions import convert_color from colormath.color_diff import delta_e_cie2000

Color reference palette with assumed RGB values

REFERENCE_COLORS = { "Yellow": (255, 255, 0), "Red": (255, 0, 0), "Blue": (0, 0, 255), "White": (255, 255, 255) }

Load and resize image

def load_image(image_file): img = Image.open(image_file) img = img.convert('RGB') img = img.resize((300, 300)) return np.array(img)

Convert RGB to LAB for better perceptual difference comparison

def rgb_to_lab(color): rgb = sRGBColor(color[0]/255, color[1]/255, color[2]/255) return convert_color(rgb, LabColor)

Get mix percentages based on closest CIEDE2000 distance

def get_mix_percentages(target_rgb, reference_rgbs): target_lab = rgb_to_lab(target_rgb) distances = [] for ref_name, ref_rgb in reference_rgbs.items(): lab = rgb_to_lab(ref_rgb) delta_e = delta_e_cie2000(target_lab, lab) distances.append((ref_name, float(delta_e)))

distances.sort(key=lambda x: x[1])
top_matches = distances[:2 if distances[0][1] < 10 else 3]

inv_distances = [1 / (d + 1e-6) for _, d in top_matches]
total = sum(inv_distances)
percentages = [round((v / total) * 100) for v in inv_distances]
correction = 100 - sum(percentages)
percentages[0] += correction

result = dict(zip([x[0] for x in top_matches], percentages))
return result

Display result table

def display_results(colors, mix_results): fig, ax = plt.subplots(figsize=(10, len(colors) * 0.6)) ax.axis('off')

for i, (color, mix) in enumerate(zip(colors, mix_results)):
    y = len(colors) - i - 1
    ax.add_patch(patches.Rectangle((0, y), 1, 1, color=np.array(color)/255))
    label = f"  {i+1:02d}  "
    mix_str = '  |  '.join([f"{k}: {v}%" for k, v in mix.items()])
    ax.text(1.1, y + 0.5, label + mix_str, va='center', fontsize=10, family='monospace')

output_path = "color_mix_result.png"
plt.savefig(output_path, bbox_inches="tight", format="png")
return output_path

Streamlit UI

st.set_page_config(page_title="Tut Analyzer", layout="centered") st.title("🎨 Tut Color Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) num_colors = st.slider("Number of tones to extract", 2, 48, 10)

if uploaded_file: img_array = load_image(uploaded_file) st.image(img_array, caption="Uploaded Image", use_container_width=True)

img_2d = img_array.reshape((-1, 3))
kmeans = KMeans(n_clusters=num_colors, n_init=10)
kmeans.fit(img_2d)
dominant_colors = np.round(kmeans.cluster_centers_).astype(int)

mix_results = [get_mix_percentages(color, REFERENCE_COLORS) for color in dominant_colors]
result_path = display_results(dominant_colors, mix_results)

st.image(result_path, caption="Mixing Guide", use_container_width=True)
with open(result_path, "rb") as f:
    st.download_button("Download Result", f, file_name="Tut_Mix_Guide.png")

