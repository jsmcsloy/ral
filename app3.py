import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import distance
import ast

# --- RAL Color Data ---
RAL_DATA = pd.read_csv("data.csv")

# Convert to Pandas DataFrame for easier handling
ral_df = pd.DataFrame(RAL_DATA)

# --- Helper Function ---
def rgb_to_lab(rgb):
    """Converts RGB to CIELAB."""
    rgb = np.array(rgb) / 255.0  # Normalize RGB to [0, 1]
    
    def srgb_to_linear(c):
        if c <= 0.04045:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4

    linear_rgb = np.array([srgb_to_linear(c) for c in rgb])

    # Convert to XYZ
    xyz_matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
    xyz = np.dot(xyz_matrix, linear_rgb)

    # Normalize XYZ
    xyz_ref_white = np.array([95.047, 100.000, 108.883])  # D65
    xyz_normalized = xyz / xyz_ref_white

    def xyz_to_lab_f(t):
        if t > 0.008856:
            return t ** (1/3)
        else:
            return (7.787 * t) + (16/116)

    f_xyz = np.array([xyz_to_lab_f(t) for t in xyz_normalized])

    # Convert to Lab
    l = (116 * f_xyz[1]) - 16
    a = 500 * (f_xyz[0] - f_xyz[1])
    b = 200 * (f_xyz[1] - f_xyz[2])

    return l, a, b

def find_closest_ral(input_lab, df):
    """Finds the RAL color in the DataFrame closest to the input CIELAB values."""
    if df.empty:
        return None, float('inf')

    min_delta_e = float('inf')
    best_match_index = -1

    delta_es = []
    for index, row in df.iterrows():
        try:
            ral_rgb = ast.literal_eval(row['RGB']) # Convert string to tuple
            ral_lab = rgb_to_lab(ral_rgb)
            delta_e = distance.euclidean(input_lab, ral_lab)
            delta_es.append(delta_e)
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Error processing RGB value: {row['RGB']}. Error: {e}")
            continue #skip to the next loop iteration.

    min_delta_e_index = np.argmin(delta_es)
    min_delta_e = delta_es[min_delta_e_index]
    closest_ral_data = df.iloc[min_delta_e_index]

    return closest_ral_data, min_delta_e

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("RAL Color Finder from RGB")
st.markdown("""
Enter RGB values to find the closest **RAL Classic** color.
""")

st.sidebar.header("Input RGB Values")

# Input fields in the sidebar
r_input = st.sidebar.number_input("Red (0-255)", min_value=0, max_value=255, value=200, step=1)
g_input = st.sidebar.number_input("Green (0-255)", min_value=0, max_value=255, value=100, step=1)
b_input = st.sidebar.number_input("Blue (0-255)", min_value=0, max_value=255, value=50, step=1)

input_color_rgb = (r_input, g_input, b_input)
input_color_lab = rgb_to_lab(input_color_rgb) #convert input to lab

st.sidebar.markdown("---")
find_button = st.sidebar.button("Find Closest RAL Color")

# Display Area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Input Color (RGB)")
    st.metric("Red", f"{input_color_rgb[0]}")
    st.metric("Green", f"{input_color_rgb[1]}")
    st.metric("Blue", f"{input_color_rgb[2]}")

with col2:
    st.subheader("Closest RAL Match")
    result_placeholder = st.empty() # Use a placeholder to update results

if find_button:
    closest_ral, delta_e = find_closest_ral(input_color_lab, ral_df)

    if closest_ral is not None:
        ral_code = closest_ral['RAL']
        ral_name = closest_ral['Name']
        ral_rgb = ast.literal_eval(closest_ral['RGB'])
        ral_lab = rgb_to_lab(ral_rgb)
        ral_hex = closest_ral.get('Hex', '#FFFFFF') # Default to white if no Hex

        with result_placeholder.container():
            st.success(f"Found closest match: **RAL {ral_code} ({ral_name})**")
            st.metric("Calculated Delta E (ΔE*₇₆)", f"{delta_e:.2f}")
            st.markdown("---")
            st.markdown(f"**RAL {ral_code} Details:**")

            # Display color swatch if Hex is available
            if ral_hex and ral_hex != '#FFFFFF':
                 st.markdown(f"""
                    <div style="width:100px; height:50px; background-color:{ral_hex}; border: 1px solid #ccc; margin-bottom: 10px;"></div>
                    """, unsafe_allow_html=True)
            elif ral_hex:
                 st.markdown(f"""
                    <div style="width:100px; height:50px; background-color:{ral_hex}; border: 1px solid #ccc; margin-bottom: 10px;">(Hex not in data)</div>
                    """, unsafe_allow_html=True)


            st.text(f"CIELAB: L*={ral_lab[0]:.2f}, a*={ral_lab[1]:.2f}, b*={ral_lab[2]:.2f}")
            if ral_hex and ral_hex != '#FFFFFF':
                st.text(f"Approx. Hex: {ral_hex}")

    else:
        with result_placeholder.container():
            st.error("Could not find a match. Is the RAL data loaded correctly?")