import streamlit as st
import pandas as pd
import numpy as np

# --- RAL Color Data (Sample) ---
# IMPORTANT: This is a SMALL SAMPLE dataset with APPROXIMATE CIELAB values.
# For a real application, you need a complete and verified dataset for RAL Classic.
# CIELAB values can vary based on illuminant, observer, and source.
# You might find CSV files online or need to compile one.
# Example structure: {'RAL': code, 'Name': name, 'L': L_val, 'a': a_val, 'b': b_val, 'Hex': hex_code}
# Hex is optional but useful for display.

RAL_DATA = [
    {'RAL': '1000', 'Name': 'Green beige', 'L': 80.96, 'a': -0.18, 'b': 13.49, 'Hex': '#CCC58F'},
    {'RAL': '1001', 'Name': 'Beige', 'L': 78.23, 'a': 2.98, 'b': 17.78, 'Hex': '#C7B89B'},
    {'RAL': '1002', 'Name': 'Sand yellow', 'L': 76.86, 'a': 4.18, 'b': 26.19, 'Hex': '#C6AA76'},
    {'RAL': '1013', 'Name': 'Oyster white', 'L': 89.61, 'a': -0.51, 'b': 9.49, 'Hex': '#EAE6CA'},
    {'RAL': '1014', 'Name': 'Ivory', 'L': 85.47, 'a': 2.17, 'b': 19.17, 'Hex': '#DFCEA1'},
    {'RAL': '1015', 'Name': 'Light ivory', 'L': 88.01, 'a': 1.99, 'b': 14.17, 'Hex': '#EADEBD'},
    {'RAL': '3000', 'Name': 'Flame red', 'L': 45.78, 'a': 48.89, 'b': 31.31, 'Hex': '#A1232A'},
    {'RAL': '3001', 'Name': 'Signal red', 'L': 42.03, 'a': 49.07, 'b': 26.54, 'Hex': '#912429'},
    {'RAL': '3020', 'Name': 'Traffic red', 'L': 48.24, 'a': 58.02, 'b': 39.88, 'Hex': '#BF1A21'},
    {'RAL': '5002', 'Name': 'Ultramarine blue', 'L': 28.11, 'a': 10.91, 'b': -41.06, 'Hex': '#1A346B'},
    {'RAL': '5010', 'Name': 'Gentian blue', 'L': 33.78, 'a': 0.49, 'b': -36.89, 'Hex': '#004B80'},
    {'RAL': '5015', 'Name': 'Sky blue', 'L': 55.07, 'a': -11.95, 'b': -30.01, 'Hex': '#0075B4'},
    {'RAL': '6005', 'Name': 'Moss green', 'L': 29.87, 'a': -18.66, 'b': 11.14, 'Hex': '#0F4336'},
    {'RAL': '6018', 'Name': 'Yellow green', 'L': 63.99, 'a': -32.77, 'b': 40.86, 'Hex': '#53A050'},
    {'RAL': '7016', 'Name': 'Anthracite grey', 'L': 33.32, 'a': -1.53, 'b': -2.74, 'Hex': '#38414A'},
    {'RAL': '7035', 'Name': 'Light grey', 'L': 79.65, 'a': -0.78, 'b': 0.95, 'Hex': '#CBD0CC'},
    {'RAL': '7037', 'Name': 'Dusty grey', 'L': 54.29, 'a': -0.89, 'b': -1.92, 'Hex': '#7F8484'},
    {'RAL': '9005', 'Name': 'Jet black', 'L': 18.04, 'a': 0.17, 'b': -1.05, 'Hex': '#0A0A0D'},
    {'RAL': '9010', 'Name': 'Pure white', 'L': 95.08, 'a': -0.38, 'b': 2.53, 'Hex': '#F5F6F0'},
    {'RAL': '9016', 'Name': 'Traffic white', 'L': 95.12, 'a': -0.59, 'b': 1.38, 'Hex': '#F6F8F4'},
]

# Convert to Pandas DataFrame for easier handling
ral_df = pd.DataFrame(RAL_DATA)

# --- Helper Function ---
def calculate_delta_e_76(lab1, lab2):
    """Calculates the simple Euclidean distance (Delta E 76) between two CIELAB colors."""
    return np.sqrt(np.sum((np.array(lab1) - np.array(lab2))**2))

def find_closest_ral(input_lab, df):
    """Finds the RAL color in the DataFrame closest to the input CIELAB values."""
    if df.empty:
        return None, float('inf')

    min_delta_e = float('inf')
    best_match_index = -1

    # Extract L, a, b columns from the DataFrame as a NumPy array for efficiency
    ral_lab_values = df[['L', 'a', 'b']].values
    input_lab_array = np.array(input_lab)

    # Calculate Delta E for all RAL colors simultaneously using NumPy broadcasting
    delta_es = np.sqrt(np.sum((ral_lab_values - input_lab_array)**2, axis=1))

    # Find the index of the minimum Delta E
    min_delta_e_index = np.argmin(delta_es)
    min_delta_e = delta_es[min_delta_e_index]

    # Get the corresponding RAL data
    closest_ral_data = df.iloc[min_delta_e_index]

    return closest_ral_data, min_delta_e

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("RAL Color Finder from CIELAB")
st.markdown("""
Enter CIELAB coordinates (L*, a*, b*) to find the closest **RAL Classic** color
from the built-in **sample dataset**.

**Disclaimer:** The CIELAB values in the dataset are approximate and may vary based
on measurement conditions (illuminant, observer angle) and specific material.
This tool provides the *nearest match* based on the simple Delta E 76 calculation
and the limited sample data provided.
""")

st.sidebar.header("Input CIELAB Values")

# Input fields in the sidebar
l_input = st.sidebar.number_input("L* (0-100)", min_value=0.0, max_value=100.0, value=80.0, step=0.1, format="%.2f")
a_input = st.sidebar.number_input("a* (-128 to 127)", min_value=-128.0, max_value=127.0, value=0.0, step=0.1, format="%.2f")
b_input = st.sidebar.number_input("b* (-128 to 127)", min_value=-128.0, max_value=127.0, value=15.0, step=0.1, format="%.2f")

input_color_lab = (l_input, a_input, b_input)

st.sidebar.markdown("---")
find_button = st.sidebar.button("Find Closest RAL Color")

# Display Area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Input Color (CIELAB)")
    st.metric("L*", f"{input_color_lab[0]:.2f}")
    st.metric("a*", f"{input_color_lab[1]:.2f}")
    st.metric("b*", f"{input_color_lab[2]:.2f}")
    # Simple visualization of input color is hard without conversion to RGB/Hex

with col2:
    st.subheader("Closest RAL Match")
    result_placeholder = st.empty() # Use a placeholder to update results

if find_button:
    closest_ral, delta_e = find_closest_ral(input_color_lab, ral_df)

    if closest_ral is not None:
        ral_code = closest_ral['RAL']
        ral_name = closest_ral['Name']
        ral_lab = (closest_ral['L'], closest_ral['a'], closest_ral['b'])
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
                    <div style="width:100px; height:50px; background-color:{ral_hex}; border: 1px solid #ccc; margin-bottom: 10px;">(Hex not in sample data)</div>
                    """, unsafe_allow_html=True)


            st.text(f"CIELAB: L*={ral_lab[0]:.2f}, a*={ral_lab[1]:.2f}, b*={ral_lab[2]:.2f}")
            if ral_hex and ral_hex != '#FFFFFF':
                st.text(f"Approx. Hex: {ral_hex}")

    else:
        with result_placeholder.container():
            st.error("Could not find a match. Is the RAL data loaded correctly?")

# Optional: Display the sample dataset used
st.markdown("---")
with st.expander("Show Sample RAL Data Used"):
    st.dataframe(ral_df)
