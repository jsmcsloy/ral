import streamlit as st
import pandas as pd
import numpy as np
import io

# --- RAL Color Data (Sample) ---
# IMPORTANT: Use the full and accurate RAL Classic dataset for a real application.
# The example data provided is for demonstration only.
# Load data from CSV.
file_name = "data.csv"  #  <- Change this if your filename is different

try:
    # Try reading the CSV with different encodings and options
    RAL_DATA = pd.read_csv(file_name, encoding='utf-8')
except FileNotFoundError:
    st.error(f"Error: '{file_name}' not found.  Make sure the file is in the same directory, or update the filename.")
    st.stop()
except Exception as e:
    st.error(f"Error reading CSV file '{file_name}': {e}")
    st.stop()

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

    # Use a dictionary to map common column names to the expected L, a, b
    lab_columns = {
        'L': ['L', 'L*', 'Lightness'],
        'a': ['a', 'a*'],
        'b': ['b', 'b*'],
    }

    # Find the actual column names in the DataFrame that match our expected names
    actual_lab_columns = {}
    for lab_key, possible_names in lab_columns.items():
        for col in possible_names:
            if col in df.columns:
                actual_lab_columns[lab_key] = col
                break  # Stop searching once a match is found
        if lab_key not in actual_lab_columns:
            st.error(f"Error: The input CSV file '{file_name}' is missing the required column for '{lab_key}'.  Checked for: {', '.join(possible_names)}.  Please check the CSV file and ensure the column names are correct.")
            st.stop()

    # Extract L, a, b columns from the DataFrame as a NumPy array for efficiency
    ral_lab_values = df[[actual_lab_columns['L'], actual_lab_columns['a'], actual_lab_columns['b']]].values
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
from the built-in dataset.

**Disclaimer:** The CIELAB values in the dataset are approximate and may vary based
on measurement conditions (illuminant, observer angle) and specific material.
This tool provides the *nearest match* based on the simple Delta E 76 calculation.
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
        ral_name = closest_ral.get('Name')
        ral_lab = (closest_ral['L'], closest_ral['a'], closest_ral['b'])
        ral_hex = closest_ral.get('Hex', '#FFFFFF')

        with result_placeholder.container():
            st.success(f"Found closest match: RAL {ral_code} ")
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
