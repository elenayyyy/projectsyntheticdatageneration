import streamlit as st
import pandas as pd


# App Title
st.title('🍪ML Generation')

st.write("""Machine Learning Project""")

df = pd.read_csv('https://raw.githubusercontent.com/elenayyyy/dataset/refs/heads/main/food_ingredients_and_allergens.csv?token=GHSAT0AAAAAAC4H5Y3DHWSB26PVOGUL3BHUZ3IBUUQ'),
df
