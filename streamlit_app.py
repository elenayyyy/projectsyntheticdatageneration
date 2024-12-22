import streamlit as st
import pandas as pd


# App Title
st.title('🍪ML Generation')

st.write("""Machine Learning Project""")

df = pd.read_csv('https://raw.githubusercontent.com/elenayyyy/data/refs/heads/master/food_ingredients_and_allergens.csv')
df
