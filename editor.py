# editor.py - streamlit editor thing
# took forever to get working lw

import streamlit as st
import pandas as pd
import numpy as np

def create_editor(defaults=None):
    if defaults is None:
        defaults = [
            {'Name': 'Sun', 'Mass': 333000.0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'Color': '#FFD700'},
            {'Name': 'Earth', 'Mass': 1.0, 'x': 1.0, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 6.28, 'vz': 0.0, 'Color': '#0000FF'},
        ]
    
    df = pd.DataFrame(defaults)
    
    st.markdown("### Body Editor")
    st.markdown("Edit stuff, right click to add rows idk")
    
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "Color": st.column_config.ColorColumn("Color"),
            # tried adding validation, gave up lol
        },
        key="body_editor"
    )
    
    return edited

def dynamics_from_df(df):
    """df -> arrays"""
    pos = df[['x', 'y', 'z']].values.astype(float)
    vel = df[['vx', 'vy', 'vz']].values.astype(float)
    mass = df['Mass'].values.astype(float)
    names = df['Name'].tolist()
    colors = df['Color'].tolist()
    return pos, vel, mass, names, colors
