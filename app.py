"""
Moon Viewer: Shape of the Moon with date and time from an observer
"""
import streamlit as st

### To use stpyvista
# import multiprocessing as mp
# mp.set_start_method("fork", force=True)

# from moon3d
import pylunar  
import numpy as np
import moon3d
# from stpyvista import stpyvista
# import pyvista as pv 
# import multiprocessing as mp
# mp.set_start_method(method="fork", force=True)  # "spawn" works fine

from ephems import plot_moonmap,plot_moonmap2,shackleton_visibility
import streamlit.components.v1 as components

### Define preset observer locations
preset_locations = {
    "Calern Observatory (France)": ((43, 44, 33), (6, 54, 1)),
    "Hawaii (Mauna Kea)": ((19, 49, 31), (-155, 28, 26)),
    "JPL Table Mountain Observatory": ((34, 22, 55.2), (-117, 40, 54.5)),
    "North Pole": ((90, 0, 0), (0, 0, 0)),
    "South Pole": ((-90, 0, 0), (0, 0, 0)),
    "Equator (Prime Meridian)": ((0, 0, 0), (0, 0, 0)),
    "Custom Location": None,
}

### Page configuration
st.set_page_config(
    page_title="Visibility of the Moon",
    page_icon="🌒",
    layout="wide",
    # initial_sidebar_state="expanded"
    # menu_items={
    #         "Moon View",
    #         "About",
    #         }
)

### Title 
st.title("Lunar forecast ㊊")

plot_tab, doc_tab = st.tabs(["🌘 Interface", "🌖 Infos"])

with plot_tab:
    col_date, col_loc = st.columns(2, width=600)
    with st.sidebar:
        st.header("Observation parameters")
        st.markdown("Choose at what time and from which location the observations are performed.")
        datetime = st.datetime_input(label="Date/time (UTC)",
                    value="now",
                    format="DD/MM/YYYY",
                    width=220)
        
        selected_preset = st.selectbox(
            "Select observer location:",
            options=list(preset_locations.keys()),
            index=0,  # Default to Calern Observatory
            key="location_preset",
            width=300,
        )
        
        forecast_days=st.slider("Days to forecast", 1, 365, 30)
        
        # Handle location selection
        if selected_preset == "Custom Location":
            st.markdown("Enter your custom location:")
            col_lat_d, col_lat_m, col_lat_s = st.columns(3, width=800)
            with col_lat_d:
                lat_deg = st.number_input("Latitude (degrees)", value=43, key="lat_deg", step=1)
            with col_lat_m:
                lat_min = st.number_input("Latitude (minutes)", value=44, key="lat_min", step=1, min_value=0, max_value=59)
            with col_lat_s:
                lat_sec = st.number_input("Latitude (seconds)", value=33.0, key="lat_sec", step=0.5, min_value=0.0, max_value=59.9)
            observer_lat = (lat_deg, lat_min, lat_sec)
            col_lon_d, col_lon_m, col_lon_s = st.columns(3, width=800)
            with col_lon_d:
                lon_deg = st.number_input("Longitude (degrees)", value=6, key="lon_deg", step=1)
            with col_lon_m:
                lon_min = st.number_input("Longitude (minutes)", value=54, key="lon_min", step=1, min_value=0, max_value=59)
            with col_lon_s:
                lon_sec = st.number_input("Longitude (seconds)", value=1.0, key="lon_sec", step=0.5, min_value=0.0, max_value=59.9)
            
            observer_lon = (lon_deg, lon_min, lon_sec)
        else:
            observer_lat, observer_lon = preset_locations[selected_preset] 
    
    # %%
    st.header("Moon State")
    colmap, colmoon = st.columns([2,0.83],gap="large")
    
    with colmap:
        ### Content 
        fig=plot_moonmap(datetime,[observer_lat,observer_lon])
        st.pyplot(fig=fig,width="content",clear_figure=True, use_container_width=True)
    with colmoon:
        # st.markdown("Shows the aspect of the Moon from a specific observer.")
        ### Generate image
        plotter = moon3d.make_3d_image()
        plotter = moon3d.get_scene_png(plotter, observer_lat, observer_lon, datetime)
        st.image("./moon_view.png", use_container_width=True)
    
    figshack,fig2=shackleton_visibility(datetime,forecast_days,[observer_lat,observer_lon])
    st.divider()
    colshak, colplots = st.columns(2,gap="large")
    st.header("Shackleton Visibility Forecast")
    colt, col, _ = st.columns([1.5, 2, 0.6])
    with colt: 
        st.markdown("<br><br>" \
                    "Shackleton is represented by the green point <span style='color: #55ff99; font-size:13px;'>&#x2B24;</span>. " \
                    "When it is in the right section of the plot (apparent latitude > -90°), it is visible from the Earth. "\
                    "The colored line indicates its position from 5 days before the current date, to the chosen number of days after the current date, `Days to forecast`.",
                    text_alignment = "justify", unsafe_allow_html=True)
    with col:
        st.pyplot(figshack, clear_figure=True)
    
    
    # with colplots:
    st.pyplot(fig2,width="content",clear_figure=True)
    # st.markdown("Interactive 3D window (scroll to zoom, click to rotate, shift+click to pan)")
    # # st.iframe("moon_view.html", height=600, width=800)
    
st.text("ⓒ 2026 - Marouchka Froment & Pierre-Yves Froissart \n Institut de Physique du Globe de Paris",text_alignment="center",width="stretch")
    
    # with open("moon_view.html", "r") as f:
    #     html_content = f.read()
    
    # components.html(html_content, height=600, scrolling=False)
    
    
    # ## Attempt at using stpyvista to render vtk window in streamlit
    # ## Unfortunately, the camera zoom stil doesn't work 
    # from stpyvista import stpyvista
    # stpyvista(plotter, width=800, horizontal_align='left')

