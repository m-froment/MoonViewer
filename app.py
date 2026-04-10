"""
Moon Viewer: Shape of the Moon with date and time from an observer
"""
import streamlit as st

### To use stpyvista
# import multiprocessing as mp
# mp.set_start_method("fork", force=True)

# from moon3d
import pylunar  
import moon3d

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
st.title("Visibility of the Moon")


# plot_tab, doc_tab = st.tabs(["📈 tab1", "📖 tab2"])
col_date, col_phase = st.columns(2, width=800)
with col_date:
    datetime = st.datetime_input(label="Date and time of observation (UTC)",
                value="now",
                format="DD/MM/YYYY",
                width=220,
                )
with col_phase:
    mi = pylunar.MoonInfo((0,0,0), (0,0,0))
    mi.update(datetime)
    st.markdown("##### Phase: {0}".format(mi.phase_name()))

### Content 
st.header("Position of the Moon")
st.markdown("Shows regions on the globe where the moon is visible, and its phase.")

st.header("Aspect of the Moon")
st.markdown("Shows the aspect of the Moon from a specific observer.")

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

# Create tabs for location selection and custom input
col1, col2 = st.columns([2, 1])

with col1:
    selected_preset = st.selectbox(
        "Select observer location:",
        options=list(preset_locations.keys()),
        index=0,  # Default to Calern Observatory
        key="location_preset",
        width=300,
    )

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

### Generate image
plotter = moon3d.make_3d_image()
plotter = moon3d.get_scene_png(plotter, observer_lat, observer_lon, datetime)
st.markdown("Still frame")
st.image("./moon_view.png", width=800)

st.markdown("Interactive 3D window (scroll to zoom, click to rotate, shift+click to pan)")
st.iframe("moon_view.html", height=600, width=800)

### Attempt at using stpyvista to render vtk window in streamlit
### Unfortunately, the camera zoom stil doesn't work 
# from stpyvista import stpyvista
# stpyvista(plotter, width=800, horizontal_align='left')