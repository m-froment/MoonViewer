import pylunar 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import matplotlib
import numpy as np
import pyvista as pv
from datetime import datetime, timedelta
import time
import rasterio
import xarray as xr
from scipy import interpolate
import time as ptime 
import ephem
from PIL import Image
import sys
import gc
import streamlit as st 
# import multiprocessing as mp
# mp.set_start_method(method="fork", force=True)  # "spawn" works fine
# Set font to Helvetica or Arial
# matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
# matplotlib.rcParams['font.family'] = 'sans-serif'
pv.OFF_SCREEN = True
pv.global_theme.font.family = 'courier'
pv.global_theme.font.size = 12

### =======================================================================================================
### Adds Shackleton to pylunar's set of features. 
### =======================================================================================================
@st.cache_data
def load_features():
    ### Total of all features available in pylunar 
    lc = pylunar.LunarFeatureContainer("Lunar")

    ### load only the visible ones: 
    shackleton = pylunar.LunarFeature(
        name="Shackleton",
        diameter=21.0,
        latitude=-89.67,
        longitude=129.78,
        delta_latitude=1.0,
        delta_longitude=1.0,
        feature_type="Crater",
        quad_name="South Polar",
        quad_code="SP",
        code_name="Both",
        lunar_club_type=None,
    )

    ### Load only the features visible from observer
    lc.load(mi)

    ### Add Shackleton manually to the in-memory feature container
    lc.features[id(shackleton)] = shackleton
    return(shackleton, lc)


### =======================================================================================================
### Tests some calculations of pylunar
### =======================================================================================================
def test_visibility_time(observer_lat, observer_lon, start_date, days=60, step_hours=1):
    """Test Shackleton visibility for a series of dates starting at start_date.

    Returns a list of 0/1 values for the next `days` days.
    """

    mi_test = pylunar.MoonInfo(observer_lat, observer_lon)

    current = datetime(*start_date)
    visibility = []
    date_list = []
    earth_moon_distances = []
    libration_lat = []
    libration_lon = []
    subsolar_colon = []

    for _ in range(int(days * 24/step_hours)):
        mi_test.update((current.year, current.month, current.day,
                        current.hour, current.minute, current.second))
        visible = mi_test.is_visible(shackleton)
        visibility.append(1 if visible else 0)
        date_list.append(current)
        earth_moon_distances.append(mi_test.earth_distance())
        libration_lat.append(mi_test.libration_lat())
        libration_lon.append(mi_test.libration_lon())
        subsolar_colon.append(mi_test.colong())
        current += timedelta(hours=step_hours)
        
    #######################################
    fig, (ax1, ax2,ax3) = plt.subplots(3,1)
    ax1.fill_between(date_list, visibility, linestyle="-", color="orangered", alpha=0.3)
    ax2.plot(date_list, earth_moon_distances, c="k")
    ax2b = ax2.twinx()
    ax2b.plot(date_list, libration_lat, c="b", label="Lat")
    ax2b.plot(date_list, libration_lon, c="r", label="Lon")
    ###
    ax1.set_title("Shackleton visibility over {} days".format(days))
    ax2.set_xlabel("Days since {}".format(datetime(*start_date).strftime("%d/%m/%Y")))
    ax1.set_ylabel("Visible (1) / Not visible (0)")
    # ax.set_yticks([0, 1])
    #ax.set_ylim(-0.1, 1.1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax1.tick_params(axis='x', labelrotation=45)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax2.tick_params(axis='x', labelrotation=45)
    ax2b.legend()
    ###
    ax2.set_ylabel("Earth-Moon distance [km]")
    ax2b.set_ylabel("Libration latitude/longitude [°]", color="b")
    ###
    ax3.plot(date_list, subsolar_colon, c="r", label="morning")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    # ax.grid(True)
    fig.tight_layout()

    return visibility


### =======================================================================================================
### Loads elevation model of the Moon (heavy file)
### =======================================================================================================
@st.cache_data
def load_lunar_dem(dem_path='data/moon_relief_06m_g.grd', image_path="data/lroc_color_16bit_srgb_4k.tif",
                   scale_factor=1, res=800):
    """Load LOLA DEM and create a displaced PyVista sphere mesh.
    
    Parameters:
    - dem_path: path to the DEM file
    - image_path: path to the texture image file (TIFF, PNG, JPG, etc.)
    - scale_factor: scale the elevation (LOLA is in meters, scale to km or appropriate units)
    - res: resolution of the sphere mesh
    
    Returns:
    - pv.PolyData: the displaced sphere mesh
    """
    radius = 1737.4e3 

    # Load texture image using PIL (more robust than PyVista's TIFF reader)
    tex_img = Image.open(image_path)
    # Convert to RGB if needed (remove alpha channel)
    if tex_img.mode == 'RGBA':
        tex_img = tex_img.convert('RGB')
    elif tex_img.mode not in ['RGB', 'L']:
        tex_img = tex_img.convert('RGB')
    
    # Convert PIL image to numpy array
    tex_array = np.array(tex_img)
    tex_array = np.transpose(tex_array, (1, 0, 2))

    ### Load lon, lat 
    lon_im = np.linspace(-180,180,tex_img.width)
    lat_im = np.linspace(90,-90,tex_img.height)

    interp_im = interpolate.RegularGridInterpolator((lon_im, lat_im), tex_array,
                                                 method="linear")

    ### Load GRD file using xarray
    ds = xr.open_dataset(dem_path)
    dem_data = ds['z'].values  
    lat = ds['lat'].values
    lon = ds['lon'].values
    ### Artificially removes elevation for tests 
    # dem_data*=0

    ### Create interpolator for DEM
    interp = interpolate.RegularGridInterpolator((lon, lat), dem_data.T,
                                                 method="linear")

    ### Create PyVista sphere
    sphere = pv.Sphere(radius=radius, 
                       theta_resolution=res, 
                       phi_resolution=res*2)

    ### Displace vertices based on DEM
    points = sphere.points.copy()
    ### Convert points to spherical coords 
    r_p = np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2)
    lat_p = np.arcsin(points[:,2] / r_p) * 180 / np.pi
    lon_p = np.arctan2(points[:,1], points[:,0]) * 180 / np.pi
    ### Interpolate elevation
    elevations = interp((lon_p, lat_p)) * scale_factor

    ### Recalculate 
    x_new = (r_p+elevations) * np.cos(np.deg2rad(lon_p)) * np.cos(np.deg2rad(lat_p))
    y_new = (r_p+elevations) * np.sin(np.deg2rad(lon_p)) * np.cos(np.deg2rad(lat_p))
    z_new = (r_p+elevations) * np.sin(np.deg2rad(lat_p))
    points = np.column_stack((x_new, y_new, z_new))
        
    sphere.points = points
    sphere['elevation'] = elevations

    interp_tex = interp_im((lon_p, lat_p))
    interp_tex = np.clip(interp_tex, 0, 255)
    interp_tex = interp_tex.astype(np.uint8)
    sphere['moon_texture'] = interp_tex
    sphere['shading'] = np.zeros(len(elevations))
    sphere.compute_normals(cell_normals=False, inplace=True)
    return sphere


### =======================================================================================================
### Build 3D model of the Moon (heavy object)
### =======================================================================================================
@st.cache_resource
def make_3d_image(dem_path='data/moon_relief_06m_g.grd', window_size=[800,600]):

    plotter = pv.Plotter(lighting=None, window_size=window_size)

    ### Load DEM
    dem_mesh=load_lunar_dem(dem_path)
    # dem_mesh.compute_normals(cell_normals=False, inplace=True)
    # normals = dem_mesh.point_data["Normals"]
    # illumination = np.clip(np.dot(normals, sun_dir), 0.0, 1.0)
    # color = np.stack([illumination, illumination, illumination], axis=1)
    # dem_mesh.point_data["shading"] = color

    ### Render the mesh 
    plotter.add_mesh(
        dem_mesh,
        # scalars='elevation',
        # cmap='gray', clim=[-5000,1000],
        scalars='moon_texture',
        rgb=True,
        lighting=True,
        diffuse=1.0, 
        specular = 0.0, 
        ambient = 0.1,   ### Adds a little bit of ambient light in the Lunar shadow, 
                         ### As if liten by the Earth reflected light. 
        scalar_bar_args={
            'position_x': 0.05,
            'position_y': 0.4,
            'width': 0.08,
            'height': 0.5,
            'title': 'Elevation (m)', 
            # 'title_font_size': 10,
            # 'label_font_size': 8,
            'vertical':True,
            'fmt':'% .4g',
        },
        # smooth_shading=True,  ### Should be off otherwise relief not visible
    )    
    ### Display Shackleton location 
    ### TODO: put back a circle. 
    # plotter.add_mesh(
    #     crater_circle,
    #     color='gold',
    #     line_width=4,
    #     lighting=False,
    #     render_lines_as_tubes=True
    # )

    return(plotter)


### =======================================================================================================
### Add sunlight and moves observer according to ephemerids
### =======================================================================================================
def update_scene(plotter, mi, start_date, lat_obs, interactive=False):
    ### lunar illumination
    ### subsolar_lat: the latitude facing the sun. 
    ### Selenographic colongitude: the longitude of morning terminator
    ### Selenographic longitude: the longitude of evening terminator.
    ###    it is the colongitude + 180 
    ### /!\ Both are counted westward !
    ### /!\ in pylunar, longitude = -colongitude !  
    ### The sun (midday) is simply at colongitude - 90° 
    subsolar_lat = np.deg2rad(mi.subsolar_lat())
    subsolar_lon = -np.deg2rad(mi.colong() - 90)

    ### To check 
    # print("Subsolar point")
    # print(np.rad2deg(subsolar_lat), "N")
    # print(np.rad2deg(subsolar_lon), "E")
    
    sun_dir = np.array([
        np.cos(subsolar_lat) * np.cos(subsolar_lon),
        np.cos(subsolar_lat) * np.sin(subsolar_lon),
        np.sin(subsolar_lat),
    ])
    ### increase sun distance 
    sun_dir = sun_dir/np.linalg.norm(sun_dir) * mi.moon.sun_distance * ephem.meters_per_au
    
    ### approximate lunar orientation from libration
    ### It is the coordinates of a vector from the center of the Moon 
    ### towards the libration vector in cartesian coordinates, 
    ### plus the paralax effect from the Earth observer 
    
    ### Distance = earth-moon distance from ephermerids, 
    ###            and paralax 
    ### Paralax angle on the moon (using km):
    earth_radius = 6471.0  
    earth_moon_distance = mi.earth_distance() * 1e3
    # denom = earth_moon_distance/earth_radius*\
    #         1/np.sin(np.deg2rad(lat_obs)) - 1/np.tan(np.deg2rad(lat_obs))
    # lat_par = 0#np.arctan2(1, denom)
    # lat_par_deg = np.rad2deg(lat_par)
    # ### todo: add lon_par from sublunar lon

    # ### Camera distance in meter 
    # camera_distance = earth_moon_distance * 1e3 * np.sin(np.deg2rad(lat_obs)) /\
    #                np.sin( np.pi - np.deg2rad(lat_obs) - lat_par )
    
    libra_lat = np.deg2rad(mi.libration_lat())# + lat_par_deg)
    libra_lon = np.deg2rad(mi.libration_lon())
    moon_view_dir = np.array([
        np.cos(libra_lat) * np.cos(libra_lon),
        np.cos(libra_lat) * np.sin(libra_lon),
        np.sin(libra_lat),
    ])
    ### To check 
    # print("Sub-Earth point")
    # print(np.rad2deg(libra_lat), "N")
    # print(np.rad2deg(libra_lon), "E")

    moon_view_dir *= earth_moon_distance   # 8e6

    ### To check 
    # print("Distance")
    # print(earth_moon_distance/1e3)

    ### To check 
    # print("Distance")
    # print(earth_moon_distance/1e3)

    ### Render the scene 
    if len(plotter.renderer.lights)>0:
        ### There is already a light, change position 
        light = plotter.renderer.lights[0]
        light.position = sun_dir
    else: 
        ### No light exists: create it. 
        light = pv.Light(
            position=sun_dir,             ### Point the light is coming from
            focal_point=(0.0, 0.0, 0.0),  ### point where the light is aiming 
            color=(1.0, 1.0, 0.95),
            intensity=1,
            positional=False,             ### Ensure light is at +infty
        )
        # plotter.enable_shadows()   ### keep for future 
        plotter.add_light(light)

    ### camera at the Moon from the Earth-facing direction
    if not interactive:
        plotter.camera.position = moon_view_dir
        ### Try for an infinite focal (telescope)
        # plotter.camera.focal_point = -moon_view_dir*1e10 #(0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 0.0, 1.0)
        ### Option two: up is according to the observer position: need Earth ecliptic. 
        # plotter.camera.up = (0.0, np.cos(np.deg2rad(lat_obs+...)), np.sin(np.deg2rad(lat_obs+...)) )
        plotter.enable_parallel_projection()
        plotter.camera.zoom(40)
        # plotter.camera.parallel_scale *= 0.02  ### same as zoom *50 
        ### NOTE: Do not modify camera distance, but zoom instead. 
        ###       Otherwise it changes the focal.  
        ### enable parallel projection can also help remove perspective
    else: 
        ### Put camera much closer for interactive view
        plotter.camera.position = moon_view_dir / 50
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.enable_parallel_projection()   ### Doesn't work 

    ### Only add text if not in streamlit 
    if not interactive:
        if "date_text" in plotter.actors:
            plotter.remove_actor(plotter.actors["date_text"])
        
        plotter.add_text(
            "Date UTC: {0}\nPhase: {1}".format(start_date.strftime("%Y-%m-%d %H:%M"), mi.phase_name()),
            font_size=12,
            color="grey",
            name="date_text",
        )
    # plotter.set_background('slategray', top="lightsteelblue")
    plotter.set_background('#1A1A30', top="#362E37")
    # plotter.set_background((0.059, 0.067, 0.086))  # ou toute autre couleur


### =======================================================================================================
### Test function to save an animation of the scene
### =======================================================================================================
def animate_moon(mi, start_date, days=30, step_hours=12, dem_path='data/moon_relief_06m_g.grd', 
                 output_file='../Figures/moon_animation.mp4', fps=10):
    """Create an animation of the Moon camera view over time with mesh rendered once.
    
    Parameters:
    - mi: pylunar.MoonInfo object
    - start_date: tuple (year, month, day, hour, minute, second) or datetime object
    - days: number of days to animate
    - step_hours: hours between frames
    - dem_path: path to DEM file
    - output_file: output video file path
    - fps: frames per second for output video
    
    Returns:
    - None (saves video file)
    """
    if isinstance(start_date, tuple):
        current = datetime(*start_date)
    else:
        current = start_date
    
    # Create plotter with mesh (only once)
    print("Setting up scene...")
    plotter = make_3d_image(dem_path=dem_path)
    
    # Calculate total number of frames
    num_frames = int(days * 24 / step_hours)
    
    # Start recording
    plotter.open_movie(output_file, framerate=fps)
    
    print(f"Animating {num_frames} frames...")
    
    for frame_idx in range(num_frames):
        # Update MoonInfo for current date/time
        mi.update((current.year, current.month, current.day,
                   current.hour, current.minute, current.second))
        
        # Update scene (camera position, lighting, text)
        update_scene(plotter, mi, current)
        
        # Write frame
        plotter.write_frame()
        
        # Progress indicator
        if (frame_idx + 1) % max(1, num_frames // 10) == 0:
            print(f"  Frame {frame_idx + 1}/{num_frames} ({(frame_idx + 1) / num_frames * 100:.0f}%)")
        
        # Increment time
        current += timedelta(hours=step_hours)
    
    # Finish and close movie
    plotter.close()
    print(f"Animation saved to: {output_file}")


### =======================================================================================================
### Shows the 3D Moon evolution in time in an interactive window. 
### =======================================================================================================
def interactive_animation(mi, start_date, lat_obs, duration_days=30, step_hours=6, dem_path='data/moon_relief_06m_g.grd', 
                          update_interval=1.0):
    """Display live animation in an interactive window that updates in real-time.
    
    Parameters:
    - mi: pylunar.MoonInfo object
    - start_date: tuple (year, month, day, hour, minute, second) or datetime object
    - duration_days: number of days to animate
    - step_hours: hours between updates
    - dem_path: path to DEM file
    - update_interval: seconds between screen updates (default 1.0)
    
    The window will update every update_interval seconds and you can close it manually.
    """
    import time
    
    if isinstance(start_date, tuple):
        current = datetime(*start_date)
    else:
        current = start_date
    
    ### Create plotter with mesh (only once)
    print("Creating interactive scene...")
    plotter = make_3d_image(dem_path=dem_path)
    
    ### Calculate total updates
    total_updates = int(duration_days * 24 / step_hours)
    
    ### Keep updating while window is still open
    update_count = 0
    try:
        while update_count < total_updates:
            ### Check if window is still open
            # if not plotter.iren or plotter.iren.GetRenderWindow().GetNeverRendered():
            #     break
            
            ### Update MoonInfo
            mi.update((current.year, current.month, current.day,
                       current.hour, current.minute, current.second))
            
            ### Update scene (this removes old text and adds new one)
            plotter.remove_actor('text')
            update_scene(plotter, mi, current, lat_obs)
            if update_count ==0 :
                print("Opening window... (Press Ctrl+C in terminal to stop)")
                plotter.show(auto_close=False, full_screen=False)
            
            ### Render the scene
            plotter.render()
            
            ### Progress
            print(f"  Frame {update_count + 1}/{total_updates}: {current.strftime('%Y-%m-%d %H:%M')}")
            
            ### Increment time
            current += timedelta(hours=step_hours)
            update_count += 1
            
            ### Wait before next update
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("Animation stopped by user")


### =======================================================================================================
### Saves a png of the Moon scene at a specific time, to be displayed on Streamlit 
### =======================================================================================================
def get_scene_png(plotter, observer_lat, observer_lon, date):
    lat_obs = observer_lat[0] + observer_lat[1]/60 + observer_lat[2]/3600
    mi = pylunar.MoonInfo(observer_lat, observer_lon)
    mi.update(date)
    plotter.off_screen = True
    update_scene(plotter, mi, date, lat_obs)
    # plotter.screenshot('./moon_view.png', window_size=[400,450])      ### Writes to disk
    img = plotter.screenshot(return_img = True, window_size=[400,450])  ### Avoids writing to disk
    return(img)

### =======================================================================================================
### Produces a 3d image of the Moon at current time, to be displayed interactively by Streamlit
### =======================================================================================================
def get_scene_3d(plotter, observer_lat, observer_lon, date):
    lat_obs = observer_lat[0] + observer_lat[1]/60 + observer_lat[2]/3600
    mi = pylunar.MoonInfo(observer_lat, observer_lon)
    mi.update(date)
    update_scene(plotter, mi, date, lat_obs, interactive=True)
    return(plotter)



### =======================================================================================================
### Call functions for tests. 
### =======================================================================================================
if __name__ == '__main__':
    pv.OFF_SCREEN = False
    ### Input observer position (latitude degree-minutes-seconds), (longitude degree-minutes-seconds)
    ### Calern : 43° 44' 33" N / 6° 54' 01" E
    # observer_lat = (43, 44, 33)
    observer_lat = (43, 44, 33)  ### North pole
    observer_lon = (6, 54, 1)
    lat_obs = observer_lat[0] + observer_lat[1]/60 + observer_lat[2]/3600
    mi = pylunar.MoonInfo(observer_lat, observer_lon)

    ### Input observer time in UTC. 
    start_date=(2026, 4, 23, 14, 00, 0)
    current = datetime(*start_date)
    mi.update(start_date)
    ##################################################################################

    ### print fractional phase (1=full light)
    fp = mi.fractional_phase()
    print("Fractional phase: {:.3f}".format(fp))

    ### Print phase name
    pn = mi.phase_name()
    print("Phase name: {}".format(pn))

    shackleton, lc = load_features()
    print(">>> Is Shackleton visible? {}\n".format(mi.is_visible(shackleton)))

    ### Test the visibility 
    test_visibility_time(observer_lat, observer_lon, start_date, days=60, step_hours=2)
    # plt.show()

    ### Option 1: Show single static frame
    plotter = make_3d_image()
    update_scene(plotter, mi, start_date, lat_obs, current)
    plotter.show(auto_close=False)
    # plotter.export_html("Figures/moon_view.html")  ### Careful, it is very large ! 
    plotter.screenshot('./moon_view.png')  

    ### Option 2: Live interactive animation (updates every 1 second)
    # interactive_animation(
    #     mi=mi,
    #     start_date=start_date,
    #     lat_obs=lat_obs, 
    #     duration_days=28,
    #     step_hours=24,  # Update every 6 hours
    #     dem_path='data/moon_relief_06m_g.grd',
    #     update_interval=2  # Update displayed frame every 1 second
    # )

    ### Option 3: Create animation video file (non-interactive)
    # animate_moon(
    #     mi=mi,
    #     start_date=(2026, 4, 12, 2, 0, 0),
    #     days=30,
    #     step_hours=6,
    #     dem_path='data/moon_relief_06m_g.grd',
    #     output_file='moon_animation_30days.mp4',
    #     fps=15
    # )


    ### Option 2: Create animation (uncomment to use)
    # animate_moon(
    #     mi=mi,
    #     start_date=(2026, 4, 12, 2, 0, 0),
    #     days=30,
    #     step_hours=6,  # Frame every 6 hours
    #     dem_path='data/moon_relief_06m_g.grd',
    #     output_file='moon_animation_30days.mp4',
    #     fps=15
    # )

    ### Option 3: Save animation figures (currently commented out)
    # start_date=(2025, 1, 1, 1, 0, 0)
    # current = datetime(*start_date)
    # end_date=(2031, 12, 31, 23, 59, 59)
    # end_date = datetime(*end_date)
    # t_delta=3 #hours
    
    # while current<end_date:
    # ### Input observer time in UTC. 
    #     mi.update(current)
    #     #################################################################################
    #     # print(current.timetuple().tm_yday)
    #     frame = ((current.timetuple().tm_yday-1) * 24 + current.hour)
    #     ### Option 1: Show single static frame
    #     plotter = make_3d_image()
    #     update_scene(plotter, mi, start_date, lat_obs, current)
    #     plotter.off_screen = True
    #     name="images_moon/moon_view_"+str(current.year)+"_"+str(frame)+".png"
    #     print(name)
    #     plotter.show(screenshot=name)
        
    #     plotter.close()
    #     del plotter
    #     gc.collect()
    
    #     current+=timedelta(hours=t_delta)
        
        
