#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lunar visibility map — beautiful night/day background
Drop-in replacement for the plotting block inside the time loop.
Requires: cartopy, matplotlib, numpy, ephem
"""

import numpy as np
import pandas as pd 
import streamlit as st     
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime
import ephem
import xarray as xr
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.image import imread
from matplotlib.lines import Line2D

from astropy.coordinates import (
    EarthLocation, AltAz, get_body, solar_system_ephemeris,
)
from astropy.time import Time
import astropy.units as u

# ── Style constants ───────────────────────────────────────────────────────────
BG       = (0.055, 0.067, 0.09)
FG       = "white"
GRID_COL = "white"
ALPHA_GRID = 0.12
CMAP     = "plasma"

    
def _style_ax(ax):
    """Apply dark theme to a single axes."""
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
        spine.set_linewidth(0.6)
    ax.tick_params(colors=FG, labelsize=8, length=3)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID_COL, alpha=ALPHA_GRID, linewidth=0.4, linestyle="--")
    

def _compute_grid_astropy(lons: np.ndarray, lats: np.ndarray,
                           date, ephemeris: str = "de432s"):
    """
    Vectorised moon/sun altitude over a lon×lat grid using astropy.

    Parameters
    ----------
    lons, lats  : 1-D arrays of longitudes / latitudes in degrees
    date        : datetime.datetime (UTC)
    ephemeris   : astropy solar_system_ephemeris string or path to .bsp kernel

    Returns
    -------
    xr.Dataset with variables alta, aza, night, moonvisible,
    moon_distance (AU), phase (%) and coords lon, lat, date.
    """
    t = Time(date, scale="utc")

    # Full grid — shape (nlon, nlat)
    LON, LAT = np.meshgrid(lons, lats, indexing="ij")
    flat_lon = LON.ravel() * u.deg
    flat_lat = LAT.ravel() * u.deg

    locations = EarthLocation(lon=flat_lon, lat=flat_lat, height=0 * u.m)
    frame = AltAz(obstime=t, location=locations)

    with solar_system_ephemeris.set(ephemeris):
        moon_body = get_body("moon", t, locations)
        sun_body  = get_body("sun",  t, locations)
        moon_altaz = moon_body.transform_to(frame)
        sun_altaz  = sun_body.transform_to(frame)

    moon_alt = moon_altaz.alt.deg.reshape(LON.shape)   # (nlon, nlat)
    moon_az  = moon_altaz.az.deg.reshape(LON.shape)
    sun_alt  = sun_altaz.alt.deg.reshape(LON.shape)

    night       = sun_alt  <  0
    moonvisible = (moon_alt > 5) & night

    # ── Scalar metadata at lon=0, lat=0 ──────────────────────────────────────
    ref_loc   = EarthLocation(lon=0*u.deg, lat=0*u.deg, height=0*u.m)
    ref_frame = AltAz(obstime=t, location=ref_loc)
    with solar_system_ephemeris.set(ephemeris):
        moon_ref = get_body("moon", t, ref_loc)

    # Distance in AU (1 AU ≈ 149 597 870.7 km)
    dist_au = moon_ref.distance.to(u.au).value

    # Illumination fraction → percentage
    # Simple geometric approximation consistent with ephem's moon.phase
    sun_ref = get_body("sun", t, ref_loc)
    elongation = moon_ref.separation(sun_ref).rad
    phase_pct  = (1 - np.cos(elongation)) / 2 * 100

    return xr.Dataset(
        {
            "alta":          (["lon", "lat"], moon_alt),
            "aza":           (["lon", "lat"], moon_az),
            "night":         (["lon", "lat"], night),
            "moonvisible":   (["lon", "lat"], moonvisible),
            "moon_distance": float(dist_au),
            "phase":         float(phase_pct),
        },
        coords={
            "lon":  lons,
            "lat":  lats,
            "date": date,
        },
    )

### =======================================================================================================
### Creates Earth map and caches it. 
### =======================================================================================================
def create_Earthmap():

        # ── Figure / axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 6), facecolor=(0.055, 0.067, 0.09))
    ax  = fig.add_axes([0.03, 0.06, 0.84, 0.88],
                       projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.set_facecolor("#0a0a1a")

    # 1. Blue Marble background
    # ax.stock_img()
    fname = './data/NE1_50M_SR_W_low.tif'
    ax.imshow(imread(fname), origin='upper', transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])


    # 2. Geographic features
    ax.add_feature(cfeature.COASTLINE,
                   linewidth=0.5, edgecolor="white", alpha=0.6, zorder=4)
    ax.add_feature(cfeature.BORDERS,
                   linewidth=0.25, edgecolor="white", alpha=0.3,
                   linestyle=":", zorder=4)

    # 3. Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.35, color="white", alpha=0.25, linestyle="--",
        zorder=5,
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator   = mticker.FixedLocator(range(-180, 181, 60))
    gl.ylocator   = mticker.FixedLocator(range(-90,   91, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "white", "alpha": 0.8}
    gl.ylabel_style = {"size": 7, "color": "white", "alpha": 0.8}

    # ── Night shading colormap ────────────────────────────────────────────────────
    # Transparent in the day, opaque dark-blue at full night.
    # Using a custom LinearSegmentedColormap so alpha varies with value.
    night_colors = [
        (0.05, 0.05, 0.20, 0.0),   # sun just below horizon → twilight, transparent
        (0.05, 0.05, 0.20, 0.25),  # civil twilight
        (0.02, 0.02, 0.15, 0.55),  # nautical twilight
        (0.00, 0.00, 0.10, 0.80),  # astronomical night
    ]
    cmap_night = mcolors.LinearSegmentedColormap.from_list(
        "night_alpha", night_colors, N=256
    )

    # ── Moon altitude colormap & norm ─────────────────────────────────────────────
    cmap_moon = plt.cm.plasma
    norm_moon  = mcolors.Normalize(vmin=5, vmax=90)

    return(fig, ax, cmap_moon, norm_moon, cmap_night)


### =======================================================================================================
### Adds Night overlay and Moon altitude to map using astropy (slower)
### =======================================================================================================
def plot_moonmap2(date, loca, dpi=300, ephemeris="de432s", space_grid=1):
    """
    Parameters
    ----------
    date        : datetime.datetime (UTC)
    loca        : [[deg,min,sec], [deg,min,sec]] — observer lat/lon in DMS
    dpi         : output resolution
    ephemeris   : astropy ephemeris string ("de432s", "de430", "builtin", …)
                  or path to a local .bsp kernel file
    space_grid  : grid spacing in degrees (default 0.5°)
    """

    observer_lat = np.sign(loca[0][0]) * (
        abs(loca[0][0]) + loca[0][1]/60 + loca[0][2]/3600
    )
    observer_lon = np.sign(loca[1][0]) * (
        abs(loca[1][0]) + loca[1][1]/60 + loca[1][2]/3600
    )

    # ── Grid & computation ────────────────────────────────────────────────────
    lons = np.arange(-180, 180, space_grid)
    lats = np.arange( -90,  90, space_grid)

    ds = _compute_grid_astropy(lons, lats, date, ephemeris=ephemeris)

    # ── Pre-process fields ────────────────────────────────────────────────────
    night_float = ds["night"].T.values.astype(float)    # (nlat, nlon)
    moon_alta   = ds["alta"].where(ds["moonvisible"]).T  # masked

    phase   = float(ds["phase"])
    dist_au = float(ds["moon_distance"])
    dist_km = dist_au * 149_597_870.7

    # ── Figure / axes ─────────────────────────────────────────────────────────
    fig, ax, cmap_moon, norm_moon, cmap_night = create_Earthmap()

    # 1. Night overlay
    ax.pcolormesh(
        ds["lon"].values, ds["lat"].values, night_float,
        cmap=cmap_night, vmin=0, vmax=1,
        shading="auto", transform=ccrs.PlateCarree(),
        zorder=2, rasterized=True,
    )

    # 2. Moon altitude overlay
    pm = ax.pcolormesh(
        ds["lon"].values, ds["lat"].values, moon_alta,
        cmap=cmap_moon, norm=norm_moon,
        alpha=0.82, shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=3, rasterized=True,
    )

    # 3. Observer position
    ax.scatter(
        observer_lon, observer_lat,
        s=20, c="red", alpha=0.7,
        transform=ccrs.PlateCarree(), zorder=10,
    )

    # 4. Colorbar
    cax = fig.add_axes([0.89, 0.10, 0.016, 0.80])
    cb  = fig.colorbar(pm, cax=cax, orientation="vertical", extend="neither")
    cb.set_label("Moon altitude (°)", fontsize=12, color="white", labelpad=6)
    cb.set_ticks([5, 15, 30, 45, 60, 75, 90])
    cb.ax.tick_params(labelsize=7.5, colors="white")
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(0.4)

    meta = (
        f"Illumination: {phase:.1f}%     "
        f"Earth–Moon: {dist_km:,.0f} km"
    )
    fig.text(0.04, 0.01, meta,
             fontsize=10, color="#cccccc", style="italic", va="bottom")

    # 5. Legend
    night_patch = mpatches.Patch(
        facecolor=(0.0, 0.0, 0.1), alpha=0.7,
        label="Night (sun < 0°)",
    )
    ax.legend(
        handles=[night_patch],
        loc="lower left", fontsize=7.5,
        framealpha=0.4, facecolor="black",
        edgecolor="#666666", labelcolor="white",
        handlelength=1.2,
    )

    return fig


### =======================================================================================================
### Adds Night overlay and Moon altitude to map.
### =======================================================================================================
def plot_moonmap(date,loca,dpi=150):
    space_grid=1
    
    observer_lat = np.sign(loca[0][0]) * (abs(loca[0][0]) + loca[0][1]/60 + loca[0][2]/3600)
    observer_lon = np.sign(loca[1][0]) * (abs(loca[1][0]) + loca[1][1]/60 + loca[1][2]/3600)
        
    lons=np.arange(-180,180,space_grid)
    lats=np.arange(-90,90,space_grid)
    #Observer
    shape = (lons.shape[0], lats.shape[0])
    here=ephem.Observer()
    here.date=date
    alta,aza,phase,nightar,moonvisiblear=[],[],[],[],[]
    for longi in lons:
        here.lon  = str(longi) #Note that lon should be in string format
        for lati in lats:
            here.lat  = str(lati) #Note that lat should be in string format
            moon=ephem.Moon(here)
            sun=ephem.Sun(here)
            
            night=True
            sunangle=np.rad2deg(sun.alt)
            if sunangle>0:night=False
            
            moonvisible=True
            moonangle=np.rad2deg(moon.alt)
            # if moonangle<5:moonvisible=False
            moonvisible = (moonangle > 5) #& night
            alta.append(np.rad2deg(moon.alt))
            aza.append(np.rad2deg(moon.az))
            nightar.append(night)
            moonvisiblear.append(moonvisible)
    
    ds = xr.Dataset(
        {
            "alta":         (["lon", "lat"], np.reshape(alta,         shape)),
            "aza":          (["lon", "lat"], np.reshape(aza,          shape)),
            "night":        (["lon", "lat"], np.reshape(nightar,      shape)),
            "moonvisible":  (["lon", "lat"], np.reshape(moonvisiblear,shape)),
            "moon_distance": moon.earth_distance,
            "phase":         moon.phase,
        },
        coords={
            "lon": lons,
            "lat": lats,
            "date": date,
        }
    )
    
    # ── Pre-process fields ────────────────────────────────────────────────────
    # Convert boolean night mask to a float in [0, 1] for the alpha colormap.
    # A simple True/False gives only two tones; if you store solar altitude
    # as a continuous variable, replace this with a proper twilight ramp:
    #   night_float = np.clip(-ds["sun_alt"].T.values / 18, 0, 1)
    night_float = ds["night"].T.values.astype(float)   # (nlat, nlon)
    moon_alta   = ds["alta"].where(ds["moonvisible"]).T # masked where not visible
    
    # Scalar metadata (stored as 0-d variables in the Dataset)
    phase   = float(ds["phase"])
    dist_au = float(ds["moon_distance"])
    dist_km = dist_au * 149_597_870.7

    # ── Figure / axes ─────────────────────────────────────────────────────────
    fig, ax, cmap_moon, norm_moon, cmap_night = create_Earthmap()

    # ── 2. Night overlay ──────────────────────────────────────────────────────
    # The colormap maps 0→transparent, 1→deep-navy-opaque, so day areas
    ax.pcolormesh(
        ds["lon"].values, ds["lat"].values, night_float,
        cmap=cmap_night, vmin=0, vmax=1,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=2, rasterized=True,
    )
    
    # ── 3. Moon altitude overlay ──────────────────────────────────────────────
    pm = ax.pcolormesh(
        ds["lon"].values, ds["lat"].values, moon_alta,
        cmap=cmap_moon, norm=norm_moon,
        alpha=0.82,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=3, rasterized=True,
    )
    
    ax.scatter(observer_lon, observer_lat, s=20, c="red",alpha=0.7,
           transform=ccrs.PlateCarree(),
           zorder=10)

    # ── 6. Colorbar ───────────────────────────────────────────────────────────
    cax = fig.add_axes([0.89, 0.10, 0.016, 0.80])
    cb  = fig.colorbar(pm, cax=cax, orientation="vertical", extend="neither")
    cb.set_label("Moon altitude (°)", fontsize=12, color="white", labelpad=6)
    cb.set_ticks([5, 15, 30, 45, 60, 75, 90])
    cb.ax.tick_params(labelsize=7.5, colors="white")
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(0.4)
    
    # ── 7. Title & metadata annotation ────────────────────────────────────────
    # date_str = date.strftime("%Y-%m-%d  %H:%M UTC")
    # ax.set_title(
    #     f"Moon visibility — {date_str}",
    #     fontsize=12, fontweight="bold", color="white", pad=8,
    # )

    meta = (
        f"Illumination: {phase:.1f}%     "
        f"Earth–Moon: {dist_km:,.0f} km"
    )
    fig.text(0.04, 0.01, meta,
             fontsize=10, color="#cccccc", style="italic", va="bottom")

    # ── 8. Legend ─────────────────────────────────────────────────────────────
    night_patch = mpatches.Patch(
        facecolor=(0.0, 0.0, 0.1), alpha=0.7,
        label="Night (sun < 0°)",
    )
    ax.legend(
        handles=[night_patch],
        loc="lower left", fontsize=7.5,
        framealpha=0.4, facecolor="black",
        edgecolor="#666666", labelcolor="white",
        handlelength=1.2,
    )
    
    return fig



### =======================================================================================================
### Creates table from observation windows
### =======================================================================================================
@st.cache_data
def extract_time_windows(cnum, condition):
    """
    Parameters
    ----------
    cnum : array-like
        Matplotlib numeric dates (same length as condition)
    condition : array-like of bool
        Boolean mask

    Returns
    -------
    DataFrame with columns:
        Start date, End date, Duration (hours)
    """
    cnum = np.asarray(cnum)
    condition = np.asarray(condition, dtype=bool)

    # Find transitions False -> True and True -> False
    padded = np.concatenate([[False], condition, [False]])
    changes = np.diff(padded.astype(int))

    starts_idx = np.where(changes == 1)[0]
    ends_idx = np.where(changes == -1)[0] - 1

    rows = []

    for i_start, i_end in zip(starts_idx, ends_idx):
        start_num = cnum[i_start]
        end_num = cnum[i_end]

        start_dt = mdates.num2date(start_num)
        end_dt = mdates.num2date(end_num)

        duration_hours = (end_num - start_num) * 24.0 
        duration_minutes = duration_hours * 60

        if duration_minutes> 0:
            rows.append({
                "Start date": start_dt.strftime("%d %B %Y %H:%M"),
                "End date": end_dt.strftime("%d %B %Y %H:%M"),
                "Duration (minutes)": round(duration_minutes, 2),
            })

    return pd.DataFrame(rows)


### =======================================================================================================
### Predicts visibility of Shackleton on visible face.
### =======================================================================================================
@st.cache_resource
def shackleton_visibility(date,forecast_days,loca):
    LAT_SHA = -89.67
    LON_SHA = 129.78
    
    observer_lat = np.sign(loca[0][0]) * (abs(loca[0][0]) + loca[0][1]/60 + loca[0][2]/3600)
    observer_lon = np.sign(loca[1][0]) * (abs(loca[1][0]) + loca[1][1]/60 + loca[1][2]/3600)
    
    here=ephem.Observer()
    here.lat  = observer_lat
    here.lon  = observer_lon
    here.date = date
    moon = ephem.Moon(here)
    
    init_poslat=LAT_SHA - np.rad2deg(moon.libration_lat)
    init_poslon=LON_SHA - np.rad2deg(moon.libration_long)
    init_distau=float(moon.earth_distance)
    init_date=date
    
    # ── Compute libration loop ────────────────────────────────────────────────────
    dates, pos_lat, pos_lon, dist_au,phasis,moon_alt, moon_visible,isnight= [], [], [], [],[],[],[],[]
    start_date=date-datetime.timedelta(days=5)
    end_date=date+datetime.timedelta(days=forecast_days)
    date=start_date
    while date < end_date:
        here.date = date
        moon = ephem.Moon(here)
        sun=ephem.Sun(here)
        dates.append(date)
        pos_lat.append(LAT_SHA - np.rad2deg(moon.libration_lat))
        pos_lon.append(LON_SHA - np.rad2deg(moon.libration_long))
        dist_au.append(float(moon.earth_distance))
        phasis.append(moon.moon_phase)
        
        moonvisible=True
        moonangle=np.rad2deg(moon.alt)
        if moonangle<15:moonvisible=False
        moon_visible.append(moonvisible)
        moon_alt.append(moonangle)
        night=True
        sunangle=np.rad2deg(sun.alt)
        if sunangle>0:night=False
        isnight.append(night)
        
        date += datetime.timedelta(hours=0.5)
        
    pos_lat = np.array(pos_lat)
    pos_lon = np.array(pos_lon)
    isnight=np.array(isnight)
    
    phasis=np.array(phasis)
    shavisible= pos_lat > - 89
    dist_km = np.array(dist_au) * 149_597_870.7 / 384_400   # units of mean dist.
    cnum    = mdates.date2num(dates)
    
    # ── Figure layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 6),facecolor=(0.055, 0.067, 0.09))
    
    # Left: libration scatter  |  Right: distance time-series
    ax = fig.add_axes([0.06, 0.11, 0.52, 0.80])
    
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
        spine.set_linewidth(0.6)
    ax.tick_params(colors=FG, labelsize=8, length=3)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID_COL, alpha=ALPHA_GRID, linewidth=0.4, linestyle="--")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Panel 1 — Libration map
    # ════════════════════════════════════════════════════════════════════════════
    
    # Shaded regions: visible face (lat > -90) / hidden face (lat < -90)
    lat_min = pos_lat.min() - 1.5
    lat_max = pos_lat.max() + 1.5
    lon_min = pos_lon.min() - 1.5
    lon_max = pos_lon.max() + 1.5
    
    ax.axvspan(lat_min,  -90,    alpha=0.18, color=BG, zorder=0)  # hidden face
    ax.axvspan(  -90, lat_max,   alpha=0.08, color="#aabbdd", zorder=0)  # visible face
    
    # Dividing line at lat = -90
    ax.axvline(-90, color="#8899bb", linewidth=0.8, linestyle="-", alpha=0.7, zorder=1)
    
    # Trailing path as a colored line (gradient = time)
    points  = np.array([pos_lat, pos_lon]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    norm_c  = mcolors.Normalize(vmin=cnum.min(), vmax=cnum.max())
    lc = LineCollection(segs, cmap=CMAP, norm=norm_c, linewidth=0.7, alpha=0.45, zorder=2)
    lc.set_array(cnum[:-1])
    ax.add_collection(lc)
    
    # Scatter (colored by date, sized by distance)
    size = 6 + 14 * (1 - (dist_km - dist_km.min()) / (dist_km + 1e-9))
    sc = ax.scatter(pos_lat, pos_lon, c=cnum, cmap=CMAP,
                        s=size, alpha=0.85, linewidths=0, zorder=3)
    
    # Nominal Shackleton position
    ax.scatter(LAT_SHA, LON_SHA, marker="*", s=120, color="#ffdd55",
                   zorder=5, linewidths=0.5, edgecolors="white", label="Nominal position")
    
    
    # Marker for current position
    ax.scatter(init_poslat,  init_poslon,  marker="o", s=60, color="#55ff99",
                   zorder=6, linewidths=0, label=f"Actual position \n{init_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Region labels
    ax.text(-90.15, (lon_min + lon_max) / 2, "Hidden face",
                color="#8899bb", fontsize=10, rotation=90,
                va="center", ha="right", alpha=0.9)
    ax.text(-89.85, (lon_min + lon_max) / 2, "Visible face",
                color="#aabbdd", fontsize=10, rotation=90,
                va="center", ha="left", alpha=0.9)
    
    # Colorbar (date axis)
    cax = fig.add_axes([0.06, 0.015, 0.52, 0.015])
    cb  = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    cb.ax.tick_params(colors=FG, labelsize=7.5, length=3)
    cb.set_label("Date", color=FG, fontsize=8, labelpad=4)
    cb.outline.set_edgecolor("#444466")
    cb.outline.set_linewidth(0.5)
    
    ax.set_xlim(lat_min, lat_max)
    ax.set_ylim(lon_min, lon_max)
    ax.set_xlabel("Apparent latitude (°)", fontsize=9)
    ax.set_ylabel("Apparent longitude (°)", fontsize=9)
    ax.set_title("Apparent position of Shackleton crater",
                     fontsize=10, fontweight="bold", pad=8)
    
    leg = ax.legend(fontsize=7.5, framealpha=0.3, facecolor=BG,
                        edgecolor="#444466", labelcolor=FG, loc="upper right",
                        handlelength=1.2)
    
    ax.set_aspect('equal')
    

    # ════════════════════════════════════════════════════════════════════════════
    # Panel 2 — Best visibility plot 
    # ════════════════════════════════════════════════════════════════════════════

    fig2, ax2 = plt.subplots(1, 1, figsize=(13, 6), facecolor=(0.055, 0.067, 0.09))
    _style_ax(ax2)
    
    # ── Twin axes ─────────────────────────────────────────────────────────────────
    ax3 = ax2.twinx()   # phase
    ax4 = ax2.twinx()   # libration lat
    for ax in (ax3, ax4):
        _style_ax(ax)
        ax.grid(False)   # avoid double grid
    
    # Offset the right spines so they don't overlap
    ax4.spines["right"].set_position(("outward", 55))
    
    # ── Green condition shading ───────────────────────────────────────────────────
    condition = isnight & shavisible & moon_visible
    ### Store good condition times 
    df_conditions = extract_time_windows(cnum[240:], condition[240:])

    ax2.fill_between(
        cnum[240:], 0, 1,
        where=condition[240:],
        transform=ax2.get_xaxis_transform(),   # spans full y regardless of scale
        color="#00ff88", alpha=0.15, zorder=0, label="Best conditions",
    )
    
    # ── Data lines ────────────────────────────────────────────────────────────────
    # Moon altitude — gradient LineCollection (time-coloured, same style as fig1)
    # Moon altitude — gradient LineCollection (colour = altitude)
    moon_alt=np.array(moon_alt)
    points = np.array([cnum[240:], moon_alt[240:]]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    norm_c = mcolors.Normalize(vmin=moon_alt[240:].min(), vmax=moon_alt[240:].max())
    lc = LineCollection(segs, cmap=CMAP, norm=norm_c, linewidth=1.2, alpha=0.9, zorder=2)
    lc.set_array(moon_alt[240:-1])   # colour driven by altitude, not date
    ax2.add_collection(lc)
    ax2.autoscale()
        
    ax3.plot(cnum[240:], phasis[240:],  color="#ffaa44", linewidth=0.9,
             alpha=0.85, label="Illumination (%)")
    ax4.plot(cnum[240:], pos_lat[240:], color="#55ff99", linewidth=0.9,
             alpha=0.85, label="Libration lat (°)")
    
    # ── Axis labels ───────────────────────────────────────────────────────────────
    ax2.set_ylabel("Moon altitude (°)",   fontsize=9, color=FG)
    ax3.set_ylabel("Illumination (%)",    fontsize=9, color="#ffaa44")
    ax4.set_ylabel("Libration lat (°)",   fontsize=9, color="#55ff99")
    ax3.tick_params(axis="y", colors="#ffaa44")
    ax4.tick_params(axis="y", colors="#55ff99")
    
    # ── X axis ────────────────────────────────────────────────────────────────────
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Hh"))
    ax2.tick_params(axis="x", labelrotation=30)
    ax2.set_xlabel("Date (UTC)", fontsize=9)
    
    # ── Title ─────────────────────────────────────────────────────────────────────
    ax2.set_title("Shackleton observability conditions",
                  fontsize=10, fontweight="bold", pad=25)
    
    # ── Legend — merge all axes ───────────────────────────────────────────────────
    handles, labels = [], []
    for ax in (ax2, ax3, ax4):
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    
    # Add a proxy for the moon altitude gradient line
    handles.append(Line2D([0], [0], color=plt.cm.get_cmap(CMAP)(0.6),
                           linewidth=1.2, label="Moon altitude"))
    labels.append("Moon altitude")
    
    ax2.legend(handles, labels, fontsize=9, framealpha=0.3,
               facecolor=BG, labelcolor=FG, edgecolor="none", #edgecolor="#444466"
               handlelength=1.4,
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4)
    
    fig2.tight_layout()
        
    
    
    # ax3=ax2.twinwx()
    # ax2.plot()
    
    
    # # Trailing path as a colored line (gradient = time)
    # points  = np.array([pos_lat, pos_lon]).T.reshape(-1, 1, 2)
    # segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    # norm_c  = mcolors.Normalize(vmin=cnum.min(), vmax=cnum.max())
    # lc = LineCollection(segs, cmap=CMAP, norm=norm_c, linewidth=0.7, alpha=0.45, zorder=2)
    # lc.set_array(cnum[:-1])
    # ax.add_collection(lc)
    
    # # Scatter (colored by date, sized by distance)
    # size = 6 + 14 * (1 - (dist_km - dist_km.min()) / (dist_km + 1e-9))
    # sc = ax.scatter(pos_lat, pos_lon, c=cnum, cmap=CMAP,
    #                     s=size, alpha=0.85, linewidths=0, zorder=3)
    
    # # Nominal Shackleton position
    # ax.scatter(LAT_SHA, LON_SHA, marker="*", s=120, color="#ffdd55",
    #                zorder=5, linewidths=0.5, edgecolors="white", label="Nominal position")
    
    
    # # Marker for current position
    # ax.scatter(init_poslat,  init_poslon,  marker="o", s=60, color="#55ff99",
    #                zorder=6, linewidths=0, label=f"Actual position \n{init_date.strftime('%Y-%m-%d %H:%M')}")
    
    # # Region labels
    # ax.text(-90.15, (lon_min + lon_max) / 2, "Hidden face",
    #             color="#8899bb", fontsize=10, rotation=90,
    #             va="center", ha="right", alpha=0.9)
    # ax.text(-89.85, (lon_min + lon_max) / 2, "Visible face",
    #             color="#aabbdd", fontsize=10, rotation=90,
    #             va="center", ha="left", alpha=0.9)
    
    # # Colorbar (date axis)
    # cax = fig.add_axes([0.06, 0.015, 0.52, 0.015])
    # cb  = fig.colorbar(sc, cax=cax, orientation="horizontal")
    # cb.ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # cb.ax.tick_params(colors=FG, labelsize=7.5, length=3)
    # cb.set_label("Date (2026)", color=FG, fontsize=8, labelpad=4)
    # cb.outline.set_edgecolor("#444466")
    # cb.outline.set_linewidth(0.5)
    
    # ax.set_xlim(lat_min, lat_max)
    # ax.set_ylim(lon_min, lon_max)
    # ax.set_xlabel("Apparent latitude (°)", fontsize=9)
    # ax.set_ylabel("Apparent longitude (°)", fontsize=9)
    # ax.set_title("Apparent position of Shackleton crater",
    #                  fontsize=10, fontweight="bold", pad=8)
    
    # leg = ax.legend(fontsize=7.5, framealpha=0.3, facecolor=BG,
    #                     edgecolor="#444466", labelcolor=FG, loc="upper right",
                        # handlelength=1.2)
    return fig,fig2, df_conditions

# def best_observation_time(loc,date):
    
    
    