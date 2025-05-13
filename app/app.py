from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import geopandas as gpd
import dash_daq as daq
import os
import warnings
import networkx as nx
import dash_bootstrap_components as dbc
import glob

warnings.filterwarnings("ignore")

# --------
# Set path
# --------
base_path = os.path.dirname(os.path.abspath(__name__))

# ------------------------------
# Load data for monitoring wells
# ------------------------------
gdfallwells = gpd.read_parquet(f"{base_path}/assets/all_wells_filtered.parquet")

# ------------------------------
# Load data for subsidence sites
# ------------------------------
gdfsites = gpd.read_parquet(f'{base_path}/assets/all_sites_filtered.parquet')

# ---------
# Check crs
# ---------
if gdfsites.crs != gdfallwells.crs:
    gdfsites = gdfsites.to_crs(gdfallwells.crs)

# ---------------------------------------
# Load the water level data for all wells
# ---------------------------------------
well_wls = pd.read_parquet(f"{base_path}/assets/wl_data_daily_filtered.parquet")


# -------------------------------------
# Grab wells based on the selected site
# -------------------------------------
def select_site(site,
                county_filter=False
                ):
    """
    
    Parameters
    ----------
    site : String
        Specify the site name you want to focus on.

    Returns
    -------
    site_data : DataFrame
        Data for all wells within 3-miles of 'site'.
    filtered_id_all : Array
        Array of site ID's corresponding to selected site.
    final_wells_gdf : GeoDataFrame
        GeaDataFrame with spatial information about wells in 'site_data'.

    """
    # Define the distance in feet (3 miles = 15840 feet)
    distance_in_feet = 15840
    
    if county_filter:
        selected_site = gpd.read_file(f"./assets/county_centroids/centroid_{site}.shp")
        selected_site = selected_site.set_crs(4326)
        selected_site = selected_site.to_crs(2227)
    else:
        # Find the specific site from gdfsites based on the "Name" column
        selected_site = gdfsites[gdfsites['Name'] == site]

    # Check if the site was found
    if selected_site.empty:
        print(f"Site '{site}' not found in gdfsites.")
    else:
        if county_filter:
            site_geometry = selected_site.geometry
        else:
            # Use the geometry of the selected site to create a buffer
            site_geometry = selected_site.iloc[0].geometry
        buffer = site_geometry.buffer(distance_in_feet)
        
        # Find all wells within this buffer from gdf3
        # Speed this up by using a bounding box filter first
        bbox_buffer = buffer.envelope
        
        if county_filter:
            bbox_candidates = gdfallwells[gdfallwells.geometry.buffer(1e-9).within(
                bbox_buffer.geometry.unary_union)]
            nearby_wells = bbox_candidates[bbox_candidates.geometry.within(buffer.geometry.unary_union)]
        else:
            bbox_candidates = gdfallwells[gdfallwells.geometry.intersects(
                bbox_buffer)]
            nearby_wells = bbox_candidates[bbox_candidates.geometry.within(buffer)]

        # Convert to GeoDataFrame for final output
        final_wells_gdf = gpd.GeoDataFrame(nearby_wells)

        # Extract the SWN values from the filtered wells GeoDataFrame
        filtered_id_all = final_wells_gdf['ID_ALL'].unique()

        # Search for rows in well_wls where ID_ALL matches filtered wells' ID_ALL
        site_data = well_wls[well_wls['ID_ALL'].isin(filtered_id_all)]

        # Sort the data by date to ensure a continuous plot
        site_data = site_data.sort_values(by=['ID_ALL', 'Date'])

    return site_data, filtered_id_all, final_wells_gdf


# --------------------------------------------------------------------
# Use graph theory to determine 3-mile connectivity filter
# Finds the largest "clique" of all the selected wells
#     --> for reference, https://en.wikipedia.org/wiki/Clique_problem
# --------------------------------------------------------------------
def filter_within_radius_networkx(gdf, distance_in_feet):
    # Initialize the Graph object
    G = nx.Graph()
    
    # Add nodes to the graph with coordinates as attributes
    for idx, point in enumerate(gdf.geometry):
        G.add_node(idx, pos=(point.x, point.y))

    # Add connections based on 3-mile threshold
    threshold = distance_in_feet
    for i in range(len(gdf)):
        for j in range(i + 1, len(gdf)):
            point1 = gdf.geometry.iloc[i]
            point2 = gdf.geometry.iloc[j]
            distance = point1.distance(point2)
            if distance < threshold:
                G.add_edge(i, j, weight=distance)
    
    # Find largest clique
    nodes = nx.approximation.max_clique(G)
    
    # Select the wells from the original GeoDataFrame based on the clique indices
    final_gdf = gdf.iloc[list(nodes)]

    return final_gdf


# ----------------------------------------------------
# Set global vars to prevent repetitive function calls
# ----------------------------------------------------
global site_data, filtered_id_all, final_wells_gdf, site_subsidence_data

site_subsidence_data = pd.read_parquet(f"{base_path}/assets/site_subsidence_data_filtered.parquet")

# -------------------
# Initialize Dash app
# -------------------
app = Dash(__name__, external_stylesheets=[f'{base_path}/assets/styles.css',
                                           dbc.themes.SANDSTONE])
server = app.server

# ----------
# App Layout
# ----------
app.layout = html.Div(
    children=[
        html.Hr(),
        dcc.Tabs(
            children=[
                dcc.Tab(label='Select Wells From Long-Term Site',
                        children=[
                            html.Div([
                                html.Hr(),
                                html.Div([
                                    html.Div([
                                        html.Div("Slide to select county-level centroids"),
                                        daq.BooleanSwitch(id='prop-118-slider',
                                                          style={'margin': '0 auto'  
                                                                 }
                                                          ),
                                        html.Div('Select Long-Term Site:',
                                             style={'margin-top': '10px'}
                                        ),
                                        dcc.Dropdown(id='site_picker_dd',
                                                     options=gdfsites['Name'].unique(),
                                                     style={'margin-bottom': '10px'},
                                                     clearable=False,
                                                     value=gdfsites['Name'].unique()[0]
                                        ),
                                    ],
                                    style={'width': '50%'}
                                    ),
                                    html.Div([
                                        html.Div('Filter by depth range (Total Depth):',
                                                 style={'textAlign': 'center',  
                                                        'width': '100%'
                                                        }
                                                 ),
                                        html.Div([
                                            dcc.Dropdown(placeholder='Min well depth',
                                                         id='min_input',
                                                         options=[x*100 for x in range(21)],
                                                         style={'margin-bottom':'5px',
                                                                'height':'20px',
                                                                'width':'200px'}
                                                         ),
                                            dcc.Dropdown(placeholder='Max well depth',
                                                         id='max_input',
                                                         options=[x*100 for x in range(21)],
                                                         style={'margin-bottom':'5px',
                                                                'height':'20px',
                                                                'width':'200px'}
                                                     ),
                                            ],
                                            style={'display':'flex',
                                                   'flex-direction':'row',
                                                   'margin-bottom':'15px'}
                                        ),
                                        html.Div([
                                            html.Button('Filter',
                                                        id='filter_depth_button',
                                                        style={'margin-right':'5px'}
                                                        ),
                                            html.Button('Reset',
                                                        id='reset_depth_button',
                                                        )
                                            ],
                                            style={'display': 'flex',
                                                   'flexDirection': 'row'
                                                   }
                                        ),
                                        ],
                                        style={'width': '100%',
                                               'display': 'flex',
                                               'flexDirection': 'column',
                                               'alignItems': 'center'
                                               }
                                    ),
                                    ],
                                    style={'display': 'flex',
                                           'flexDirection': 'row',
                                           }
                                ),
                                html.Hr(),
                                dcc.Graph(id='picker_graph',
                                          style={'height': '80vh'}
                                ),
                                ],
                                style={'width':'95%',
                                       'margin': '0 auto', 
                                       }
                            ),
                            html.Div(
                                children=[
                                    html.Div('Downsample large datasets to monthly interval to improve performance (default is True): ',
                                            style={'margin-bottom':'0',
                                                   'margin-top':'0',
                                                   'margin-right':'10px'}),
                                    daq.BooleanSwitch(
                                        on=True,
                                        id='resample_switch',
                                        style={'margin-top':'0'}),
                                ],
                                style={'text-align':'center',
                                       'display':'flex',
                                       'flex-direction':'row'}
                            ),
                        ],
                ),
                dcc.Tab(label='View and Check Selected Wells',
                        children=[
                            html.Div([
                                html.Hr(),
                                html.Div([
                                    html.Div(
                                        children=[
                                            html.Ul([
                                                html.Li("Click on legend entries to remove wells, or use the lasso tool to select a subset."),
                                                html.Li("Activate the switch to view lasso-selected subset."),
                                                html.Li("Removed time series will also be excluded from the downloaded wells."),
                                            ], style={"margin": "0", "padding-left": "20px", "line-height": "1.5"})
                                        ],
                                        style={'display':'flex', 
                                               'width':'35%',
                                               'flexDirection':'column', 
                                               #'alignItems':'center',
                                               'margin-right':'15px',
                                               'border': '2px solid #ccc',  # Box border
                                               'border-radius': '5px',  # Rounded corners
                                               'background-color': '#f9f9f9',  # Light background color
                                               'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',  # Subtle shadow
                                               'padding': '10px'}
                                    ),
                                    html.Div(
                                        [
                                        html.Div('Slide to view selected subset',
                                                 style={'textAlign': 'center',  
                                                        'width': '100%'     
                                                        }
                                                 ),
                                        daq.BooleanSwitch(id='selected_view_switch',
                                                          style={'margin': '0 auto'  
                                                                 }
                                                          ),
                                        ],
                                        style={'display':'flex', 
                                               'flexDirection':'column', 
                                               'alignItems':'center',
                                               'margin-right':'15px',
                                               'border': '2px solid #ccc',  # Box border
                                               'border-radius': '5px',  # Rounded corners
                                               'background-color': '#f9f9f9',  # Light background color
                                               'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',  # Subtle shadow
                                               'padding': '10px'
                                               }
                                    ),
                                    html.Div(
                                        [
                                        html.Div('Slide to filter by 3-mile radius.'),
                                        html.Div('Returns largest subset of wells.'),
                                        daq.BooleanSwitch(
                                            on=False,
                                            id='filter_by_radius_slider',
                                            style={'margin-top':'0',
                                                  })
                                        ],
                                        style={'border': '2px solid #ccc',  # Box border
                                               'border-radius': '5px',  # Rounded corners
                                               'background-color': '#f9f9f9',  # Light background color
                                               'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',  # Subtle shadow
                                               'padding': '15px',
                                               'margin-right':'15px'
                                               }
                                    ),
                                    html.Div(
                                        [
                                        html.Button("Download Filtered Well Data",
                                                    id='export_button',
                                                    ),
                                        ],
                                        style={'border': '2px solid #ccc',  # Box border
                                               'border-radius': '5px',  # Rounded corners
                                               'background-color': '#f9f9f9',  # Light background color
                                               'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',  # Subtle shadow
                                               'padding': '15px'}
                                    ),
                                    
                                    # 3D view launch button
                                    html.Div(
                                        [
                                        html.Button("Open 3D Well View",
                                                    id='view_3d_button',
                                                    ),
                                        ],
                                        style={'display':'flex', 
                                                'flexDirection':'column', 
                                                'border': '2px solid #ccc',  # Box border
                                                'border-radius': '5px',  # Rounded corners
                                                'background-color': '#f9f9f9',  # Light background color
                                                'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',  # Subtle shadow
                                                'padding': '10px',
                                                'margin-left':'15px',
                                                'width':'9%'}
                                    ),
                                    ],
                                    style={'display': 'flex',
                                           'flexDirection': 'row',
                                           'alignItems': 'center'
                                    },
                                ),
                                dcc.Download(id="download-data"),
                                html.Hr(),
                                # Figure to show subset of selected wells
                                dcc.Graph(id='selected_well_graph',
                                          style={'height': '60vh'}
                                ),
                                html.Hr(),
                                # Subsidence figure and mapbox plot
                                html.Div(
                                    id='subplot_container',
                                    children=[
                                        dcc.Graph(id='site_subsidence_graph',
                                                  style={'height': '60vh',
                                                         'margin': '0', 
                                                         'padding': '0'
                                                         }
                                        ),
                                        dcc.Graph(id='selected_well_map',
                                                  style={'height': '60vh',
                                                         #'margin-right': '10px' 
                                                         }
                                        ),
                                    ],
                                    style={'display': 'grid',
                                           'grid-template-columns': 'minmax(300px, 1fr) minmax(300px, 1fr)',  # Ensures columns stay balanced, with a minimum size
                                           'grid-gap': '10px',  # Spacing between grid items
                                           'width': '100%',  # Full width of the parent container
                                           'overflow': 'hidden',  # Prevents horizontal overflow
                                           'box-sizing': 'border-box'  # Ensures padding/borders are included in dimensions
                                    }
                                ),
                                ],
                                style={'width':'95%',
                                       'margin': '0 auto', 
                                       'margin-bottom':'200px'
                                       }
                            )  
                        ]
                ),
                dcc.Tab(label='View Tabular Data',
                        children=[
                            html.Div(id='table_container',
                                     style={'border': '2px solid #ccc',  # Box border
                                            'border-radius': '5px',  # Rounded corners
                                            'background-color': '#f9f9f9',  # Light background color
                                            'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',  # Subtle shadow
                                            'padding': '15px',
                                            'width':'95%',
                                            'margin': '0 auto', 
                                     }
                            )
                        ],
                ),
               
            ]
        ),
        # Data storage containers
        dcc.Store(id='downloadable-wells', data=[]),
        dcc.Store(id='selected-wells', data=[]),
        dcc.Store(id='color_values',data=[]),
        dcc.Store(id='selectedData',data=[]),
        dcc.Store(id='selected_IDs',data=[]),
        dcc.Store(id='site_data',data=[]),
        
        # 3D view modal
        dbc.Modal(id='view_3d_modal',
                  fullscreen=True,
                  children=[
                      dbc.ModalHeader("Current Wells Selected for Download - Click the 'X' or press esc to close"),
                      dbc.ModalBody(
                          dcc.Graph(id='view_3d_graph',
                                    style={'height': '80vh',
                                           'width':'160vh'}
                                    ),
                          className="align-self-center",
                          ),
                      ]
                  ),
        ],
)

# ---------------------------
# Update the dropdown options
# ---------------------------
@app.callback(Output("site_picker_dd","options"),
              Output("site_picker_dd","value"),
              Input("prop-118-slider",'on')
              )
def update_dropdown_items(on):
    if on:
        paths = glob.glob(f"{base_path}/assets/county_centroids/*")
        paths = [x.split("centroid_")[1].split(".")[0] for x in paths]
        paths = list(set(paths))
        return paths, paths[0]
    else:
        return gdfsites['Name'].unique(),gdfsites['Name'].unique()[0]

# -----------------------------------
# Load the data for the selected site
# ------------------------------------
@app.callback(Output('site_data','data'),
              Input('site_picker_dd', 'value'),
              State("prop-118-slider",'on'),
              )
def load_initial_data(site,county):
    global site_data, filtered_id_all, final_wells_gdf
    site_data, filtered_id_all, final_wells_gdf = select_site(site,
                                                              county_filter=county)
    
    return site_data.head(5).to_dict()

# ----------------------------------
# Update the well picker figure
# Also will reset the boolean switch
# ----------------------------------
@app.callback(Output('picker_graph', 'figure'),
              Input('site_data', 'data'),
              Input('resample_switch','on'),
              State('site_picker_dd', 'value'),
              State("prop-118-slider",'on'),
              )
def render_selected_site_figure(site_data_trigger,resample,site,county):

    fig = go.Figure()
    colorscale = 'Turbo'
    # Create the main figure with all SWNs
    for i, swn in enumerate(filtered_id_all):
        color_value = i / (len(filtered_id_all) - 1)
        swn_data = site_data[site_data['ID_ALL']==swn][['Date','wse']]

        # Fetch additional attributes from `final_wells_gdf` for each well
        well_info = final_wells_gdf[final_wells_gdf['ID_ALL'] == swn].iloc[0]
        well_depth = well_info.get('WELL_DEPTH', 'N/A')
        top_prf_bg = well_info.get('TOP_PRF_BG', 'N/A')
        bot_prf_bg = well_info.get('BOT_PRF_BG', 'N/A')
        
        # Grab start and end dates
        # Input the start and end date of the measured data at each site
        s = site_data.set_index('Date')
        # Catching NaT error
        try:
            min_date = s.loc[s['ID_ALL']==swn].index.min().normalize()
        except:
            min_date = s.loc[s['ID_ALL']==swn].index.min()
        try:
            max_date = s.loc[s['ID_ALL']==swn].index.max().normalize()
        except:
            max_date = s.loc[s['ID_ALL']==swn].index.max()
        
        # Define hover template to display only the specific trace information
        hover_template = (
            f"ID: {swn}<br>"
            f"Depth: {well_depth}<br>Top Perf: {top_prf_bg}<br>Bot Perf: {bot_prf_bg}<br>Start Date: {min_date}<br>End Date: {max_date}"
        )  
        
        swn_data = swn_data.set_index(pd.to_datetime(swn_data['Date']))
        
        if resample:
            if len(swn_data) > 1000:
                swn_data = swn_data.resample('ME').mean()
        
        fig.add_trace(go.Scatter(
            x=swn_data.index,
            y=swn_data['wse'],
            mode='lines+markers',
            name=f'ID_ALL: {swn}',
            line=dict(color=px.colors.sample_colorscale(
                colorscale, color_value)[0]),
            text=hover_template,  # Add the hover text here
            )
        )

    if len(site_data) == 0:
        title = f'Site data not found for {site}'
    else:
        title = f'Continuous Water Levels for ID_ALL Within 3 Miles of {site}'

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='wse',
        legend_title='ID_ALLs',
        hovermode='closest',
        template='plotly_white',
        dragmode='lasso',
        modebar_add=['select', 'lasso2d', 'zoom', 'pan']
    )
    
    return fig

# ----------------------------------------------
# Rreset dcc.Store data when the site is changed
# ----------------------------------------------
@app.callback(Output('selected-wells','data',
                     allow_duplicate=True),
              Output('color_values','data',
                     allow_duplicate=True),
              Output('selected_IDs','data',
                     allow_duplicate=True),
              Output('selectedData', 'data',
                     allow_duplicate=True),
              Output('selected_view_switch','on'),
              Input('picker_graph','selectedData'),
              prevent_initial_call=True
              )
def reset_all_stored_data(trigger):
    return [],[],[],[],False

# -----------------------
# Store the selected data
# -----------------------
@app.callback(Output('selectedData','data'),
              Input('picker_graph','selectedData'),
              prevent_initial_call=True
              )
def updated_selected_data(dat):
    return dat

# ----------------------------------------------------
# Update the selected data based on the boolean switch
# ----------------------------------------------------
@app.callback(Output('selectedData','data',
                     allow_duplicate=True),
              Input('selected_view_switch','on'),
              State('picker_graph','selectedData'),
              State('selected_well_graph','selectedData'),
              prevent_initial_call=True
              )
def switch_to_show_well_subset(switch_state,dat,dat_subset):
    
    if dat_subset == None:
        return dat
    else:
        if switch_state:
            if dat_subset is None:
                return dat
            else:
                return dat_subset
        else:
            return dat

# -------------------------------------------------
# Update the selected data based on depth filtering
# -------------------------------------------------
@app.callback(Output('site_data','data',
                     allow_duplicate=True),
              Input('filter_depth_button','n_clicks'),
              State('min_input','value'),
              State('max_input','value'),
              State('site_picker_dd', 'value'),
              State("prop-118-slider",'on'),
              prevent_initial_call=True
              )

def filter_by_depth(clicked,min_depth,max_depth,site,county):
    global site_data, filtered_id_all, final_wells_gdf
    
    # First reload the data
    site_data, filtered_id_all, final_wells_gdf = select_site(site,
                                                              county_filter=county)
    
    # Catch some edge cases
    # Set to very low or very high if nothing is selected
    if min_depth == None or min_depth == '':
        min_depth = -10000
    
    if max_depth == None or max_depth == '':
        max_depth = 10000
    
    # Then slice by depth
    site_data = site_data.loc[(site_data['WELL_DEPTH']>=float(min_depth))&(site_data['WELL_DEPTH']<=float(max_depth))]
    
    return site_data.head(5).to_dict()

# ------------------------------
# Reset the data depth filtering
# ------------------------------
@app.callback(Output('site_data','data',
                     allow_duplicate=True),
              Output('min_input','value'),
              Output('max_input','value'),
              Input('reset_depth_button','n_clicks'),
              State('site_picker_dd', 'value'),
              State("prop-118-slider",'on'),
              prevent_initial_call=True
              )
def reset_site_data(cicked,site,county):
    global site_data, filtered_id_all, final_wells_gdf
    
    # First reload the data
    site_data, filtered_id_all, final_wells_gdf = select_site(site,
                                                              county_filter=county)
    
    return site_data.head(5).to_dict(),None,None

# -------------------------------------------------------------------------
# Render the selected wells plot
# This will also handle the secondary filtering of data and then update all
# relevant dcc.Store items to refelct the new subset of data.
# -------------------------------------------------------------------------
@app.callback(Output('selected_well_graph','figure'),
              Output('downloadable-wells','data'),
              Output('selected-wells','data'),
              Output('color_values','data'),
              Output('selected_IDs','data'),
              Input('selectedData', 'data'),
              Input('resample_switch','on'),
              State('site_picker_dd', 'value'),
              State('picker_graph','selectedData'),
              State('selected_IDs','data'),
              State('selected_view_switch','on'),
              prevent_initial_call=True
              )
def render_selected_wells_figure(selectedData,resample,site,selectedData_ref,selected_ids,switch_state):
    global site_data, filtered_id_all, final_wells_gdf
            
    # If no data --> return nothing
    if selectedData is None:
        return go.Figure(), [], [], [], []
    # If no data --> return nothing
    if len(selectedData) == 0:
        return go.Figure(), [], [], [], []
    
    # If first rendering --> Keep all the IDs
    # Or if the switch is not active --> Keep all the IDs
    # Else, use the subset in order to find the new lines
    if len(selected_ids) == 0:
        IDs = filtered_id_all
    elif not switch_state:
        IDs = filtered_id_all
    else:
        IDs = selected_ids
    
    # Extract selected data for plotting
    selected_info = []
    for pt in selectedData['points']:
        curve_number = pt['curveNumber']
        # Get the corresponding ID_ALL using the index from filtered_id_all
        selected_info.append({
            'ID_ALL': IDs[curve_number],  # Correct mapping from curveNumber
            'x': pt['x'],
            'y': pt['y']
        })

    # Create DataFrame and figure from the selected points
    selected_df = pd.DataFrame(selected_info, 
                               columns=['ID_ALL', 'Date', 'wse']
                               )
    
    selected_wells = selected_df['ID_ALL'].unique()
    
    fig = go.Figure()
    colorscale = 'Turbo'
    
    if len(selected_wells) == 1:
        color_values = [0.1]
    else:
        color_values = [i / (len(selected_wells) - 1) for i in range(len(selected_wells))]
                
    for i, swn in enumerate(selected_wells):
        swn_data = site_data[site_data['ID_ALL'] == swn]

        swn_data = swn_data.set_index(pd.to_datetime(swn_data['Date']))[['Date','wse']]
        
        if resample:
            if len(swn_data) > 1000:
                swn_data = swn_data.resample('ME').mean()
        
        # Fetch well attributes from final_wells_gdf for additional info
        well_info = final_wells_gdf[final_wells_gdf['ID_ALL'] == swn].iloc[0]
        well_depth = well_info.get('WELL_DEPTH', 'N/A')
        top_prf_bg = well_info.get('TOP_PRF_BG', 'N/A')
        bot_prf_bg = well_info.get('BOT_PRF_BG', 'N/A')

        # Grab start and end dates
        # Input the start and end date of the measured data at each site
        s = site_data.set_index('Date')
        # Catching NaT error
        try:
            min_date = s.loc[s['ID_ALL']==swn].index.min().normalize()
        except:
            min_date = s.loc[s['ID_ALL']==swn].index.min()
        try:
            max_date = s.loc[s['ID_ALL']==swn].index.max().normalize()
        except:
            max_date = s.loc[s['ID_ALL']==swn].index.max()
        
        # Define hover template to display only the specific trace information
        hover_template = (
            f"ID: {swn}<br>"
            f"Depth: {well_depth}<br>Top Perf: {top_prf_bg}<br>Bot Perf: {bot_prf_bg}<br>Start Date: {min_date}<br>End Date: {max_date}"
        )

        fig.add_trace(go.Scatter(
            x=swn_data.index,
            y=swn_data['wse'],
            mode='lines+markers',
            name=f'ID_ALL: {swn}',
            line=dict(color=px.colors.sample_colorscale(colorscale, color_values[i])[0]),
            text=hover_template
            )
        )

    fig.update_layout(
        title=f'Selected Data Points for site {site}',
        xaxis_title='Date',
        yaxis_title='wse',
        legend_title='ID_ALLs',
        template='plotly_white',
        dragmode='lasso',
        modebar_add=['select', 'lasso2d', 'zoom', 'pan'],
        hovermode='closest',  # Show hover info only for the closest point
        #legend_itemdoubleclick=False,
        paper_bgcolor='rgb(230, 230, 230)',  # Slightly darker gray for the figure
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # If the switch is active, return selected ids to 'selected_IDs'
    # Else, return all the selected wells
    if switch_state:
        return fig,selected_wells,selected_wells,color_values,selected_ids
    else:
        return fig,selected_wells,selected_wells,color_values,selected_wells

# --------------------------------------
# Render selected site subsidence figure
# --------------------------------------
@app.callback(Output('site_subsidence_graph','figure'),
              Input('site_picker_dd', 'value'),
              )
def render_subsidence_figure(site):
    global site_subsidence_data
    
    # Grab subsidence data for specified site
    if site == 'T_88':
        t = 'T88'
    else:
        t = site
    dat = site_subsidence_data.loc[site_subsidence_data['Site']==t]
    
    fig = go.Figure()
    # If no subsidence data for site, return empty figure
    if len(dat) == 0:
        fig.update_layout(
            paper_bgcolor='rgb(230, 230, 230)',
            title='No Subsidence Data Available'
            )
        return fig
    
    # Format dat
    dat = dat.set_index(pd.to_datetime(dat['FINAL LEVELING DATA']))
    dat = dat.sort_index()
    
    fig.add_trace(
        go.Scatter(
            x=dat.index,
            y=dat['Sub_ft'],
            mode='lines+markers',
        )
    )
    
    # Reverse y-axis
    fig.update_yaxes(autorange="reversed")

    # Layout
    fig.update_layout(
        title=f'Subsidence Data for site {site}',
        xaxis_title='Date',
        yaxis_title='Subsidence (ft)',
        template='plotly_white',
        hovermode='closest', 
        legend_itemdoubleclick=False,
        paper_bgcolor='rgb(230, 230, 230)',
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig
    
# ------------------------------
# Render selected wells map-plot
# ------------------------------
@app.callback(Output('selected_well_map', 'figure'),
              Input('downloadable-wells', 'data'),
              State('site_picker_dd', 'value'),
              State('selected-wells', 'data'),
              State('color_values','data'),
              prevent_initial_call=True
              )
def render_map_figure(downloadable_wells,site,selectedWellNames,color_values):
    global site_data, filtered_id_all, final_wells_gdf

    if downloadable_wells is None or len(downloadable_wells) == 0:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgb(230, 230, 230)',
            )
        return fig
    
    # Create the selected wells gdf subset
    # And add selected site point
    swns = downloadable_wells
    dat = final_wells_gdf.loc[final_wells_gdf['ID_ALL'].isin(swns)]
    dat = dat.to_crs(crs=4326)
    dat['x'] = dat.geometry.x
    dat['y'] = dat.geometry.y
    dat = dat[['x', 'y', 'ID_ALL', 'WELL_DEPTH', 'TOP_PRF_BG', 'BOT_PRF_BG']]
    
    # Create single entry df for selected site
    site_point = gdfsites[gdfsites['Name'] == site].copy()
    site_point = site_point.to_crs(crs=4326)
    site_point['x'] = site_point.geometry.x
    site_point['y'] = site_point.geometry.y
    site_point = site_point[['x', 'y']]
    site_point['ID_ALL'] = [f'Selected Site: {site}']
    site_point['WELL_DEPTH'] = 'N/A'
    site_point['TOP_PRF_BG'] = 'N/A'
    site_point['BOT_PRF_BG'] = 'N/A'
    # Assign a color to the site point that is unlikely to be shared by any others
    site_point['color_value'] = [0.97564352]
    
    # Create a colorscale relative to all selected wells
    indices = [selectedWellNames.index(entry) for entry in downloadable_wells]
    dat['color_value'] = [color_values[i] for i in indices]
    
    # Concat the downloadable well sites with the site of interest point
    dat = pd.concat([dat, site_point])
    
    # Assign colors in the turbo colorscale to each numeric value
    colorscale = 'Turbo'
    def get_color(value):
        return px.colors.sample_colorscale(colorscale, [value])[0]
    dat['color'] = dat['color_value'].apply(lambda x: get_color(x))
    
    # Plot selected wells
    fig = px.scatter_mapbox(
        dat,
        lat='y',
        lon='x',
        color='color',
        zoom=11,
        color_discrete_sequence=dat['color'].tolist(),
        hover_name='ID_ALL',
        hover_data={
            'WELL_DEPTH': True,
            'TOP_PRF_BG': True,
            'BOT_PRF_BG': True,
            'color': False,  # Hide color from hover data
            'x': False,  # Hide x-coord from hover data
            'y': False   # Hide y-coord from hover data
            }
    )
    
    # Name the traces based on ID_ALL
    newnames = dict(zip(dat['color'].values, dat['ID_ALL'].values))
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name]))
    
    # Plot formatting
    fig.update_layout(
        title='Plotted wells represent those that will be downloaded.',
        mapbox_style="carto-positron",
        legend_itemdoubleclick=False,
        showlegend=False,
        paper_bgcolor='rgb(230, 230, 230)',  # Slightly darker gray for the figure
        #margin=dict(l=10, r=10, t=40, b=10),
    )

    fig.update_traces(
        marker=dict(size=12),
    )

    return fig

# -------------------------------------------------
# Edit the downloaded wells based on legend entries
# -------------------------------------------------
@app.callback(Output('downloadable-wells', 'data',
                     allow_duplicate=True),
              Input('selected_well_graph', 'restyleData'),
              Input('downloadable-wells', 'data'),
              State('selected-wells', 'data'),
              State('selected_well_graph','figure'),
              prevent_initial_call=True
              )
def update_wells_with_legend(restyleData, current_wells, selected_wells, fig):
    if restyleData is None:
        return current_wells  # No change if nothing was changed
    
    # ------------------------------------------------------------
    # Check if the visibility of the clicked trace is 'legendonly'
    # --> 'legendonly' means trace is not currently rendered
    # ------------------------------------------------------------
    updated_wells = [
        selected_wells[i] for i, trace in enumerate(fig['data'])
        if trace.get('visible', True) != 'legendonly'  # Include if visible or not explicitly set to 'legendonly'
    ]
            
    updated_wells.sort(key=lambda x: selected_wells.index(x))
    
    return updated_wells

# ------------------------------------------------
# Filter by 3-mile radius
# Turn visibility for wells outside radius off....
# ------------------------------------------------
@app.callback(Output('downloadable-wells','data',
                     allow_duplicate=True),
              Output('selected_well_graph','figure',
                     allow_duplicate=True),
              Input('filter_by_radius_slider','on'),
              State('downloadable-wells', 'data'),
              State('selected-wells', 'data'),
              State('selected_well_graph','figure'),
              prevent_initial_call=True
              )
def filter_wells_by_radius(on,downloadable_wells,selected_wells,fig):
    global final_wells_gdf
    
    if on:
        swns = downloadable_wells
        dat = final_wells_gdf.loc[final_wells_gdf['ID_ALL'].isin(swns)]
        
        # Filter the subset by radius
        radius = 3 * 5280
        dat = filter_within_radius_networkx(dat,radius)        
        
        for trace in fig['data']:
            if trace['name'].split(' ')[1] not in dat['ID_ALL'].values:
                if trace.get('visible', True) == 'legendonly':
                    pass
                else:
                    trace['visible'] = 'legendonly'
                    trace['filter_slider'] = True
                
        # Return ID_ALL as downloadable wells
        return dat['ID_ALL'].values, fig

    else:
        for trace in fig['data']:
            if trace.get('filter_slider', False) == True:
                trace['visible'] = True
                trace['filter_slider'] = False
                    
        return selected_wells, fig

# -----------------------
# Download selected wells
# -----------------------
@app.callback(Output('download-data', 'data'),
              Input('export_button', 'n_clicks'),
              State('downloadable-wells', 'data'),
              State('site_picker_dd', 'value'),
              prevent_initial_call=True
              )
def export_selected_data(n_clicks, downloadable_wells, site):
    global site_data, filtered_id_all, final_wells_gdf

    if n_clicks == 0 or not downloadable_wells:
        return None  # If no data is available for export
    
    # Specify the folder for exported files
    export_folder = f'{base_path}/export_filtered_data'  # Replace with the actual folder path

    # Ensure the folder exists, creating it if necessary
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    
    # Extract selected data for export
    selected_info = site_data.loc[site_data['ID_ALL'].isin(
        downloadable_wells), :]
    selected_df = selected_info[['ID_ALL', 'Date', 'wse','wl_bgs']] #'WCRLinks']]
    selected_df = selected_df.reset_index(drop=True)
    #selected_df.columns = ['ID_ALL', 'Date', 'wse','wl_bgs']
    
    # Generate the file name based on the site name
    file_name = f"filtered_selected_data_{site}.csv"
    file_path = os.path.join(export_folder, file_name)
    
    # Save the DataFrame to CSV in the export folder
    selected_df.to_csv(file_path, index=False)
    
    # Provide a downloadable link for the saved file
    return dcc.send_file(file_path)

# -------------------------------------------------------
# Open a 3D view of the downloadable wells in a dbc.Modal
# -------------------------------------------------------
@app.callback(Output('view_3d_modal','is_open'),
              Input('view_3d_button','n_clicks'),
              State('view_3d_modal','is_open'),
              )
def open_3D_view(clicked,is_open):
    if clicked:
        return not is_open
    return is_open

# ----------------------------
# Render the 3D well view plot
# ----------------------------
@app.callback(Output('view_3d_graph','figure'),
              Input('view_3d_button','n_clicks'),
              State('downloadable-wells','data'),
              )
def render_3D_plot(clicked,downloadable_wells):
    global final_wells_gdf
    
    # Create the selected wells gdf subset
    swns = downloadable_wells
    dat = final_wells_gdf.loc[final_wells_gdf['ID_ALL'].isin(swns)]
    dat = dat.to_crs(crs=4326)
    dat['x'] = dat.geometry.x
    dat['y'] = dat.geometry.y
    dat = dat[['x', 'y', 'ID_ALL', 'WELL_DEPTH', 'TOP_PRF_BG', 'BOT_PRF_BG']]
    fig = go.Figure()
    for idx,well in enumerate(dat['ID_ALL']):
        # Skip plotting wells with no depth info
        if dat['WELL_DEPTH'].iloc[idx] == -9999:
            pass
        else:
            z = [0,dat['WELL_DEPTH'].iloc[idx] * -1]
            x = [dat['x'].iloc[idx]] * 2
            y = [dat['y'].iloc[idx]] * 2
            
            # Plot with no screen depth info
            if dat['TOP_PRF_BG'].iloc[idx] == -9999 or dat['BOT_PRF_BG'].iloc[idx] == -9999:
                # Plot well
                fig.add_trace(
                    go.Scatter3d(x=x,
                                 y=y, 
                                 z=z,
                                 name=well,
                                 line=dict(
                                     width=4
                                     ),
                                 marker=dict(
                                     size=0
                                     )
                        )
                    )
                
            # Plot with screen depth information
            else:
                screen_interval = [-1 * dat['TOP_PRF_BG'].iloc[idx],
                                   -1 * dat['BOT_PRF_BG'].iloc[idx]]
                # Plot well
                fig.add_trace(
                    go.Scatter3d(x=x,
                                  y=y, 
                                  z=z,
                                  name=well,
                                  line=dict(
                                      width=4
                                      ),
                                  marker=dict(
                                      size=0
                                      )
                        )
                    )
                # Plot screen interval
                fig.add_trace(
                    go.Scatter3d(x=x,
                                 y=y, 
                                 z=screen_interval,
                                 line=dict(
                                     color='black',
                                     width=6,
                                     dash='dash'
                                     ),
                                 marker=dict(
                                     size=0
                                     ),
                                 showlegend=False
                        )
                    )
            
    fig.update_layout(
        scene=dict(
            zaxis=dict(
                title='Depth (bgs)'
                )
            )
        )
    
    return fig
                

# -------------------------------
# Create the tabular data section
# -------------------------------
@app.callback(Output('table_container','children'),
              Input('downloadable-wells', 'data'),
              Input('site_picker_dd', 'value'),
              Input('selectedData','data'),
              State('selected-wells', 'data'),
              prevent_initial_call=True
              )
def update_table(downloadable_wells,site,temp,selectedWellNames):
    global site_data, filtered_id_all, final_wells_gdf

    if downloadable_wells is None or len(downloadable_wells) == 0:
        swns = filtered_id_all
    else:
        swns = downloadable_wells
        
    # Create the selected wells gdf subset
    dat = final_wells_gdf.loc[final_wells_gdf['ID_ALL'].isin(swns)]
    # Input the start and end date of the measured data at each site
    min_dates = []
    max_dates = []
    s = site_data.set_index('Date')
    for well in dat['ID_ALL']:
        # Catch NaT error
        try:
            min_dates.append(s.loc[s['ID_ALL']==well].index.min().normalize())
        except:
            min_dates.append(s.loc[s['ID_ALL']==well].index.min())
        try:
            max_dates.append(s.loc[s['ID_ALL']==well].index.max().normalize())
        except:
            max_dates.append(s.loc[s['ID_ALL']==well].index.max())
        
    dat = dat.to_crs(crs=4326)
    dat['x'] = dat.geometry.x
    dat['y'] = dat.geometry.y
    dat['Max Date'] = max_dates
    dat['Min Date'] = min_dates
    dat = dat[['x', 'y', 'ID_ALL', 'WELL_DEPTH', 'TOP_PRF_BG', 'BOT_PRF_BG','Min Date','Max Date']]
    
    dat = dat.astype('object')
    
    dat[dat==-9999] = 'NA'
    
    # Build the table
    table = html.Table(
        style={'width': '100%', 'borderCollapse': 'collapse', 'margin': '20px 0'},
        children=[
        # Header
        html.Thead(
            html.Tr([html.Th(col,style={'border': '1px solid black', 'padding': '8px', 'backgroundColor': '#f2f2f2'}) for col in dat.columns])
        ),
        # Body
        html.Tbody([
            html.Tr([
                html.Td(dat.iloc[i][col],style={'border': '1px solid black', 'padding': '8px', 'textAlign': 'center'}) for col in dat.columns
                ]
            ) for i in range(len(dat))
            ]
        )
        ]
    )      
    
    return table

# ------------------------------------
# Start the dashboard
# Go to localhost:8050 to view
# ------------------------------------
if __name__ == '__main__':
    app.run(debug=False, port=8050)
