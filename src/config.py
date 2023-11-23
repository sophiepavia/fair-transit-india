# Objective function (options: 'MaxMin', 'Utilitarian', 'LinearCombo')
OBJ = 'Utilitarian'

# Priority type (options: 'EP', 'P')
PRIORITY = "EP"

# Algorithm parameters
ALPHA = 2          # Tolerance parameter
GAMMA = 0.01       # Scaling parameter (options: 0.01, 0.3, 0.5, 0.7)

# Distance calculation type (options: 'center', 'osrm')
DISTANCE_TYPE = 'center' 

# Network and demand data
CITY_EDGE_LIST = 'india_edge_list'   # Edge list file for the city
CITY_OD = 'india_od'                # Origin-Destination file for the city

# Run results identifiers
RUN = 'r0'
RESULT_FOLDER = 'india_center'
