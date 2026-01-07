"""
Handball Court Geometry - Single Source of Truth

This module defines the official IHF handball court geometry with correct D-shaped arcs
for the 6m and 9m lines. All other modules should import from this file.

Coordinate System:
- Origin (0,0) is at the Left Goal Line, Near Sideline Corner
- X-axis: 0 to 40 meters (court length)
- Y-axis: 0 to 20 meters (court width)

Key Geometry:
- Court: 40m x 20m
- Goal: 3m wide, centered at Y=10m
- Goal posts: Y=8.5m (near) and Y=11.5m (far)
- 6m line: D-shaped arc centered at goal posts
  - Meets goal line at Y=2.5m and Y=17.5m (6m + 1.5m from center)
  - 3m flat section at Y=8.5m to Y=11.5m
- 9m line: D-shaped arc centered at goal posts
  - Meets sideline at X=2.958m (left) or X=37.042m (right)
  - 3m flat section at Y=8.5m to Y=11.5m
"""

from typing import List, Tuple, Dict
import numpy as np

# Court Dimensions (meters)
COURT_LENGTH = 40.0
COURT_WIDTH = 20.0
GOAL_WIDTH = 3.0
GOAL_CENTER_Y = 10.0  # Center of court width
GOAL_POST_NEAR_Y = 8.5  # Near post (closer to Y=0)
GOAL_POST_FAR_Y = 11.5  # Far post (closer to Y=20)

# Arc Radii
RADIUS_6M = 6.0
RADIUS_9M = 9.0

# Goal post positions
GOAL_POST_LEFT_NEAR = (0.0, GOAL_POST_NEAR_Y)
GOAL_POST_LEFT_FAR = (0.0, GOAL_POST_FAR_Y)
GOAL_POST_RIGHT_NEAR = (COURT_LENGTH, GOAL_POST_NEAR_Y)
GOAL_POST_RIGHT_FAR = (COURT_LENGTH, GOAL_POST_FAR_Y)

# 6m Line Geometry
# D-shaped arc: quarter circle - 3m straight - quarter circle
# Centers at goal posts: (0, 8.5) and (0, 11.5) for left goal
SIX_M_ARC_CENTER_LEFT_NEAR = (0.0, GOAL_POST_NEAR_Y)
SIX_M_ARC_CENTER_LEFT_FAR = (0.0, GOAL_POST_FAR_Y)
SIX_M_ARC_CENTER_RIGHT_NEAR = (COURT_LENGTH, GOAL_POST_NEAR_Y)
SIX_M_ARC_CENTER_RIGHT_FAR = (COURT_LENGTH, GOAL_POST_FAR_Y)

# 6m line meets goal line at: 6m radius + 1.5m from center = 7.5m from center
# Center is at Y=10, so: 10 - 7.5 = 2.5 (near) and 10 + 7.5 = 17.5 (far)
SIX_M_GOAL_LINE_NEAR_Y = 2.5  # 6m + 1.5m from center
SIX_M_GOAL_LINE_FAR_Y = 17.5

# 6m line straight section (3m wide, parallel to goal line)
# At X = 6.0m from goal line
SIX_M_STRAIGHT_X_LEFT = 6.0
SIX_M_STRAIGHT_X_RIGHT = 34.0
SIX_M_STRAIGHT_Y_START = GOAL_POST_NEAR_Y  # 8.5
SIX_M_STRAIGHT_Y_END = GOAL_POST_FAR_Y    # 11.5

# 9m Line Geometry
# D-shaped arc: quarter circle - 3m straight - quarter circle
# Centers at goal posts
NINE_M_ARC_CENTER_LEFT_NEAR = (0.0, GOAL_POST_NEAR_Y)
NINE_M_ARC_CENTER_LEFT_FAR = (0.0, GOAL_POST_FAR_Y)
NINE_M_ARC_CENTER_RIGHT_NEAR = (COURT_LENGTH, GOAL_POST_NEAR_Y)
NINE_M_ARC_CENTER_RIGHT_FAR = (COURT_LENGTH, GOAL_POST_FAR_Y)

# 9m line meets sideline where arc (9m radius + 1.5m goal distance) intersects
# Calculation: sqrt(9^2 - 1.5^2) = sqrt(81 - 2.25) = sqrt(78.75) ≈ 8.874
# Distance from goal line: 9.0 - 8.874 = 0.126m? No, that's wrong.
# Actually: The arc center is at (0, 8.5) or (0, 11.5)
# The arc radius is 9m. At Y=0 (sideline), X = sqrt(9^2 - 8.5^2) = sqrt(81 - 72.25) = sqrt(8.75) ≈ 2.958
# At Y=20 (far sideline), same calculation: X = sqrt(9^2 - 8.5^2) = 2.958
NINE_M_SIDELINE_X_LEFT = 2.958  # Where 9m arc meets near/far sideline
NINE_M_SIDELINE_X_RIGHT = 37.042  # 40 - 2.958

# 9m line straight section (3m wide, parallel to goal line)
# At X = 9.0m from goal line
NINE_M_STRAIGHT_X_LEFT = 9.0
NINE_M_STRAIGHT_X_RIGHT = 31.0
NINE_M_STRAIGHT_Y_START = GOAL_POST_NEAR_Y  # 8.5
NINE_M_STRAIGHT_Y_END = GOAL_POST_FAR_Y     # 11.5

# 7m Penalty Marks
SEVEN_M_X_LEFT = 7.0
SEVEN_M_X_RIGHT = 33.0
SEVEN_M_Y = GOAL_CENTER_Y  # 10.0

# Center Line
CENTER_LINE_X = COURT_LENGTH / 2.0  # 20.0


def get_world_keypoints_origin_corner() -> Dict[str, Dict[str, float]]:
    """
    Returns world keypoints with origin at Left Goal Line, Near Sideline Corner.
    This matches the handball_cd project coordinate system.
    
    Returns:
        Dictionary mapping keypoint names to {x_m, y_m, desc}
    """
    return {
        # Corners
        "P01_L_GLC": {"x_m": 0.0, "y_m": 0.0, "desc": "Left Goal Line Corner (Near Sideline)"},
        "P02_L_GLF": {"x_m": 0.0, "y_m": 20.0, "desc": "Left Goal Line Corner (Far Sideline)"},
        "P03_R_GLC": {"x_m": 40.0, "y_m": 0.0, "desc": "Right Goal Line Corner (Near Sideline)"},
        "P04_R_GLF": {"x_m": 40.0, "y_m": 20.0, "desc": "Right Goal Line Corner (Far Sideline)"},
        
        # Goal Posts
        "P05_L_GP-N": {"x_m": 0.0, "y_m": 8.5, "desc": "Left Goal Post (Near Post)"},
        "P06_L_GP-F": {"x_m": 0.0, "y_m": 11.5, "desc": "Left Goal Post (Far Post)"},
        "P07_R_GP-N": {"x_m": 40.0, "y_m": 8.5, "desc": "Right Goal Post (Near Post)"},
        "P08_R_GP-F": {"x_m": 40.0, "y_m": 11.5, "desc": "Right Goal Post (Far Post)"},
        
        # 6m Line - Goal Line Intersections
        "P09_L_6M-GL-N": {"x_m": 0.0, "y_m": 2.5, "desc": "Left 6m Arc meets Goal Line (Near Sideline side)"},
        "P10_L_6M-GL-F": {"x_m": 0.0, "y_m": 17.5, "desc": "Left 6m Arc meets Goal Line (Far Sideline side)"},
        "P11_R_6M-GL-N": {"x_m": 40.0, "y_m": 2.5, "desc": "Right 6m Arc meets Goal Line (Near Sideline side)"},
        "P12_R_6M-GL-F": {"x_m": 40.0, "y_m": 17.5, "desc": "Right 6m Arc meets Goal Line (Far Sideline side)"},
        
        # 6m Line - Straight Section Ends
        "P13_L_6M-C": {"x_m": 6.0, "y_m": 10.0, "desc": "Left 6m Line Center (Midpoint)"},
        "P14_L_6M-Arc-N": {"x_m": 6.0, "y_m": 8.5, "desc": "Left 6m Line (Near Post side straight segment end)"},
        "P15_L_6M-Arc-F": {"x_m": 6.0, "y_m": 11.5, "desc": "Left 6m Line (Far Post side straight segment end)"},
        "P16_R_6M-C": {"x_m": 34.0, "y_m": 10.0, "desc": "Right 6m Line Center (Midpoint)"},
        "P17_R_6M-Arc-N": {"x_m": 34.0, "y_m": 8.5, "desc": "Right 6m Line (Near Post side straight segment end)"},
        "P18_R_6M-Arc-F": {"x_m": 34.0, "y_m": 11.5, "desc": "Right 6m Line (Far Post side straight segment end)"},
        
        # 9m Line - Sideline Intersections
        "P19_L_9M-NS": {"x_m": 2.958, "y_m": 0.0, "desc": "Left 9m Arc meets Near Sideline"},
        "P20_L_9M-FS": {"x_m": 2.958, "y_m": 20.0, "desc": "Left 9m Arc meets Far Sideline"},
        "P21_R_9M-NS": {"x_m": 37.042, "y_m": 0.0, "desc": "Right 9m Arc meets Near Sideline"},
        "P22_R_9M-FS": {"x_m": 37.042, "y_m": 20.0, "desc": "Right 9m Arc meets Far Sideline"},
        
        # 9m Line - Straight Section Ends
        "P23_L_9M-Arc-N": {"x_m": 9.0, "y_m": 8.5, "desc": "Left 9m Line (Near Post side straight segment end)"},
        "P24_L_9M-Arc-F": {"x_m": 9.0, "y_m": 11.5, "desc": "Left 9m Line (Far Post side straight segment end)"},
        "P25_R_9M-Arc-N": {"x_m": 31.0, "y_m": 8.5, "desc": "Right 9m Line (Near Post side straight segment end)"},
        "P26_R_9M-Arc-F": {"x_m": 31.0, "y_m": 11.5, "desc": "Right 9m Line (Far Post side straight segment end)"},
        
        # 7m Penalty Marks
        "P30_L_7M": {"x_m": 7.0, "y_m": 10.0, "desc": "Left 7m Penalty Mark (Midpoint)"},
        "P33_R_7M": {"x_m": 33.0, "y_m": 10.0, "desc": "Right 7m Penalty Mark (Midpoint)"},
        
        # Center Line
        "P27_CENTER-NS": {"x_m": 20.0, "y_m": 0.0, "desc": "Center Line (Near Sideline)"},
        "P28_CENTER-FS": {"x_m": 20.0, "y_m": 20.0, "desc": "Center Line (Far Sideline)"},
    }


def get_world_keypoints_origin_center() -> List[Tuple[float, float]]:
    """
    Returns world keypoints with origin at court center.
    X: -20 to +20 (left to right)
    Y: -10 to +10 (near to far sideline)
    
    Returns:
        List of (x, y) tuples in meters
    """
    keypoints_corner = get_world_keypoints_origin_corner()
    keypoints_center = []
    
    for name, point in keypoints_corner.items():
        # Convert from corner origin to center origin
        x_center = point["x_m"] - COURT_LENGTH / 2.0  # Shift X by -20
        y_center = point["y_m"] - COURT_WIDTH / 2.0   # Shift Y by -10
        keypoints_center.append((x_center, y_center))
    
    return keypoints_center


def get_court_vertices_center_origin() -> List[List[float]]:
    """
    Returns handball court vertices in center-origin coordinate system
    for use with roboflow/sports library CourtConfiguration.
    
    Coordinate System:
    - Origin (0,0) at court center
    - X: -20 to +20 meters (left to right)
    - Y: -10 to +10 meters (near to far sideline)
    
    Returns:
        List of [x, y] lists suitable for CourtConfiguration.vertices
    """
    keypoints = get_world_keypoints_origin_corner()
    
    # Select key vertices for homography (similar to basketball's 33 keypoints)
    # Order: corners, goal posts, 6m intersections, 9m intersections, 7m marks, center line
    vertex_names = [
        # Corners
        "P01_L_GLC", "P02_L_GLF", "P03_R_GLC", "P04_R_GLF",
        # Goal Posts
        "P05_L_GP-N", "P06_L_GP-F", "P07_R_GP-N", "P08_R_GP-F",
        # 6m line intersections with goal line
        "P09_L_6M-GL-N", "P10_L_6M-GL-F", "P11_R_6M-GL-N", "P12_R_6M-GL-F",
        # 6m line straight section ends
        "P14_L_6M-Arc-N", "P15_L_6M-Arc-F", "P17_R_6M-Arc-N", "P18_R_6M-Arc-F",
        # 9m line intersections with sideline
        "P19_L_9M-NS", "P20_L_9M-FS", "P21_R_9M-NS", "P22_R_9M-FS",
        # 9m line straight section ends
        "P23_L_9M-Arc-N", "P24_L_9M-Arc-F", "P25_R_9M-Arc-N", "P26_R_9M-Arc-F",
        # 7m penalty marks
        "P30_L_7M", "P33_R_7M",
        # Center line
        "P27_CENTER-NS", "P28_CENTER-FS",
    ]
    
    vertices = []
    for name in vertex_names:
        if name in keypoints:
            point = keypoints[name]
            # Convert to center-origin
            x_center = point["x_m"] - COURT_LENGTH / 2.0
            y_center = point["y_m"] - COURT_WIDTH / 2.0
            vertices.append([x_center, y_center])
    
    return vertices


def get_6m_arc_points(is_left: bool, num_points: int = 50) -> List[Tuple[float, float]]:
    """
    Generate points along the 6m D-shaped arc.
    
    Args:
        is_left: True for left goal, False for right goal
        num_points: Number of points to sample along the arc
        
    Returns:
        List of (x, y) tuples in meters (corner origin system)
    """
    points = []
    
    if is_left:
        center_near = SIX_M_ARC_CENTER_LEFT_NEAR  # (0, 8.5)
        center_far = SIX_M_ARC_CENTER_LEFT_FAR     # (0, 11.5)
        goal_x = 0.0
        straight_x = SIX_M_STRAIGHT_X_LEFT  # 6.0
    else:
        center_near = SIX_M_ARC_CENTER_RIGHT_NEAR  # (40, 8.5)
        center_far = SIX_M_ARC_CENTER_RIGHT_FAR   # (40, 11.5)
        goal_x = COURT_LENGTH
        straight_x = SIX_M_STRAIGHT_X_RIGHT  # 34.0
    
    # Arc 1: From goal line to straight section (near post side)
    # Start: (goal_x, 2.5), Center: center_near, End: (straight_x, 8.5)
    # Angle: -90° to 0° (left) or -90° to 180° (right)
    start_angle = -np.pi / 2
    end_angle = 0.0 if is_left else np.pi
    
    for i in range(num_points // 3):
        theta = start_angle + (end_angle - start_angle) * i / (num_points // 3)
        x = center_near[0] + RADIUS_6M * np.cos(theta)
        y = center_near[1] + RADIUS_6M * np.sin(theta)
        points.append((x, y))
    
    # Straight section: 3m wide, parallel to goal line
    for i in range(num_points // 3):
        y = GOAL_POST_NEAR_Y + (GOAL_POST_FAR_Y - GOAL_POST_NEAR_Y) * i / (num_points // 3)
        points.append((straight_x, y))
    
    # Arc 2: From straight section to goal line (far post side)
    # Start: (straight_x, 11.5), Center: center_far, End: (goal_x, 17.5)
    # Angle: 0° to 90° (left) or 180° to 90° (right)
    start_angle = 0.0 if is_left else np.pi
    end_angle = np.pi / 2
    
    for i in range(num_points // 3):
        theta = start_angle + (end_angle - start_angle) * i / (num_points // 3)
        x = center_far[0] + RADIUS_6M * np.cos(theta)
        y = center_far[1] + RADIUS_6M * np.sin(theta)
        points.append((x, y))
    
    return points


def get_9m_arc_points(is_left: bool, num_points: int = 50) -> List[Tuple[float, float]]:
    """
    Generate points along the 9m D-shaped arc.
    
    Args:
        is_left: True for left goal, False for right goal
        num_points: Number of points to sample along the arc
        
    Returns:
        List of (x, y) tuples in meters (corner origin system)
    """
    points = []
    
    if is_left:
        center_near = NINE_M_ARC_CENTER_LEFT_NEAR  # (0, 8.5)
        center_far = NINE_M_ARC_CENTER_LEFT_FAR     # (0, 11.5)
        sideline_x = NINE_M_SIDELINE_X_LEFT  # 2.958
        straight_x = NINE_M_STRAIGHT_X_LEFT  # 9.0
    else:
        center_near = NINE_M_ARC_CENTER_RIGHT_NEAR  # (40, 8.5)
        center_far = NINE_M_ARC_CENTER_RIGHT_FAR   # (40, 11.5)
        sideline_x = NINE_M_SIDELINE_X_RIGHT  # 37.042
        straight_x = NINE_M_STRAIGHT_X_RIGHT  # 31.0
    
    # Arc 1: From sideline to straight section (near post side)
    # Start: (sideline_x, 0), Center: center_near, End: (straight_x, 8.5)
    # Calculate start angle from center to sideline intersection
    dx_start = sideline_x - center_near[0]
    dy_start = 0.0 - center_near[1]
    start_angle = np.arctan2(dy_start, dx_start)
    
    # End angle: from center to straight section end
    dx_end = straight_x - center_near[0]
    dy_end = GOAL_POST_NEAR_Y - center_near[1]
    end_angle = np.arctan2(dy_end, dx_end)
    
    for i in range(num_points // 3):
        theta = start_angle + (end_angle - start_angle) * i / (num_points // 3)
        x = center_near[0] + RADIUS_9M * np.cos(theta)
        y = center_near[1] + RADIUS_9M * np.sin(theta)
        points.append((x, y))
    
    # Straight section: 3m wide, parallel to goal line
    for i in range(num_points // 3):
        y = GOAL_POST_NEAR_Y + (GOAL_POST_FAR_Y - GOAL_POST_NEAR_Y) * i / (num_points // 3)
        points.append((straight_x, y))
    
    # Arc 2: From straight section to sideline (far post side)
    # Start: (straight_x, 11.5), Center: center_far, End: (sideline_x, 20)
    dx_start = straight_x - center_far[0]
    dy_start = GOAL_POST_FAR_Y - center_far[1]
    start_angle = np.arctan2(dy_start, dx_start)
    
    dx_end = sideline_x - center_far[0]
    dy_end = COURT_WIDTH - center_far[1]
    end_angle = np.arctan2(dy_end, dx_end)
    
    for i in range(num_points // 3):
        theta = start_angle + (end_angle - start_angle) * i / (num_points // 3)
        x = center_far[0] + RADIUS_9M * np.cos(theta)
        y = center_far[1] + RADIUS_9M * np.sin(theta)
        points.append((x, y))
    
    return points

