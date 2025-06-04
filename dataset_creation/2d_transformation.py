#!/usr/bin/env python3
"""
Math-based Visual Analogy Task Generator (with Boundary Adjustment)

In this script all transformations are computed mathematically on the vertices of shapes.
Shapes are approximated as polygons via a helper function. Transformation instructions
(e.g., rotate, translate, scale, shear, and flip) are applied to these vertices using math–based
functions. The instructions are randomized and filtered based on the shape’s symmetry.
After transformation, the resulting shape (or distractor) is adjusted so that its entire
bounding box is within the drawing boundary (here, [-50,50] for both axes).
An HTML report is generated at the end.
"""

import math, random, os, json
import matplotlib.pyplot as plt
from azfuse import File
from tqdm import tqdm

### UTILITY FUNCTIONS ###

def random_color():
    """Generate a random RGB color."""
    return (random.random(), random.random(), random.random())

def random_center():
    """Generate a random center position."""
    return (random.uniform(-5, 5), random.uniform(-5, 5))

def random_size(shape_name):
    """Generate a random size for a given shape type."""
    if shape_name == 'circle':
        return random.uniform(30, 35)
    elif shape_name == 'square':
        return random.uniform(30, 35)
    elif shape_name == 'rectangle':
        return  (random.uniform(30, 35), random.uniform(20, 25))
    elif shape_name == 'triangle':
        return random.uniform(30, 35)
    elif shape_name == 'ellipse':
        return (random.uniform(30, 35), random.uniform(20, 25))
    elif shape_name in ['hexagon', 'pentagon']:
        return random.uniform(30, 35)
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

def generate_shapes():
    """Generate a list of shapes with randomized properties."""
    shapes = [
        {'name': 'circle', 'facecolor': random_color()},
        {'name': 'square', 'facecolor': random_color()},
        {'name': 'rectangle', 'facecolor': random_color()},
        {'name': 'triangle', 'facecolor': random_color()},
        {'name': 'ellipse', 'facecolor': random_color()},
        {'name': 'hexagon', 'facecolor': random_color()},
        {'name': 'pentagon', 'facecolor': random_color()},
    ]
    for shape in shapes:
        shape['edgecolor'] = 'black'
        shape['center'] = (0, 0)  # centered by default
        shape['size'] = random_size(shape['name'])
    return shapes

#########################
# MATH-BASED TRANSFORMATIONS
#########################

import numpy as np

# def rotate_shape(points, angle_deg, origin=(0,0)):
#     """
#     Rotate the shape (given as an array-like set of points) by angle_deg around origin.
#     Uses NumPy vectorized operations.
#     """
#     pts = np.asarray(points, dtype=float)  # shape (n,2)
#     origin = np.asarray(origin, dtype=float)
#     angle = np.deg2rad(angle_deg)
#     # Build the rotation matrix.
#     R = np.array([[np.cos(angle), -np.sin(angle)],
#                   [np.sin(angle),  np.cos(angle)]])
#     # Subtract origin, apply rotation, then add origin.
#     rotated = (pts - origin) @ R.T + origin
#     return np.round(rotated, 3)

def rotate_shape_around_center(points, angle_deg, center=(0,0)):
    """
    Rotate the shape (given as an array-like set of points) by angle_deg around center.
    Now properly maintains the center point.
    """
    pts = np.asarray(points, dtype=float)  # shape (n,2)
    center = np.asarray(center, dtype=float)
    angle = np.deg2rad(angle_deg)
    
    # Build the rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    
    # Subtract center, apply rotation, then add center back
    rotated = (pts - center) @ R.T + center
    
    # Calculate current centroid
    current_centroid = np.mean(rotated, axis=0)
    
    # Apply correction to maintain the original center
    correction = center - current_centroid
    rotated = rotated + correction

    # print(center, current_centroid, correction)
    
    return np.round(rotated, 3)

def scale_shape(points, scale_factor, pivot=(0,0)):
    pts = np.asarray(points, dtype=float)
    centroid = np.mean(pts, axis=0)
    # pivot = np.asarray(pivot, dtype=float)
    
    # Scale about pivot
    scaled = (pts - centroid) * scale_factor + centroid

    # print(pts, scaled)
    
    return scaled


def reflect_shape_vertically(points, axis_x=0):
    """
    Reflect the shape (array of points) vertically about the vertical line x = axis_x.
    """
    pts = np.asarray(points, dtype=float)
    pts[:, 0] = 2 * axis_x - pts[:, 0]
    return np.round(pts, 3)

def reflect_shape_horizontally(points, axis_y=0):
    """
    Reflect the shape (array of points) horizontally about the horizontal line y = axis_y.
    """
    pts = np.asarray(points, dtype=float)
    pts[:, 1] = 2 * axis_y - pts[:, 1]
    return np.round(pts, 3)

def translate_shape(points, dx, dy):
    """
    Translate the shape by (dx, dy) using vectorized addition.
    """
    pts = np.asarray(points, dtype=float)
    translated = pts + np.array([dx, dy])
    return np.round(translated, 3)

# def scale_shape(points, scale_factor, origin=(0,0)):
#     """
#     Scale the shape by scale_factor with respect to origin.
#     """
#     pts = np.asarray(points, dtype=float)
#     origin = np.asarray(origin, dtype=float)
#     scaled = (pts - origin) * scale_factor + origin

#     # print(pts, scaled)
#     print(max(pts[:, 0]), min(pts[:, 0]), max(pts[:, 1]), min(pts[:, 1]))
#     print(max(scaled[:, 0]), min(scaled[:, 0]), max(scaled[:, 1]), min(scaled[:, 1]))
#     return np.round(scaled, 3)

def shear_shape(points, shear_x=0, shear_y=0):
    """
    Apply shear transformation to the shape.
    For each point: new_x = x + shear_x * y and new_y = y + shear_y * x.
    """
    pts = np.asarray(points, dtype=float)
    new_x = pts[:, 0] + shear_x * pts[:, 1]
    new_y = pts[:, 1] + shear_y * pts[:, 0]
    sheared = np.column_stack((new_x, new_y))
    return np.round(sheared, 3)

def apply_transformation_sequence(points, instructions, origin=(0,0), center=(0,0)):
    """
    Apply sequence of transformations while maintaining proper centering.
    """
    pts = np.asarray(points, dtype=float)
    current_center = np.array([0., 0.])  # Start at origin
    
    for op, param in instructions:
        if op == "rotate":
            pts = rotate_shape_around_center(pts, param, center=current_center)
        elif op == "flip":
            if param == "horizontal":
                pts = reflect_shape_horizontally(pts, axis_y=origin[1])
            elif param == "vertical":
                pts = reflect_shape_vertically(pts, axis_x=origin[0])
            # # Re-center after flip
            # centroid = np.mean(pts, axis=0)
            # pts = pts - centroid
        elif op == "translate":
            dx, dy = param
            pts = translate_shape(pts, dx, dy)
            current_center = current_center + np.array([dx, dy])
        elif op == "scale":
            pts = scale_shape(pts, param)
        elif op == "shear":
            shear_x, shear_y = param
            pts = shear_shape(pts, shear_x, shear_y)
            # Re-center after shear
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid
        
        # if op not in ["translate", "flip"]:
        #     # For all operations except translate, ensure shape stays centered
        #     centroid = np.mean(pts, axis=0)
        #     pts = pts - centroid + current_center

    
    return np.round(pts, 3)

def compose_transformation_sequence(image_files, output_filename, labels):
    """Compose several images horizontally with labels."""
    fig, axes = plt.subplots(1, len(image_files), figsize=(4*len(image_files), 4))
    for ax, img_file, label in zip(axes, image_files, labels):
        with File.open(img_file, 'rb') as f:
            img = plt.imread(f)
        # img = plt.imread(img_file)
        ax.imshow(img)
        ax.set_title(label, fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    with File.open(output_filename, 'wb') as f:
        plt.savefig(f, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()

#########################
# BOUNDARY ADJUSTMENT
#########################

def adjust_to_boundary(points, boundary=50):
    """
    Adjust the list of (x,y) points so that the shape's bounding box fits
    within [-boundary, boundary] in both x and y.
    
    This is done by computing the bounding box, then translating the shape so that
    its center is at (0,0), and scaling it down (if necessary) so that its width and height
    do not exceed 2*boundary.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    # Compute the center of the bounding box.
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    # Compute scaling factor if the shape is too large.
    scale_factor = 1.0
    if width > 0 and height > 0:
        scale_factor = min(1.0, (2*boundary)/width, (2*boundary)/height)
    # Translate points so that the center is at (0,0) and scale.
    new_points = []
    for x, y in points:
        new_x = (x - cx) * scale_factor
        new_y = (y - cy) * scale_factor
        new_points.append((new_x, new_y))
    return new_points

    
def get_boundary(points, original_boundary=50):
    """
    Adjust the list of (x,y) points so that the shape's bounding box fits
    within [-boundary, boundary] in both x and y.
    
    This is done by computing the bounding box, then translating the shape so that
    its center is at (0,0), and scaling it down (if necessary) so that its width and height
    do not exceed 2*boundary.

    get the new boundary based on the points
    """
    pts = points + [points[0]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # if not exceed the original boundary, return the original boundary
    if max_x <= original_boundary and min_x >= -original_boundary and max_y <= original_boundary and min_y >= -original_boundary:
        return original_boundary
    # print(min_x, max_x, min_y, max_y)

    # new boundary
    boundary = int(max([original_boundary, abs(max_x), abs(min_x), abs(max_y), abs(min_y)]) )+ 10
    return boundary




#########################
# SHAPE POLYGON GENERATION
#########################


def recenter_points(points, center=(0,0)):
    """
    Given a list of (x,y) points, shift them so that their centroid (average)
    equals the given center.
    """
    avg_x = sum(x for x, y in points) / len(points)
    avg_y = sum(y for x, y in points) / len(points)
    dx = center[0] - avg_x
    dy = center[1] - avg_y
    return [(x + dx, y + dy) for (x, y) in points]

def get_shape_points(shape, num_points_circle=40):
    """
    Return a list of (x,y) points approximating the boundary of the shape.
    The shape dictionary should have keys:
       - name: one of circle, square, rectangle, triangle, ellipse, hexagon, pentagon
       - center: (x,y)
       - size: for circle, triangle, hexagon, pentagon, a single value (radius);
               for rectangle and ellipse, a tuple (width, height);
               for square, a single value (side length).
    This function makes sure that the final set of points is recentered
    so that the polygon's centroid equals the provided center.
    """
    name = shape['name'].lower()
    cx, cy = shape.get('center', (0, 0))
    
    if name == 'circle':
        r = shape.get('size', 10)
        pts = []
        for i in range(num_points_circle):
            theta = 2 * math.pi * i / num_points_circle
            pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    
    elif name == 'square':
        s = shape.get('size', 25)
        half = s / 2
        pts = [(cx - half, cy - half),
               (cx + half, cy - half),
               (cx + half, cy + half),
               (cx - half, cy + half)]
    
    elif name == 'rectangle':
        width, height = shape.get('size', (30, 15))
        half_w = width / 2
        half_h = height / 2
        pts = [(cx - half_w, cy - half_h),
               (cx + half_w, cy - half_h),
               (cx + half_w, cy + half_h),
               (cx - half_w, cy + half_h)]
    
    elif name == 'triangle':
        # Generate three random angles and compute the corresponding points on a circle
        r = shape.get('size', 25)
        pts = []
        random_angles = [random.uniform(0, 2 * math.pi) for _ in range(3)]
        while True:
            pts = [(cx + r * math.cos(angle), cy + r * math.sin(angle)) for angle in random_angles]
            # Check if the triangle is non-degenerate (using a small margin)
            a, b, c = [math.dist(pts[i], pts[j]) for i in range(3) for j in range(i+1, 3)]
            if a + b > c + 5 and a + c > b + 5 and b + c > a + 5:
                break
            random_angles = [random.uniform(0, 2 * math.pi) for _ in range(3)]
    
    elif name == 'ellipse':
        width, height = shape.get('size', (30, 15))
        a = width / 2
        b = height / 2
        pts = []
        for i in range(num_points_circle):
            theta = 2 * math.pi * i / num_points_circle
            pts.append((cx + a * math.cos(theta), cy + b * math.sin(theta)))
    
    elif name == 'hexagon':
        r = shape.get('size', 25)
        pts = []
        for i in range(6):
            theta = 2 * math.pi * i / 6
            pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    
    elif name == 'pentagon':
        r = shape.get('size', 25)
        pts = []
        for i in range(5):
            theta = 2 * math.pi * i / 5
            pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    
    else:
        raise ValueError(f"Unknown shape: {shape['name']}")
    
    # Recenter the points so that the centroid is exactly (cx, cy)
    pts = recenter_points(pts, (cx, cy))
    return pts

#########################
# PLOTTING FUNCTION
#########################

def plot_polygon(points, filename, color='skyblue', boundary=50):
    # Close the polygon by appending the first point.
    # print(points[0])
    pts_to_plot = points # + [points[0]]
    xs, ys = zip(*pts_to_plot)
    # print(filename, pts_to_plot[0], xs, ys)
    # print("boundary:", boundary)
    plt.figure(figsize=(4, 4))
    plt.fill(xs, ys, color=color, edgecolor='black', linewidth=2, alpha=0.8)
    # Draw a boundary box for reference.
    plt.gca().add_patch(plt.Rectangle((-boundary, -boundary), boundary*2, boundary*2, fill=False, edgecolor='black', linewidth=1))

    # draw the origin
    plt.plot(0, 0, 'ko', markersize=5)
    plt.xlim(-boundary, boundary)
    plt.ylim(-boundary, boundary)
    plt.axis('off')

    with File.open(filename, 'wb') as f:
        plt.savefig(f, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()

#########################
# RANDOM INSTRUCTIONS (SYMMETRY AWARE)
#########################

def random_instructions(num_steps=3, shape=None, center=(0,0), debug=False):
    """
    Generate a randomized list of transformation instructions with reduced parameter ranges,
    discrete rotation angles, and symmetry restrictions.
    
    Each instruction is a tuple (operation, parameter) where:
      - "rotate": For:
           • a square: choose from [30, 60] (90° is symmetric)
           • a hexagon: choose from [30, 90] (avoid 60°)
           • otherwise: choose from [30, 60, 90]
      - "flip": parameter is None.
      - "translate": a tuple (dx, dy) ∈ [-2,2], avoiding trivial moves.
      - "scale": a factor between 0.8 and 1.2 (avoiding nearly 1).
      - "shear": a tuple (shear_x, shear_y) in degrees ∈ [-10, 10].
    
    Additionally, if a shape is provided:
      - For a circle: exclude "rotate" and "flip".
      - For a square or triangle: exclude "flip".
    """
    shape_name = shape.get('name', '').lower() if shape is not None else None
    if shape_name == 'circle':
        allowed_ops = ["translate", "scale", "shear"]
    elif shape_name in ['square']:
        allowed_ops = ["rotate", "translate", "scale"]  # exclude flip
    # elif shape_name == 'hexagon':
    #     allowed_ops = ["rotate", "flip", "translate", "scale", "shear"]
    else:
        allowed_ops = ["rotate", "flip", "translate", "scale"]

    
    shape_size = shape.get('size', 10)
    if isinstance(shape_size, tuple):
        shape_size = min(shape_size)
    
    
    instructions = []
    while len(instructions) < num_steps:
        op = random.choice(allowed_ops)
        if op == "flip" and center == (0,0):
            op = random.choice(["rotate", "translate", "scale", "shear"])
        if op == "rotate":
            if shape_name == 'square':
                angle = random.choice([-30, -60, 30, 60])
            elif shape_name == 'hexagon':
                angle = random.choice([-30, -90, 30, 90])
            else:
                angle = random.choice([-30, -60, -90, 30, 60, 90])
            instructions.append((op, angle))
            center = center
        elif op == "flip":
            param = random.choice(["horizontal", "vertical"])
            instructions.append((op, param))
            if center != (0,0):
                center = random_center()
        elif op == "translate":
            dx = random.choice([-10, -30, 0, 10, 30])
            dy = random.choice([-10, -30, 0, 10, 30])
            while dx == 0 and dy == 0:
                dx = random.choice([-10, -30, 0, 10, 30])
                dy = random.choice([-10, -30, 0, 10, 30])
            instructions.append((op, (dx, dy)))
            center = (center[0] + dx, center[1] + dy)
        elif op == "scale":
            factor = random.choice([0.5, 2.0])
            new_size = shape_size * factor
            if new_size < 10 or new_size > 40:
                continue
            instructions.append((op, factor))
            if center != (0,0):
                center = random_center()
            shape_size = new_size
        elif op == "shear":
            shear_x = random.uniform(-1, 1)
            shear_y = random.uniform(-1, 1)
            while abs(abs(shear_x) - abs(shear_y)) < .05 or abs(abs(shear_x) - abs(shear_y)) > 0.25:
                shear_x = random.uniform(-1, 1)
                shear_y = random.uniform(-1, 1)
            instructions.append((op, (shear_x, shear_y)))
            if center != (0,0):
                center = random_center()

    if debug:
        instructions = [
            ("translate", (10, 10)),
            ("flip", "horizontal"),
            ("scale", 2.0),
        ]
    return instructions

#########################
# DISTRACTOR GENERATION HELPERS
#########################

def get_shape_size_from_pts(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    return width, height
    
def points_are_close(pts1, pts2, tol=1e-2):
    if len(pts1) != len(pts2):
        return -1, False
    # for (x1, y1), (x2, y2) in zip(pts1, pts2):
    #     if abs(x1 - x2) > tol or abs(y1 - y2) > tol:
    #         return False, min(abs(x1 - x2), abs(y1 - y2))

    # min_dist = 50 * 2
    # for p1 in pts1:
    #     for p2 in pts2:
    #         min_dist = min(min_dist, math.dist(p1, p2))
    # return min_dist, min_dist < tol

    # check whether the shape after plotting on canvas is too close to each other, use the bounding box
    xs1 = [p[0] for p in pts1]
    ys1 = [p[1] for p in pts1]
    min_x1, max_x1 = min(xs1), max(xs1)
    min_y1, max_y1 = min(ys1), max(ys1)

    xs2 = [p[0] for p in pts2]
    ys2 = [p[1] for p in pts2]
    min_x2, max_x2 = min(xs2), max(xs2)
    min_y2, max_y2 = min(ys2), max(ys2)

    # check whether the bounding box is too close
    # check upper left and lower right points distance
    ul_dist = math.dist((min_x1, max_y1), (min_x2, max_y2))
    lr_dist = math.dist((max_x1, min_y1), (max_x2, min_y2))
    min_dist = min(ul_dist, lr_dist)
    return min_dist, min_dist < tol

#########################
# INSTANCE GENERATION (ALL MATH-BASED)
#########################

def generate_visual_analogy_instance(instance_id=0, output_dir="output", num_steps=3, overwrite=False, debug=False):
    question_info_path = os.path.join(output_dir, f"question_info_{instance_id}.json")
    if not overwrite and File.isfile(question_info_path):
        return
    os.makedirs(output_dir, exist_ok=True)
    
    ### Compose intermediate transformation images for Shape B ###
    # Create a blank image to indicate a missing intermediate step
    blank_path = os.path.join(output_dir, "blank_image.png")
    if not File.isfile(blank_path):
        plt.figure(figsize=(4,4))
        plt.axis('off')
        with File.open(blank_path, 'wb') as f:
            plt.savefig(f, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()

    question_mark_path = os.path.join(output_dir, "question_mark.png")
    if not File.isfile(question_mark_path):
        plt.figure(figsize=(4,4))
        plt.text(0.5, 0.5, "?", fontsize=48, ha='center', va='center')
        plt.axis('off')
        with File.open(question_mark_path, 'wb') as f:
            plt.savefig(f, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()
    
    # Generate two different shapes.
    shapes = generate_shapes()
    shape_A = random.choice(shapes)
    shape_B = random.choice(shapes)
    while shape_A['name'] == shape_B['name']:
        shape_B = random.choice(shapes)
    
    # Set centers explicitly (here, we keep them at (0,0)).
    shape_A['center'] = (0, 0)
    shape_B['center'] = (0, 0)
    
    # Get polygon points for each shape.
    pts_A = get_shape_points(shape_A)
    pts_B = get_shape_points(shape_B)
    # print(pts_A[0], pts_B[0])

    initial_boundary = 50
    new_boundary = get_boundary(pts_B, original_boundary=initial_boundary)
    new_boundary = get_boundary(pts_A, original_boundary=new_boundary)
    
    is_close_A, is_close_B = True, True
    max_iters = 5

    while (is_close_A or is_close_B) and max_iters > 0:
        max_iters -= 1
        # Generate transformation instructions based on Shape A.
        instructions = random_instructions(num_steps=num_steps, shape=shape_A, debug=debug)
        
        # composite the transformation sequence for Shape A
        tranformed_A_sequence = []
        center = shape_A.get('center', (0,0))
        for i in range(1, len(instructions)+1):
            step = apply_transformation_sequence(pts_A, instructions[:i], origin=(0, 0), center=center)
            new_boundary = get_boundary(step, original_boundary=new_boundary)
            tranformed_A_sequence.append(step)
            if i == 1:
                prev = pts_A
            else:
                prev = tranformed_A_sequence[i-2]
            dist, is_close_A = points_are_close(tranformed_A_sequence[i-1], prev)
            if is_close_A:
                print("close A", dist)
                continue


        transform_B_sequence = []
        center = shape_B.get('center', (0,0))
        for i in range(1, len(instructions)+1):
            stepB = apply_transformation_sequence(pts_B, instructions[:i], origin=(0, 0), center=center)
            new_boundary = get_boundary(stepB, original_boundary=new_boundary)
            # step_path = os.path.join(output_dir, f"shape_B_step{i}_{instance_id}.png")
            transform_B_sequence.append(stepB)
            if i == 1:
                prev = pts_B
            else:
                prev = transform_B_sequence[i-2]
            dist, is_close_B = points_are_close(transform_B_sequence[i-1], prev)
            if is_close_B:
                print("close B", dist)
                continue
    if max_iters == 0 and (is_close_A or is_close_B):
        print("max_iters reached, but still close")
        return {}
    # print(new_boundary)
    
    # Save images for original and transformed shapes.
    base_A_path = os.path.join(output_dir, f"shape_A_{instance_id}.png")
    plot_polygon(pts_A, base_A_path, color=shape_A['facecolor'], boundary=new_boundary)
    
    base_B_path = os.path.join(output_dir, f"shape_B_{instance_id}.png")
    plot_polygon(pts_B, base_B_path, color=shape_B['facecolor'], boundary=new_boundary)
    
    trans_B_path = os.path.join(output_dir, f"shape_B_prime_{instance_id}.png")
    plot_polygon(transform_B_sequence[-1], trans_B_path, color=shape_B['facecolor'], boundary=new_boundary)

    tranformed_A_sequence_info = []

    tranformed_A_sequence_path = []
    for i, stepA in enumerate(tranformed_A_sequence):
        step_path = os.path.join(output_dir, f"shape_A_step{i}_{instance_id}.png")
        tranformed_A_sequence_path.append(step_path)

        tranformed_A_sequence_info.append({
            "name": shape_A["name"],
            "size": get_shape_size_from_pts(stepA), # get shape size from the points,
            "center": np.mean(stepA, axis=0).tolist(),# get shape center from the points
            "color": shape_A['facecolor'],
        })

        plot_polygon(stepA, step_path, color=shape_A['facecolor'], boundary=new_boundary)

    compose_labels = [f"step = {i}" for i in range(num_steps+1)]
    compose_transformation_sequence(
        [base_A_path] + tranformed_A_sequence_path,
        os.path.join(output_dir, f"composed_transformation_A_{instance_id}.png"),
        compose_labels
    )

    tranformed_B_sequence_info = []
    tranformed_B_sequence_path = []
    for i, stepB in enumerate(transform_B_sequence):
        step_path = os.path.join(output_dir, f"shape_B_step{i}_{instance_id}.png")
        tranformed_B_sequence_path.append(step_path)
        plot_polygon(stepB, step_path, color=shape_B['facecolor'], boundary=new_boundary)

        tranformed_B_sequence_info.append({
            "name": shape_B["name"],
            "size": get_shape_size_from_pts(stepB), # get shape size from the points,
            "center": np.mean(stepB, axis=0).tolist(),# get shape center from the points
            "color": shape_B['facecolor'],
        })

    
    
    # Variant 1: Without intermediate steps
    compose_labels = [f"step = {i}" for i in range(num_steps+1)]
    compose_image_list = [base_B_path] + [blank_path] * max(0, num_steps - 1) + [question_mark_path]
    compose_transformation_sequence(
        compose_image_list,
        os.path.join(output_dir, f"composed_transformation_B_no_{instance_id}.png"),
        compose_labels
    )
    # Variant 2: Partial intermediate (only first step)
    if num_steps > 2:
        num_intermediate = random.choice(range(1, num_steps - 1))
        compose_image_list = [base_B_path] + tranformed_B_sequence_path[:num_intermediate]+ [blank_path] * max(0, num_steps - num_intermediate - 1) + [question_mark_path]
        compose_transformation_sequence(
            compose_image_list,
            os.path.join(output_dir, f"composed_transformation_B_partial_{instance_id}.png"),
            compose_labels
        )
    # Variant 3: All intermediate steps (using first two instructions, if available)
    if num_steps >= 2:
        compose_image_list = [base_B_path] + tranformed_B_sequence_path[:-1] + [question_mark_path]
        compose_transformation_sequence(
            compose_image_list,
            os.path.join(output_dir, f"composed_transformation_B_all_{instance_id}.png"),
            compose_labels
        )
        # compose_image_list = [base_B_path] + tranformed_B_sequence_path 
        # compose_transformation_sequence(
        #     compose_image_list,
        #     os.path.join(output_dir, f"composed_transformation_B_all_w_ans_{instance_id}.png"),
        #     compose_labels
        # )
    
    # For the answer choices, the correct answer is transformed_B.
    # correct_pts = transformed_B
    answer_info = []
    answer_choices_paths = {}
    answer_choices_infos = {}
    ind_answer_choices_paths = {}
    gt_answers = {}
    distractor_instructions = random_instructions(num_steps=3, shape=shape_B)

    gt_size = get_shape_size_from_pts(transform_B_sequence[-1])
    gt_centroid = np.mean(transform_B_sequence[-1], axis=0)
    answer_info.append({
        "name": shape_B["name"],
        "size": gt_size, # get shape size from the points,
        "center": gt_centroid.tolist(),# get shape center from the points
        "color": shape_B['facecolor'],
    })
    print(shape_B)
    print(answer_info)

    # generate distractors based on levels
    for level in ["easy", "medium", "hard"]:
        if level == "easy":
            distractor_colors = [random_color(), random_color(), shape_B['facecolor']]
        elif level == "medium":
            distractor_colors = [random_color(), shape_B['facecolor'], shape_B['facecolor']]
        elif level == "hard":
            distractor_colors = [shape_B['facecolor'], shape_B['facecolor'], shape_B['facecolor']]
        
        random.shuffle(distractor_colors)

        distractor_sequence = []
        distractor_answer_info = []
        
        for i in range(1, len(distractor_instructions)+1):
            step = apply_transformation_sequence(pts_B, instructions[:i], origin=shape_B.get('center', (0,0)))
            step = adjust_to_boundary(step, boundary=new_boundary)
            is_too_close_all = []
            is_too_close = points_are_close(step, transform_B_sequence[-1], tol=5)
            is_too_close_all.append(is_too_close)
            for stepcand in distractor_sequence:
                is_too_close = points_are_close(step, stepcand, tol=5)
                is_too_close_all.append(is_too_close)
            # print(is_too_close_all)

            is_too_close = [close for _, close in is_too_close_all]
            is_too_close = any(is_too_close)

            
            max_iters = 10
            while is_too_close and max_iters > 0:
                max_iters -= 1
                distractor_instructions = random_instructions(num_steps=3, shape=shape_B)
                step = apply_transformation_sequence(pts_B, distractor_instructions[:i], origin=shape_B.get('center', (0,0)))


                if level == "medium":
                    resize_p = random.random()
                    if resize_p < 0.3:
                        # make it similar size to the correct answer
                        width, height = get_shape_size_from_pts(step)
                        scale_factor = min(gt_size[0]/width, gt_size[1]/height)
                        step = scale_shape(step, scale_factor)
                elif level == "hard":
                    resize_p = random.random()
                    if resize_p < 0.6:
                        # make it similar size to the correct answer
                        width, height = get_shape_size_from_pts(step)
                        scale_factor = min(gt_size[0]/width, gt_size[1]/height)
                        step = scale_shape(step, scale_factor)
                        cand_centroid = np.mean(step, axis=0)
                        # assert np.allclose(cand_centroid, gt_centroid, atol=1e-2)
                        
                step = adjust_to_boundary(step, boundary=new_boundary)
                is_too_close_all = []
                is_too_close = points_are_close(step, transform_B_sequence[-1], tol=5)
                is_too_close_all.append(is_too_close)
                for stepcand in distractor_sequence:
                    is_too_close = points_are_close(step, stepcand, tol=5)
                    is_too_close_all.append(is_too_close)

                is_too_close = [close for _, close in is_too_close_all]
                is_too_close = any(is_too_close)
                # print(is_too_close_all)
            if max_iters == 0 and is_too_close:
                print("Failed to generate a distractor")
                distractor_sequence = None
                break
            distractor_sequence.append(step)
            info = {
                    "name": shape_B["name"],
                    "size": get_shape_size_from_pts(step), # get shape size from the points
                    "center": np.mean(step, axis=0).tolist(),# get shape center from the points
                }
            distractor_answer_info.append(info)

        if distractor_sequence is not None:
            distractor2_path = os.path.join(output_dir, f"distractor_B_{level}_{instance_id}.png")
            plot_polygon(distractor_sequence[0], distractor2_path, color=distractor_colors[0])
            distractor_answer_info[0]["color"] = distractor_colors[0]


            
            distractor3_path = os.path.join(output_dir, f"distractor_C_{level}_{instance_id}.png")
            plot_polygon(distractor_sequence[1], distractor3_path, color=distractor_colors[1])
            distractor_answer_info[1]["color"] = distractor_colors[1]
            
            distractor4_path = os.path.join(output_dir, f"distractor_D_{level}_{instance_id}.png")
            plot_polygon(distractor_sequence[2], distractor4_path, color=distractor_colors[2])
            distractor_answer_info[2]["color"] = distractor_colors[2]
            
            # Compose an answer choices image (side-by-side)
            answer_choices_path = os.path.join(output_dir, f"answer_choices_{level}_{instance_id}.png")

            answer_image_list = [trans_B_path, distractor2_path, distractor3_path, distractor4_path]
            answer_choices_info = [answer_info[0], distractor_answer_info[0], distractor_answer_info[1], distractor_answer_info[2]]
            answer_image_idx = [0, 1, 2, 3]

            shuffled_answer_image_idx = answer_image_idx.copy()
            random.shuffle(shuffled_answer_image_idx)

            answer_image_list = [answer_image_list[i] for i in shuffled_answer_image_idx]
            answer_choices_info = [answer_choices_info[i] for i in shuffled_answer_image_idx]
            # ["(A)", "(B)", "(C)", "(D)"]

            gt_ans = answer_image_list.index(trans_B_path)
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for ax, img_path, label in zip(axes, 
                                        answer_image_list,
                                        ["(A)", "(B)", "(C)", "(D)"]):
                with File.open(img_path, 'rb') as f:
                    img = plt.imread(f)
                ax.imshow(img)
                ax.set_title(label, fontsize=18)
                ax.axis('off')
            plt.tight_layout()
            with File.open(answer_choices_path, 'wb') as f:
                plt.savefig(f, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
            answer_choices_paths[level] = answer_choices_path
            ind_answer_choices_paths[level] = answer_image_list
            gt_answers[level] = gt_ans
            answer_choices_infos[level] = answer_choices_info
    
    # Save question and ground-truth info as JSON.
    shape_A['size'] = get_shape_size_from_pts(pts_A)
    shape_B['size'] = get_shape_size_from_pts(pts_B)
    shape_A['center'] = np.mean(pts_A, axis=0).tolist()
    shape_B['center'] = np.mean(pts_B, axis=0).tolist()
    question_info = {
        "instance_id": instance_id,
        "shape_A": shape_A,
        "shape_B": shape_B,
        "individual_shape_B_transformations": [base_B_path] + tranformed_B_sequence_path,
        "question": f"Observe the transformation pattern of Shape A through steps 0 to {num_steps}. <question_image> Apply the same transformation sequence to Shape B and determine the final shape at step {num_steps}. <image_for_B> For reference, the black dots in each panel of the figures indicate the origin. Select the correct answer choice that matches the expected transformation result. <answer_choices>",
        "composed_B_no": os.path.join(output_dir, f"composed_transformation_B_no_{instance_id}.png"),
        "composed_B_partial": os.path.join(output_dir, f"composed_transformation_B_partial_{instance_id}.png"),
        "composed_B_all": os.path.join(output_dir, f"composed_transformation_B_all_{instance_id}.png"),
        # "composed_B_all_w_ans": os.path.join(output_dir, f"composed_transformation_B_all_w_ans_{instance_id}.png"),
        "answer_choices": answer_choices_paths,
        "answer_choices_info": answer_choices_infos,
        "transformed_A_sequence_info": tranformed_A_sequence_info,
        "tranformed_B_sequence_info": tranformed_B_sequence_info,
        "individual_answer_choices_path": ind_answer_choices_paths,
        "question_image": os.path.join(output_dir, f"composed_transformation_A_{instance_id}.png"),
        "instructions": instructions,
        "correct_answer": {key: chr(ord("A")+gt_ans) for key, gt_ans in gt_answers.items()},
        "text_instructions": generate_natural_text_task(shape_B['name'], instructions) if "shear" not in [op for op, _ in instructions] else "N/A"
    }
    with File.open(question_info_path, "w") as f:
        json.dump(question_info, f, indent=2)
    return question_info


def generate_natural_text_task(shape_name, transformations):
    steps = []
    
    for op, value in transformations:
        if op == "rotate":
            direction = "counter-clockwise" if value > 0 else "clockwise"
            steps.append(f"Rotate the {shape_name} by {abs(value)} degrees {direction} around its center.")
        elif op == "flip":
            if value == "vertical":
                steps.append(f"Flip the {shape_name} as if there's a mirror along the vertical line x = 0.")
            else:
                steps.append(f"Flip the {shape_name} as if there's a mirror along the horizontal line y = 0.")
        elif op == "translate":
            dx, dy = value
            if dx > 0:
                if dx <= 10:
                    direction_x = "a little to the right"
                else:
                    direction_x = "to the right by a significant amount"
            elif dx < 0:
                if dx >= -10:
                    direction_x = "a little to the left"
                else:
                    direction_x = "to the left by a significant amount"
            
            if dy > 0:
                if dy <= 10:
                    direction_y = "slightly upward"
                else:
                    direction_y = "significantly upward"
            elif dy < 0:
                if dy >= -10:
                    direction_y = "slightly downward"
                else:
                    direction_y = "significantly downward"
            if dx == 0:
                steps.append(f"Shift the {shape_name} {direction_y}.")
            elif dy == 0:
                steps.append(f"Shift the {shape_name} {direction_x}.")
            else:
                steps.append(f"Shift the {shape_name} {direction_x} and {direction_y}.")
            # direction_x = "a little to the right" if dx > 0 else "a little to the left"
            # direction_y = "slightly upward" if dy > 0 else "slightly downward"
            # steps.append(f"Shift the {shape_name} {direction_x} and {direction_y}.")
        elif op == "scale":
            if value > 1:
                steps.append(f"Increase the size of the {shape_name}, making it larger.")
            else:
                steps.append(f"Reduce the size of the {shape_name}, making it smaller.")
        elif op == "shear":
            steps.append(f"Skew the {shape_name}, stretching it slightly in a diagonal direction.")

    appended_image_steps = []
    for i, step in enumerate(steps):
        appended_image_steps.append(f"{step} <shapeB_step_{i}>")

    question_text = f"""
    Imagine transforming a {shape_name} step by step. <shapeB_image> Follow these changes:

    {chr(10).join(['- ' + step for step in appended_image_steps])}

    After these transformations, which of the following shapes best represents the final result?

    For reference, the black dots in each panel of the figures indicate the origin.
    """

    return question_text

#########################
# HTML REPORT GENERATION
#########################

def generate_html_report(instances, output_html="all_variants_analogy.html", output_dir="output_instances"):
    html_parts = []
    html_parts.append("<html><head><title>2D task</title></head><body>")
    html_parts.append("<h1>Visual Analogy Task Variants</h1>")
    for inst in instances:
        html_parts.append("<hr>")
        html_parts.append(f"<h2>Instance {inst['instance_id']}</h2>")
        html_parts.append("<h3>Transformed Shape A (A'):</h3>")
        img_path = inst["question_image"].replace(output_dir+"/", "./")
        html_parts.append(f'<div><img src="{img_path}" style="max-width:800px;"></div>')
        html_parts.append("<h3>Transformation Sequence for Shape B Variants:</h3>")
        for variant in ["composed_B_no", "composed_B_partial", "composed_B_all"]:
            img_path = inst[variant].replace(output_dir+"/", "./")
            html_parts.append(f'<div><img src="{img_path}" style="max-width:800px;"></div>')

        html_parts.append("<h3>Answer Choices:</h3>")

        for level in ["easy", "medium", "hard"]:
            html_parts.append(f"<h4>Level: {level.capitalize()}</h4>")
            img_path = inst["answer_choices"][level].replace(output_dir+"/", "./")
            html_parts.append(f'<div><img src="{img_path}" style="max-width:800px;"></div>')
            html_parts.append(f"<strong>Correct Answer:</strong> {inst['correct_answer'][level]}")


    html_parts.append("<h1>Text Instruction Task Variants</h1>")
    for inst in instances:
        if inst["text_instructions"] == "N/A":
            continue
        html_parts.append("<hr>")
        html_parts.append(f"<h2>Instance {inst['instance_id']}</h2>")
        html_parts.append(f"<h3>Text Instructions:</h3>")
        html_parts.append(inst['text_instructions'])
        # img_path = inst["question_image"].replace(output_dir+"/", "./")
        shapeB_images = inst["individual_shape_B_transformations"]

        for i, step in enumerate(inst["instructions"]):
            img_path = shapeB_images[i].replace(output_dir+"/", "./")
            html_parts.append(f'<div><img src="{img_path}" style="max-width:200px;"></div>')

        for level in ["easy", "medium", "hard"]:
            html_parts.append(f"<h4>Level: {level.capitalize()}</h4>")
            img_path = inst["answer_choices"][level].replace(output_dir+"/", "./")
            html_parts.append(f'<div><img src="{img_path}" style="max-width:800px;"></div>')
            html_parts.append(f"<strong>Correct Answer:</strong> {inst['correct_answer'][level]}")

    html_parts.append("</body></html>")
    with File.open(output_html, "w") as f:
        f.write("\n".join(html_parts))
    print(f"HTML report generated: {output_html}")



#########################
# MAIN FUNCTION
#########################
def main(num_instances=1, output_dir="output_instances", num_steps=3, seed=42, overwrite=False, debug=False):
    random.seed(seed)
    np.random.seed(seed)
    output_dir = os.path.join(output_dir, f"2d_visual_analogy_{num_steps}_steps")
    print(f"Generating {num_instances} instances with {num_steps} steps each. Saving to {output_dir}")
    if debug:
        output_dir += "_debug"
        num_instances = 5
    os.makedirs(output_dir, exist_ok=True)
    instances = []
    for i in tqdm(range(num_instances)):
        if num_instances > 120:
            random.seed(seed*10 + i)
            np.random.seed(seed*10 + i)
        inst = generate_visual_analogy_instance(instance_id=i, output_dir=output_dir, num_steps=num_steps, overwrite=overwrite, debug=debug)
        if inst is not None:
            instances.append(inst)
    if num_instances <= 5:
        html_path = os.path.join(output_dir, f"all_variants_analogy.html")
        generate_html_report(instances, output_html=html_path, output_dir=output_dir)
    return output_dir

def create_va_hf_dataset(output_dir, dataset_name, num_instances=1, num_steps=3, seed=42, overwrite=False,num_samples=100):
    from PIL import Image
    import datasets
    from datasets import Dataset, Features
    import json
    data_dict = {
        'qid': [], 'A_image': [], 'B_image': [], 'choices': [], 
        # 'individual_choices': [],
        'answer': [], 'transformations': [], 'difficulty_level': [], 'question_info': [], 'answer_info':[]}
    for steps in range(1, num_steps+1):
        saved_output_dir = main(num_instances=num_instances, output_dir=output_dir, num_steps=steps, seed=seed, overwrite=overwrite)
        random.seed(seed)
        np.random.seed(seed)

        for difficulty in ['easy', 'medium', 'hard']:
            # for variant in ['no', 'partial', 'all']:
            # sampled_indices = random.sample(range(num_instances), num_instances//2)
            
            sampled_indices = random.sample(range(120), 120//2)

            if 'train' in dataset_name:
                remaining_indices = [i for i in range(num_instances) if i not in sampled_indices]
                sampled_indices = remaining_indices
                all_variants = ['no']

            if 'train' in dataset_name:
                remaining_indices = [i for i in range(num_instances) if i not in sampled_indices]
                sampled_indices = remaining_indices
                all_variants = ['no']
            elif 'vissim' in dataset_name:
                all_variants = ['partial', 'all']
            else:
                all_variants = ['no']
            for variant in all_variants:
                if variant == 'partial' and steps <= 2:
                    continue
                if variant == 'all' and steps < 2:
                    continue
                num_added_samples = 0
                # for i in tqdm(range(100)):
                for i in sampled_indices:
                    if num_added_samples >= num_samples // 3 + 1:
                        break
                    question_info_path = os.path.join(saved_output_dir, f'question_info_{i}.json')
                    composed_A_image_path = os.path.join(saved_output_dir, f'composed_transformation_A_{i}.png')
                    composed_B_image_path = os.path.join(saved_output_dir, f'composed_transformation_B_{variant}_{i}.png')
                    answer_choices_path = os.path.join(saved_output_dir, f'answer_choices_{difficulty}_{i}.png')
                    if not File.isfile(composed_A_image_path) or not File.isfile(composed_B_image_path) or not File.isfile(answer_choices_path):
                        # print(f"Skipping {composed_A_image_path}, {composed_B_image_path}, {answer_choices_path}")
                        continue
                    with open(question_info_path, 'r') as f:
                        question_info = json.load(f)
                    if difficulty not in question_info['correct_answer']:
                        continue
                    # check if shape A is a circle and skip if any of the transformation is rotation while not shear in front of the rotation
                    if question_info['shape_A']['name'] == 'circle' and any([op == 'rotate' for op, _ in question_info['instructions']]) and 'shear' not in [op for op, _ in question_info['instructions']]:
                        print("Skipping circle with rotation, ", question_info_path)
                        continue
                    num_added_samples += 1
                    with File.open(composed_A_image_path, 'rb') as f:
                        composed_A_image = Image.open(f).convert('RGB')
                    with File.open(composed_B_image_path, 'rb') as f:
                        composed_B_image = Image.open(f).convert('RGB')
                    with File.open(answer_choices_path, 'rb') as f:
                        answer_choices_image = Image.open(f).convert('RGB')
                    data_dict['qid'].append(f'{steps}steps_{difficulty}_{variant}_{i}')
                    data_dict['A_image'].append(composed_A_image)
                    data_dict['B_image'].append(composed_B_image)
                    data_dict['choices'].append(answer_choices_image)
                    # data_dict['individual_choices'].append(individual_choice_images)
                    data_dict['answer'].append(question_info['correct_answer'][difficulty])
                    data_dict['transformations'].append(question_info['instructions'])
                    data_dict['difficulty_level'].append(difficulty)
                    data_dict['question_info'].append(json.dumps(question_info))
                    data_dict['answer_info'].append(json.dumps(question_info['correct_answer']))
                # print("Added", num_added_samples, "samples for ", difficulty, variant, steps, "steps")
    # print("sampled {} samples".format(len(data_dict['qid'])))
    
    print(f"Total number of instances: {len(data_dict['qid'])}")
    print(f"Unique instances: {len(set(data_dict['qid']))}")
    assert len(data_dict['qid']) == len(set(data_dict['qid'])), "Duplicate qid found in dataset"
    
    features = Features({
        'qid': datasets.Value('string'),
        'A_image': datasets.Image(),
        'B_image': datasets.Image(),
        'choices': datasets.Image(),
        # 'individual_choices': datasets.Sequence(datasets.Image()),
        'answer': datasets.Value('string'),
        'transformations': datasets.Value('string'),
        'difficulty_level': datasets.Value('string'),
        'question_info': datasets.Value('string'),
        'answer_info': datasets.Value('string')
    })
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    with open('aux_data/credentials/hf_token.txt', 'r') as f:
        hf_token = f.read().strip()
    hf_dataset.push_to_hub(
        f"VisSim/{dataset_name}",
        create_pr=False,
        token=hf_token)
    


def create_text_instruction_hf_dataset(output_dir, dataset_name, num_instances=1, num_steps=3, seed=42, overwrite=False, individual_choices=False):
    data_dict = {
        'qid': [], 'question': [], 'choices': [], 
        'images': [],
        # 'individual_choices': [],
        'answer': [], 'transformations': [], 'difficulty_level': [], 'question_info': [], 'answer_info':[]}
    if individual_choices:
        data_dict['individual_choices'] = []
    from PIL import Image
    import datasets
    from datasets import Dataset, Features
    import json
    for steps in range(1, num_steps+1):
        saved_output_dir = main(num_instances=num_instances, output_dir=output_dir, num_steps=steps, seed=seed, overwrite=overwrite)
        random.seed(seed)
        np.random.seed(seed)

        for difficulty in ['easy', 'medium', 'hard']:
            # for variant in ['no', 'partial', 'all']:
            sampled_indices = random.sample(range(120), 120//2)

            if 'train' in dataset_name:
                remaining_indices = [i for i in range(num_instances) if i not in sampled_indices]
                sampled_indices = remaining_indices
                all_variants = ['no']
            elif 'vissim' in dataset_name:
                all_variants = ['partial', 'all', 'all_w_ans', 'all_last']
            else:
                all_variants = ['no']

            # for variant in ['no', 'partial', 'all', 'all_w_ans']:
            for variant in all_variants:
            # for variant in ['partial', 'all', 'all_w_ans', 'all_last']:
                if variant == 'partial' and steps <= 2:
                    continue
                if variant == 'all' and steps < 2:
                    continue
                if variant == 'all_last' and steps < 2:
                    continue
                # for i in tqdm(range(100)):
                for i in sampled_indices:
                    question_info_path = os.path.join(saved_output_dir, f'question_info_{i}.json')
                    if not File.isfile(question_info_path):
                        continue
                    with File.open(question_info_path, 'r') as f:
                        question_info_path = json.load(f)
                    if difficulty not in question_info_path['correct_answer']:
                        continue
                    question = question_info_path['text_instructions']
                    if question == "N/A":
                        continue
                    individual_transformed_image_path = question_info_path['individual_shape_B_transformations']
                    individual_transformed_images = []
                    for j in range(len(individual_transformed_image_path)):
                        path = individual_transformed_image_path[j]
                        if 'vigstandard_data/' in path:
                            path = path.replace('vigstandard_data/', 'vigstandard_data_2/')
                        individual_transformed_images.append(Image.open(path).convert('RGB'))
                    
                    if variant == 'no':
                        data_dict['images'].append([individual_transformed_images[0]])
                    elif variant == 'partial' and steps > 2:
                        num_intermediate = random.choice(range(1, steps - 1))
                        data_dict['images'].append([individual_transformed_images[0], individual_transformed_images[num_intermediate]])
                    elif variant == 'all':
                        data_dict['images'].append(individual_transformed_images[:-1])
                    elif variant == 'all_last':
                        data_dict['images'].append([individual_transformed_images[0],individual_transformed_images[-2]])
                        for s_ in list(range(steps-2)) + [steps-1]:
                            question_info_path['text_instructions'] = question_info_path['text_instructions'].replace(f'<shapeB_step_{s_}>', '')
                        question_info_path['text_instructions'] = question_info_path['text_instructions'].replace(f'<shapeB_step_{s_}>', '<shapeB_step_0>')
                        
                    elif variant == 'all_w_ans':
                        data_dict['images'].append(individual_transformed_images)
                    data_dict['qid'].append(f'{steps}steps_{difficulty}_{variant}_{i}')
                    data_dict['question'].append(question_info_path['text_instructions'])
                    path = question_info_path['answer_choices'][difficulty]
                    if 'vigstandard_data/' in path:
                        path = path.replace('vigstandard_data/', 'vigstandard_data_2/')

                    answer_choise_image = Image.open(path).convert('RGB')
                    data_dict['choices'].append(answer_choise_image)
                    if individual_choices:
                        individual_choices_list = []
                        individual_answer_choices_path = question_info_path['individual_answer_choices_path']
                        for j in range(len(individual_answer_choices_path[difficulty])):
                            path = individual_answer_choices_path[difficulty][j]
                            if 'vigstandard_data/' in path:
                                path = path.replace('vigstandard_data/', 'vigstandard_data_2/')
                            # individual_answer_choices_path[j] = path
                            individual_choices_list.append(Image.open(path).convert('RGB'))
                        data_dict['individual_choices'].append(individual_choices_list)
                    data_dict['answer'].append(question_info_path['correct_answer'][difficulty])
                    data_dict['transformations'].append(question_info_path['instructions'])
                    data_dict['difficulty_level'].append(difficulty)
                    data_dict['question_info'].append(json.dumps(question_info_path))
                    data_dict['answer_info'].append(json.dumps(question_info_path['correct_answer']))
    
    print(f"Total number of instances: {len(data_dict['qid'])}")
    print(f"Unique instances: {len(set(data_dict['qid']))}")
    assert len(data_dict['qid']) == len(set(data_dict['qid'])), "Duplicate qid found in dataset"
    
    print("sampled {} samples".format(len(data_dict['qid'])))
    print("unique qid", len(set(data_dict['qid'])))
    feature_dict = {
        'qid': datasets.Value('string'),
        'question': datasets.Value('string'),
        'images': datasets.Sequence(datasets.Image()),
        'choices': datasets.Image(),
        # 'individual_choices': datasets.Sequence(datasets.Image()),
        'answer': datasets.Value('string'),
        'transformations': datasets.Value('string'),
        'difficulty_level': datasets.Value('string'),
        'question_info': datasets.Value('string'),
        'answer_info': datasets.Value('string')
    }
    if individual_choices:
        feature_dict['individual_choices'] = datasets.Sequence(datasets.Image())
    features = Features(feature_dict)
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    with open('aux_data/credentials/hf_token.txt', 'r') as f:
        hf_token = f.read().strip()
    hf_dataset.push_to_hub(
        f"VisSim/{dataset_name}",
        create_pr=False,
        token=hf_token)
    


if __name__ == "__main__":
    from fire import Fire
    Fire()

'''
AZFUSE_USE_FUSE=1 python mmcot/va_tasks/script_w_all_shapes_v1.py --num_instances 100 --output_dir /datadrive_a/linjie/blob/vigstandard_data/linjli/debug_output/UW/mental_simulation/ --num_steps 2 --seed 4202

AZFUSE_USE_FUSE=1 python mmcot/va_tasks/script_w_all_shapes_v1.py --num_instances 100 --output_dir /datadrive_a/linjie/blob/vigstandard_data/linjli/debug_output/UW/mental_simulation/ --num_steps 4 --seed 4204
'''