import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import random
import math
title_fontsize = 20
label_fontsize = 16
ticks_fontsize = 14

##############################################################################
# 1. SEGMENTATION ALGORITHM WITH STEP RECORDING
##############################################################################
def segment_rectangle_steps_max(r0, r1, c0, c1, min_size=2, max_pieces=10):
    """
    Iteratively segment a rectangle (rows r0 to r1 and cols c0 to c1) into smaller pieces,
    stopping once the total number of pieces reaches max_pieces or no further splits are possible.
    
    This approach picks a splittable piece (based on area) and splits it, recording each split step.
    
    Returns:
      pieces: a list of final rectangles as tuples (r0, r1, c0, c1)
      steps:  a list of dictionaries recording each split.
    """
    pieces = [(r0, r1, c0, c1)]
    steps = []
    
    while len(pieces) < max_pieces:
        # Identify pieces that are splittable (i.e. dimension greater than 2*min_size)
        splittable_indices = [i for i, (pr0, pr1, pc0, pc1) in enumerate(pieces)
                              if (pr1 - pr0) > 2 * min_size or (pc1 - pc0) > 2 * min_size]
        if not splittable_indices:
            break  # no more splittable pieces
        
        # Choose the largest splittable piece (by area)
        idx_to_split = max(splittable_indices, key=lambda i: (pieces[i][1]-pieces[i][0])*(pieces[i][3]-pieces[i][2]))
        piece = pieces.pop(idx_to_split)
        pr0, pr1, pc0, pc1 = piece
        height = pr1 - pr0
        width = pc1 - pc0
        
        # Decide splitting direction: choose the longer dimension if possible.
        if height >= width and height > 2 * min_size:
            split_line = random.randint(pr0 + min_size, pr1 - min_size)
            piece1 = (pr0, split_line, pc0, pc1)
            piece2 = (split_line, pr1, pc0, pc1)
            direction = "horizontal"
        elif width > 2 * min_size:
            split_line = random.randint(pc0 + min_size, pc1 - min_size)
            piece1 = (pr0, pr1, pc0, split_line)
            piece2 = (pr0, pr1, split_line, pc1)
            direction = "vertical"
        else:
            # Should not reach here, but in case the piece is not splittable, put it back.
            pieces.append(piece)
            break
        
        pieces.append(piece1)
        pieces.append(piece2)
        steps.append({
            "action": "split",
            "direction": direction,
            "rectangle": piece,
            "split_line": split_line,
            "instruction": f"Split rectangle {piece} {direction} at line {split_line}"
        })
        
    return pieces, steps



##############################################################################
# 2. VISUALIZATION FUNCTIONS: SEGMENTATION & SCRAMBLED PIECES
##############################################################################
def generate_piece_colors(num_pieces):
    """
    Generate a dictionary mapping piece index to a random color.
    """
    colors = {}
    for idx in range(num_pieces):
        colors[idx] = np.random.rand(3,)
    return colors

def animate_segmentation(n, segmentation_steps, output_dir="", prefix=""):
    """
    Animate the segmentation process on an n x n board.
    Each step displays the current segmentation with light-gray grid lines;
    each piece is drawn in a unique color and labeled with its index and area.
    """
    current_segmentation = [(0, n, 0, n)]
    history = [current_segmentation.copy()]
    instructions = [f"Start with full board (0,{n},0,{n})"]

    for step in segmentation_steps:
        rect = step["rectangle"]
        direction = step["direction"]
        split_line = step["split_line"]
        if rect in current_segmentation:
            current_segmentation.remove(rect)
            if direction == "horizontal":
                rect1 = (rect[0], split_line, rect[2], rect[3])
                rect2 = (split_line, rect[1], rect[2], rect[3])
            else:
                rect1 = (rect[0], rect[1], rect[2], split_line)
                rect2 = (rect[0], rect[1], split_line, rect[3])
            current_segmentation.append(rect1)
            current_segmentation.append(rect2)
            history.append(current_segmentation.copy())
            instructions.append(step["instruction"])
        else:
            continue

    max_pieces = max(len(state) for state in history)
    # colors = generate_piece_colors(max_pieces)

    for i, seg in enumerate(history):
        plt.figure(figsize=(6,6))
        ax = plt.gca()
        for r in range(n+1):
            ax.plot([0, n], [r, r], color="lightgray", lw=1)
        for c in range(n+1):
            ax.plot([c, c], [0, n], color="lightgray", lw=1)
        for idx, (r0, r1, c0, c1) in enumerate(seg):
            area = (r1 - r0) * (c1 - c0)
            rect_patch = patches.Rectangle((c0, n - r1), c1 - c0, r1 - r0,
                                           facecolor=colors[idx],
                                           alpha=0.6,
                                           edgecolor="black", lw=2)
            ax.add_patch(rect_patch)
            center = ((c0+c1)/2, n - (r0+r1)/2)
            ax.text(center[0], center[1], f"{idx}: {area}",
                    ha="center", va="center", fontsize=label_fontsize, color="black")
        plt.title(f"Segmentation Step {i}\n{instructions[i]}")
        plt.xlim(0, n)
        plt.ylim(0, n)
        plt.gca().set_yticks(range(n+1))
        plt.gca().set_yticklabels(range(n+1), fontsize=ticks_fontsize)
        plt.gca().set_xticks(range(n+1))
        plt.gca().set_xticklabels(range(n+1), fontsize=ticks_fontsize)
        ax.set_aspect("equal")
        
        output_path = os.path.join(output_dir, f"{prefix}segmentation_step_{i}.png")
        plt.savefig(output_path)
    image_paths = [os.path.join(output_dir, f"{prefix}segmentation_step_{i}.png") for i in range(len(history))]
    return image_paths

def plot_empty_and_scrambled(n, pieces, draw_piece_grid=True):
    """
    Plot a single figure with two subplots:
      - Left: an empty n x n board with light-gray grid lines.
      - Right: the scrambled pieces arranged in non-overlapping cells.
    
    Each piece is:
      - Randomly rotated by one of the angles in {0, 30, 60, 90}.
      - Placed in a grid cell (non overlapping) on a larger canvas.
      - Drawn in its assigned color and labeled with its index and area.
    
    If draw_piece_grid is True, a light-gray grid is overlaid on each scrambled piece.
    The canvas size is recalculated based on the rotated bounding box dimensions.
    """
    num_pieces = len(pieces)
    
    # For each piece, choose a random angle from {0, 30, 60, 90} and recalc bounding box.
    scrambled = []
    for idx, (r0, r1, c0, c1) in enumerate(pieces):
        orig_width = c1 - c0
        orig_height = r1 - r0
        angle = random.choice([0, 30, 60, 90])
        theta = np.deg2rad(angle)
        # Compute rotated bounding box dimensions.
        new_w = abs(orig_width * np.cos(theta)) + abs(orig_height * np.sin(theta))
        new_h = abs(orig_width * np.sin(theta)) + abs(orig_height * np.cos(theta))
        original_area = orig_width * orig_height
        scrambled.append({
            'index': idx,
            'piece': (r0, r1, c0, c1),
            'angle': angle,
            'orig_width': orig_width,
            'orig_height': orig_height,
            'w': new_w,
            'h': new_h,
            'area': original_area,
            'label': f"{orig_width}x{orig_height}" if angle != 90 else f"{orig_height}x{orig_width}"
        })
    random.shuffle(scrambled)
    
    # Determine grid layout for scrambled pieces.
    num_cols = math.ceil(math.sqrt(num_pieces))
    num_rows = math.ceil(num_pieces / num_cols)
    # Use the maximum rotated bounding box dimensions.
    max_w = max(item['w'] for item in scrambled)
    max_h = max(item['h'] for item in scrambled)
    margin = 1
    cell_width = max_w + margin
    cell_height = max_h + margin
    canvas_width = num_cols * cell_width
    canvas_height = num_rows * cell_height

    # colors = generate_piece_colors(num_pieces)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,8))
    
    # Left subplot: empty board.
    for r in range(n+1):
        ax1.plot([0, n], [r, r], color="lightgray", lw=1)
    for c in range(n+1):
        ax1.plot([c, c], [0, n], color="lightgray", lw=1)
    ax1.set_xlim(0, n)
    ax1.set_ylim(0, n)
    plt.gca().set_yticks(range(n+1))
    plt.gca().set_yticklabels(range(n+1), fontsize=ticks_fontsize)
    plt.gca().set_xticks(range(n+1))
    plt.gca().set_xticklabels(range(n+1), fontsize=ticks_fontsize)
    # set y ticks to be in reverse order
    ax1.set_aspect("equal")
    ax1.set_title(f"The {n}x{n} Tangram Board", fontsize=title_fontsize)
    # add the title as text
    # ax1.text(0.5, 1.05, f"The {n}x{n} Tangram Board", ha="center", va="center", fontsize=16, transform=ax1.transAxes)
    
    # Right subplot: scrambled pieces.
    for i, item in enumerate(scrambled):
        w = item['w']
        h = item['h']
        angle = item['angle']
        print(f"Piece {i}: {item['piece']} rotated by {angle} degrees")
        idx = item['index']
        # Determine grid cell position (top row first).
        grid_row = i // num_cols
        grid_col = i % num_cols
        cell_x = grid_col * cell_width
        cell_y = canvas_height - (grid_row + 1) * cell_height
        cell_center = (cell_x + cell_width/2, cell_y + cell_height/2)
        # For drawing, use the original piece dimensions.
        orig_w = item['orig_width']
        orig_h = item['orig_height']
        # Create a rectangle patch centered at (0,0) with size (orig_w, orig_h).
        rect_patch = patches.Rectangle((-orig_w/2, -orig_h/2), orig_w, orig_h,
                                       facecolor=colors[idx],
                                       edgecolor="black",
                                       lw=2,
                                       alpha=0.7)
        # Create an affine transform: rotate by 'angle' (in degrees) about (0,0)
        # then translate to cell_center.
        trans = (transforms.Affine2D().rotate_deg(angle).translate(cell_center[0], cell_center[1])
                 + ax2.transData)
        rect_patch.set_transform(trans)
        ax2.add_patch(rect_patch)
        # Label the piece at the cell center.
        ax2.text(cell_center[0], cell_center[1]+ item['h']/2+0.2, f"{idx}: {item['label']}",
                 ha="center", va="center", fontsize=label_fontsize, color="black")
        
        # Optionally, overlay a light-gray grid on the piece.
        if draw_piece_grid:
            # In local coordinates, draw grid lines for the piece's original size.
            for xi in range(0, orig_w + 1):
                x = -orig_w/2 + xi
                line = plt.Line2D([x, x], [-orig_h/2, orig_h/2], color="black", lw=1, alpha=0.8)
                line.set_transform(trans)
                ax2.add_line(line)
            for yi in range(0, orig_h + 1):
                y = -orig_h/2 + yi
                line = plt.Line2D([-orig_w/2, orig_w/2], [y, y], color="black", lw=1, alpha=0.8)
                line.set_transform(trans)
                ax2.add_line(line)
    
    ax2.set_xlim(0, canvas_width)
    ax2.set_ylim(0, canvas_height)
    # do not show axes
    ax2.axis("off")
    ax2.set_aspect("equal")
    ax2.set_title("Avaialable Pieces", fontsize=title_fontsize)
    
    plt.tight_layout()
    plt.savefig("scrambled_pieces.png")
    plt.close()


def animate_reassembly(n, pieces, output_dir="", prefix=""):
    """
    Animate the reassembly process by placing each segmented piece onto an empty board (left panel)
    while displaying the available scrambled pieces (right panel) in a fixed layout.
    In the left panel, pieces placed in earlier steps are drawn in gray, while the most recently
    placed piece is highlighted in its original color.
    """
    instructions = []
    image_paths = []
    # Sort pieces for assembly (by top-left coordinate)
    sorted_pieces = sorted(pieces, key=lambda x: (x[0], x[2]))
    num_pieces = len(sorted_pieces)
    # colors = generate_piece_colors(num_pieces)
    
    # Precompute scrambled details for each piece based on the sorted order.
    # We'll assign a fixed random rotation and compute the rotated bounding box.
    scrambled_list = []
    for idx, piece in enumerate(sorted_pieces):
        r0, r1, c0, c1 = piece
        orig_width = c1 - c0
        orig_height = r1 - r0
        angle = random.choice([0, 30, 60, 90])
        theta = np.deg2rad(angle)
        new_w = abs(orig_width * np.cos(theta)) + abs(orig_height * np.sin(theta))
        new_h = abs(orig_width * np.sin(theta)) + abs(orig_height * np.cos(theta))
        label = f"{orig_width}x{orig_height}" if angle != 90 else f"{orig_height}x{orig_width}"
        scrambled_list.append({
            'index': idx,
            'angle': angle,
            'orig_width': orig_width,
            'orig_height': orig_height,
            'w': new_w,
            'h': new_h,
            'label': label,
            'piece': piece,
            'color': colors[idx].tolist()
        })
    
    shuffled_shapes = scrambled_list.copy()
    random.shuffle(shuffled_shapes)

    idx2label = {item['index']: item['label'] for item in scrambled_list}

    
    # Compute a fixed grid layout for the right panel based on all pieces.
    num_cols = math.ceil(math.sqrt(num_pieces))
    num_rows = math.ceil(num_pieces / num_cols)
    max_w = max(item['w'] for item in scrambled_list)
    max_h = max(item['h'] for item in scrambled_list)
    margin = 1
    cell_width = max_w + margin
    cell_height = max_h + margin
    canvas_width = num_cols * cell_width
    canvas_height = num_rows * cell_height
    
    # Assign a fixed grid cell center for each scrambled piece.
    for order, item in enumerate(scrambled_list):
        grid_row = order // num_cols
        grid_col = order % num_cols
        cell_x = grid_col * cell_width
        cell_y = canvas_height - (grid_row + 1) * cell_height
        cell_center = (cell_x + cell_width/2, cell_y + cell_height/2)
        item['cell_center'] = cell_center
    
    # For each assembly step, draw the board (left) and the fixed right panel.
    for step in range(num_pieces + 1):
        plt.figure(figsize=(14,8))
        
        # -----------------------
        # Left panel: Tangram board with placed pieces
        # -----------------------
        ax1 = plt.subplot(1, 2, 1)
        # Draw grid lines
        for r in range(n+1):
            ax1.plot([0, n], [r, r], color="lightgray", lw=1)
        for c in range(n+1):
            ax1.plot([c, c], [0, n], color="lightgray", lw=1)
        ax1.set_xlim(0, n)
        ax1.set_ylim(0, n)
        ax1.set_yticks(range(n+1))
        ax1.set_yticklabels(range(n+1), fontsize=ticks_fontsize)
        ax1.set_xticks(range(n+1))
        ax1.set_xticklabels(range(n+1), fontsize=ticks_fontsize)
        ax1.set_aspect("equal")
        ax1.set_title(f"The {n}x{n} Tangram Board", fontsize=title_fontsize)
        
        # Draw the placed pieces (those with sorted index < step)
        for idx in range(step):
            r0, r1, c0, c1 = shuffled_shapes[idx]['piece']
            index = shuffled_shapes[idx]['index']
            area = (r1 - r0) * (c1 - c0)
            # Use gray for pieces that were placed in earlier steps.
            # Only the current (most recently placed) piece uses its assigned color.
            facecolor = colors[idx] if idx == step - 1 else "lightgray"
            edgecolor = "black" if idx == step - 1 else "black"
            rect_patch = patches.Rectangle((c0, n - r1), c1 - c0, r1 - r0,
                                           facecolor=facecolor,
                                           alpha=0.6,
                                           edgecolor=edgecolor, lw=2)
            ax1.add_patch(rect_patch)
            center = ((c0+c1)/2, n - (r0+r1)/2)
            # ax1.text(center[0], center[1], f"{index}",
            #          ha="center", va="center", fontsize=label_fontsize, color="black")
            part_name = chr(65 + index)
            # fill all cells in the piece with part_name
            for r in range(r0, r1):
                for c in range(c0, c1):
                    ax1.text(c+0.5, n-r-0.5, part_name, ha="center", va="center", fontsize=label_fontsize, color="black")
        if step > 0:
            part_name = chr(65 + index)
            piece = shuffled_shapes[idx]['piece']
            r0, r1, c0, c1 = piece
            angle = shuffled_shapes[idx]['angle']
            rotate_instruction = f"Rotate piece {part_name} by about {angle} degrees clockwise, and p" if angle > 0 else "P"
            instruction = rotate_instruction + f"lace piece {part_name} with its upper-left corner at (x, y) = ({int(c0):d}, {n - int(r0):d})."

            instructions.append(instruction)
        else:
            instructions.append("")
        # ax1.set_xlabel(instruction)
        
        # -----------------------
        # Right panel: Available scrambled pieces (fixed layout)
        # -----------------------
        ax2 = plt.subplot(1, 2, 2)
        # Iterate through the fixed scrambled_list.
        # If a piece's sorted index is less than step, it has been placed and is skipped.
        for item_idx, item in enumerate(shuffled_shapes):
            if item_idx < step:
                continue  # piece already placed; leave its cell empty.
            index = item['index']
            part_name = chr(65 + index)
            angle = item['angle']
            orig_w = item['orig_width']
            orig_h = item['orig_height']
            cell_center = item['cell_center']
            rect_patch = patches.Rectangle((-orig_w/2, -orig_h/2), orig_w, orig_h,
                                           facecolor=colors[item_idx],
                                           edgecolor="black",
                                           lw=2,
                                           alpha=0.7)
            trans = (transforms.Affine2D().rotate_deg(angle)
                     .translate(cell_center[0], cell_center[1])
                     + ax2.transData)
            rect_patch.set_transform(trans)
            ax2.add_patch(rect_patch)
            ax2.text(cell_center[0], cell_center[1]+ item['h']/2+0.2, f"{part_name}: {idx2label[index]}",
                     ha="center", va="center", fontsize=label_fontsize, color="black")
            # Optionally, overlay a light-gray grid on the piece.
            for xi in range(0, int(orig_w) + 1):
                x = -orig_w/2 + xi
                line = plt.Line2D([x, x], [-orig_h/2, orig_h/2], color="black", lw=1, alpha=0.8)
                line.set_transform(trans)
                ax2.add_line(line)
            for yi in range(0, int(orig_h) + 1):
                y = -orig_h/2 + yi
                line = plt.Line2D([-orig_w/2, orig_w/2], [y, y], color="black", lw=1, alpha=0.8)
                line.set_transform(trans)
                ax2.add_line(line)
        
        ax2.set_xlim(0, canvas_width)
        ax2.set_ylim(0, canvas_height)
        ax2.axis("off")
        ax2.set_aspect("equal")
        ax2.set_title("Available Pieces", fontsize=title_fontsize)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"{prefix}assembly_step_{step}.png")
        plt.savefig(output_path)
        plt.close()
        image_paths.append(output_path)
    return shuffled_shapes, instructions, image_paths


# (Don't forget to import shapely.affinity for translation)
import shapely.affinity
from shapely.geometry import box
from shapely.ops import unary_union

def is_adjacent(rect1, rect2):
    """
    Check if two rectangles (r0, r1, c0, c1) are adjacent.
    Two rectangles are considered adjacent if they share a boundary segment with positive length.
    """
    r0, r1, c0, c1 = rect1
    s0, s1, t0, t1 = rect2
    # Check vertical adjacency: share a vertical edge
    if (abs(c1 - t0) < 1e-9 or abs(t1 - c0) < 1e-9):
        overlap = max(0, min(r1, s1) - max(r0, s0))
        if overlap > 0:
            return True
    # Check horizontal adjacency: share a horizontal edge
    if (abs(r1 - s0) < 1e-9 or abs(s1 - r0) < 1e-9):
        overlap = max(0, min(c1, t1) - max(c0, t0))
        if overlap > 0:
            return True
    return False


def randomly_combine_two(pieces, ignore_rect=False):
    """
    Randomly select two adjacent pieces from a list of rectangles (each as (r0, r1, c0, c1))
    and combine them into an irregular shape using Shapely's union.
    
    If no adjacent pair is found, the original pieces (converted to Shapely boxes) are returned.
    
    Returns:
      new_shapes: a list of Shapely geometries, where two adjacent pieces have been merged.
    """
    # Convert each rectangle into a Shapely box (treating columns as x and rows as y).
    shapely_pieces = [box(c0, r0, c1, r1) for (r0, r1, c0, c1) in pieces]
    
    # Find all pairs of indices for which the corresponding rectangles are adjacent.
    adjacent_pairs = []
    n = len(pieces)
    for i in range(n):
        for j in range(i+1, n):
            if is_adjacent(pieces[i], pieces[j]):
                adjacent_pairs.append((i, j))
    
    # If no adjacent pairs exist, return the original list as Shapely geometries.
    if not adjacent_pairs:
        return shapely_pieces
    
    trials = 0
    while trials < 10:
        # Randomly choose one adjacent pair.
        i, j = random.choice(adjacent_pairs)
        
        # Combine the two chosen pieces using union.
        combined_shape = shapely_pieces[i].union(shapely_pieces[j])

        # check if the union is a rectangle?


        if not combined_shape.equals(combined_shape.minimum_rotated_rectangle) or ignore_rect:
            break
        trials += 1
    
    if trials == 10 and combined_shape.equals(combined_shape.minimum_rotated_rectangle):
        return None, None
    
    # Build a new list of shapes: omit the two that were combined and add the new merged shape.
    new_shapes = []
    newidx2oldidx = {}
    for idx in range(n):
        if idx == i or idx == j:
            continue
        new_shapes.append(shapely_pieces[idx])
        newidx2oldidx[len(new_shapes)-1] = idx
    new_shapes.append(combined_shape)
    newidx2oldidx[len(new_shapes)-1] = (i, j)
    
    return new_shapes, newidx2oldidx

import matplotlib.patches as patches
import matplotlib.pyplot as plt

def draw_shape(ax, shape, facecolor, edgecolor="black", lw=2, alpha=0.6):
    """
    Draw a shapely geometry (Polygon or MultiPolygon) on the given axis using a Polygon patch.
    """
    if shape.geom_type == "Polygon":
        x, y = shape.exterior.xy
        patch = patches.Polygon(xy=list(zip(x, y)), facecolor=facecolor, edgecolor=edgecolor, lw=lw, alpha=alpha)
        ax.add_patch(patch)
    elif shape.geom_type == "MultiPolygon":
        for poly in shape:
            x, y = poly.exterior.xy
            patch = patches.Polygon(xy=list(zip(x, y)), facecolor=facecolor, edgecolor=edgecolor, lw=lw, alpha=alpha)
            ax.add_patch(patch)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely.affinity
from shapely.geometry import LineString
import math
import random

def draw_shape(ax, shape, facecolor, edgecolor="black", lw=2, alpha=0.6):
    """
    Draw a shapely geometry (Polygon or MultiPolygon) on the given axis using a Polygon patch.
    """
    if shape.geom_type == "Polygon":
        x, y = shape.exterior.xy
        patch = patches.Polygon(xy=list(zip(x, y)), facecolor=facecolor, edgecolor=edgecolor, lw=lw, alpha=alpha)
        ax.add_patch(patch)
    elif shape.geom_type == "MultiPolygon":
        for poly in shape:
            x, y = poly.exterior.xy
            patch = patches.Polygon(xy=list(zip(x, y)), facecolor=facecolor, edgecolor=edgecolor, lw=lw, alpha=alpha)
            ax.add_patch(patch)

from shapely.geometry import LineString, Point, MultiPoint

def get_composite_transform(angle, flip, target_center, shape):
    """
    Compute a composite affine transformation (as a 6-tuple) that:
      1. Translates the shape so its centroid is at (0,0)
      2. Applies a flip (if any) about (0,0)
      3. Rotates by 'angle' degrees about (0,0)
      4. Translates so that the shape is moved to target_center.
    The returned 6-tuple is in the form (a, b, d, e, xoff, yoff), suitable for shapely.affinity.affine_transform.
    """
    centroid = shape.centroid
    trans = transforms.Affine2D()
    trans.translate(-centroid.x, -centroid.y)
    if flip == "horizontal":
        trans.scale(-1, 1)
    elif flip == "vertical":
        trans.scale(1, -1)
    trans.rotate_deg(angle)
    trans.translate(target_center[0], target_center[1])
    # Extract the 3x3 matrix and convert to the 6-tuple form.
    M = trans.get_matrix()
    a = M[0,0]
    b = M[0,1]
    d = M[1,0]
    e = M[1,1]
    xoff = M[0,2]
    yoff = M[1,2]
    return (a, b, d, e, xoff, yoff)

from shapely.affinity import affine_transform
from shapely.geometry import LineString

def draw_transformed_grid_lines(ax, original_shape, angle, flip, target_center, spacing=1, color="lightgray", lw=1, alpha=0.8):
    """
    Compute grid lines (vertical and horizontal) on the original shape (before transformation),
    then apply a composite transformation (computed once) to all grid lines, and finally draw the resulting segments.
    """
    # Compute the composite transform once.
    transform_tuple = get_composite_transform(angle, flip, target_center, original_shape)
    
    minx, miny, maxx, maxy = original_shape.bounds
    # Draw vertical grid lines.
    for x in np.arange(math.floor(minx), math.ceil(maxx)+1, spacing):
        line = LineString([(x, miny), (x, maxy)])
        clipped = original_shape.intersection(line)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "LineString":
            transformed_line = affine_transform(clipped, transform_tuple)
            xs, ys = transformed_line.xy
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)
        elif clipped.geom_type == "MultiLineString":
            for seg in clipped.geoms:
                transformed_line = affine_transform(seg, transform_tuple)
                xs, ys = transformed_line.xy
                ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)
    # Draw horizontal grid lines.
    for y in np.arange(math.floor(miny), math.ceil(maxy)+1, spacing):
        line = LineString([(minx, y), (maxx, y)])
        clipped = original_shape.intersection(line)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "LineString":
            transformed_line = affine_transform(clipped, transform_tuple)
            xs, ys = transformed_line.xy
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)
        elif clipped.geom_type == "MultiLineString":
            for seg in clipped.geoms:
                transformed_line = affine_transform(seg, transform_tuple)
                xs, ys = transformed_line.xy
                ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)
        # print("clipped", clipped)
        # print("transformed_line", transformed_line)


def find_actual_corner_point(shape, corner_type="upper-left"):
    """
    Find the actual corner point of a shape based on the requested corner type.
    
    Parameters:
        shape: The Shapely geometry
        corner_type: One of "upper-left", "upper-right", "lower-left", "lower-right"
        n: Board size (only needed for translating y-coordinates)
        
    Returns:
        A tuple (r, c) representing the corner point coordinates
    """
    # Get the boundary points of the shape
    if shape.geom_type == "Polygon":
        boundary_points = list(shape.exterior.coords)
    else:
        # For MultiPolygon, get points from all parts
        boundary_points = []
        for poly in shape.geoms:
            boundary_points.extend(list(poly.exterior.coords))
    
    # Convert boundary points to a list of (r, c) tuples
    points = [(x, y) for x, y in boundary_points]
    
    # Sort points based on corner type
    if corner_type == "upper-left":
        # Lowest X (column), highest Y (lowest row)
        sorted_points = sorted(points, key=lambda p: (p[0], -p[1]))
    elif corner_type == "upper-right":
        # Highest X (column), highest Y (lowest row)
        sorted_points = sorted(points, key=lambda p: (-p[0], -p[1]))
    elif corner_type == "lower-left":
        # Lowest X (column), lowest Y (highest row)
        sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    elif corner_type == "lower-right":
        # Highest X (column), lowest Y (highest row)
        sorted_points = sorted(points, key=lambda p: (-p[0], p[1]))
    
    # Return the first point from the sorted list
    point = sorted_points[0]
    
    # Convert to integer coordinates
    r, c = point[1], point[0]
    
    return (c, r)


def animate_reassembly_irregular(n, shapes, output_dir="", prefix="", invalid_idx=None):
    """
    Animate the reassembly process using irregular (merged) shapes.
    In the left panel, shapes placed in earlier steps are drawn in gray and only the current shape is highlighted.
    The right panel displays available shapes in a fixed grid layout.
    Each available shape is randomly rotated and flipped, and then grid lines are drawn clipped onto the shape.
    
    Parameters:
      n: size of the Tangram board (n x n)
      shapes: list of shapely geometries representing irregular pieces.
    """
    instructions = []
    image_paths = []
    results = []
    first_invalid_step = -1

    #record the merged shape index
    is_merged = [False] * len(shapes)
    if invalid_idx is None:
        is_merged[-1] = True
    else:
        is_merged[invalid_idx] = True 
    # Sort and prepare transforms (same as before)...
        
    # sort both shapes and is_merged using the y coordinate of the bounding box
        
    sorted_shapes, sorted_is_merged = zip(*sorted(zip(shapes, is_merged), key=lambda shp: shp[0].bounds[1]))

    # how to know the idx after sorting?

    num_shapes = len(sorted_shapes)
    transform_params = {}
    shuffled_shapes = []
    for order in range(num_shapes):
        angle = random.choice([0,30,60,90])
        flip = (random.choice(["none","horizontal","vertical"]) 
                if not sorted_shapes[order].equals(sorted_shapes[order].minimum_rotated_rectangle) else "none")
        transform_params[order] = (angle, flip)
        shuffled_shapes.append({"index":order, "shape":sorted_shapes[order],
                                "angle":angle, "flip":flip,
                                "color":colors[order].tolist(),
                                "is_merged":sorted_is_merged[order]})

    
    # For the right panel, define a fixed grid layout.
    # First, precompute each shape’s transformed extents (without translation) to determine max width/height.
    transformed_extents = []
    for order, shape in enumerate(sorted_shapes):
        centroid = shape.centroid
        temp = shapely.affinity.translate(shape, xoff=-centroid.x, yoff=-centroid.y)
        angle, flip = transform_params[order]
        if flip == "horizontal":
            temp = shapely.affinity.scale(temp, xfact=-1, yfact=1, origin=(0,0))
        elif flip == "vertical":
            temp = shapely.affinity.scale(temp, xfact=1, yfact=-1, origin=(0,0))
        temp = shapely.affinity.rotate(temp, angle, origin=(0,0))
        minx, miny, maxx, maxy = temp.bounds
        transformed_extents.append((maxx - minx, maxy - miny))
    max_width = max(w for w, h in transformed_extents)
    max_height = max(h for w, h in transformed_extents)
    margin = 1
    cell_width = max_width + margin
    cell_height = max_height + margin
    num_cols = math.ceil(math.sqrt(num_shapes))
    num_rows = math.ceil(num_shapes / num_cols)
    canvas_width = num_cols * cell_width
    canvas_height = num_rows * cell_height
    
    # Assign fixed grid cell centers.
    grid_positions = []
    for order in range(num_shapes):
        grid_row = order // num_cols
        grid_col = order % num_cols
        cell_x = grid_col * cell_width
        cell_y = canvas_height - (grid_row + 1) * cell_height
        cell_center = (cell_x + cell_width/2, cell_y + cell_height/2)
        grid_positions.append(cell_center)
    
    def apply_transform(shape, angle, flip, target_center):
        """
        Apply a composite transformation to a shape:
          1. Translate so that its centroid is at (0,0)
          2. Apply the flip (if any) about (0,0)
          3. Rotate by the given angle about (0,0)
          4. Translate so that the shape’s centroid goes to target_center.
        """
        centroid = shape.centroid
        transformed = shapely.affinity.translate(shape, xoff=-centroid.x, yoff=-centroid.y)
        if flip == "horizontal":
            transformed = shapely.affinity.scale(transformed, xfact=-1, yfact=1, origin=(0,0))
        elif flip == "vertical":
            transformed = shapely.affinity.scale(transformed, xfact=1, yfact=-1, origin=(0,0))
        transformed = shapely.affinity.rotate(transformed, angle, origin=(0,0))
        transformed = shapely.affinity.translate(transformed, xoff=target_center[0], yoff=target_center[1])
        return transformed
    

    
    random.shuffle(shuffled_shapes)
    # Animate each assembly step.
    for step in range(num_shapes + 1):
        plt.figure(figsize=(14,8))
        
        # -----------------------
        # Left panel: Tangram board with placed shapes.
        # -----------------------
        ax1 = plt.subplot(1, 2, 1)
        # Draw board grid.
        for r in range(n+1):
            ax1.plot([0, n], [r, r], color="lightgray", lw=1)
        for c in range(n+1):
            ax1.plot([c, c], [0, n], color="lightgray", lw=1)
        ax1.set_xlim(0, n)
        ax1.set_ylim(0, n)
        ax1.set_yticks(range(n+1))
        ax1.set_yticklabels(range(n+1), fontsize=ticks_fontsize)
        ax1.set_xticks(range(n+1))
        ax1.set_xticklabels(range(n+1), fontsize=ticks_fontsize)
        ax1.set_aspect("equal")
        ax1.set_title(f"The {n}x{n} Tangram Board", fontsize=title_fontsize)
        
        for idx in range(step):
            # Draw previously placed shapes in gray; highlight the current shape.
            index = shuffled_shapes[idx]["index"]
            shape = shuffled_shapes[idx]["shape"]
            facecolor = colors[index] if idx == step - 1 else "lightgray"
            draw_shape(ax1, shape, facecolor=facecolor, edgecolor="black", lw=2, alpha=0.6)
            center = shape.centroid
            # ax1.text(center.x, center.y, f"{index}",
            #          ha="center", va="center", fontsize=label_fontsize, color="black")
            part_name = chr(65 + index)
            # fill all cells in the piece with part_name, note that the shape is not necessarily a rectangle
            # check if the shape is a rectangle
            # if shape.equals(shape.minimum_rotated_rectangle):
            #     r0, r1, c0, c1 = shape.bounds
            #     for r in range(int(r0), int(r1)):
            #         for c in range(int(c0), int(c1)):
            #             ax1.text(r+0.5, c+0.5, part_name, ha="center", va="center", fontsize=label_fontsize, color="black")
            # else:
            #     # if the shape is not a rectangle, fill the cells with the part_name

            # get all the points contained in the shape, also need to consider the points on the boundary
            # print("

            for r in range(n):
                for c in range(n):
                    # check if the point is within the patch?
                    point = Point(r+0.5, c+0.5)
                    if shape.contains(point):
                        ax1.text(r+0.5, c+0.5, part_name, ha="center", va="center", fontsize=label_fontsize, color="black")
        # upperleft corner of the shape
        if step > 0:
            shape = shuffled_shapes[step-1]["shape"]
            points = list(shape.exterior.coords)
            r0, r1, c0, c1 = shape.bounds
            # need to check whether the shape is occupying the space at the upperleft corner
            if shape.equals(shape.minimum_rotated_rectangle):
                desc_point = (r0, c1)
                desc_point_location = "upper-left"
            else:
                desc_point = (r0, c1)
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>shape.bounds", shape.bounds)
                desc_point_location = "upper-left"
                actual_point = find_actual_corner_point(shape, corner_type=desc_point_location)
                # check if upperleft is in the shape
                if desc_point != actual_point:
                    print("upper-left mismatch", "desc_point", desc_point, "actual_point", actual_point)
                    desc_point = (r0, c0)
                    desc_point_location = "upper-right"
                    actual_point = find_actual_corner_point(shape, corner_type=desc_point_location)
                    if desc_point != actual_point:
                        print("upper-right mismatch", "desc_point", desc_point, "actual_point", actual_point)
                        desc_point = (r1, c0)
                        desc_point_location = "lower-right"
                        actual_point = find_actual_corner_point(shape, corner_type=desc_point_location)
                        if desc_point != actual_point:
                            print("lower-right mismatch", "desc_point", desc_point, "actual_point", actual_point)
                            desc_point = (r0, c0)
                            desc_point_location = "lower-left"
                            actual_point = find_actual_corner_point(shape, corner_type=desc_point_location)
                            if desc_point != actual_point:
                                print("lower-left mismatch", "desc_point", desc_point, "actual_point", actual_point)

            # if not, may need to use upperright corner
            # print("shape.bounds", shape.bounds)
            angle, flip = shuffled_shapes[step-1]["angle"], shuffled_shapes[step-1]["flip"]
            if angle > 0:
                rotate_instruction = f"Rotate piece {part_name} by about {angle} degrees clockwise."
            else:
                rotate_instruction = ""
            if flip == "horizontal":
                flip_instruction = f"Flip piece {part_name} horizontally."
            elif flip == "vertical":
                flip_instruction = f"Flip piece {part_name} vertically."
            else:
                flip_instruction = ""
            instruction = f"{rotate_instruction} {flip_instruction} "
            if rotate_instruction or flip_instruction:
                instruction = instruction + f"Place piece {part_name} with its {desc_point_location} corner at (x,y) = ({int(desc_point[0]):d}, {int(desc_point[1]):d})."
            else:
                instruction = f"Place piece {part_name} with its {desc_point_location} corner at (x,y) = ({int(desc_point[0]):d}, {int(desc_point[1]):d})."
            instructions.append(instruction)
        else:
            instructions.append("")
        # instruction = f"Placed shape {step-1}" if step > 0 else "No shape placed yet"
        # ax1.set_xlabel(instruction)
            
        # Invalid detection
        res = ''
        if step>0 and (step-1)==invalid_idx:
            res = 'Invalid irregular'
            ax1.text(n/2, -0.5, 'Puzzle unsolvable', ha='center', color='red', fontsize=14)
        results.append(res)
        if res and first_invalid_step < 0:
            first_invalid_step = step
        if first_invalid_step >= 0 and step > first_invalid_step:
            plt.close(); continue
        
        # -----------------------
        # Right panel: Available shapes with random rotate/flip and grid lines onto the pieces.
        # -----------------------
        ax2 = plt.subplot(1, 2, 2)
        for idx, item in enumerate(shuffled_shapes):

            if idx < step:
                continue  # Skip shapes already placed.
            index = item["index"]
            shape = item["shape"]
            part_name = chr(65 + index)
            angle, flip = item["angle"], item["flip"]
            cell_center = grid_positions[index]
            trans_func = lambda geom: apply_transform(geom, angle, flip, cell_center)
            transformed_shape = trans_func(shape)
            height = transformed_shape.bounds[3] - transformed_shape.bounds[1]
            
            # top_y = transformed_shape.bounds[1]


            # shape_center = shape.centroid

            # trans_line_func = lambda geom: apply_line_transform(geom, angle, cell_center, shape_center)
            # Draw the transformed shape.
            draw_shape(ax2, transformed_shape, facecolor=colors[index], edgecolor="black", lw=2, alpha=0.7)
            ax2.text(cell_center[0], cell_center[1] + height/2. + 0.2, f"{part_name}", ha="center", va="center",
                     fontsize=label_fontsize, color="black")
            # Draw grid lines clipped to the shape.
            draw_transformed_grid_lines(ax2, shape, angle, flip, cell_center,
                                spacing=1, color="black", lw=1, alpha=0.8)
        
        ax2.set_xlim(0, canvas_width)
        ax2.set_ylim(0, canvas_height)
        ax2.axis("off")
        ax2.set_aspect("equal")
        ax2.set_title("Avaialable Pieces", fontsize=title_fontsize)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"{prefix}irregular_assembly_step_{step}.png")
        plt.savefig(output_path)
        plt.close()
        image_paths.append(output_path)
    return shuffled_shapes, instructions, image_paths, results



def perturb_pieces(pieces, n):
    """
    Given a list of rectangular pieces (each as (r0, r1, c0, c1)) that exactly tile the board,
    randomly perturb them so that the puzzle becomes unsolvable.
    
    Three options:
      1. Add an extra piece.
      2. Enlarge one piece (thus increasing its area).
      3. Transform one piece: For a piece with area > 2, transform it by doubling its shortest side
         and halving its longest side—only if the side to be halved is divisible by 2.
    
    Returns the new list of pieces (a mix of tuples for unmodified pieces and polygons for perturbed ones).
    """
    new_pieces = pieces.copy()
    option = random.choice([1, 2, 3])
    
    if option == 1:
        # Option 1: add an extra piece.
        extra_w = random.choice([1, 2])
        extra_h = random.choice([1, 2])
        max_c0 = n - extra_w
        max_r0 = n - extra_h
        c0 = random.randint(0, max_c0)
        r0 = random.randint(0, max_r0)
        extra_piece = (r0, r0 + extra_h, c0, c0 + extra_w)
        new_pieces.append(extra_piece)
        print("Perturbation option 1 applied: extra piece added.")
        
    elif option == 2:
        # Option 2: enlarge one piece.
        idx = random.randrange(len(new_pieces))
        piece = new_pieces[idx]
        r0, r1, c0, c1 = piece
        new_piece = (r0, r1 + 1, c0, c1 + 1)
        new_pieces[idx] = new_piece
        print(f"Perturbation option 2 applied: piece {idx} enlarged.")
        
    elif option == 3:
        # Option 3: transform one piece by doubling the shortest side and halving the longest side.
        # Only consider pieces with area > 2 and where the side to be halved is divisible by 2.
        eligible_indices = []
        for i, piece in enumerate(new_pieces):
            area = (piece[1] - piece[0]) * (piece[3] - piece[2])
            if area > 2:
                width = piece[3] - piece[2]
                height = piece[1] - piece[0]
                if width == height:
                    # For a square, require at least one side even.
                    if width % 2 == 0:
                        eligible_indices.append(i)
                else:
                    if width < height:
                        # Will double width and halve height; ensure height is even.
                        if height % 2 == 0:
                            eligible_indices.append(i)
                    else:
                        # width >= height; will double height and halve width; ensure width is even.
                        if width % 2 == 0:
                            eligible_indices.append(i)
        if not eligible_indices:
            print("No eligible piece found for option 3; no perturbation applied.")
            return None
        
        idx = random.choice(eligible_indices)
        piece = new_pieces[idx]
        r0, r1, c0, c1 = piece
        width = c1 - c0
        height = r1 - r0
        
        if width == height:
            # Square: randomly choose one transformation.
            if random.choice([True, False]):
                new_width = width * 2
                new_height = height // 2  # height is even
            else:
                new_width = width // 2
                new_height = height * 2
        else:
            if width < height:
                new_width = width * 2
                new_height = height // 2  # height is even per our check
            else:
                new_width = width // 2  # width is even per our check
                new_height = height * 2
        
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        new_piece = (r0, r0 + new_height, c0, c0 + new_width)
        new_pieces[idx] = new_piece
        print(f"Perturbation option 3 applied: piece {idx} transformed (shortest side doubled, longest side halved).")
        print(f"Old piece: {piece}; New piece: {new_piece}")
        
    return new_pieces


def get_piece_bbox(piece, n):
    """
    Return the bounding box of a piece as (r0, r1, c0, c1).
    If piece is a tuple (regular rectangle), return it directly.
    If piece is a list (irregular polygon given as vertices in board coordinates,
    where x = column and y = n - row), compute the bounding box.
    """
    if isinstance(piece, tuple):
        return piece
    else:
        xs = [p[0] for p in piece]
        ys = [p[1] for p in piece]
        c0 = min(xs)
        c1 = max(xs)
        # Convert y back to row coordinate: row = n - y.
        r1 = n - min(ys)
        r0 = n - max(ys)
        return (r0, r1, c0, c1)

def animate_assembly_perturbed(n, pieces, output_dir="", prefix=""):
    """
    Animate the assembly process for perturbed pieces.
    
    In the left panel (the board), pieces are placed in sorted order (by their bounding box’s top‐left).
    – For options 1 and 2, either an extra piece remains on the right panel or a piece is drawn partly over the board edge.
    – For option 3, if after placing all pieces the board is not exactly covered, a "Puzzle unsolvable" annotation is shown.
    
    The right panel displays the remaining pieces (based on their bounding boxes) arranged in a fixed grid.
    """
    instructions = []
    image_paths = []
    results = []
    # Sort pieces by their bounding box's top-left coordinate.
    sorted_pieces = sorted(pieces, key=lambda p: (get_piece_bbox(p, n)[0], get_piece_bbox(p, n)[2]))
    num = len(sorted_pieces)
    # colors = generate_piece_colors(num)
    
    # For the right panel, precompute a scrambled layout based on each piece’s bounding box.
    scrambled_list = []
    for idx, piece in enumerate(sorted_pieces):
        r0, r1, c0, c1 = get_piece_bbox(piece, n)
        orig_w = c1 - c0
        orig_h = r1 - r0
        # Choose a random rotation angle for display.
        angle = random.choice([0, 30, 60, 90])
        theta = np.deg2rad(angle)
        new_w = abs(orig_w * np.cos(theta)) + abs(orig_h * np.sin(theta))
        new_h = abs(orig_w * np.sin(theta)) + abs(orig_h * np.cos(theta))
        label = f"{orig_w}x{orig_h}" if angle != 90 else f"{orig_h}x{orig_w}"
        scrambled_list.append({
            'index': idx,
            'angle': angle,
            'orig_width': orig_w,
            'orig_height': orig_h,
            'w': new_w,
            'h': new_h,
            'label': label,
            'piece': piece,
            'color': colors[idx].tolist()
        })
    
    shuffled_shapes = scrambled_list.copy()
    random.shuffle(shuffled_shapes)

    idx2label = {item['index']: item['label'] for item in shuffled_shapes}
    
    # Determine a fixed grid layout for the right panel.
    num_cols = math.ceil(math.sqrt(num))
    num_rows = math.ceil(num / num_cols)
    max_w = max(item['w'] for item in shuffled_shapes)
    max_h = max(item['h'] for item in shuffled_shapes)
    margin = 1
    cell_width = max_w + margin
    cell_height = max_h + margin
    canvas_width = num_cols * cell_width
    canvas_height = num_rows * cell_height
    grid_positions = []
    for order in range(num):
        grid_row = order // num_cols
        grid_col = order % num_cols
        cell_x = grid_col * cell_width
        cell_y = canvas_height - (grid_row + 1) * cell_height
        grid_positions.append((cell_x + cell_width/2, cell_y + cell_height/2))
    
    # Assembly animation: one step per number of placed pieces.

    result = ""
    perturb_step = -1
    for step in range(num + 1):
        plt.figure(figsize=(14,8))
        ax1 = plt.subplot(1, 2, 1)
        # Draw board grid.
        for r in range(n+1):
            ax1.plot([0, n], [r, r], color="lightgray", lw=1)
        for c in range(n+1):
            ax1.plot([c, c], [0, n], color="lightgray", lw=1)
        ax1.set_xlim(0, n)
        ax1.set_ylim(0, n)
        ax1.set_yticks(range(n+1))
        ax1.set_yticklabels(range(n+1), fontsize=ticks_fontsize)
        ax1.set_xticks(range(n+1))
        ax1.set_xticklabels(range(n+1), fontsize=ticks_fontsize)
        ax1.set_aspect("equal")
        ax1.set_title(f"The {n}x{n} Tangram Board", fontsize=title_fontsize)
        
        # Place pieces 0 to step-1 on the board.
        for idx in range(step):
            piece = shuffled_shapes[idx]['piece']
            index = shuffled_shapes[idx]['index']
            bbox = get_piece_bbox(piece, n)
            r0, r1, c0, c1 = bbox
            print("bbox", bbox)
            # Check if piece lies completely within the board.
            over_board = (r0 < 0 or r1 > n or c0 < 0 or c1 > n)
            # Draw the piece: current piece in its assigned color, earlier pieces in gray.
            facecolor = colors[idx] if idx == step - 1 else "lightgray"
            if isinstance(piece, tuple):
                patch = patches.Rectangle((c0, n - r1), c1 - c0, r1 - r0,
                                           facecolor=facecolor, edgecolor="black", lw=2, alpha=0.6)
            else:
                patch = patches.Polygon(piece, closed=True, facecolor=facecolor, edgecolor="black", lw=2, alpha=0.6)
            ax1.add_patch(patch)
            center = ((c0 + c1) / 2, n - (r0 + r1) / 2)
            # ax1.text(center[0], center[1], f"{index}", ha="center", va="center", fontsize=label_fontsize, color="black")
            part_name = chr(65 + index)
            # fill all cells in the piece with part_name
            for r in range(r0, r1):
                for c in range(c0, c1):
                    ax1.text(c+0.5, n-r-0.5, part_name, ha="center", va="center", fontsize=label_fontsize, color="black")
            if over_board:
                # ax1.text(center[0], center[1]-0.5, "Over board", ha="center", va="center", fontsize=10, color="red")
                result = "Over board"

            # check overlap
            for i in range(idx):
                piece_i = shuffled_shapes[i]['piece']
                if isinstance(piece_i, tuple):
                    bbox_i = get_piece_bbox(piece_i, n)
                    r0_i, r1_i, c0_i, c1_i = bbox_i
                    if r0_i < r1 and r1_i > r0 and c0_i < c1 and c1_i > c0:
                        # ax1.text(n/2, -0.5, "Overlap", ha="center", va="center", fontsize=14, color="red")
                        result = "Overlap"
                        break
        
        # # At final step, check if the union of pieces covers the board.
        # if step == num:
        #     # (A simple heuristic: if any piece was perturbed, assume unsolvability.)
        #     ax1.text(n/2, -0.5, "Puzzle unsolvable", ha="center", va="center", fontsize=14, color="red")
        results.append(result)
        if result != "" and perturb_step == -1:
            perturb_step = step
        if step > 0:
            part_name = chr(65 + index)
            angle = scrambled_list[idx]['angle']
            rotate_instruction = f"Rotate piece {part_name} by about {angle} degrees clockwise, and p" if angle > 0 else "P"
            instruction = rotate_instruction + f"lace piece {part_name} with its upper-left corner at (x, y) = ({int(c0):d}, {n - int(r0):d})."

            instructions.append(instruction)
        else:
            instructions.append("")
        
        # Right panel: display available (unplaced) pieces.
        ax2 = plt.subplot(1, 2, 2)
        for item_i, item in enumerate(shuffled_shapes):
            if item_i < step:
                continue  # already placed.
            pos = grid_positions[item['index']]
            part_name = chr(65 + item['index'])
            orig_w = item['orig_width']
            orig_h = item['orig_height']
            rect_patch = patches.Rectangle((-item['orig_width']/2, -item['orig_height']/2),
                                           item['orig_width'], item['orig_height'],
                                           facecolor=colors[item_i],
                                           edgecolor="black", lw=2, alpha=0.7)
            trans = transforms.Affine2D().rotate_deg(item['angle']).translate(pos[0], pos[1]) + ax2.transData
            rect_patch.set_transform(trans)
            ax2.add_patch(rect_patch)
            ax2.text(pos[0], pos[1]+ item['h']/2+0.2, f"{part_name}: {item['label']}", ha="center", va="center", fontsize=label_fontsize, color="black")
            # Optionally, overlay a light-gray grid on the piece.
            for xi in range(0, int(orig_w) + 1):
                x = -orig_w/2 + xi
                line = plt.Line2D([x, x], [-orig_h/2, orig_h/2],
                                  color="black", lw=1, alpha=0.8)
                line.set_transform(trans)
                ax2.add_line(line)
            for yi in range(0, int(orig_h) + 1):
                y = -orig_h/2 + yi
                line = plt.Line2D([-orig_w/2, orig_w/2], [y, y],
                                  color="black", lw=1, alpha=0.8)
                line.set_transform(trans)
                ax2.add_line(line)
        ax2.set_xlim(0, canvas_width)
        ax2.set_ylim(0, canvas_height)
        ax2.axis("off")
        ax2.set_aspect("equal")
        ax2.set_title("Avaialable Pieces", fontsize=title_fontsize)
        
        plt.tight_layout()
        # plt.savefig(f"perturbed_assembly_step_{step}.png")
        
        output_path = os.path.join(output_dir, f"{prefix}perturbed_part_assembly_step_{step}.png")
        plt.savefig(output_path)
        plt.close()
        if step > perturb_step and perturb_step != -1:
            continue
        image_paths.append(output_path)
    return shuffled_shapes, instructions, image_paths, results


def perturb_step(piece, board_size, max_offset=1):
    """
    Given a piece (as a tuple (r0, r1, c0, c1)),
    return a perturbed version by shifting all coordinates by a random integer offset 
    between -max_offset and max_offset.
    This will change its position so that it no longer fits correctly.
    """
    r0, r1, c0, c1 = piece
    new_r0, new_r1, new_c0, new_c1 = r0, r1, c0, c1
    left_corner_constraint = new_r0 < 0 or new_c0 <0 or new_r0 > board_size or new_c0 > board_size

    same_place = new_r0 == r0 and new_c0 == c0
    while same_place or left_corner_constraint:
        delta_r, delta_c = 0, 0
        while delta_r == 0 and delta_c == 0:
            delta_r = random.randint(-max_offset, max_offset)
            delta_c = random.randint(-max_offset, max_offset)
        new_r0 = r0 + delta_r
        new_r1 = r1 + delta_r
        new_c0 = c0 + delta_c
        new_c1 = c1 + delta_c
        left_corner_constraint = new_r0 < 0 or new_c0 <0 or new_r0 > board_size or new_c0 > board_size
        same_place = new_r0 == r0 and new_c0 == c0
    return (new_r0, new_r1, new_c0, new_c1)

def animate_assembly_perturbed_step(n, pieces, perturbed_index, max_offset=1, output_dir="", prefix=""):
    """
    Animate the assembly process until (and including) a perturbed step.
    
    Parameters:
      n: board size (n x n)
      pieces: list of pieces (each as a tuple (r0, r1, c0, c1))
      perturbed_index: the index (in the assembly order) at which to apply a perturbation.
                       The piece at that step will be replaced by a perturbed version.
    
    In the left panel (the board), pieces placed in earlier steps are drawn in gray,
    except the current piece is drawn in its assigned color.
    At the perturbed step the selected piece is misaligned.
    The right panel shows the remaining (unplaced) pieces in a fixed grid.
    """
    instructions = []
    image_paths = []
    results = []
    # Sort pieces for assembly (by top-left coordinate)
    sorted_pieces = sorted(pieces, key=lambda x: (x[0], x[2]))
    num_pieces = len(sorted_pieces)
    # colors = generate_piece_colors(num_pieces)
    
    # Precompute a fixed scrambled layout for the right panel (same as before).
    scrambled_list = []
    for idx, piece in enumerate(sorted_pieces):
        r0, r1, c0, c1 = piece
        orig_width = c1 - c0
        orig_height = r1 - r0
        angle = random.choice([0, 30, 60, 90])
        theta = np.deg2rad(angle)
        new_w = abs(orig_width * np.cos(theta)) + abs(orig_height * np.sin(theta))
        new_h = abs(orig_width * np.sin(theta)) + abs(orig_height * np.cos(theta))
        label = f"{orig_width}x{orig_height}"
        scrambled_list.append({
            'index': idx,
            'angle': angle,
            'orig_width': orig_width,
            'orig_height': orig_height,
            'w': new_w,
            'h': new_h,
            'label': label,
            'piece': piece,
            'color': colors[idx].tolist()
        })
    idx2label = {item['index']: item['label'] for item in scrambled_list}
    random.shuffle(scrambled_list)
    
    # Compute fixed grid layout for the right panel.
    num_cols = math.ceil(math.sqrt(num_pieces))
    num_rows = math.ceil(num_pieces / num_cols)
    max_w = max(item['w'] for item in scrambled_list)
    max_h = max(item['h'] for item in scrambled_list)
    margin = 1
    cell_width = max_w + margin
    cell_height = max_h + margin
    canvas_width = num_cols * cell_width
    canvas_height = num_rows * cell_height
    for order, item in enumerate(scrambled_list):
        grid_row = order // num_cols
        grid_col = order % num_cols
        cell_x = grid_col * cell_width
        cell_y = canvas_height - (grid_row + 1) * cell_height
        item['cell_center'] = (cell_x + cell_width/2, cell_y + cell_height/2)
    

    is_perturbed_step = False
    perturbed_step_index = -1
    # Assembly animation: iterate step-by-step.
    for step in range(num_pieces + 1):
        plt.figure(figsize=(14,8))
        
        # Left panel: the board.
        ax1 = plt.subplot(1, 2, 1)
        # Draw board grid.
        for r in range(n+1):
            ax1.plot([0, n], [r, r], color="lightgray", lw=1)
        for c in range(n+1):
            ax1.plot([c, c], [0, n], color="lightgray", lw=1)
        ax1.set_xlim(0, n)
        ax1.set_ylim(0, n)
        ax1.set_yticks(range(n+1))
        ax1.set_yticklabels(range(n+1), fontsize=ticks_fontsize)
        ax1.set_xticks(range(n+1))
        ax1.set_xticklabels(range(n+1), fontsize=ticks_fontsize)
        ax1.set_aspect("equal")
        ax1.set_title(f"The {n}x{n} Tangram Board", fontsize=title_fontsize)

        curr_step = step - 1
        curr_step_index = scrambled_list[curr_step]['index']
        is_perturbed_step = curr_step_index == perturbed_index  or is_perturbed_step

        if curr_step_index == perturbed_index:
            perturbed_step_index = step
        
        
        # Place pieces 0 to step-1.

        result = ""
        for idx in range(step):
            piece = scrambled_list[idx]['piece']
            index = scrambled_list[idx]['index']
            part_name = chr(65 + index)
            # If this is the perturbed step, use the perturbed version.
            if index == perturbed_index:
                print(f"Perturbing piece {index} at step {step}")
                print(f"Original: {piece}")
                piece = perturb_step(piece, max_offset=max_offset, board_size=n)
                print(f"Perturbed: {piece}")
            r0, r1, c0, c1 = piece
            # Draw the current (most recently placed) piece in color; earlier ones in gray.
            facecolor = colors[index] if idx == step - 1 else "lightgray"
            if isinstance(piece, tuple):
                patch = patches.Rectangle((c0, n - r1), c1 - c0, r1 - r0,
                                           facecolor=facecolor,
                                           edgecolor="black", lw=2, alpha=0.6)
            else:
                patch = patches.Polygon(piece, closed=True, facecolor=facecolor,
                                        edgecolor="black", lw=2, alpha=0.6)
            ax1.add_patch(patch)
            center = ((c0 + c1)/2, n - (r0 + r1)/2)
            # ax1.text(center[0], center[1], f"{part_name}: {idx2label[index]}", ha="center", va="center", fontsize=label_fontsize, color="black")

            # fill all cells in the piece with part_name
            for r in range(r0, r1):
                for c in range(c0, c1):
                    ax1.text(c+0.5, n-r-0.5, part_name, ha="center", va="center", fontsize=label_fontsize, color="black")
            over_board = (r0 < 0 or r1 > n or c0 < 0 or c1 > n)
            if over_board:
                # ax1.text(center[0], center[1]-0.5, "Over board", ha="center", va="center", fontsize=10, color="red")
                result = "Over board"

            # check overlap
            for i in range(idx):
                piece_i = scrambled_list[i]['piece']
                if isinstance(piece_i, tuple):
                    bbox_i = get_piece_bbox(piece_i, n)
                    r0_i, r1_i, c0_i, c1_i = bbox_i
                    if r0_i < r1 and r1_i > r0 and c0_i < c1 and c1_i > c0:
                        # ax1.text(n/2, -0.5, "Overlap", ha="center", va="center", fontsize=14, color="red")
                        result = "Overlap"
                        break
        results.append(result)
        
        # if step == num_pieces:
        #     # Final step: if union of pieces does not cover board exactly, annotate unsolvability.
        #     ax1.text(n/2, -0.5, "Puzzle unsolvable", ha="center", va="center", fontsize=14, color="red")
        
        # instruction = f"Step {step}: " + ("Perturbed" if perturbed_index < step else "Valid")
        
        # revert the rotation
        if step > 0:
            part_name = chr(65 + index)
            angle = scrambled_list[curr_step]['angle']
            if curr_step_index == perturbed_index:
                angle = random.choice([0, 30, 60, 90])
                
            rotate_instruction = f"Rotate piece {part_name} by about {angle} degrees clockwise, and p" if angle > 0 else "P"
            instruction = rotate_instruction + f"lace piece {part_name} with its upper-left corner at (x, y) = ({int(c0):d}, {n - int(r0):d})."

            instructions.append(instruction)
        else:
            instructions.append("")
        # ax1.set_xlabel(instruction)
        
        # Right panel: available pieces.
        ax2 = plt.subplot(1, 2, 2)
        for item_idx, item in enumerate(scrambled_list):
            if item_idx < step:
                continue  # skip already placed pieces.
            pos = item['cell_center']
            rect_patch = patches.Rectangle((-item['orig_width']/2, -item['orig_height']/2),
                                           item['orig_width'], item['orig_height'],
                                           facecolor=colors[item['index']],
                                           edgecolor="black", lw=2, alpha=0.7)
            trans = transforms.Affine2D().rotate_deg(item['angle']).translate(pos[0], pos[1]) + ax2.transData
            rect_patch.set_transform(trans)
            ax2.add_patch(rect_patch)
            # put label below the shape
            part_name = chr(65 + item['index'])
            ax2.text(pos[0], pos[1]+ item['h']/2+0.2, f"{part_name}: {item['label']}",
                     ha="center", va="center", fontsize=label_fontsize, color="black")
            # Optionally, overlay a grid.
            for xi in range(0, int(item['orig_width']) + 1):
                x = -item['orig_width']/2 + xi
                line = plt.Line2D([x, x], [-item['orig_height']/2, item['orig_height']/2],
                                  color="black", lw=1, alpha=0.8)
                line.set_transform(trans)
                ax2.add_line(line)
            for yi in range(0, int(item['orig_height']) + 1):
                y = -item['orig_height']/2 + yi
                line = plt.Line2D([-item['orig_width']/2, item['orig_width']/2], [y, y],
                                  color="black", lw=1, alpha=0.8)
                line.set_transform(trans)
                ax2.add_line(line)
        ax2.set_xlim(0, canvas_width)
        ax2.set_ylim(0, canvas_height)
        ax2.axis("off")
        ax2.set_aspect("equal")
        ax2.set_title("Avaialable Pieces", fontsize=title_fontsize)
        
        output_path = os.path.join(output_dir, f"{prefix}perturbed_step_assembly_step_{step}.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        if step > perturbed_step_index and perturbed_step_index != -1:
            continue
        image_paths.append(output_path)
    return scrambled_list, instructions, image_paths, results


##############################################################################
# 3. MAIN PROCESS: SEGMENTATION, SCRAMBLED PIECES, & REASSEMBLY ANIMATION
##############################################################################
# def main(n=6, seed=42, min_size=1, max_pieces=10):
#     # n = 6  # Board size: 6x6
#     random.seed(seed)
#     np.random.seed(seed)
    
#     final_pieces, seg_steps = segment_rectangle_steps_max(0, n, 0, n, min_size=min_size, max_pieces=max_pieces)
#     print(f"Board of size {n}x{n} segmented into {len(final_pieces)} pieces.")
#     for piece in final_pieces:
#         area = (piece[1] - piece[0]) * (piece[3] - piece[2])
#         print(f"Piece {piece} with area {area}")
    
#     # Animate segmentation.
#     animate_segmentation(n, seg_steps)
    
#     # Plot empty board and scrambled pieces (with random angle from {0,30,60,90}) on one canvas.
#     # plot_empty_and_scrambled(n, final_pieces, draw_piece_grid=True)
    
#     # Animate reassembly.
#     animate_reassembly(n, final_pieces)


#     if len(final_pieces) >= 4: 
#         merged_shapes = randomly_combine_two(final_pieces)
#         if merged_shapes is not None:
#             print(f"Two adjacent pieces merged into a single shape.")
#             animate_reassembly_irregular(n, merged_shapes)

    
#     # Perturb the pieces to make the puzzle unsolvable.
#     final_pieces_copy = final_pieces.copy()
#     perturbed_pieces = perturb_pieces(final_pieces_copy, n)
#     # print(f"Perturbed pieces: {perturbed_pieces}")
#     # Animate reassembly with perturbed pieces.
#     animate_assembly_perturbed(n, perturbed_pieces)

#     perturb_index = random.choice(range(len(final_pieces)))
#     print(f"Piece {perturb_index} perturbed.")
#     # Animate reassembly until the perturbed step.
#     animate_assembly_perturbed_step(n, final_pieces, perturb_index)
        
import os
import json

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def piece_to_ascii_local(shape, letter, rotate_deg=0, flip=None):
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely import affinity
    """
    Render a single piece into its own minimal ASCII grid, after optionally
    rotating (90° multiples) and/or flipping it.

    Args:
      shape: either
        - a tuple (r0,r1,c0,c1) for an axis-aligned rectangle in board coords, or
        - a Shapely Polygon or MultiPolygon in board coords, or
        - a list of (x,y) points representing polygon vertices in board coords,
        - a list of tuples (r0,r1,c0,c1) representing multiple rectangles.
      letter: single-character label to fill the piece.
      rotate_deg: rotation about the origin, in degrees (must be 0, 90, 180 or 270).
      flip: one of None, "horizontal", or "vertical".

    Returns:
      A list of strings (top row first), with '.' for empty cells.
    """
    # 1) Convert tuple→Polygon, so we can treat everything uniformly.
    if isinstance(shape, tuple):
        r0, r1, c0, c1 = shape
        poly = box(c0, r0, c1, r1)
    elif isinstance(shape, list):
        if not shape:
            raise ValueError("Empty shape list")
        
        # Check if list contains rectangles (tuples of 4 values) or points (tuples of 2 values)
        if all(isinstance(p, tuple) and len(p) == 4 for p in shape):
            # List of rectangles: convert each to a box and merge
            boxes = [box(rect[2], rect[0], rect[3], rect[1]) for rect in shape]
            poly = unary_union(boxes)
        elif all(isinstance(p, (list, tuple)) and len(p) == 2 for p in shape):
            # List of points: convert to Polygon
            poly = Polygon(shape)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid polygons
        else:
            raise ValueError(f"Invalid shape format: {shape}. Expected list of (x,y) points or list of (r0,r1,c0,c1) rectangles.")
    elif isinstance(shape, (Polygon, MultiPolygon)):
        poly = shape
    else:
        raise TypeError(f"Unsupported shape type: {type(shape)}")

    # 2) Translate so lower-left of bounding box sits at (0,0).
    minx, miny, maxx, maxy = poly.bounds
    poly = affinity.translate(poly, xoff=-math.floor(minx), yoff=-math.floor(miny))

    # 3) Apply flip (about origin).
    if flip == "horizontal":
        poly = affinity.scale(poly, xfact=-1, yfact=1, origin=(0,0))
    elif flip == "vertical":
        poly = affinity.scale(poly, xfact=1, yfact=-1, origin=(0,0))

    # 4) Apply rotation (about origin).
    if rotate_deg  == 90:
        poly = affinity.rotate(poly, rotate_deg, origin=(0,0))

    # 5) Re-compute bounds, shift again so lower-left is non-negative.
    minx, miny, maxx, maxy = poly.bounds
    poly = affinity.translate(poly, xoff=-math.floor(minx), yoff=-math.floor(miny))
    width  = int(math.ceil(maxx) - math.floor(minx))
    height = int(math.ceil(maxy) - math.floor(miny))

    # 6) Rasterize into a local grid of size (height × width).
    grid = [['0' for _ in range(width)] for _ in range(height)]
    for dr in range(height):
        for dc in range(width):
            pt = Point(dc + 0.5, dr + 0.5)
            if poly.intersects(pt):
                # row 0 in list = top, so invert dr
                grid[height - 1 - dr][dc] = '1'

    # 7) Return as list of strings
    return [''.join(row) for row in grid]


def generate_text_puzzle_from_scrambled(scrambled_list, board_size):
    """
    Given scrambled_list (as from animate_reassembly or irregular variants)
    and the target board_size, produce a complete text puzzle prompt.
    """
    # Header
    lines = [
        "Tangram-Style Text Puzzle",
        "",
        f"Can you use all of the following pieces—rotating or flipping them as needed—to exactly fill a {board_size}x{board_size} square without overlaps or gaps?",
        "",
        f"Target board: a {board_size}x{board_size} grid",
        "",
        "Available pieces:",
        ""
    ]

    # For each piece in index order
    for item in sorted(scrambled_list, key=lambda x: x['index']):
        idx     = item['index']
        shape   = item.get('shape', item.get('piece'))
        angle   = item.get('angle', 0)
        flip    = item.get('flip', None)
        letter  = chr(65 + idx)

        ascii_art = piece_to_ascii_local(shape, letter, rotate_deg=angle, flip=flip)
        lines.append(f"Piece {letter}:")
        lines.extend(ascii_art)
        lines.append("")  # blank line

    # Instructions & question
    lines.extend([
        # add that for the pieces above, 0 means empty, 1 means filled
        "For each piece, the grid cells it occupies are marked with 1s and the rest with 0s.",
        "",
        "Instructions:",
        "• You must place every cell of each piece somewhere in the grid.",
        "• Pieces may be rotated in 90° increments or flipped horizontally/vertically.",
        "",
        f"Question: Is it possible to fill the {board_size}x{board_size} board exactly with these pieces? (Yes or No)"
    ])

    return "\n".join(lines)


def generate_instance(board_size, instance_id, output_dir, overwrite_image=False, overwrite_json=True):
    """
    Generate one instance for a board of given size.
    Runs segmentation, assembly (non-perturbed) and perturbed assembly.
    Saves the images (segmentation steps, assembly steps, perturbed assembly steps)
    with filenames that include board_size and instance_id.
    Returns a dictionary recording the image paths and instructions.
    
    Parameters:
        board_size: Size of the board (board_size x board_size)
        instance_id: Identifier for this instance
        output_dir: Directory to save images and records
        overwrite_image: Whether to regenerate images if they already exist
        overwrite_json: Whether to regenerate JSON if it already exists
    """
    
    # Use a fixed seed for reproducibility for each instance (could incorporate instance_id).
    seed = 1000 * board_size + instance_id
    random.seed(seed)
    np.random.seed(seed)
    instance_record = {
        "board_size": board_size,
        "seed": seed,
        "instance_id": instance_id,
        "segmentation": {},
        "assembly": {},
        "perturbed_part": {},
        "perturbed_step": {},
        "irregular_assembly": {},
        "irregular_invalid_assembly": {}
    }
    
    prefix = f"{board_size}x{board_size}_inst{instance_id}"
    seg_dir = os.path.join(output_dir, prefix, "segmentation")
    asm_dir = os.path.join(output_dir, prefix, "assembly")
    irg_dir = os.path.join(output_dir, prefix, "irregular_assembly")
    pert_asm_dir = os.path.join(output_dir, prefix, "perturbed_part")
    pert_step_dir = os.path.join(output_dir, prefix, "perturbed_step")
    for d in [seg_dir, asm_dir, irg_dir, pert_asm_dir, pert_step_dir]:
        ensure_dir(d)

    max_pieces = random.randint(3, max(3, min(board_size * board_size // 2, 8)))
    
    # Run segmentation.
    final_pieces, seg_steps = segment_rectangle_steps_max(0, board_size, 0, board_size, min_size=1, max_pieces=max_pieces)
    instance_record["pieces"] = final_pieces


    global colors
    colors = generate_piece_colors(len(final_pieces)+1)

    # Check if segmentation images already exist
    seg_last_image = os.path.join(seg_dir, f"{prefix}segmentation_step_{len(seg_steps)}.png")
    if not overwrite_image and os.path.exists(seg_last_image):
        # If images exist, just record the paths without regenerating
        seg_image_paths = [os.path.join(seg_dir, f"{prefix}segmentation_step_{i}.png") 
                          for i in range(len(seg_steps) + 1)]
    else:
        # Generate segmentation images
        seg_image_paths = animate_segmentation(board_size, seg_steps, seg_dir, prefix)
    
    instance_record["segmentation"]["images"] = seg_image_paths
    
    # Animate non-perturbed assembly.
    asm_last_image = os.path.join(asm_dir, f"{prefix}assembly_step_{len(final_pieces)}.png")
    if not overwrite_image and os.path.exists(asm_last_image):
        # If images exist, use existing ones but we need to reconstruct the instructions and shuffled_list
        # For now, we'll regenerate the assembly data but skip saving images
        with open(os.devnull, 'w') as f:
            scrambled_list, instructions, _ = animate_reassembly(board_size, final_pieces, "", "")
        asm_image_paths = [os.path.join(asm_dir, f"{prefix}assembly_step_{i}.png") 
                          for i in range(len(final_pieces) + 1)]
    else:
        # Generate assembly images
        scrambled_list, instructions, asm_image_paths = animate_reassembly(board_size, final_pieces, asm_dir, prefix)
    
    instance_record["assembly"]["images"] = asm_image_paths
    instance_record["assembly"]["instructions"] = instructions
    instance_record["assembly"]["valid"] = True
    instance_record["assembly"]["config"] = scrambled_list
    instance_record["assembly"]["text_question"] = generate_text_puzzle_from_scrambled(scrambled_list, board_size) 

    if len(final_pieces) >= 3:
        tries = 1
        while tries > 0:

            if "images" not in instance_record["irregular_assembly"]:
                merged_shapes, newidx2oldidx = randomly_combine_two(final_pieces, ignore_rect=False)
                if merged_shapes is None:
                    print("No adjacent pieces found to merge.")
                    pass
                else:
                    print(f"Two adjacent pieces merged into a single shape.")
                    # Check if irregular assembly images already exist
                    irg_last_image = os.path.join(irg_dir, f"{prefix}irregular_assembly_step_{len(merged_shapes)}.png")
                    if not overwrite_image and os.path.exists(irg_last_image):
                        # Use existing images but regenerate other data
                        with open(os.devnull, 'w') as f:
                            scrambled_list, instructions, _, results = animate_reassembly_irregular(board_size, merged_shapes, "", "")
                        image_paths = [os.path.join(irg_dir, f"{prefix}irregular_assembly_step_{i}.png") 
                                      for i in range(len(merged_shapes) + 1)]
                    else:
                        # Generate new images
                        scrambled_list, instructions, image_paths, results = animate_reassembly_irregular(board_size, merged_shapes, irg_dir, prefix)
                    
                    num_shapes = len(scrambled_list)
                    for item in scrambled_list:
                        new_idx = item['index']
                        old_idx = newidx2oldidx[new_idx]
                        if isinstance(old_idx, tuple):
                            item['shape'] = [final_pieces[old_idx[0]], final_pieces[old_idx[1]]]
                        else:
                            item['shape'] = final_pieces[old_idx]

                    instance_record["irregular_assembly"]["images"] = image_paths
                    instance_record["irregular_assembly"]["instructions"] = instructions
                    instance_record["irregular_assembly"]["valid"] = True
                    instance_record["irregular_assembly"]["config"] = scrambled_list
                    instance_record["irregular_assembly"]["results"] = results

                    q_scrambled_list = []
                    for item in scrambled_list:
                        if item["is_merged"]:
                            new_item = item.copy()
                            new_item["shape"] = merged_shapes[-1]
                        else:
                            new_item = item.copy()
                        q_scrambled_list.append(new_item)
                    instance_record["irregular_assembly"]["text_question"] = generate_text_puzzle_from_scrambled(q_scrambled_list, board_size) 
            else:
                merged_shapes, newidx2oldidx = randomly_combine_two(final_pieces, ignore_rect=True)

            if "images" not in instance_record["irregular_invalid_assembly"] and merged_shapes is not None:
                # Build invalid variant by shifting the merged pair shape
                bad_idx = next(i for i, v in newidx2oldidx.items() if isinstance(v, tuple))
                bad_orig_idx = newidx2oldidx[bad_idx]
                
                # Check if invalid irregular assembly images already exist
                invalid_irg_last_image = os.path.join(irg_dir, f"{prefix}invalid_irregular_assembly_step_{len(merged_shapes)}.png")
                
                # Skip the whole section if images don't exist and we're not supposed to overwrite
                if not overwrite_image and not os.path.exists(invalid_irg_last_image):
                    print(f"Invalid irregular assembly images don't exist for {prefix} and overwrite_image=False, skipping")
                    tries -= 1
                    continue
                
                # Create invalid shapes by shifting the merged pair shape
                r_orig1, r_orig2 = bad_orig_idx[0], bad_orig_idx[1]  # Original pieces to be merged
                print(f"Shape {bad_idx} is a merged shape of {bad_orig_idx[0]} and {bad_orig_idx[1]}")
                piece1, piece2 = final_pieces[r_orig1], final_pieces[r_orig2]

                # Convert to shapely geometries for checking adjacency
                shapely_piece1 = box(piece1[2], piece1[0], piece1[3], piece1[1])  # Convert to box(xmin,ymin,xmax,ymax)
                shapely_piece2 = box(piece2[2], piece2[0], piece2[3], piece2[1])

                # Try different shifts until we find one where pieces still touch but form invalid shape

                orig_merged = merged_shapes[bad_idx]
                def get_touching_side(rect1, rect2):
                    # rect: (ymin, ymax, xmin, xmax)
                    # Returns 'left', 'right', 'top', 'bottom', or None
                    if rect1[2] == rect2[3]:  # rect1 left touches rect2 right
                        return 'left'
                    if rect1[3] == rect2[2]:  # rect1 right touches rect2 left
                        return 'right'
                    if rect1[1] == rect2[0]:  # rect1 top touches rect2 bottom
                        return 'top'
                    if rect1[0] == rect2[1]:  # rect1 bottom touches rect2 top
                        return 'bottom'
                    return None
                
                from shapely import affinity, wkt
                def normalize(poly):
                    minx, miny, _, _ = poly.bounds
                    return affinity.translate(poly, xoff=-minx, yoff=-miny)

                def same_up_to_rigid(poly_a, poly_b, tol=1e-6):
                    a = normalize(poly_a)
                    for angle in (0, 90, 180, 270):
                        rotated = affinity.rotate(poly_b, angle, origin='centroid')
                        for flip in (False, True):
                            if flip:
                                # mirror over vertical axis through centroid
                                candidate = affinity.scale(rotated, xfact=-1, yfact=1, origin='centroid')
                            else:
                                candidate = rotated
                            cand_n = normalize(candidate)
                            if a.equals_exact(cand_n, tol) or a.symmetric_difference(cand_n).area < tol:
                                return True #, angle, flip
                    return False #, None, None
                
                def get_shifted_merged_shape(piece1, piece2):
                    merged_invalid = None
                    side = get_touching_side(piece1, piece2)
                    if side in ('left', 'right'):
                        # Touching vertically, shift along y
                        shift_axis = 'y'
                    elif side in ('top', 'bottom'):
                        # Touching horizontally, shift along x
                        shift_axis = 'x'
                    else:
                        shift_axis = None  # Not touching

                    if shift_axis is None:
                        raise ValueError("Pieces are not touching, cannot create invalid shape.")
                    

                    for offset in [-1, 1]:
                        if shift_axis == 'y':
                            shifted_piece1 = (piece1[0]+offset, piece1[1]+offset, piece1[2], piece1[3])
                        else:
                            shifted_piece1 = (piece1[0], piece1[1], piece1[2]+offset, piece1[3]+offset)
                        if is_adjacent(shifted_piece1, piece2):
                            shifted_shape = box(shifted_piece1[2], shifted_piece1[0], shifted_piece1[3], shifted_piece1[1])
                            merged = shifted_shape.union(shapely_piece2)
                            total_area = shifted_shape.area + shapely_piece2.area
                            if (merged.geom_type == 'Polygon' and
                                abs(merged.area - total_area) < 1e-6 and
                                not merged.equals(merged.minimum_rotated_rectangle)):
                                if same_up_to_rigid(orig_merged, merged):
                                    print("original merged shape and shifted merged shape are the same up to rigid transformation", orig_merged, merged)
                                else:
                                    merged_invalid = merged
                                    # print(shapely_piece1, shapely_piece2)
                                    # print(orig_merged, merged_invalid)
                                    break
                    return merged_invalid
                
                merged_invalid = get_shifted_merged_shape(piece1, piece2)
                if merged_invalid is None:
                    merged_invalid = get_shifted_merged_shape(piece2, piece1)

                if merged_invalid is None:
                    print("Failed to create an invalid shape after max tries.")
                else:
                    # Create invalid shapes list by replacing the merged shape with our invalid merged shape
                    invalid_shapes = []
                    for idx, shape in enumerate(merged_shapes):
                        if idx == bad_idx:
                            invalid_shapes.append(merged_invalid)
                        else:
                            invalid_shapes.append(shape)
                                
                    # Check if invalid irregular assembly images already exist
                    invalid_irg_last_image = os.path.join(irg_dir, f"{prefix}invalid_irregular_assembly_step_{len(invalid_shapes)}.png")
                    if not overwrite_image and os.path.exists(invalid_irg_last_image):
                        # Use existing images but regenerate other data
                        with open(os.devnull, 'w') as f:
                            sh2, instr2, _, results2 = animate_reassembly_irregular(
                                board_size, invalid_shapes, "", "", invalid_idx=bad_idx)
                        imgs2 = [os.path.join(irg_dir, f"{prefix}invalid_irregular_assembly_step_{i}.png") 
                                for i in range(len(invalid_shapes) + 1)]
                    else:
                        # Generate new images
                        sh2, instr2, imgs2, results2 = animate_reassembly_irregular(
                            board_size, invalid_shapes, irg_dir, prefix+"invalid_", invalid_idx=bad_idx)
                    
                    print(imgs2)

                    num_shapes = len(sh2)
                    for item in sh2:
                        new_idx = item['index']
                        old_idx = newidx2oldidx[new_idx]
                        if isinstance(old_idx, tuple):
                            item['shape'] = [final_pieces[old_idx[0]], final_pieces[old_idx[1]]]
                        else:
                            item['shape'] = final_pieces[old_idx]
                            
                    instance_record['irregular_invalid_assembly'] = {
                        'images': imgs2,
                        'instructions': instr2,
                        'results': results2,
                        'valid': False,
                        'config': sh2
                    }

                    q_scrambled_list = []
                    for item in sh2:
                        if item["is_merged"]:
                            new_item = item.copy()
                            new_item["shape"] = merged_invalid
                        else:
                            new_item = item.copy()
                        q_scrambled_list.append(new_item)
                    instance_record["irregular_invalid_assembly"]["text_question"] = generate_text_puzzle_from_scrambled(q_scrambled_list, board_size) 
                    break
            tries -= 1

    # Generate perturbed pieces.
    perturbed = perturb_pieces(final_pieces.copy(), board_size)
    if perturbed is not None:
        # Check if perturbed part images already exist
        pert_asm_last_image = os.path.join(pert_asm_dir, f"{prefix}perturbed_part_assembly_step_{len(perturbed)}.png")
        if not overwrite_image and os.path.exists(pert_asm_last_image):
            # Use existing images but regenerate other data
            with open(os.devnull, 'w') as f:
                scrambled_list, instructions, _, results = animate_assembly_perturbed(board_size, perturbed, "", "")
            pert_asm_image_paths = [os.path.join(pert_asm_dir, f"{prefix}perturbed_part_assembly_step_{i}.png") 
                                   for i in range(len(perturbed) + 1)]
        else:
            # Generate new images
            scrambled_list, instructions, pert_asm_image_paths, results = animate_assembly_perturbed(board_size, perturbed, pert_asm_dir, prefix)
        
        instance_record["perturbed_part"]["images"] = pert_asm_image_paths
        instance_record["perturbed_part"]["instructions"] = instructions
        instance_record["perturbed_part"]["results"] = results
        instance_record["perturbed_part"]["valid"] = results[-1] == ""
        instance_record["perturbed_part"]["config"] = scrambled_list
        instance_record["perturbed_part"]["text_question"] = generate_text_puzzle_from_scrambled(scrambled_list, board_size) 
        
    # Choose a random assembly step index at which to perturb a piece.
    perturbed_index = random.choice(range(len(final_pieces)))
    max_offset = random.choice([1, 2])
    
    # Check if perturbed step images already exist
    pert_step_last_image = os.path.join(pert_step_dir, f"{prefix}perturbed_step_assembly_step_{len(final_pieces)}.png")
    if not overwrite_image and os.path.exists(pert_step_last_image):
        # Use existing images but regenerate other data
        with open(os.devnull, 'w') as f:
            scrambled_list, instructions, _, results = animate_assembly_perturbed_step(
                board_size, final_pieces, perturbed_index, max_offset=max_offset, output_dir="", prefix="")
        pert_step_image_paths = [os.path.join(pert_step_dir, f"{prefix}perturbed_step_assembly_step_{i}.png") 
                                for i in range(len(final_pieces) + 1)]
    else:
        # Generate new images
        scrambled_list, instructions, pert_step_image_paths, results = animate_assembly_perturbed_step(
            board_size, final_pieces, perturbed_index, max_offset=max_offset, output_dir=pert_step_dir, prefix=prefix)
    
    instance_record["perturbed_step"]["images"] = pert_step_image_paths
    instance_record["perturbed_step"]["instructions"] = instructions
    instance_record["perturbed_step"]["results"] = results
    instance_record["perturbed_step"]["valid"] = results[-1] == ""
    instance_record["perturbed_step"]["config"] = scrambled_list
    return instance_record

def generate_instances(min_board=3, max_board=4, instances_per_size=2, output_dir="output_dir", overwrite=False):
    """
    Generate puzzle instances for board sizes from min_board x min_board to max_board x max_board.
    For each board size, generate instances_per_size instances.
    Save images individually for each instance and record image paths and instructions.
    
    Returns a list of instance records.
    """
    output_dir = os.path.join(output_dir, "Tangram_puzzle_v2")
    print(f"Output directory: {output_dir}")
    ensure_dir(output_dir)
    all_instances = []
    for board_size in range(min_board, max_board + 1):
        for inst in range(instances_per_size):
            json_path = os.path.join(output_dir, f"{board_size}x{board_size}_inst{inst}_record.json")
            all_instances.append(json_path)
            if os.path.exists(json_path) and not overwrite:
                print(f"Instance record already exists for board {board_size}x{board_size}, instance {inst}")
                continue
            print(f"Generating instance for board {board_size}x{board_size}, instance {inst}")
            record = generate_instance(board_size, inst, output_dir, overwrite_image=False, overwrite_json=overwrite)
            # all_instances.append(record)
            with open(json_path, "w") as f:
                json.dump(record, f, indent=4)
    # Save all instance records to a JSON file.
    # json_path = os.path.join(output_dir, "instance_records.json")
    # with open(json_path, "w") as f:
    #     json.dump(all_instances, f, indent=4)
    # print(f"Instance records saved to {json_path}")
    # return all_instances
    return all_instances
    


def create_text_instruction_hf_dataset(output_dir="output_dir", dataset_name="Tangram_puzzle", instances_per_size=50,overwrite=False, vissim=False):
    all_json_paths = generate_instances(output_dir=output_dir, overwrite=overwrite, instances_per_size=instances_per_size)
    # print(all_json_paths)
    random.seed(0)
    random.shuffle(all_json_paths)
    data_dict = {
        'qid': [], 'question': [], 
        'images': [],
        'answer': [], 'question_info': [], 'type': [],
        'choices': []}
    from PIL import Image
    import datasets
    from datasets import Dataset, Features
    import json
    from azfuse import File
    for json_idx, json_path in enumerate(all_json_paths):
        with File.open(json_path, 'r') as f:
            question_info_path = json.load(f)
        id_ = "_".join(os.path.basename(json_path).split('_')[:-1])
        if '3x3' not in id_ and '4x4' not in id_:
            continue

        inst_id = int(id_.split('_')[-1].replace("inst", ""))

        # if inst_id > 30:
        #     continue

        variants = ['assembly', 'perturbed_part', 'perturbed_step', 'irregular_assembly', 'irregular_invalid_assembly']
        random.shuffle(variants)


        for v_idx, variant in enumerate(variants):
            if variant != 'irregular_invalid_assembly':
                if inst_id > 30:
                    continue

            if len(question_info_path[variant]) == 0:
                continue

            question_text = "Check out an Tangram puzzle below. The left panel is an empty Tangram puzzle, while the right panel shows available pieces to complete the puzzle. \n<image_0>"

            instructions = question_info_path[variant]['instructions'][1:]
            image_paths = question_info_path[variant]['images']
            num_valid_vis = len(image_paths) - 1

            # answer_text = f"Choose from the following answer choices:\nA. {answer_choices[0]}\nB. {answer_choices[1]}"

            is_valid = question_info_path[variant]['valid']

            # for type in ['q+steps+partial_vis', 'q+steps+all_intermediate_vis', 'q+steps+all_for_valid_vis', 'q+steps+all_intermediate_last_vis']:
            if vissim:
                q_types = ['q+steps+partial_vis', 'q+steps+all_intermediate_vis', 'q+steps+all_for_valid_vis', 'q+steps+all_intermediate_last_vis']
            else:
                q_types = ['q_only', 'q+steps']
            for type in q_types:
                if variant == 'perturbed_step':
                    if type == 'q_only':
                        continue
                if variant == 'perturbed_part' or variant == 'perturbed_step':
                    if is_valid:
                        print("Invalid variant")
                        continue
                # if type == 'q+steps+all_for_valid_vis':
                #     continue
                if type == 'q+steps+partial_vis' and num_valid_vis <= 1:
                    continue
                if type == 'q+steps+all_intermediate_last_vis' and num_valid_vis <= 1:
                    continue

                data_dict['qid'].append(f'{id_}_var{v_idx}_{type}')

            
                answer_choices = ['yes', 'no']
                random.shuffle(answer_choices)

                answer = 'yes' if is_valid else 'no'
                answer_idx = answer_choices.index(answer)
                data_dict['answer'].append(chr(ord('A') + answer_idx))
                # answer_text = f"Choose from the following answer choices:\nA. Yes\nB. No"

                images = []

                images.append(Image.open(image_paths[0]).convert('RGB'))
                if type == 'q_only':
                    full_question = question_text+ "Keep in mind that you can rotate or flip the pieces. Can the Tangram puzzle be completed with the available pieces, yes or no?"
                elif type == 'q+steps':
                    full_question = question_text + f"Below are the steps to complete the Tangram puzzle:\n" 
                    for i in range(len(instructions)):
                        curr_step = instructions[i]
                        full_question += f"Step {i+1}: {curr_step}\n"
                    full_question = full_question+"Based on the above steps, can the Tangram puzzle be completed with the available pieces, yes or no?"
                    
                elif type == 'q+steps+all_for_valid_vis':
                    full_question = question_text + f"Below are the steps to complete the Tangram puzzle:\n" 
                    for i in range(1, len(image_paths)):
                        images.append(Image.open(image_paths[i]).convert('RGB'))
                        curr_step = instructions[i-1]
                        full_question += f"Step {i}: {curr_step}\n<image_{i}>\n"
                    
                    for i in range(len(image_paths)-1, len(instructions)):
                        curr_step = instructions[i]
                        full_question += f"Step {i+1}: {curr_step}\n"
                    full_question = full_question+"Based on the above steps, can the Tangram puzzle be completed with the available pieces, yes or no?"
                    # +answer_text
                
                elif type == 'q+steps+all_intermediate_vis':
                    num_vis = len(image_paths) - 1

                    full_question = question_text + f"Below are the steps to complete the Tangram puzzle:\n" 
                    for i in range(1, num_vis):
                        images.append(Image.open(image_paths[i]).convert('RGB'))
                        curr_step = instructions[i-1]
                        full_question += f"Step {i}: {curr_step}\n<image_{i}>\n"
                    
                    for i in range(num_vis-1, len(instructions)):
                        curr_step = instructions[i]
                        full_question += f"Step {i+1}: {curr_step}\n"
                    full_question = full_question+"Based on the above steps, can the Tangram puzzle be completed with the available pieces, yes or no?"
                # +answer_text
                
                elif type == 'q+steps+all_intermediate_last_vis':
                    num_vis = len(image_paths) - 1

                    full_question = question_text + f"Below are the steps to complete the Tangram puzzle:\n" 
                    for i in range(1, num_vis):
                        curr_step = instructions[i-1]
                        if i == num_vis - 1:
                            images.append(Image.open(image_paths[i]).convert('RGB'))
                            full_question += f"Step {i}: {curr_step}\n<image_1>\n"
                        else:
                            full_question += f"Step {i}: {curr_step}\n"
                    
                    for i in range(num_vis-1, len(instructions)):
                        curr_step = instructions[i]
                        full_question += f"Step {i+1}: {curr_step}\n"
                    full_question = full_question+"Based on the above steps, can the Tangram puzzle be completed with the available pieces, yes or no?"

                elif type == 'q+steps+partial_vis':
                    random_num_vis = random.randint(1, num_valid_vis - 1)
                    full_question = question_text + f"Below are the steps to complete the Tangram puzzle:\n"  
                    for i in range(1, random_num_vis+1):
                        images.append(Image.open(image_paths[i]).convert('RGB'))
                        curr_step = instructions[i-1]
                        full_question += f"Step {i}: {curr_step}\n<image_{i}>\n"
                    
                    for i in range(random_num_vis, len(instructions)):
                        curr_step = instructions[i]
                        full_question += f"Step {i+1}: {curr_step}\n"
                    full_question = full_question+"Based on the above steps, can the Tangram puzzle be completed with the available pieces, yes or no?"
                data_dict['images'].append(images)
                data_dict['question'].append(full_question)
                question_info_path[variant]['variant'] = variant
                data_dict['question_info'].append(json.dumps(question_info_path[variant]))
                data_dict['type'].append(type)
                data_dict['choices'].append(answer_choices)
    # shuffle data_dict as well
    num_q = len(data_dict['qid'])
    shuffled_indices = list(range(num_q))
    random.shuffle(shuffled_indices)
    for key in data_dict:
        data_dict[key] = [data_dict[key][i] for i in shuffled_indices]
    feature_dict = {
        'qid': datasets.Value('string'),
        'question': datasets.Value('string'),
        'images': datasets.Sequence(datasets.Image()),
        'answer': datasets.Value('string'),
        'type': datasets.Value('string'),
        'question_info': datasets.Value('string'),
        'choices': datasets.Sequence(datasets.Value('string'))
    }

    features = Features(feature_dict)
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    with open('aux_data/credentials/hf_token.txt', 'r') as f:
        hf_token = f.read().strip()
    hf_dataset.push_to_hub(
        f"VisSim/{dataset_name}",
        create_pr=False,
        token=hf_token)

if __name__ == '__main__':
    from fire import Fire
    Fire(create_text_instruction_hf_dataset)
