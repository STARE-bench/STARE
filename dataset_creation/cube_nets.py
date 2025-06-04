import numpy as np

# ASCII‑art layouts for all nets (name → 2D ASCII)
ASCII_Nets = {
    # — Valid Nets —
    "t-shaped Net":        " 1\n2345\n 6",
    "t-shaped Net Variant 1":"  1\n2345\n 6",
    "t-shaped Net Variant 2":"   1\n2345\n 6",
    "Big T Net":           "123\n 4\n 5\n 6",
    "Big T - Variant 1":   "12\n 43\n 5\n 6",
    "Big T - Variant 2":   "12\n 4\n 53\n 6",
    "Big T - Variant 3":   "12\n 4\n 5\n 63",
    "3-3 Net":             "1\n2\n34\n 5\n 6",
    "2-2 net":             "1\n23\n 45\n  6",
    "1-3-2 Net":           "12\n 3\n 45\n  6",  # base
    "1-3-2 Net (v1)":      " 2\n13\n 45\n  6",
    "1-3-2 Net (v2)":      " 2\n 3\n145\n  6",

    # — Invalid Nets —
    "Linear 6-Face Row":          "123456",
    "Net B: Row 5 + Face 6 (shift=0)":"12345\n6",
    "Net B: Row 5 + Face 6 (shift=1)":"12345\n 6",
    "Net B: Row 5 + Face 6 (shift=2)":"12345\n  6",
    "Net B: Row 5 + Face 6 (shift=3)":"12345\n   6",
    "Net B: Row 5 + Face 6 (shift=4)":"12345\n    6",
    "2 Rows of 3":                "123\n456",
    "Invalid D":                  " 456\n23\n1",
    "invalid_G":                  "1  6\n2345",
    "invalid_F":                  "1234\n  56",
    "invalid_E":                  "1  6\n2345",
    "Net H: Two rows with bottom row shifted right":"123\n 456",
}


def create_big_t_variants_adjusted():
    """
    Creates three variants of a Big T-shaped net by shifting face 3 downward.
    
    Original Big T net (before shifting):
      Top row: faces 1, 2, 3 (from left to right)
      Vertical column (below face 2): faces 4, 5, 6
       
    In the variants, we remove face 3 from the top row and reattach it
    to the vertical column:
      - Variant 1: face 3 is shifted down by one block; it touches face 4.
      - Variant 2: face 3 is shifted down by two blocks; it touches face 5.
      - Variant 3: face 3 is shifted down by three blocks; it touches face 6.
    """
    # Define the common squares for faces 1, 2, 4, 5, and 6.
    # All squares are 1×1 in the plane z=0.
    # Coordinates are given as [x,y,z].
    faces_common = {
        '1': np.array([
            [0,  0, 0],
            [1,  0, 0],
            [1,  1, 0],
            [0,  1, 0]
        ]),
        '2': np.array([
            [1,  0, 0],
            [2,  0, 0],
            [2,  1, 0],
            [1,  1, 0]
        ]),
        '4': np.array([
            [1, -1, 0],
            [2, -1, 0],
            [2,  0, 0],
            [1,  0, 0]
        ]),
        '5': np.array([
            [1, -2, 0],
            [2, -2, 0],
            [2, -1, 0],
            [1, -1, 0]
        ]),
        '6': np.array([
            [1, -3, 0],
            [2, -3, 0],
            [2, -2, 0],
            [1, -2, 0]
        ])
    }
    
    # Variant 1: Shift face 3 down by one block.
    # Original face 3 (in top row) is at x from 2 to 3, y from 0 to 1.
    # In Variant 1, we shift it down by 1 unit: x remains the same, y becomes -1 to 0.
    face3_variant1 = np.array([
        [2, -1, 0],
        [3, -1, 0],
        [3,  0, 0],
        [2,  0, 0]
    ])
    # Update connectivity:
    # Now the top row has faces 1 and 2 only.
    # The vertical column (starting at face 4) will include face 3 at the top.
    face_connections_variant1 = {
        '1': ['2'],
        '2': ['1','4'],
        '3': ['4'],     # face 3 attaches to face 4 now.
        '4': ['2','3','5'],
        '5': ['4','6'],
        '6': ['5']
    }
    net_variant1 = {
        "name": "Big T - Variant 1",
        "faces": {
            '1': faces_common['1'],
            '2': faces_common['2'],
            '3': face3_variant1,
            '4': faces_common['4'],
            '5': faces_common['5'],
            '6': faces_common['6']
        },
        "face_connections": face_connections_variant1,
        "valid": True
    }
    
    # Variant 2: Shift face 3 down by two blocks.
    # Now face 3: x from 2 to 3, y from -2 to -1.
    face3_variant2 = np.array([
        [2, -2, 0],
        [3, -2, 0],
        [3, -1, 0],
        [2, -1, 0]
    ])
    # Update connectivity: Now face 3 should connect to face 5.
    face_connections_variant2 = {
        '1': ['2'],
        '2': ['1','4'],
        '3': ['5'],    # face 3 attaches to face 5 now.
        '4': ['2','5'],
        '5': ['4','3','6'],
        '6': ['5']
    }
    net_variant2 = {
        "name": "Big T - Variant 2",
        "faces": {
            '1': faces_common['1'],
            '2': faces_common['2'],
            '3': face3_variant2,
            '4': faces_common['4'],
            '5': faces_common['5'],
            '6': faces_common['6']
        },
        "face_connections": face_connections_variant2,
        "valid": True
    }
    
    # Variant 3: Shift face 3 down by three blocks.
    # Now face 3: x from 2 to 3, y from -3 to -2.
    face3_variant3 = np.array([
        [2, -3, 0],
        [3, -3, 0],
        [3, -2, 0],
        [2, -2, 0]
    ])
    # Update connectivity: Now face 3 should connect to face 6.
    face_connections_variant3 = {
        '1': ['2'],
        '2': ['1','4'],
        '3': ['6'],    # face 3 attaches to face 6 now.
        '4': ['2','5'],
        '5': ['4','6'],
        '6': ['5','3']
    }
    net_variant3 = {
        "name": "Big T - Variant 3",
        "faces": {
            '1': faces_common['1'],
            '2': faces_common['2'],
            '3': face3_variant3,
            '4': faces_common['4'],
            '5': faces_common['5'],
            '6': faces_common['6']
        },
        "face_connections": face_connections_variant3,
        "valid": True
    }
    
    return [net_variant1, net_variant2, net_variant3]

def create_t_variants():
    net_variant_1 = {
        "name": "t-shaped Net Variant 1",   # move 1 to next to 4
        "faces": {
            '3': np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),   # center (base)
            '2': np.array([[-1,0,0],[0,0,0],[0,1,0],[-1,1,0]]), # left
            '4': np.array([[1,0,0],[2,0,0],[2,1,0],[1,1,0]]),   # right
            '5': np.array([[2,0,0],[3,0,0],[3,1,0],[2,1,0]]),   # attached to face 4
            '6': np.array([[0,-1,0],[1,-1,0],[1,0,0],[0,0,0]]),   # below
            # 1 is next to 4
            '1': np.array([[1,1,0],[2,1,0],[2,2,0],[1,2,0]])   # above
        },
        "face_connections": {
            '1': ['4'],
            '2': ['3'],
            '3': ['2','4','6'],
            '4': ['1','3','5'],
            '5': ['4'],
            '6': ['3']
        },
        "valid": True
    }

    net_variant_2 = {
        "name": "t-shaped Net Variant 2",   # move 1 to next to 5
        "faces": {
            '3': np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),   # center (base)
            '2': np.array([[-1,0,0],[0,0,0],[0,1,0],[-1,1,0]]), # left
            '4': np.array([[1,0,0],[2,0,0],[2,1,0],[1,1,0]]),   # right
            '5': np.array([[2,0,0],[3,0,0],[3,1,0],[2,1,0]]),   # attached to face 4
            '6': np.array([[0,-1,0],[1,-1,0],[1,0,0],[0,0,0]]),   # below
            # 1 is next to 5
            '1': np.array([[2,1,0],[3,1,0],[3,2,0],[2,2,0]])   # above
        },
        "face_connections": {
            '1': ['5'],
            '2': ['3'],
            '3': ['2','4','6'],
            '4': ['3','5'],
            '5': ['1','4'],
            '6': ['3']
        },
        "valid": True
    }
    return [net_variant_1, net_variant_2]


def create_1_3_2_variant():
    net_1_3_2 = {
        "name": "1-3-2 Net",
        "faces": {
            # Face 1: top left, from (0,2) to (1,3)
            '1': np.array([
                [0, 2, 0],
                [1, 2, 0],
                [1, 3, 0],
                [0, 3, 0]
            ]),
            # Face 2: top right, from (1,2) to (2,3)
            '2': np.array([
                [1, 2, 0],
                [2, 2, 0],
                [2, 3, 0],
                [1, 3, 0]
            ]),
            # Face 3: second row, below face 2, from (1,1) to (2,2)
            '3': np.array([
                [1, 1, 0],
                [2, 1, 0],
                [2, 2, 0],
                [1, 2, 0]
            ]),
            # Face 4: third row, left square, from (1,0) to (2,1)
            '4': np.array([
                [1, 0, 0],
                [2, 0, 0],
                [2, 1, 0],
                [1, 1, 0]
            ]),
            # Face 5: third row, right square, from (2,0) to (3,1)
            '5': np.array([
                [2, 0, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0]
            ]),
            # Face 6: fourth row, from (2,-1) to (3,0)
            '6': np.array([
                [2, -1, 0],
                [3, -1, 0],
                [3,  0, 0],
                [2,  0, 0]
            ])
        },
        "face_connections": {
            '1': ['2'],
            '2': ['1', '3'],
            '3': ['2', '4'],
            '4': ['3', '5'],
            '5': ['4', '6'],
            '6': ['5']
        },
        "valid": True
    }
    # variant 1: move 1 down to next to 3
    from copy import deepcopy
    net_1_3_2_variant_1 = deepcopy(net_1_3_2)
    # from (0,0) to (1,1)
    net_1_3_2_variant_1['faces']['1'] = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [1, 2, 0],
        [0, 2, 0]
    ])
    net_1_3_2_variant_1['face_connections']['1'] = ['3']
    net_1_3_2_variant_1['face_connections']['2'] = ['3']
    net_1_3_2_variant_1['face_connections']['3'] = ['1', '2', '4']

    # variant 2: move 1 down to next to 4
    net_1_3_2_variant_2 = deepcopy(net_1_3_2)
    net_1_3_2_variant_2['faces']['1'] = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ])
    net_1_3_2_variant_2['face_connections']['1'] = ['4']
    net_1_3_2_variant_2['face_connections']['2'] = ['3']
    net_1_3_2_variant_2['face_connections']['4'] = ['1', '3', '5']

    return [net_1_3_2_variant_1, net_1_3_2_variant_2]


def generate_cube_nets():
    """
    Returns a list of cube net definitions.
    Each net is a dictionary with the following keys:
      - "name": a string name.
      - "faces": a dict mapping face key (string) to a 4x3 numpy array of vertices.
      - "face_connections": a dict mapping face key to a list of connected face keys.
      - "valid": a Boolean indicating whether the net is intended to be foldable into a cube.
    
    The following examples are provided:
      1. T-shaped Net
      2. Cross-shaped Net
      3. Row Net
      4. Zigzag Net
    """
    nets = []
    
    # Net 1: t-shaped Net.
    net1 = {
        "name": "t-shaped Net",
        "faces": {
            '3': np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),   # center (base)
            '1': np.array([[0,1,0],[1,1,0],[1,2,0],[0,2,0]]),   # above
            '2': np.array([[-1,0,0],[0,0,0],[0,1,0],[-1,1,0]]), # left
            '4': np.array([[1,0,0],[2,0,0],[2,1,0],[1,1,0]]),   # right
            '5': np.array([[2,0,0],[3,0,0],[3,1,0],[2,1,0]]),   # attached to face 4
            '6': np.array([[0,-1,0],[1,-1,0],[1,0,0],[0,0,0]])   # below
        },
        "face_connections": {
            '1': ['3'],
            '2': ['3'],
            '3': ['1','2','4','6'],
            '4': ['3','5'],
            '5': ['4'],
            '6': ['3']
        },
        "valid": True
    }
    nets.append(net1)
    nets.extend(create_t_variants())

    big_t_shaped_net = {
        "name": "Big T Net",
        "faces": {
            # Top-left square (1)
            '1': np.array([
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0]
            ]),
            # Top-middle square (2)
            '2': np.array([
                [1, 0, 0],
                [2, 0, 0],
                [2, 1, 0],
                [1, 1, 0]
            ]),
            # Top-right square (3)
            '3': np.array([
                [2, 0, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0]
            ]),
            # Middle column, first square under (2) -> square (4)
            '4': np.array([
                [1, -1, 0],
                [2, -1, 0],
                [2,  0, 0],
                [1,  0, 0]
            ]),
            # Next square down (5)
            '5': np.array([
                [1, -2, 0],
                [2, -2, 0],
                [2, -1, 0],
                [1, -1, 0]
            ]),
            # Bottom square (6)
            '6': np.array([
                [1, -3, 0],
                [2, -3, 0],
                [2, -2, 0],
                [1, -2, 0]
            ])
        },
        "face_connections": {
            '1': ['2'],
            '2': ['1','3','4'],
            '3': ['2'],
            '4': ['2','5'],
            '5': ['4','6'],
            '6': ['5']
        },
        "valid": True
    }
    nets.append(big_t_shaped_net)
    # Example usage:
    variants = create_big_t_variants_adjusted()
    nets.extend(variants)
    
    # Net 2: Cross-shaped Net.
    net_3_3 = {
        "name": "3-3 Net",
        "faces": {
            # Face 1: top square, lower-left at (0,2)
            '1': np.array([
                [0, 2, 0],
                [1, 2, 0],
                [1, 3, 0],
                [0, 3, 0]
            ]),
            # Face 2: middle square of left column, lower-left at (0,1)
            '2': np.array([
                [0, 1, 0],
                [1, 1, 0],
                [1, 2, 0],
                [0, 2, 0]
            ]),
            # Face 3: bottom square of left column, lower-left at (0,0)
            '3': np.array([
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0]
            ]),
            # Face 4: top square of right column, attached to face 3, lower-left at (1,0)
            '4': np.array([
                [1, 0, 0],
                [2, 0, 0],
                [2, 1, 0],
                [1, 1, 0]
            ]),
            # Face 5: middle square of right column, below face 4, lower-left at (1,-1)
            '5': np.array([
                [1, -1, 0],
                [2, -1, 0],
                [2, 0, 0],
                [1, 0, 0]
            ]),
            # Face 6: bottom square of right column, below face 5, lower-left at (1,-2)
            '6': np.array([
                [1, -2, 0],
                [2, -2, 0],
                [2, -1, 0],
                [1, -1, 0]
            ])
        },
        "face_connections": {
            '1': ['2'],
            '2': ['1', '3'],
            '3': ['2', '4'],
            '4': ['3', '5'],
            '5': ['4', '6'],
            '6': ['5']
        },
        "valid": True
    }
    nets.append(net_3_3)
    
    net_2_2 = {
        "name": "2-2 net",
        "faces": {
            '1': np.array([
                [0, 3, 0],
                [1, 3, 0],
                [1, 4, 0],
                [0, 4, 0]
            ]),
            '2': np.array([
                [0, 2, 0],
                [1, 2, 0],
                [1, 3, 0],
                [0, 3, 0]
            ]),
            '3': np.array([
                [1, 2, 0],
                [2, 2, 0],
                [2, 3, 0],
                [1, 3, 0]
            ]),
            '4': np.array([
                [1, 1, 0],
                [2, 1, 0],
                [2, 2, 0],
                [1, 2, 0]
            ]),
            '5': np.array([
                [2, 1, 0],
                [3, 1, 0],
                [3, 2, 0],
                [2, 2, 0]
            ]),
            '6': np.array([
                [2, 0, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0]
            ])
        },
        "face_connections": {
            '1': ['2'],
            '2': ['1', '3'],
            '3': ['2', '4'],
            '4': ['3', '5'],
            '5': ['4', '6'],
            '6': ['5']
        },
        "valid": True
    }
    nets.append(net_2_2)

    net_1_3_2 = {
        "name": "1-3-2 Net",
        "faces": {
            # Face 1: top left, from (0,2) to (1,3)
            '1': np.array([
                [0, 2, 0],
                [1, 2, 0],
                [1, 3, 0],
                [0, 3, 0]
            ]),
            # Face 2: top right, from (1,2) to (2,3)
            '2': np.array([
                [1, 2, 0],
                [2, 2, 0],
                [2, 3, 0],
                [1, 3, 0]
            ]),
            # Face 3: second row, below face 2, from (1,1) to (2,2)
            '3': np.array([
                [1, 1, 0],
                [2, 1, 0],
                [2, 2, 0],
                [1, 2, 0]
            ]),
            # Face 4: third row, left square, from (1,0) to (2,1)
            '4': np.array([
                [1, 0, 0],
                [2, 0, 0],
                [2, 1, 0],
                [1, 1, 0]
            ]),
            # Face 5: third row, right square, from (2,0) to (3,1)
            '5': np.array([
                [2, 0, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0]
            ]),
            # Face 6: fourth row, from (2,-1) to (3,0)
            '6': np.array([
                [2, -1, 0],
                [3, -1, 0],
                [3,  0, 0],
                [2,  0, 0]
            ])
        },
        "face_connections": {
            '1': ['2'],
            '2': ['1', '3'],
            '3': ['2', '4'],
            '4': ['3', '5'],
            '5': ['4', '6'],
            '6': ['5']
        },
        "valid": True
    }
    nets.append(net_1_3_2)
    nets.extend(create_1_3_2_variant())
    for net in nets:
        net["ascii"] = ASCII_Nets[net["name"]]
    return nets


def make_net_B(shift):
    # Top row: faces 1 to 5, spanning x=0 to 5, all with y from 1 to 2.
    faces = {
        '1': np.array([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]]),
        '2': np.array([[1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]]),
        '3': np.array([[2, 1, 0], [3, 1, 0], [3, 2, 0], [2, 2, 0]]),
        '4': np.array([[3, 1, 0], [4, 1, 0], [4, 2, 0], [3, 2, 0]]),
        '5': np.array([[4, 1, 0], [5, 1, 0], [5, 2, 0], [4, 2, 0]])
    }
    # Face 6: placed in second row with lower-left corner at (shift, 0) (width 1, height 1).
    faces['6'] = np.array([[shift, 0, 0],
                        [shift+1, 0, 0],
                        [shift+1, 1, 0],
                        [shift, 1, 0]])
    # Define connectivity.
    # Assume top row faces connect consecutively.
    face_connections = {
        '1': ['2'],
        '2': ['1', '3'],
        '3': ['2', '4'],
        '4': ['3', '5'],
        '5': ['4'],
        # We set connectivity only if face 6 shares an edge with one of the top row faces.
        # For these variants, let’s connect face 6 with the face whose x-range overlaps.
    }
    # Determine connection for face 6:
    # Compute its x-range.
    x6_min, x6_max = shift, shift+1
    # Check each top face:
    for key in ['1','2','3','4','5']:
        face = faces[key]
        x_min = np.min(face[:,0])
        x_max = np.max(face[:,0])
        # If there is an overlap of at least 0.5, consider them connected.
        if (x6_min < x_max - 0.5) and (x6_max > x_min + 0.5):
            face_connections.setdefault(key, []).append('6')
            face_connections.setdefault('6', []).append(key)
    return {
        "name": f"Net B: Row 5 + Face 6 (shift={shift})",
        "faces": faces,
        "face_connections": face_connections,
        "valid": False
    }
    

def gather_invalid_cubes():
    # ------------------------------
    # Net A: Linear Net of 6 Faces
    # ------------------------------
    net_A = {
        "name": "Linear 6-Face Row",
        "faces": {
            '1': np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            '2': np.array([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]]),
            '3': np.array([[2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]]),
            '4': np.array([[3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0]]),
            '5': np.array([[4, 0, 0], [5, 0, 0], [5, 1, 0], [4, 1, 0]]),
            '6': np.array([[5, 0, 0], [6, 0, 0], [6, 1, 0], [5, 1, 0]])
        },
        "face_connections": {
            '1': ['2'],
            '2': ['1', '3'],
            '3': ['2', '4'],
            '4': ['3', '5'],
            '5': ['4', '6'],
            '6': ['5']
        },
        "valid": False
    }

    # ------------------------------
    # Net B: Row of 5 with Face 6 on Second Row
    # Variants differ by horizontal shift of face 6.
    def make_net_B(shift):
        # Top row: faces 1 to 5, spanning x=0 to 5, all with y from 1 to 2.
        faces = {
            '1': np.array([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]]),
            '2': np.array([[1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]]),
            '3': np.array([[2, 1, 0], [3, 1, 0], [3, 2, 0], [2, 2, 0]]),
            '4': np.array([[3, 1, 0], [4, 1, 0], [4, 2, 0], [3, 2, 0]]),
            '5': np.array([[4, 1, 0], [5, 1, 0], [5, 2, 0], [4, 2, 0]])
        }
        # Face 6: placed in second row with lower-left corner at (shift, 0) (width 1, height 1).
        faces['6'] = np.array([[shift, 0, 0],
                            [shift+1, 0, 0],
                            [shift+1, 1, 0],
                            [shift, 1, 0]])
        # Define connectivity.
        # Assume top row faces connect consecutively.
        face_connections = {
            '1': ['2'],
            '2': ['1', '3'],
            '3': ['2', '4'],
            '4': ['3', '5'],
            '5': ['4'],
            # We set connectivity only if face 6 shares an edge with one of the top row faces.
            # For these variants, let’s connect face 6 with the face whose x-range overlaps.
        }
        # Determine connection for face 6:
        # Compute its x-range.
        x6_min, x6_max = shift, shift+1
        # Check each top face:
        for key in ['1','2','3','4','5']:
            face = faces[key]
            x_min = np.min(face[:,0])
            x_max = np.max(face[:,0])
            # If there is an overlap of at least 0.5, consider them connected.
            if (x6_min < x_max - 0.5) and (x6_max > x_min + 0.5):
                face_connections.setdefault(key, []).append('6')
                face_connections.setdefault('6', []).append(key)
        return {
            "name": f"Net B: Row 5 + Face 6 (shift={shift})",
            "faces": faces,
            "face_connections": face_connections,
            "valid": False
        }
    '''
    1-2-3-4-5
    6

    1-2-3-4-5
      6

    1-2-3-4-5
        6
    1-2-3-4-5
          6

    1-2-3-4-5
            6
    '''

    net_B1 = make_net_B(shift=0)   # Face 6 left-aligned with face 1.
    net_B2 = make_net_B(shift=1)   # Face 6 aligned with face 2.
    net_B3 = make_net_B(shift=2)   # Face 6 aligned with face 3.
    net_B4 = make_net_B(shift=3)   # Face 6 aligned with face 4.
    net_B5 = make_net_B(shift=4)   # Face 6 aligned with face 5.

    # ------------------------------
    # Net C: Two Rows of Three Faces
    # Top row: faces 1-2-3; Bottom row: faces 4-5-6.

    '''
    1-2-3
    4-5-6
    '''
    net_C = {
        "name": "2 Rows of 3",
        "faces": {
            '1': np.array([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]]),
            '2': np.array([[1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]]),
            '3': np.array([[2, 1, 0], [3, 1, 0], [3, 2, 0], [2, 2, 0]]),
            '4': np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            '5': np.array([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]]),
            '6': np.array([[2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]])
        },
        "face_connections": {
            '1': ['2','4'],
            '2': ['1','3','5'],
            '3': ['2','6'],
            '4': ['1','5'],
            '5': ['2','4','6'],
            '6': ['3','5']
        },
        "valid": False
    }

    net_D = {
        "name": "Invalid D",
        "faces": {
                    '1': np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
                    '2': np.array([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]]),
                    '3': np.array([[1, 1, 0], [1, 2, 0], [2, 2, 0], [2, 1, 0]]),
                    '4': np.array([[1, 2, 0], [2, 2, 0], [2, 3, 0], [1, 3, 0]]),
                    '5': np.array([[2, 2, 0], [3, 2, 0], [3, 3, 0], [2, 3, 0]]),
                    '6': np.array([[3, 2, 0], [4, 2, 0], [4, 3, 0], [3, 3, 0]])
                },
        "face_connections": {
            '1': ['2'],
            '2': ['1','3'],
            '3': ['2','4'],
            '4': ['3','5'],
            '5': ['4','6'],
            '6': ['5']
        },
        "valid": False
    }

    '''
    1-2-3
      4-5-6
    '''
    net_D1 = {
        "name": "Net H: Two rows with bottom row shifted right",
        "faces": {
            # Top row: faces 1,2,3 (y from 1 to 2)
            '1': np.array([
                [0, 1, 0],
                [1, 1, 0],
                [1, 2, 0],
                [0, 2, 0]
            ]),
            '2': np.array([
                [1, 1, 0],
                [2, 1, 0],
                [2, 2, 0],
                [1, 2, 0]
            ]),
            '3': np.array([
                [2, 1, 0],
                [3, 1, 0],
                [3, 2, 0],
                [2, 2, 0]
            ]),
            # Bottom row: faces 4,5,6 (shifted right by 1; y from 0 to 1)
            '4': np.array([
                [1, 0, 0],
                [2, 0, 0],
                [2, 1, 0],
                [1, 1, 0]
            ]),
            '5': np.array([
                [2, 0, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0]
            ]),
            '6': np.array([
                [3, 0, 0],
                [4, 0, 0],
                [4, 1, 0],
                [3, 1, 0]
            ])
        },
        "face_connections": {
            '1': ['2'],
            '2': ['1', '3', '4'],
            '3': ['2', '5'],
            '4': ['2', '5'],
            '5': ['3', '4', '6'],
            '6': ['5']
        },
        "valid": False
    }
    
    '''
    1      6
    2 - 3 -4 - 5
    '''

    net_G = {
        "name": "invalid_G",
        "faces": {
            # Bottom row (y from 0 to 1): faces 2,3,4,5
            '2': np.array([
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0]
            ]),
            '3': np.array([
                [1, 0, 0],
                [2, 0, 0],
                [2, 1, 0],
                [1, 1, 0]
            ]),
            '4': np.array([
                [2, 0, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0]
            ]),
            '5': np.array([
                [3, 0, 0],
                [4, 0, 0],
                [4, 1, 0],
                [3, 1, 0]
            ]),
            # Top row: face 1 above face 2, face 6 above face 5.
            '1': np.array([
                [0, 1, 0],
                [1, 1, 0],
                [1, 2, 0],
                [0, 2, 0]
            ]),
            '6': np.array([
                [3, 1, 0],
                [4, 1, 0],
                [4, 2, 0],
                [3, 2, 0]
            ])
        },
        "face_connections": {
            # Bottom row: consecutive connections.
            '2': ['3', '1'],
            '3': ['2','4'],
            '4': ['3','5'],
            '5': ['4', '6'],
            # Top row:
            '1': ['2'],
            '6': ['5']
        },
        "valid": False
    }

    '''
    1-2-3-4
      5-6
    '''

    net_F = {
        "name": "invalid_F",
        "faces": {
            # Top row (y from 1 to 2)
            '1': np.array([
                [0, 1, 0],
                [1, 1, 0],
                [1, 2, 0],
                [0, 2, 0]
            ]),
            '2': np.array([
                [1, 1, 0],
                [2, 1, 0],
                [2, 2, 0],
                [1, 2, 0]
            ]),
            '3': np.array([
                [2, 1, 0],
                [3, 1, 0],
                [3, 2, 0],
                [2, 2, 0]
            ]),
            '4': np.array([
                [3, 1, 0],
                [4, 1, 0],
                [4, 2, 0],
                [3, 2, 0]
            ]),
            # Bottom row (y from 0 to 1) – indented: aligned with faces 3-4
            '5': np.array([
                [2, 0, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0]
            ]),
            '6': np.array([
                [3, 0, 0],
                [4, 0, 0],
                [4, 1, 0],
                [3, 1, 0]
            ])
        },
        "face_connections": {
            # Top row: adjacent horizontally.
            '1': ['2'],
            '2': ['1','3'],
            '3': ['2','4','5'],  # also touches bottom row face 5.
            '4': ['3','6'],      # touches bottom row face 6.
            # Bottom row:
            '5': ['3','6'],
            '6': ['4','5']
        },
        "valid": False
    }


    '''
    1         6
    2 - 3 -4 -5
    '''
    net_E = {
        "name": "invalid_E",
        "faces": {
            # Top row
            '1': np.array([
                [0, 1, 0],    # face 1: lower-left at (0,1)
                [1, 1, 0],
                [1, 2, 0],
                [0, 2, 0]
            ]),
            '6': np.array([
                [3, 1, 0],    # face 6: lower-left at (3,1)
                [4, 1, 0],
                [4, 2, 0],
                [3, 2, 0]
            ]),
            # Bottom row: contiguous row of 4 faces
            '2': np.array([
                [0, 0, 0],    # face 2: lower-left at (0,0)
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0]
            ]),
            '3': np.array([
                [1, 0, 0],    # face 3: from (1,0) to (2,1)
                [2, 0, 0],
                [2, 1, 0],
                [1, 1, 0]
            ]),
            '4': np.array([
                [2, 0, 0],    # face 4: from (2,0) to (3,1)
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0]
            ]),
            '5': np.array([
                [3, 0, 0],    # face 5: from (3,0) to (4,1)
                [4, 0, 0],
                [4, 1, 0],
                [3, 1, 0]
            ])
        },
        "face_connections": {
            # Top row connections (vertical only)
            '1': ['2'],      # face 1 touches face 2 below
            '6': ['5'],      # face 6 touches face 5 below
            # Bottom row: consecutive connections
            '2': ['1', '3'],
            '3': ['2', '4'],
            '4': ['3', '5'],
            '5': ['4', '6']
        },
        "valid": False
    }

    nets = [net_A, net_B1, net_B2, net_B3, net_B4, net_B5, net_C, net_D, net_G, net_F, net_E, net_D1]
    for net in nets:
        net["ascii"] = ASCII_Nets[net["name"]]
    return nets
