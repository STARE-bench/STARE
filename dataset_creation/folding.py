import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import json
from azfuse import File
# --- Function to Randomize Folding Steps Based on the Net Connections ---
import random
from tqdm import tqdm


def plot_faces(faces, step, output_file_prefix=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for face_key, face in faces.items():
        verts = [face]
        face_collection = Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=0.75)
        try:
            color = plt.cm.Set3(int(face_key) / 12)
        except:
            color = plt.cm.Set3(0)
        face_collection.set_facecolor(color)
        ax.add_collection3d(face_collection)
        centroid = np.mean(face, axis=0)
        ax.text(centroid[0], centroid[1], centroid[2], face_key,
                color='black', fontsize=12, ha='center', va='center')
        # print(face_key)
    ax.set_xlim([-2, 4])
    ax.set_ylim([-2, 4])
    ax.set_zlim([-2, 4])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    plt.tight_layout()
    with File.open(f"{output_file_prefix}_step{step}.png", "wb") as f:
        plt.savefig(f)
    plt.close()


def face_normal(face):
    """Compute a unit normal for the face (assumes face has at least 3 vertices)."""
    v1 = face[1] - face[0]
    v2 = face[2] - face[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return np.array([0,0,1])
    return n / norm

def rotation_matrix(axis, angle):
    """Return the rotation matrix for a rotation around 'axis' by 'angle' radians."""
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [c + ux*ux*C,      ux*uy*C - uz*s,  ux*uz*C + uy*s],
        [uy*ux*C + uz*s,   c + uy*uy*C,     uy*uz*C - ux*s],
        [uz*ux*C - uy*s,   uz*uy*C + ux*s,  c + uz*uz*C]
    ])


def find_shared_edge(folded_face, candidate_face, faces, tol=1e-6):
    """
    Returns two shared vertices (as NumPy arrays) between folded_face and candidate_face,
    or (None, None) if no proper edge is found.
    """
    shared = []
    for vA in faces[folded_face]:
        for vB in faces[candidate_face]:
            if np.allclose(vA, vB, atol=tol):
                shared.append(vA)
    if len(shared) >= 2:
        return np.array(shared[0]), np.array(shared[1])
    return None, None


def compute_fold_axis_and_angle(folded_face, candidate_face, base_face, faces, tol=1e-6):
    """
    Compute the rotation parameters to fold candidate_face using folded_face as the hinge,
    aligning candidate_face so its center moves toward the cube's center (computed from base_face).
    The rotation angle is always 90° (i.e. ±π/2); the function only determines the correct sign.
    
    Parameters:
      folded_face: key of the already folded face (used to detect the shared edge)
      candidate_face: key of the face to be folded
      base_face: key of the face that defines the cube's base (used to compute cube center)
      faces: dictionary mapping face key -> Nx3 numpy array of vertices
      tol: tolerance for floating point comparisons
      
    Returns:
      axis_point: a point on the shared edge (pivot)
      rotation_axis: the normalized shared edge (rotation axis)
      angle: either +π/2 or -π/2, chosen so that candidate_face’s center moves toward the cube center.
    """
    # 1. Find the shared edge between folded_face and candidate_face.
    v1, v2 = find_shared_edge(folded_face, candidate_face, faces, tol)
    if v1 is None:
        # Fallback: if no shared edge is found, use the first edge of candidate_face.
        v1, v2 = faces[candidate_face][0], faces[candidate_face][1]
    axis_point = v1
    # Compute the default rotation axis from the shared edge.
    default_axis = v2 - v1
    if np.linalg.norm(default_axis) < tol:
        default_axis = np.array([0, 1, 0])
    else:
        default_axis = default_axis / np.linalg.norm(default_axis)
    
    # 2. Compute the cube's center based on the base_face.
    base_center = np.mean(faces[base_face], axis=0)
    # Assume the base is square; get side length from the first edge.
    side_length = np.linalg.norm(faces[base_face][1] - faces[base_face][0])
    n_base = face_normal(faces[base_face])
    # For a cube with the base as the bottom face:
    cube_center = base_center + 0.5 * side_length * n_base

    # 3. Compute the candidate face's center.
    candidate_center = np.mean(faces[candidate_face], axis=0)
    
    # 4. From the pivot, compute the vectors toward cube_center and candidate_center.
    v_desired = cube_center - axis_point   # desired direction from pivot
    v_candidate = candidate_center - axis_point  # current direction from pivot

    # 5. Project both vectors onto the plane perpendicular to default_axis.
    def project_onto_plane(v, axis):
        return v - np.dot(v, axis) * axis

    p_desired = project_onto_plane(v_desired, default_axis)
    p_candidate = project_onto_plane(v_candidate, default_axis)

    # If either projection is nearly zero, return a fixed rotation.
    if np.linalg.norm(p_desired) < tol or np.linalg.norm(p_candidate) < tol:
        return axis_point, default_axis, -np.pi/2

    p_desired /= np.linalg.norm(p_desired)
    p_candidate /= np.linalg.norm(p_candidate)
    
    # 6. Apply both a +90° and a -90° rotation to p_candidate.
    R_pos = rotation_matrix(default_axis, np.pi/2)
    R_neg = rotation_matrix(default_axis, -np.pi/2)
    p_candidate_pos = np.dot(R_pos, p_candidate)
    p_candidate_neg = np.dot(R_neg, p_candidate)
    
    # 7. Choose the rotation that brings p_candidate closer to p_desired.
    dot_pos = np.dot(p_candidate_pos, p_desired)
    dot_neg = np.dot(p_candidate_neg, p_desired)
    if dot_pos > dot_neg:
        angle = np.pi/2
    else:
        angle = -np.pi/2

    return axis_point, default_axis, angle



def rotate_points_around_axis(points, axis_point, axis_direction, angle):
    """
    Rotates an array of points (Nx3) around the given axis by the specified angle.
    """
    translated = points - axis_point
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    ux, uy, uz = axis_direction
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [c + ux*ux*C,      ux*uy*C - uz*s,  ux*uz*C + uy*s],
        [uy*ux*C + uz*s,   c + uy*uy*C,     uy*uz*C - ux*s],
        [uz*ux*C - uy*s,   uz*uy*C + ux*s,  c + uz*uz*C]
    ])
    return np.dot(translated, R.T) + axis_point

# def rotate_face_and_connected(face_key, axis_point, axis_direction, angle, faces, face_connections, rotated_faces, folded, base_face):
#     """
#     Recursively rotates face_key and all connected (unvisited) faces.
#     The rotation uses the given pivot (axis_point) and axis_direction.
#     """
#     if face_key in rotated_faces:
#         return
#     faces[face_key] = rotate_points_around_axis(faces[face_key], axis_point, axis_direction, angle)
#     rotated_faces.append(face_key)
#     # Propagate the rotation to connected faces that haven't been folded yet.
#     for connected_face in face_connections.get(face_key, []):
#         if connected_face not in folded:
#             rotate_face_and_connected(connected_face, axis_point, axis_direction, angle,
#                                         faces, face_connections, rotated_faces, folded)


def rotate_face_and_connected(face_key, axis_point, axis_direction, angle, 
                                faces, face_connections, rotated_faces, folded, base_face):
    """
    Recursively rotates face_key and all connected (unvisited) faces,
    except for any face that is connected with the base face.
    
    The rotation uses the given pivot (axis_point) and axis_direction.
    
    Parameters:
      face_key: the current face to rotate.
      axis_point, axis_direction, angle: rotation parameters.
      faces: dictionary mapping face keys to their vertex arrays.
      face_connections: dictionary mapping face keys to a list of connected face keys.
      rotated_faces: list of faces already rotated in this recursive call.
      folded: set of faces that have been explicitly folded.
      base_face: the key of the base face that must remain fixed.
    """
    if face_key in rotated_faces or face_key == base_face:
        return

    # Rotate this face.
    faces[face_key] = rotate_points_around_axis(faces[face_key], axis_point, axis_direction, angle)
    rotated_faces.append(face_key)
    
    # Propagate the rotation to connected faces.
    for connected_face in face_connections.get(face_key, []):
        # If the connected face is the base face or is directly connected with the base face,
        # then skip rotating it.
        if connected_face == base_face or (base_face in face_connections.get(connected_face, [])):
            continue
        if connected_face not in folded:
            rotate_face_and_connected(connected_face, axis_point, axis_direction, angle,
                                        faces, face_connections, rotated_faces, folded, base_face)



def describe_folding_step(face_key, neighbor_face, axis_direction, angle,
                            candidate_center_before, candidate_center_after,
                            base_center, n_base, faces, face_connections, special=False):
    """
    Returns a natural-language description of the folding step using relative distance.
    The candidate face’s center is compared (before and after rotation) relative to the base face’s plane.
    
    Parameters:
      - face_key: the face being folded.
      - neighbor_face: the face toward which it folds.
      - axis_direction, angle: rotation details.
      - candidate_center_before: candidate's center before rotation.
      - candidate_center_after: candidate's center after rotation.
      - base_center: center of the base face.
      - n_base: unit normal of the base face.
      - special: if True, use a fixed description.
      
    The function computes:
      d_before = dot(candidate_center_before - base_center, n_base)
      d_after  = dot(candidate_center_after  - base_center, n_base)
      delta  = d_after - d_before
      
    Then, if the base is roughly horizontal (|n_base[2]| > 0.5):
      delta > 0 implies "upwards", delta < 0 implies "downwards".
    Otherwise:
      delta > 0 implies "outwards", delta < 0 implies "inwards".
    """
    # tol = 1e-6
    if special:
        return f"Fold face {face_key} inward to close the cube."
    
    # Compute distances relative to the base face.
    d_before = np.dot(candidate_center_before - base_center, n_base)
    d_after  = np.dot(candidate_center_after  - base_center, n_base)
    delta = d_after - d_before

    distance = candidate_center_before - candidate_center_after


    # Choose word pair based on base orientation.
    # if abs(n_base[2]) > 0.5:
    if np.abs(distance[-1]) > 0:
        simple_direction = "upwards" if delta > 0 else "downwards"
    else:
        simple_direction = "outwards" if delta > 0 else "inwards"
    
    # Format angle and axis.
    angle_deg = abs(np.degrees(angle))
    axis_str = f"({axis_direction[0]:.2f}, {axis_direction[1]:.2f}, {axis_direction[2]:.2f})"

    previous_face_connection = face_connections[face_key]
    new_conn = None
    for other_key in faces:
        if other_key == face_key or other_key == neighbor_face:
            continue
        v1, v2 = find_shared_edge(face_key, other_key, faces)
        if v1 is not None:
            if other_key not in previous_face_connection:
                new_conn = other_key
                break
    
    if new_conn is None:
        return (f"Fold face {face_key} {simple_direction}")
                # f"(angle: {angle_deg:.0f}°, axis: {axis_str}).")
    else:
        return (f"Fold face {face_key} {simple_direction} towards face {new_conn}")
                # f"(angle: {angle_deg:.0f}°, axis: {axis_str}).")


# In iterative_folding:
def iterative_folding(faces, face_connections, base_face='3', output_file_prefix="CubeNet"):
    print(f"Starting folding simulation from face {base_face}.")
    # visited = {base_face}
    folded = {base_face}
    steps = []
    step_descs = []
    step_counter = 0

    step_counter += 1
    plot_faces(faces, step=step_counter, output_file_prefix=output_file_prefix)

    base_center = np.mean(faces[base_face], axis=0)
    n_base = face_normal(faces[base_face])

    available_faces = list(faces.keys())

    is_visually_valid = True
    while len(folded) < len(faces):
        candidate = None
        connecting_face = None
        available_faces = list(set(available_faces) - folded)
        for face_key in available_faces:
            neighbors = face_connections.get(face_key, [])
            common = [n for n in neighbors if n in folded]
            if common:
                candidate = face_key
                connecting_face = random.choice(common)
                break
        if candidate is None:
            break

        axis_point, axis_direction, angle = compute_fold_axis_and_angle(connecting_face, candidate, base_face, faces)
        # print(f"Fold face {candidate} using neighbor {connecting_face} with axis {axis_point}, "
        #       f"direction {axis_direction}, angle {angle:.2f}")
        # Append step with neighbor_face info.
        # visited.add(candidate)

        candidate_center_before = np.mean(faces[candidate].copy(), axis=0)

        rotated_faces = list()
        rotate_face_and_connected(candidate, np.array(axis_point), np.array(axis_direction), angle,
                                    faces, face_connections, rotated_faces, folded, base_face)
        
        
        candidate_center_after = np.mean(faces[candidate].copy(), axis=0)
        
        curr_rotated_faces = [rotated_faces[0]]
        # print(f"Rotated faces: {rotated_faces[0]} -> {set(rotated_faces)}")
        folded.update(set(curr_rotated_faces))
        # print(f"Folded face {candidate} using neighbor {connecting_face} with axis {axis_point}, "
        #       f"direction {axis_direction}, angle {angle:.2f}")
        # print("Check if the cube is valid:", check_valid_cube(faces))

        
        # print(step_desc)
        # is_face_overlap = check_face_overlap(candidate, faces)
        # is_conflicting_shared_edges = check_conflicting_shared_edges(candidate, faces)
        if is_face_overlap(candidate, faces):
            print(f"Face {candidate} overlaps with another face.")
            is_visually_valid = False
        if is_conflicting_shared_edges(candidate, faces):
            print(f"Face {candidate} has conflicting shared edges.")
            is_visually_valid = False
        if is_connection_disconnected(faces, face_connections):
            print(f"Face {candidate} has disconnected connections.")
            is_visually_valid = False
        
        step_counter += 1
        if is_visually_valid:
            plot_faces(faces, step=step_counter, output_file_prefix=output_file_prefix)

        step_desc = describe_folding_step(face_key, connecting_face, axis_direction, angle,
                                    candidate_center_before, candidate_center_after, base_center, n_base, faces, face_connections)
        steps.append((candidate, connecting_face, axis_point.tolist(), axis_direction.tolist(), angle))
        step_descs.append(step_desc)

    return steps, step_descs


# In perturbe_folding_steps:
def perturbe_folding_steps(selected_net, folding_steps, n_perturbe=1, base_face='3'):
    from copy import deepcopy
    faces = deepcopy(selected_net["faces"])
    perturbed_folding_steps = []

    perturb_index = np.random.choice(len(folding_steps), n_perturbe, replace=False)
    face_connections = selected_net["face_connections"]
    folded = set([base_face])
    plot_faces(faces, step=0, net_name="wrong_"+selected_net["name"])

    base_center = np.mean(faces[base_face], axis=0)
    n_base = face_normal(faces[base_face])
    min_perterb_index = min(perturb_index)
    for i in range(len(folding_steps)):
        # Unpack the updated tuple with neighbor_face.
        face_key, neighbor_face, axis_point, axis_direction, angle = folding_steps[i]
        if i in perturb_index:
            if np.random.rand() > 0.5:
                angle = -angle
            else:
                axis_direction_perturbed = axis_direction
                while np.allclose(np.abs(axis_direction), np.abs(axis_direction_perturbed)):
                    which_axis = np.random.randint(0, 3, 1)[0]
                    axis_direction_perturbed = [0, 0, 0]
                    axis_direction_perturbed[which_axis] = 1
                axis_direction = axis_direction_perturbed
        
        if i > min_perterb_index:
            candidate = None
            new_neighbor = None
            for key in faces:
                if key not in folded:
                    neighbors = face_connections.get(key, [])
                    common = [n for n in neighbors if n in folded]
                    if common:
                        candidate = key
                        new_neighbor = random.choice(common)
                        break
            if candidate is None:
                break
            axis_point, _ = find_shared_edge(new_neighbor, candidate, faces)
            # Optionally update the face_key and neighbor_face:
            face_key = candidate
            neighbor_face = new_neighbor

        candidate_center_before = np.mean(faces[face_key].copy(), axis=0)

        rotated_faces = list()
        rotate_face_and_connected(face_key, np.array(axis_point), np.array(axis_direction), angle,
                                  faces, face_connections, rotated_faces, folded)
        
        candidate_center_after = np.mean(faces[face_key].copy(), axis=0)
        curr_rotated_faces = [rotated_faces[0]]
        folded.update(set(curr_rotated_faces))
        plot_faces(faces, step=i+1, net_name="wrong_"+selected_net["name"]) 
        # print(describe_folding_step(face_key, neighbor_face, axis_direction, angle))
        perturbed_folding_steps.append((face_key, neighbor_face, axis_point, axis_direction, angle))

        print(describe_folding_step(face_key, neighbor_face, axis_direction, angle,
      candidate_center_before, candidate_center_after, base_center, n_base, faces))
    return perturbed_folding_steps


def is_face_overlap(face_key, faces, tol=1e-6):
    """
    Check if the face identified by 'face_key' overlaps with any other face in 'faces'.

    Overlap is detected if any vertex of the candidate face lies in the interior
    (or on the plane) of another face's polygon, or vice versa.

    Parameters:
      face_key: key identifying the candidate face (e.g., "5").
      faces: dictionary mapping face keys to an (N x 3) NumPy array of vertices.
      tol: tolerance for numerical comparisons.

    Returns:
      True if an overlap is found, otherwise False.
    """
    import numpy as np

    # def face_normal(face):
    #     """Compute a unit normal for a face (assumes at least 3 vertices)."""
    #     v1 = face[1] - face[0]
    #     v2 = face[2] - face[0]
    #     n = np.cross(v1, v2)
    #     norm = np.linalg.norm(n)
    #     return n / norm if norm > tol else np.array([0, 0, 1])
    
    # def point_in_polygon(point, polygon):
    #     """
    #     Check if 'point' is inside the polygon.
    #     Assumes the polygon is convex and the point lies in the polygon's plane.
    #     Uses the "half-space" method.
    #     """
    #     n = face_normal(polygon)
    #     num_vertices = len(polygon)
    #     for i in range(num_vertices):
    #         a = polygon[i]
    #         b = polygon[(i + 1) % num_vertices]
    #         edge = b - a
    #         v = point - a
    #         # Compute the cross product and then the dot with the normal.
    #         if np.dot(np.cross(edge, v), n) < -tol:
    #             return False
    #     return True

    # candidate = faces[face_key]
    # candidate_normal = face_normal(candidate)
    
    # # Check candidate's vertices against every other face.
    # for other_key, other_face in faces.items():
    #     if other_key == face_key:
    #         continue
        
    #     other_normal = face_normal(other_face)
    #     other_point = other_face[0]  # a point on the other face
        
    #     # For each vertex in the candidate face, check if it lies in the plane of other_face
    #     # and then if it is inside the polygon.
    #     for vertex in candidate:
    #         # Distance from vertex to the plane defined by other_face.
    #         d = abs(np.dot(vertex - other_point, other_normal))
    #         if d < tol:
    #             if point_in_polygon(vertex, other_face):
    #                 return True  # Overlap detected
        
    #     # Also check the reverse: any vertex of the other face that might lie in candidate.
    #     candidate_point = candidate[0]
    #     for vertex in other_face:
    #         d = abs(np.dot(vertex - candidate_point, candidate_normal))
    #         if d < tol:
    #             if point_in_polygon(vertex, candidate):
    #                 return True

    # simply check if the face center overlaps with any other face

    candidate = faces[face_key]
    candidate_center = np.mean(candidate, axis=0)
    for other_key, other_face in faces.items():
        if other_key == face_key:
            continue
        other_center = np.mean(other_face, axis=0)
        if np.linalg.norm(candidate_center - other_center) < tol:
            return True
    return False


# def check_if_connection_disconnected(face_key, faces, face_connections):
#     """
#     Check if the connections to the face identified by 'face_key' are disconnected.
#     """
#     connected_faces = face_connections.get(face_key, [])
#     for connected_face in connected_faces:
#         if connected_face not in face_connections:
#             return True
#     return False

def is_conflicting_shared_edges(face_key, faces):
    # first find the shared edges of the face to each other face

    shared_edges = {}
    for other_key, other_face in faces.items():
        if other_key == face_key:
            continue
        v1, v2 = find_shared_edge(face_key, other_key, faces)
        if v1 is not None:
            shared_edges[other_key] = (v1, v2)
    # print(f"Face Key: {face_key} Shared edges: {shared_edges}")
    # check if any shared edges are conflicting
    for key1, (v1, v2) in shared_edges.items():
        for key2, (v3, v4) in shared_edges.items():
            if key1 == key2:
                continue
            if np.allclose(v1, v3) and np.allclose(v2, v4):
                return True
    return False


def is_connection_disconnected(faces, face_connections):
    """
    Check if the connections to the face identified by 'face_key' are disconnected.
    """
    for face_key in faces:
        connected_faces = face_connections.get(face_key, [])
        for connected_face in connected_faces:
            # find the shared edge between face_key and connected_face
            v1, v2 = find_shared_edge(face_key, connected_face, faces)
            if v1 is None:
                return True
    return False


def gen_example_per_net(selected_net, base_face, output_file_prefix, overwrite=False):

    output_file_prefix = output_file_prefix + f"_base{base_face}"
    output_json_path = output_file_prefix + ".json"
    if not overwrite and File.isfile(output_json_path):
        return output_json_path

    # # You can select one by name or at random. For example:
    # selected_net = random.choice(nets)
    # base_face = random.choice(list(selected_net["faces"].keys()))
    print("Selected net:", selected_net["name"])

    from copy import deepcopy
    # Use its faces and face_connections for the folding simulation.
    faces = deepcopy(selected_net["faces"])       # copy to avoid modifying the original
    face_connections = selected_net["face_connections"]
    net_is_valid = selected_net["valid"]
    
    # Run the iterative folding procedure and plot intermediate steps.
    folding_steps, folding_step_descs = iterative_folding(
        faces, face_connections, base_face=base_face, output_file_prefix=output_file_prefix)

    # record all to a json file
    # serialize selected_net["faces"] to list
    serialized_faces = {k: v.tolist() for k, v in faces.items()}
    
    output_json = {
        "initial_net": {
            "name": selected_net["name"],
            "faces": serialized_faces,
            "face_connections": face_connections,
            "valid": net_is_valid
        },
        "base_face": base_face,
        "folding_steps": folding_steps,
        "folding_step_descs": folding_step_descs,
        "images_per_step": [f"{output_file_prefix}_step{i+1}.png" for i in range(len(folding_steps) + 1) if File.isfile(f"{output_file_prefix}_step{i+1}.png")],
        "id": output_file_prefix.split("/")[-1]
    }
    with File.open(output_json_path, "w") as f:
        json.dump(output_json, f)
    return output_json_path

def main(output_dir="output_dir", debug=False, overwrite=False, num_randomness=1):
    import os
    output_dir = os.path.join(output_dir, "folding_nets")
    all_json_paths = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    from cube_nets import generate_cube_nets, gather_invalid_cubes
    # Generate a list of nets.
    valid_nets = generate_cube_nets()

    invalid_nets = gather_invalid_cubes()
    print(len(valid_nets), "valid nets and", len(invalid_nets), "invalid nets found.")

    nets = valid_nets + invalid_nets
    if debug:
        nets = valid_nets
    for randomness in range(num_randomness):
        random.seed(randomness)
        for idx, net in tqdm(enumerate(nets), total=len(nets)):
            # for base_face in net["faces"]:
            available_faces = list(net["faces"].keys())

            # shuffle the names of the faces and the connections
            if randomness != 0:
                new_key = list(net["faces"].keys())
                random.shuffle(new_key)

                old2new = {key: str(i+1) for i, key in enumerate(new_key)}
                new_net = {
                    "name": net["name"],
                    "faces": {old2new[key]: net["faces"][key] for key in net["faces"]},
                    "face_connections": {old2new[key]: [old2new[j] for j in net["face_connections"][key]] for key in net["face_connections"]},
                    "valid": net["valid"]
                }
                net = new_net
                # overwrite = False

                # print("Randomized net:", net)
            # else:
            #     overwrite = False
            if debug:
                available_faces = ["3"]
            for base_face in available_faces:
                if randomness == 0:
                    output_file_prefix = f"{output_dir}/net{idx}"
                else:
                    output_file_prefix = f"{output_dir}/random{randomness}_net{idx}"
                output_json_path = gen_example_per_net(net, base_face, output_file_prefix, overwrite=overwrite)
                all_json_paths.append(output_json_path)
    return all_json_paths
    


def create_text_instruction_hf_dataset(output_dir, dataset_name, overwrite=False, num_randomness=4):
    all_json_paths = main(output_dir=output_dir, overwrite=overwrite, num_randomness=num_randomness)
    # print(all_json_paths)
    data_dict = {
        'qid': [], 'question': [], 
        'images': [],
        'answer': [], 'question_info': [], 'type': [],
        'choices': []}
    random.seed(0)
    random.shuffle(all_json_paths)
    from PIL import Image
    import datasets
    from datasets import Dataset, Features
    import json
    for json_idx, json_path in enumerate(all_json_paths):
        with File.open(json_path, 'r') as f:
            question_info_path = json.load(f)
        base_face = question_info_path['base_face']
        num_valid_vis = len(question_info_path['images_per_step']) - 1
        for type in ['q+steps+partial_vis', 'q+steps+all_intermediate_vis', 'q+steps+all_vis', 'q+steps+all_intermediate_last_vis']:
        # for type in ['q_only', 'q+steps']:
            selected_base_face = str(random.choice(list(range(1, 7))))
            # if type == 'q_only' and base_face != '3':
            #     continue
            if type == 'q+steps+partial_vis' and num_valid_vis == 1:
                continue
            if type == 'q+steps+all_for_valid_vis' and num_valid_vis < 5:
                continue

            if base_face != selected_base_face:
                continue

            data_dict['qid'].append(f'{json_idx}_{type}')
            answer_choices = ['yes', 'no']
            random.shuffle(answer_choices)

            is_valid = question_info_path['initial_net']['valid']

            answer = 'yes' if is_valid else 'no'
            answer_idx = answer_choices.index(answer)
            data_dict['answer'].append(chr(ord('A') + answer_idx))

            question_text = "Check out a net with 6 faces below: \n<image_0>"

            answer_text = f"Choose from the following answer choices:\nA. {answer_choices[0]}\nB. {answer_choices[1]}"
            # answer_text = f"Choose from the following answer choices:\nA. Yes\nB. No"

            images = []

            images.append(Image.open(question_info_path['images_per_step'][0]).convert('RGB'))
            if type == 'q_only':
                full_question = question_text+ "Can the net be folded to form a cube, yes or no?"
            elif type == 'q+steps':
                full_question = question_text + f"Below are the steps to fold the net with face {base_face} as the base:\n" 
                for i in range(len(question_info_path['folding_step_descs'])):
                    curr_step = question_info_path['folding_step_descs'][i]
                    full_question += f"Step {i+1}: {curr_step}\n"
                full_question = full_question+"Based on the above steps, can the net be folded to form a cube, yes or no?"
                
            elif type == 'q+steps+all_intermediate_vis':
                full_question = question_text + f"Here are the steps to fold the net with face {base_face} as the base:\n" 
                if num_valid_vis == 5:
                    vis_images = question_info_path['images_per_step'][:-1]
                else:
                    vis_images = question_info_path['images_per_step']
                for i in range(1, len(vis_images)):
                    images.append(Image.open(vis_images[i]).convert('RGB'))
                    curr_step = question_info_path['folding_step_descs'][i-1]
                    full_question += f"Step {i}: {curr_step}\n<image_{i}>\n"
                
                for i in range(len(vis_images)-1, len(question_info_path['folding_step_descs'])):
                    curr_step = question_info_path['folding_step_descs'][i]
                    full_question += f"Step {i+1}: {curr_step}\n"
                full_question = full_question+"Based on the above steps, can the net be folded to form a cube, yes or no?"
                # +answer_text
                
            elif type == 'q+steps+all_intermediate_last_vis':
                full_question = question_text + f"Here are the steps to fold the net with face {base_face} as the base:\n" 
                if num_valid_vis == 5:
                    vis_images = question_info_path['images_per_step'][:-1]
                else:
                    vis_images = question_info_path['images_per_step']
                for i in range(1, len(vis_images)):
                    curr_step = question_info_path['folding_step_descs'][i-1]
                    if i == len(vis_images) - 1:
                        images.append(Image.open(vis_images[i]).convert('RGB'))
                        full_question += f"Step {i}: {curr_step}\n<image_1>\n"
                    else:
                        full_question += f"Step {i}: {curr_step}\n"
                
                for i in range(len(vis_images)-1, len(question_info_path['folding_step_descs'])):
                    curr_step = question_info_path['folding_step_descs'][i]
                    full_question += f"Step {i+1}: {curr_step}\n"
                full_question = full_question+"Based on the above steps, can the net be folded to form a cube, yes or no?"
                # +answer_text
                
            elif type == 'q+steps+all_vis':
                full_question = question_text + f"Here are the steps to fold the net with face {base_face} as the base:\n" 
                for i in range(1, len(question_info_path['images_per_step'])):
                    images.append(Image.open(question_info_path['images_per_step'][i]).convert('RGB'))
                    curr_step = question_info_path['folding_step_descs'][i-1]
                    full_question += f"Step {i}: {curr_step}\n<image_{i}>\n"
                
                for i in range(len(question_info_path['images_per_step'])-1, len(question_info_path['folding_step_descs'])):
                    curr_step = question_info_path['folding_step_descs'][i]
                    full_question += f"Step {i+1}: {curr_step}\n"
                full_question = full_question+"Based on the above steps, can the net be folded to form a cube, yes or no?"
                # +answer_text
            
            elif type == 'q+steps+partial_vis':
                random_num_vis = random.randint(1, num_valid_vis-1)
                full_question = question_text + f"Here are the steps to fold the net with face {base_face} as the base:\n" 
                for i in range(1, random_num_vis+1):
                    images.append(Image.open(question_info_path['images_per_step'][i]).convert('RGB'))
                    curr_step = question_info_path['folding_step_descs'][i-1]
                    full_question += f"Step {i}: {curr_step}\n<image_{i}>\n"
                
                for i in range(random_num_vis, len(question_info_path['folding_step_descs'])):
                    curr_step = question_info_path['folding_step_descs'][i]
                    full_question += f"Step {i+1}: {curr_step}\n"
                full_question = full_question+"Based on the above steps, can the net be folded to form a cube, yes or no?"
            data_dict['images'].append(images)
            data_dict['question'].append(full_question)
            data_dict['question_info'].append(json.dumps(question_info_path))
            data_dict['type'].append(type)
            data_dict['choices'].append(answer_choices)
    # shuffle data_dict as well
    num_q = len(data_dict['qid'])
    shuffled_indices = list(range(num_q))
    random.shuffle(shuffled_indices)
    for key in data_dict:
        print(key)
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
    

def get_face_color_name(face_key):
    """
    Get the color name of a face based on its key using the same logic as the rendering code.
    
    Args:
        face_key: The key of the face (string like "1", "2", "3", etc.)
        
    Returns:
        String name of the color
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    
    # Use the same formula as in the rendering code
    color_value = plt.cm.Set3(int(face_key) / 12)
    
    # Convert RGB to color name
    rgb = tuple(color_value[:3])  # Get RGB values (ignore alpha)
    
    # Define color ranges for Set3 colormap
    color_names = {
        0: "light blue",
        1: "yellow",
        2: "light purple",
        3: "light red",
        4: "light blue",
        5: "light orange",
        6: "light green",
        7: "tan",
        8: "light teal",
        9: "light brown",
        10: "light purple",
        11: "cream"
    }
    
    return color_names[int(face_key) % 12]

import os
from PIL import Image
import json
import datasets
from datasets import Dataset, Features
def create_2d_perception_hf_dataset(output_dir, dataset_name, source_dataset_path=None, num_samples=100, token_path='aux_data/credentials/hf_token.txt', num_randomness=4):
    """
    Create a HuggingFace dataset for cube net perception tasks focusing on:
    1. Color of specific faces
    2. Connectivity between faces (yes/no questions)
    
    Args:
        output_dir: Directory containing rendered images
        dataset_name: Name for the new HuggingFace dataset
        source_dataset_path: Path to the directory containing source data (optional, will use output_dir if None)
        num_samples: Number of examples to include in the dataset
        token_path: Path to HuggingFace token file
    """
    all_json_paths = main(output_dir=output_dir, overwrite=False, num_randomness=num_randomness)
    
    # # Get all JSON paths
    # all_json_paths = []
    # for root, dirs, files in File.walk(source_dataset_path):
    #     for file in files:
    #         if file.endswith(".json") and file != "metadata.json":
    #             all_json_paths.append(os.path.join(root, file))
    
    # Shuffle and sample JSON paths
    random.seed(42)
    random.shuffle(all_json_paths)
    sampled_json_paths = all_json_paths[:min(len(all_json_paths), num_samples*2)]  # Sample extra for filtering
    
    # Define the dictionary to hold all examples
    data_dict = {
        'qid': [], 
        'question': [], 
        'images': [],
        'answer': [], 
        'type': [],
        'choices': [],
        'question_info': []
    }
    
    perception_question_types = ["color", "connectivity"]
    
    count = 0
    for json_path in tqdm(sampled_json_paths):
        if count >= num_samples:
            break
            
        try:
            with File.open(json_path, 'r') as f:
                question_info = json.load(f)
                
            # Skip if no images are available
            if 'images_per_step' not in question_info or not question_info['images_per_step']:
                continue
                
            # Load the first image (original net)
            first_image_path = question_info['images_per_step'][0]
            if not os.path.exists(first_image_path):
                continue
                
            image = Image.open(first_image_path).convert('RGB')
            
            # Select a random question type
            question_type = random.choice(perception_question_types)
            
            # Get face information and connections
            faces = question_info['initial_net']['faces']
            face_connections = question_info['initial_net']['face_connections']
            
            # List of face keys
            face_keys = list(faces.keys())
            
            # In create_perception_hf_dataset function:
            question_type = "color"
            # Question about the color of a specific face
            for target_face in random.sample(face_keys, 2):
                face_color = get_face_color_name(target_face)
                
                question = f"What color is face {target_face} in the cube net shown in the image?\n<image_0>\n"
                
                # Create choices with the correct answer and distractors
                all_colors = [get_face_color_name(str(i+1)) for i in range(min(12, len(faces)))]
                incorrect_colors = [c for c in all_colors if c != face_color]
                random.shuffle(incorrect_colors)
                
                choices = [face_color] + incorrect_colors[:3]
                random.shuffle(choices)
                
                # Find the index of the correct answer
                correct_idx = choices.index(face_color)
                answer_letter = chr(ord('A') + correct_idx)
            
                # Format the choices as A, B, C, D options
                formatted_choices = [f"{chr(ord('A') + i)}. {choices[i]}" for i in range(len(choices))]
                question += "\n".join(formatted_choices)

                # Add to dataset
                data_dict['qid'].append(f"2d_perception_{question_type}_{count}_{target_face}")
                data_dict['question'].append(question)
                data_dict['images'].append([image])
                data_dict['answer'].append(answer_letter)
                data_dict['type'].append(question_type)
                data_dict['choices'].append(formatted_choices)
                data_dict['question_info'].append(json.dumps(question_info))
                
            question_type = "connectivity"

            import itertools
            face_combinations = list(itertools.combinations(face_keys, 2))
            random.shuffle(face_combinations)
            face_combinations = face_combinations[:2] # Sample 4 combinations
            # face1, face2 = random.sample(face_keys, 2)
            for face1, face2 in face_combinations:
                
                # Check if they're connected
                is_connected = face2 in face_connections.get(face1, [])
                
                question = f"In the cube net shown in the image, is face {face1} directly connected to face {face2}, yes or no?\n<image_0>"
                
                # Yes/No choices
                choices = ["Yes", "No"]

                random.shuffle(choices)
                
                # Set the correct answer
                if is_connected:
                    answer_letter = chr(ord('A') + choices.index("Yes"))  # Yes
                else:
                    answer_letter = chr(ord('A') + choices.index("No"))

                # Add to dataset
                data_dict['qid'].append(f"2d_perception_{question_type}_{count}_{face1}_{face2}")
                data_dict['question'].append(question)
                data_dict['images'].append([image])
                data_dict['answer'].append(answer_letter)
                data_dict['type'].append(question_type)
                data_dict['choices'].append(choices)
                data_dict['question_info'].append(json.dumps(question_info))
            count += 1
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue
    
    print(f"Created dataset with {len(data_dict['qid'])} examples")
    if len(data_dict['qid']) == 0:
        print("No valid examples found. Dataset not created.")
        return
        
    # Create HuggingFace dataset
    feature_dict = {
        'qid': datasets.Value('string'),
        'question': datasets.Value('string'),
        'images': datasets.Sequence(datasets.Image()),
        'answer': datasets.Value('string'),
        'type': datasets.Value('string'),
        'choices': datasets.Sequence(datasets.Value('string')),
        'question_info': datasets.Value('string')
    }
    
    features = Features(feature_dict)
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    
    # Push to hub if token exists
    if token_path and os.path.exists(token_path):
        with open(token_path, 'r') as f:
            hf_token = f.read().strip()
        print(f"Pushing dataset to HuggingFace Hub as VisSim/{dataset_name}")
        hf_dataset.push_to_hub(
            f"VisSim/{dataset_name}",
            create_pr=False,
            token=hf_token)
        print(f"Dataset successfully pushed to HuggingFace Hub as VisSim/{dataset_name}")
    return


def create_3d_perception_hf_dataset(output_dir, dataset_name, num_samples=200, token_path='aux_data/credentials/hf_token.txt', num_randomness=4):
    """
    Create a HuggingFace dataset for 3D perception tasks that asks whether a specific face
    has been folded at randomly chosen intermediate folding steps.
    
    Args:
        output_dir: Directory containing rendered images
        dataset_name: Name for the new HuggingFace dataset
        source_dataset_path: Path to the directory containing source data (optional, will use output_dir if None)
        num_samples: Number of examples to include in the dataset
        token_path: Path to HuggingFace token file
        num_randomness: Number of random net configurations to generate
    """
    # Generate or use existing JSON paths
    all_json_paths = main(output_dir=output_dir, overwrite=False, num_randomness=num_randomness)
    
    # Shuffle and sample JSON paths
    random.seed(42)
    random.shuffle(all_json_paths)
    sampled_json_paths = all_json_paths
    # [:min(len(all_json_paths), num_samples*2)]  # Sample extra for filtering
    
    # Define the dictionary to hold all examples
    data_dict = {
        'qid': [], 
        'question': [], 
        'images': [],
        'answer': [], 
        'choices': [],
        'question_info': [],
        'type': []
    }
    
    count = 0
    for json_path in tqdm(sampled_json_paths):
        if count >= num_samples:
            break
            
        try:
            with File.open(json_path, 'r') as f:
                question_info = json.load(f)
                
            # Skip if not enough folding steps
            if 'folding_steps' not in question_info or len(question_info['folding_steps']) < 2:
                continue
                
            # Skip if not enough images
            if 'images_per_step' not in question_info or len(question_info['images_per_step']) < 3:
                continue
                
            # Load a random intermediate image (not the first or last one)
            num_steps = len(question_info['images_per_step'])
            if num_steps <= 2:  # Need at least 3 steps (initial, intermediate, final)
                continue
                
            # Choose a random intermediate step
            step_idx = random.randint(1, num_steps - 1)  # Skip first step (0), include last step
            step_image_path = question_info['images_per_step'][step_idx]
            
            if not os.path.exists(step_image_path):
                continue
                
            step_image = Image.open(step_image_path).convert('RGB')
            
            # Get all faces that are in the net
            faces = list(question_info['initial_net']['faces'].keys())
            base_face = question_info['base_face']
            # Get all faces that have been folded up to the current step
            folded_faces = set([question_info['base_face']])  # Base face is always folded (it's the starting point)
            
            # Add faces that have been folded in previous steps
            for i in range(min(step_idx, len(question_info['folding_steps']))):
                step_data = question_info['folding_steps'][i]
                if isinstance(step_data, list):
                    folded_face = step_data[0]  # First element is the face being folded
                elif isinstance(step_data, tuple):
                    folded_face = step_data[0]  # First element is the face being folded
                else:
                    continue
                folded_faces.add(folded_face)
            
            # Randomly select a face to ask about
            target_face = random.choice(faces)
            
            # Determine if this face has been folded
            is_folded = target_face in folded_faces
            
            # Create the question
            current_step_num = step_idx + 1  # Convert to 1-indexed step number
            question = f"In the cube net folding process shown below, has face {target_face} been folded towards {base_face} yet, yes or no?\n<image_0>"
            
            # Create choices
            choices = ["Yes", "No"]
            random.shuffle(choices)
            
            # Set the correct answer
            if is_folded:
                answer_letter = chr(ord('A') + choices.index("Yes"))
            else:
                answer_letter = chr(ord('A') + choices.index("No"))
            
            # Format the choices
            formatted_choices = [f"{chr(ord('A') + i)}. {choices[i]}" for i in range(len(choices))]
            # question += "\n" + "\n".join(formatted_choices)
            
            # Add extra information to question_info for this dataset
            step_info = {
                "step_idx": step_idx,
                "step_num": current_step_num,
                "target_face": target_face,
                "folded_faces": list(folded_faces),
                "is_folded": is_folded,
            }
            
            # Store original question_info plus our step-specific info
            combined_info = {**question_info, "step_info": step_info}
                
            # Add to dataset
            data_dict['qid'].append(f"3d_perception_fold_{count}_{target_face}_step{current_step_num}")
            data_dict['question'].append(question)
            data_dict['images'].append([step_image])
            data_dict['answer'].append(answer_letter)
            data_dict['choices'].append(choices)
            data_dict['type'].append("3d_perception_fold")
            data_dict['question_info'].append(json.dumps(combined_info))
            
            count += 1
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue
    
    print(f"Created dataset with {len(data_dict['qid'])} examples")
    if len(data_dict['qid']) == 0:
        print("No valid examples found. Dataset not created.")
        return
        
    # Create HuggingFace dataset
    feature_dict = {
        'qid': datasets.Value('string'),
        'question': datasets.Value('string'),
        'images': datasets.Sequence(datasets.Image()),
        'answer': datasets.Value('string'),
        'choices': datasets.Sequence(datasets.Value('string')),
        'question_info': datasets.Value('string'),
        'type': datasets.Value('string')
    }
    
    features = Features(feature_dict)
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    
    # Push to hub if token exists
    if token_path and os.path.exists(token_path):
        with open(token_path, 'r') as f:
            hf_token = f.read().strip()
        print(f"Pushing dataset to HuggingFace Hub as VisSim/{dataset_name}")
        hf_dataset.push_to_hub(
            f"VisSim/{dataset_name}",
            create_pr=False,
            token=hf_token)
        print(f"Dataset successfully pushed to HuggingFace Hub as VisSim/{dataset_name}")
    
    return


if __name__ == "__main__":
    from fire import Fire
    Fire(create_3d_perception_hf_dataset)
