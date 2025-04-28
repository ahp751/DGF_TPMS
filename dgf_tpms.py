"""
===================================================
DGF TPMS Solver
===================================================

This script generates a TPMS (Triply Periodic Minimal Surface) geometry using an SDF-based approach,
extracts its surface using marching cubes, and computes slice-by-slice convective heat transfer
parameters using a Discrete Green's Function (DGF) method.

Features:
- Voxelized domain generation
- Surface extraction and 3D visualization
- Slice-by-slice area and volume computation
- Hydraulic diameter calculation
- Building and exporting the convection DGF matrix
- Local convective heat transfer coefficient (h) and heat transfer rate (Q) evaluation

Usage:
- Run the script and follow interactive prompts for setup.
- Visualization and CSV exports are optional based on user input.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

# Importing scikit-image for marching cubes algorithm
try:
    from skimage import measure
except ImportError:
    print("scikit-image not installed. Please install via 'pip install scikit-image'")
    sys.exit(1)


#===========================================#
#           USER INPUT FUNCTION            #
#===========================================#

def get_user_input(prompt, dtype=str, default=None, validator=lambda x: True):
    """
    Utility to safely get user input with a default and validator.
    """
    while True:
        raw_inp = input(f"{prompt} [{default}]: ") or str(default)
        try:
            val = dtype(raw_inp)
            if not validator(val):
                raise ValueError
            return val
        except (ValueError, TypeError):
            print(f"Invalid input. Please enter a valid {dtype.__name__}.")


#===========================================#
#       BOUNDING BOX & PARAMS SETUP        #
#===========================================#

def adjust_bounding_box(bounds, scale_factor):
    """
    Scale the bounding box about its center by the provided scale_factor.
    """
    center = (bounds[0] + bounds[1]) / 2.0
    half_extent = (bounds[1] - bounds[0]) / 2.0
    new_half_extent = half_extent * scale_factor
    return np.array([center - new_half_extent, center + new_half_extent])


def initialize_parameters():
    """
    Interactively gather bounding-box data, number of slices, pitch,
    fluid properties and convection setup. Returns a params dict.
    """
    print("\n--- SDF-BASED TPMS PARAMETERS ---")
    x_min = float(get_user_input("Enter x_min", float, default="-1.0"))
    x_max = float(get_user_input("Enter x_max", float, default="1.0"))
    y_min = float(get_user_input("Enter y_min", float, default="-1.0"))
    y_max = float(get_user_input("Enter y_max", float, default="1.0"))
    z_min = float(get_user_input("Enter z_min", float, default="0.0"))
    z_max = float(get_user_input("Enter z_max", float, default="1.0"))
    bounds = np.array([[x_min, y_min, z_min],
                       [x_max, y_max, z_max]])

    if get_user_input("Scale bounding box? (y/n)", str, default="n",
                      validator=lambda x: x.lower() in ['y', 'n']).lower() == 'y':
        sf = float(get_user_input("Enter scale factor", float, default="2.0"))
        bounds = adjust_bounding_box(bounds, sf)
        print("New bounding box:", bounds)

    n_slices = int(get_user_input("Enter number of slices in Z", int, default="10",
                                  validator=lambda x: x > 0))
    height = float(get_user_input("Enter total geometry height (m)", float, default="1.0",
                                  validator=lambda x: x > 0))
    pitch = float(get_user_input("Enter voxel pitch (m)", float, default="0.01",
                                 validator=lambda x: x > 0))
    unit_cell = float(get_user_input("Enter unit cell size (m)", float, default="0.01",
                                     validator=lambda x: x > 0))

    print("\n--- Convection Model Setup ---")
    approach_h = get_user_input("Approach ('manual' or 'tpms-corr')", str,
                                default="tpms-corr",
                                validator=lambda x: x.lower() in ["manual", "tpms-corr"])
    velocity = float(get_user_input("Enter fluid velocity (m/s)", float, default="1.0",
                                    validator=lambda x: x > 0))

    if approach_h.lower() == "manual":
        h_user = float(get_user_input("Enter convective coeff h (W/m^2.K)", float,
                                      default="100.0", validator=lambda x: x > 0))
        m_dot = float(get_user_input("Enter mass flow rate (kg/s)", float,
                                     default="0.1", validator=lambda x: x > 0))
        c_p = float(get_user_input("Enter specific heat c_p (J/kg.K)", float,
                                   default="1000.0", validator=lambda x: x > 0))
        T_inf = float(get_user_input("Enter inlet T∞ (K)", float, default="300.0"))
        rho, mu, lam = 1.2, 1.8e-5, 0.025
    else:
        rho = float(get_user_input("Fluid density ρ (kg/m^3)", float, default="1.2",
                                   validator=lambda x: x > 0))
        mu = float(get_user_input("Viscosity μ (Pa.s)", float, default="1.8e-5",
                                  validator=lambda x: x > 0))
        lam = float(get_user_input("Conductivity λ (W/m.K)", float, default="0.025",
                                   validator=lambda x: x > 0))
        c_p = float(get_user_input("Specific heat c_p (J/kg.K)", float, default="1005.0",
                                   validator=lambda x: x > 0))
        m_dot = float(get_user_input("Mass flow rate (kg/s)", float, default="0.1",
                                     validator=lambda x: x > 0))
        T_inf = float(get_user_input("Inlet T∞ (K)", float, default="300.0"))
        h_user = None

    Pr = float(get_user_input("Enter Prandtl number", float,
                              default=str(c_p * mu / lam),
                              validator=lambda x: x > 0))

    scale_val = 2 * np.pi / unit_cell
    surface_area_scale = float(get_user_input("Enter surface area scale factor", float,
                                              default="2.0", validator=lambda x: x > 0))

    return {
        'bounds': bounds,
        'n_slices': n_slices,
        'height': height,
        'pitch': pitch,
        'scale': scale_val,
        'thickness': 0.0,
        'slice_height': height / n_slices,
        'approach_h': approach_h.lower(),
        'h_user': h_user,
        'm_dot': m_dot,
        'c_p': c_p,
        'rho': rho,
        'mu': mu,
        'lambda': lam,
        'T_infinity': T_inf,
        'velocity': velocity,
        'Pr': Pr,
        'surface_area_scale': surface_area_scale
    }


#===========================================#
#        VISUALIZATION: BOUNDING BOX       #
#===========================================#

def visualize_bounding_box(bounds):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin, ymin, zmin = bounds[0]
    xmax, ymax, zmax = bounds[1]
    corners = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax]
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for e in edges:
        ax.plot(*zip(corners[e[0]], corners[e[1]]), 'k-')
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title("Bounding Box"); plt.show()


#===========================================#
#      SDF & MARCHING CUBES FUNCTIONS      #
#===========================================#

def base_gyroid_sdf(x, y, z, scale, C=0):
    return np.cos(scale * x) + np.cos(scale * y) + np.cos(scale * z) - C

def thick_sdf(x, y, z, scale, C=0, thickness=0.0):
    val = base_gyroid_sdf(x, y, z, scale, C)
    return abs(val) - thickness if thickness > 1e-12 else val

def build_thick_gyroid_sdf_grid(params, iso_C=0):
    b = params['bounds']
    x0, y0, z0 = b[0]; x1, y1, z1 = b[1]
    pitch = params['pitch']; scale = params['scale']; thick = params['thickness']

    Nx = int(np.ceil((x1 - x0) / pitch))
    Ny = int(np.ceil((y1 - y0) / pitch))
    Nz = int(np.ceil((z1 - z0) / pitch))

    sdf = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    for iz in range(Nz):
        cz = z0 + (iz + 0.5) * pitch
        for iy in range(Ny):
            cy = y0 + (iy + 0.5) * pitch
            for ix in range(Nx):
                cx = x0 + (ix + 0.5) * pitch
                sdf[iz, iy, ix] = thick_sdf(cx, cy, cz, scale, iso_C, thick)
    return sdf

def marching_cubes_thick_gyroid(params):
    sdf = build_thick_gyroid_sdf_grid(params, iso_C=0)
    vmin, vmax = sdf.min(), sdf.max()
    iso = 0.0 if (vmin < 0 < vmax) else 0.5 * (vmin + vmax)
    try:
        verts, faces, normals, values = measure.marching_cubes(
            sdf, level=iso, spacing=(params['pitch'],) * 3
        )
    except Exception as e:
        print("Error in marching_cubes:", e)
        sys.exit(1)

    if len(verts) == 0 or len(faces) == 0:
        print("No surface extracted. Check bounding box/pitch.")
        sys.exit(1)

    print(f"MarchingCubes => {len(verts)} verts, {len(faces)} faces.")
    return verts, faces, sdf


#===========================================#
#     PER-SLICE AREA & VOLUME CALCS        #
#===========================================#

def accumulate_slice_area_from_surface(verts, faces, params, area_tol=1e-8):
    n = params['n_slices']
    # Use first coordinate as 'z'
    z_vals = verts[:, 0]
    zmin, zmax = z_vals.min(), z_vals.max()
    edges = np.linspace(zmin, zmax, n + 1)

    slice_area = np.zeros(n)
    def tri_area(p0, p1, p2):
        return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))

    for f in faces:
        p0, p1, p2 = verts[f]
        A = tri_area(p0, p1, p2)
        if A < area_tol:
            continue
        centroid = (p0 + p1 + p2) / 3.0
        cz = centroid[0]
        idx = np.searchsorted(edges, cz) - 1
        if 0 <= idx < n:
            slice_area[idx] += A

    slice_area *= params['surface_area_scale']
    for i, a in enumerate(slice_area):
        print(f"Slice {i}: area={a:.6f} m^2 (scaled)")
    return slice_area

def accumulate_slice_volume_from_sdf(sdf, params):
    n = params['n_slices']
    z0 = params['bounds'][0][2]; z1 = params['bounds'][1][2]
    edges = np.linspace(z0, z1, n + 1)
    Nz, Ny, Nx = sdf.shape

    slice_vol = np.zeros(n)
    for iz in range(Nz):
        cz = z0 + (iz + 0.5) * params['pitch']
        idx = np.searchsorted(edges, cz) - 1
        if 0 <= idx < n:
            count = np.count_nonzero(sdf[iz] > 0)
            slice_vol[idx] += count * (params['pitch'] ** 3)

    for i, v in enumerate(slice_vol):
        print(f"Slice {i}: volume={v:.6e} m^3")
    return slice_vol


#===========================================#
#     BUILD THE 1D DGF FOR CONVECTION       #
#===========================================#

def build_convection_dgf_new(slice_area, slice_vol, params):
    n = params['n_slices']
    cp = params['c_p']; md = params['m_dot']; Tinf = params['T_infinity']
    vel = params['velocity']; rho = params['rho']; mu = params['mu']
    k = params['lambda']; Pr = params['Pr']

    A_const = 0.0964
    C1 = A_const * k * (Pr ** 0.4) * ((rho * vel) ** 0.7136) / (mu ** 0.7136)
    print(f"\nUsing new eqn's C1={C1:.3e}")

    Dh = np.zeros(n)
    for i in range(n):
        Dh[i] = 4.0 * slice_vol[i] / slice_area[i] if slice_area[i] > 0 else 0.0

    G = np.zeros((n, n))
    for j in range(n):
        Tnode = np.zeros(n); Tnode[j] = 1.0
        Tfl = Tinf
        for i in range(n):
            Tw = Tinf + Tnode[i]
            dT = Tw - Tfl
            q = C1 * (Dh[i] ** -0.2864) * dT * slice_area[i]
            G[i, j] = q
            Tfl += q / (md * cp)
    return G


#===========================================#
#        EXPORT TO CSV UTILITIES            #
#===========================================#

def export_matrix_to_csv(matrix, filename="matrix.csv", label="Matrix"):
    if not filename.endswith(".csv"):
        filename += ".csv"
    n = matrix.shape[0]
    try:
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([label] + [f"j={j}" for j in range(n)])
            for i in range(n):
                w.writerow([f"i={i}"] + list(matrix[i, :]))
        print(f"{label} saved to {filename}")
    except Exception as e:
        print(f"Error saving {label} => {e}")

def export_final_data_to_csv(params, Tnode, Tfl, deltaT, q, h, filename="final_data.csv"):
    if not filename.endswith(".csv"):
        filename += ".csv"
    n = params['n_slices']
    z0 = params['bounds'][0][2]; z1 = params['bounds'][1][2]
    edges = np.linspace(z0, z1, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    try:
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Slice","Center(m)","T_node","T_fluid","ΔT","q(W)","h"])
            for i in range(n):
                w.writerow([i, centers[i], Tnode[i], Tfl[i], deltaT[i], q[i], h[i]])
        print(f"Final data saved to {filename}")
    except Exception as e:
        print(f"Error exporting final data => {e}")


#===========================================#
#         VISUALIZATION FUNCTIONS           #
#===========================================#

def visualize_sdf_slice_2D(params, z_level):
    bounds = params['bounds']; pitch = params['pitch']; scale = params['scale']
    x0, y0, _ = bounds[0]; x1, y1, _ = bounds[1]
    Nx = int(np.ceil((x1 - x0)/pitch)); Ny = int(np.ceil((y1 - y0)/pitch))
    X = np.linspace(x0, x1, Nx); Y = np.linspace(y0, y1, Ny)
    XX, YY = np.meshgrid(X, Y); ZZ = np.full_like(XX, z_level)
    sdf_base = base_gyroid_sdf(XX, YY, ZZ, scale, 0)
    plt.contourf(XX, YY, sdf_base, 50, cmap='RdBu_r')
    plt.colorbar(label='SDF'); plt.title(f"SDF slice z={z_level}")
    plt.xlabel("X"); plt.ylabel("Y"); plt.show()

def visualize_isosurface(verts, faces):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.3, facecolor='cyan', edgecolor='k')
    ax.add_collection3d(mesh)
    ax.set_title("Isosurface"); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.show()

def plot_slice_area(slice_area):
    plt.bar(range(len(slice_area)), slice_area)
    plt.xlabel("Slice"); plt.ylabel("Area"); plt.title("Per-slice Area"); plt.show()

def plot_local_h_vs_distance(params, h):
    n = params['n_slices']
    z0 = params['bounds'][0][2]; z1 = params['bounds'][1][2]
    centers = 0.5*(np.linspace(z0, z1, n+1)[:-1] + np.linspace(z0, z1, n+1)[1:])
    plt.plot(centers, h, marker='o')
    plt.xlabel("Z (m)"); plt.ylabel("h (W/m²·K)"); plt.title("Local h vs Distance"); plt.grid(True); plt.show()

def plot_Q_vs_distance(params, q):
    n = params['n_slices']
    z0 = params['bounds'][0][2]; z1 = params['bounds'][1][2]
    centers = 0.5*(np.linspace(z0, z1, n+1)[:-1] + np.linspace(z0, z1, n+1)[1:])
    plt.plot(centers, q, marker='s')
    plt.xlabel("Z (m)"); plt.ylabel("Q (W)"); plt.title("Q vs Distance"); plt.grid(True); plt.show()


#===========================================#
#               MAIN WORKFLOW               #
#===========================================#

def main():
    params = initialize_parameters()

    if get_user_input("Visualize bounding box? (y/n)", str, default="n",
                      validator=lambda x: x.lower() in ['y','n']).lower() == 'y':
        visualize_bounding_box(params['bounds'])

    verts, faces, sdf = marching_cubes_thick_gyroid(params)

    slice_area = accumulate_slice_area_from_surface(verts, faces, params)
    slice_vol = accumulate_slice_volume_from_sdf(sdf, params)

    G = build_convection_dgf_new(slice_area, slice_vol, params)
    print("G matrix shape:", G.shape)

    if get_user_input("Export G to CSV? (y/n)", str, default="n",
                      validator=lambda x: x.lower() in ['y','n']).lower() == 'y':
        fname = get_user_input("Filename", str, default="dgf_matrix.csv")
        export_matrix_to_csv(G, fname, "DGF")

    if get_user_input("Compute local h & Q? (y/n)", str, default="n",
                      validator=lambda x: x.lower() in ['y','n']).lower() == 'y':
        # Base/solid profile
        T_base = float(get_user_input("Base T (K)", float, default="350.0"))
        base_idx = get_user_input("Base slices (comma-separated)", str, default="0")
        base_idx = [int(i) for i in base_idx.split(",") if i.strip().isdigit()]

        T_solid = float(get_user_input("Solid T (K)", float, default="500.0"))
        sol_idx = get_user_input("Solid slices (comma-separated)", str, default="7,8,9")
        sol_idx = [int(i) for i in sol_idx.split(",") if i.strip().isdigit()]

        n = params['n_slices']
        Tnode = np.zeros(n)
        bmax = max(base_idx); smin = min(sol_idx)
        for i in range(n):
            if i <= bmax:
                Tnode[i] = T_base
            elif i >= smin:
                Tnode[i] = T_solid
            else:
                Tnode[i] = T_base + (T_solid - T_base)*(i - bmax)/(smin - bmax)

        Tfl = np.zeros(n+1); Tfl[0] = params['T_infinity']
        deltaT = np.zeros(n)
        for i in range(n):
            deltaT[i] = Tnode[i] - Tfl[i]
            Tfl[i+1] = Tfl[i] + np.dot(G[i], deltaT)/(params['m_dot']*params['c_p'])

        q = G.dot(deltaT)

        if get_user_input("Use single-surface area? (y/n)", str, default="n",
                          validator=lambda x: x.lower() in ['y','n']).lower() == 'y':
            area_used = slice_area/params['surface_area_scale']
        else:
            area_used = slice_area

        local_h = np.zeros(n)
        for i in range(n):
            local_h[i] = q[i]/(area_used[i]*deltaT[i]) if area_used[i]*abs(deltaT[i])>1e-12 else 0.0
            print(f"Slice {i}: q={q[i]:.2f}, h={local_h[i]:.2f}")

        if get_user_input("Export final data? (y/n)", str, default="n",
                          validator=lambda x: x.lower() in ['y','n']).lower() == 'y':
            fname = get_user_input("Filename", str, default="final_data.csv")
            export_final_data_to_csv(params, Tnode, Tfl, deltaT, q, local_h, filename=fname)

        plot_local_h_vs_distance(params, local_h)
        plot_Q_vs_distance(params, q)

    print("All done.")

if __name__ == "__main__":
    main()
