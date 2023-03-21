import bpy
import blender_plots as bplt
import blender_plots.blender_utils as bu
import numpy as np
from tqdm import tqdm

from geometry_utils import geometry

ydown2zup = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

def srgb_to_linearrgb(c):
    if   c < 0:       return 0
    elif c < 0.04045: return c/12.92
    else:             return ((c+0.055)/1.055)**2.4


def hex_to_rgb(h,alpha=1):
    # source: https://blender.stackexchange.com/questions/153094/blender-2-8-python-how-to-set-material-color-using-hex-value-instead-of-rgb
    r = (h & 0xff0000) >> 16
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    return tuple([srgb_to_linearrgb(c/0xff) for c in (r,g,b)] + [alpha])

orange1 = hex_to_rgb(0xff9a00)
orange2 = hex_to_rgb(0xff5d00)
blue1 = hex_to_rgb(0x00a2ff)
blue2 = hex_to_rgb(0x0065ff)
white = [1, 1, 1]


def get_frustum(intrinsics, height, width, image_depth, name="", with_fill=True):
    frustum_points = np.array([
        [0, height, 1],
        [width, height, 1],
        [0, 0, 1],
        [width, 0, 1]
    ]) * image_depth

    frustum_points = np.einsum('ij,...j->...i',
        np.linalg.inv(intrinsics),
        frustum_points,
    )
    frustum_edges = np.array([
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [0, 4],
        [1, 4],
        [2, 4],
        [3, 4]
    ])

    frustum_faces = [
        [0, 1, 4],
        [1, 3, 4],
        [3, 2, 4],
        [2, 0, 4],
    ]

    mesh = bpy.data.meshes.new("frustum")
    mesh.from_pydata(np.vstack([frustum_points, np.zeros(3)]), frustum_edges, frustum_faces)
    frustum = bu.new_empty("frustum" + name, mesh)
    modifier = bu.add_modifier(frustum, "WIREFRAME", use_crease=True, crease_weight=0.6, thickness=0.03, use_boundary=True)
    bpy.context.view_layer.objects.active = frustum
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    if with_fill:
        mesh_fill = bpy.data.meshes.new("fill")
        mesh_fill.from_pydata(np.vstack([frustum_points, np.zeros(3)]), frustum_edges, frustum_faces + [[0, 1, 3, 2]])
        fill = bu.new_empty("fill" + name, mesh_fill)
    else:
        fill = None
    return fill, frustum
    
def set_color(mesh, color):
    if len(color) == 3:
        color = [*color, 1.]
    mesh.materials.append(bpy.data.materials.new("color"))
    mesh.materials[0].use_nodes = True
    mesh.materials[0].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color

def make_line(start, end, width=0.05, name="line", color=None):
    line_mesh = bpy.data.meshes.new(name)
    line_mesh.from_pydata([start, end], [[0, 1]], [])
    line = bplt.blender_utils.new_empty(name, object_data=line_mesh)
    
    bplt.blender_utils.add_modifier(line, "SKIN")
    bplt.blender_utils.add_modifier(line, "SUBSURF", levels=3, render_levels=3)
    
    line.data.skin_vertices[''].data[0].radius = (width, width)
    line.data.skin_vertices[''].data[1].radius = (width, width)
    
    if color is not None:
        set_color(line_mesh, color)
    return line

def plot_cameras(poses, intrinsics, height, width, image_depth, name="", with_fill=True):
    fill, frustum = get_frustum(intrinsics, height, width, image_depth, name, with_fill)
    if with_fill:
        s = bplt.Scatter(
            poses.t,
            marker_rotation=poses.R,
            marker_type=fill,
            color=orange2,
            name="fill_scatter" + name,
        )
    s = bplt.Scatter(
        poses.t,
        marker_rotation=poses.R,
        marker_type=frustum,
        name="frustum_scatter" + name,
        color=orange1,
    )
    
def plot_observations(poses, observations, intrinsics, image_depth, size=1., name="observations"):
    s = bplt.Scatter(
        np.einsum('...ij, ...j->...i', poses.T, geometry.homogenize(np.einsum('...ij, ...j->...i', np.linalg.inv(intrinsics), geometry.homogenize(observations) * image_depth)))[:, :-1],
        marker_type="cubes",
        color=np.zeros(3),
        size=np.array([0.05, 0.05, 0.005]) * size,
        marker_rotation=poses.R,
        name=name
    )

def plot_problem(poses, observations, intrinsics, shapes, image_depth, with_fill=True, line_color=None, inlier_mask=None, neg_line=False, line_width=0.01):
    if line_color is None:
        line_color = np.ones(3) * 0.1
    if inlier_mask is None:
        inlier_mask = [True for _ in range(len(poses))]
    for i in tqdm(range(len(poses))):
        plot_cameras(poses[i], intrinsics[i], *shapes[i], image_depth, f"_{i}", with_fill=with_fill)
    
    plot_observations(poses, observations, intrinsics, image_depth, name="2d observations")
    for i, (pose, observation) in enumerate(zip(poses, observations)):
        make_line(
            pose.t,
            (pose.T @ geometry.homogenize(np.linalg.inv(intrinsics[i]) @ geometry.homogenize(observation) * 100))[:-1],
            width=line_width,
            color=line_color if inlier_mask[i] else [1, 0, 0],
            name=f'observation_{i}',
        )
        if neg_line:
            make_line(
                pose.t,
                (pose.T @ geometry.homogenize(-np.linalg.inv(intrinsics[i]) @ geometry.homogenize(observation) * 100))[:-1],
                width=line_width,
                color=line_color if inlier_mask[i] else [1, 0, 0],
                name=f'observation_negative_{i}',
            )

def plot_point(point, color=[0, 0, 1], point_radius=0.06, name='point'):
    bplt.Scatter(
        point,
        color=color,
        marker_type='spheres',
        name=name,
        radius=point_radius
    )

def setup_scene(floor_z=None, resolution=None, sun_energy=1):
    if floor_z is not None:
        floor_size = 500
        floor = bplt.Scatter([0, 0, floor_z], marker_type='cubes', size=(floor_size, floor_size, 0.1), name='floor')
        floor.base_object.is_shadow_catcher = True

    if resolution == 'thin':
        resolution = (1363, 2592)
    if resolution is not None:
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]

    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.scenes["Scene"].cycles.samples = 256


    if "Sun" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Sun"])
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.data.objects["Sun"].data.energy = sun_energy
    bpy.data.objects["Sun"].data.angle = 0
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0