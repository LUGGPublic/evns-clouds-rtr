import bpy
import datetime
import mathutils
import numpy as np
import time
import os


NUMBER_OF_FRAMES = 200
RESOLUTION = 1024
LOWER_SUN_ELEVATION = -5.0  # degrees
UPPER_SUN_ELEVATION = 90.0  # degrees
LOWER_SUN_ROTATION = 0.0    # degrees
UPPER_SUN_ROTATION = 360.0  # degrees


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def get_object(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise Exception(f"No object named '{name}' found in the scene.")
    return obj


def set_fileoutput(path):
    for node in bpy.context.scene.node_tree.nodes:
        if node.type == 'OUTPUT_FILE':
            node.base_path = path


def render(timestamp):
    # Get renderer
    renderer = bpy.context.scene.render
    renderer.engine = 'CYCLES'  # Use Cycles renderer
    renderer.resolution_x = RESOLUTION  # Set render resolution
    renderer.resolution_y = RESOLUTION
    renderer.resolution_percentage = 100  # Set resolution percentage to 100%
    renderer.use_border = False
    renderer.use_crop_to_border = False

    # Get the mesh
    mesh = get_object("Mesh")

    # Save bounding box
    coords = [mesh.matrix_world @ mathutils.Vector(corner) for corner in mesh.bound_box]
    min_corner = mathutils.Vector((min(v.x for v in coords),
                                min(v.y for v in coords),
                                min(v.z for v in coords)))
    max_corner = mathutils.Vector((max(v.x for v in coords),
                                max(v.y for v in coords),
                                max(v.z for v in coords)))
    with open(f"{timestamp}/bounds.csv", "w") as file:
        file.write("min_x, min_y, min_z, max_x, max_y, max_z\n")
        file.write(f"{min_corner.x}, {min_corner.y}, {min_corner.z}, {max_corner.x}, {max_corner.y}, {max_corner.z}\n")

    # Get the front camera
    camera = get_object("Camera")

    # Get the pivot object of both scenes
    pivot = get_object("Pivot")
    pivot_back = get_object("PivotBack")
    pivot_vis = get_object("PivotVis")

    # Get the visibility test sun
    sun_vis = get_object("SunVis")
    pivot_sun_vis = get_object("PivotSunVis")
    
    # Get the sky texture (Nishita)
    world = bpy.context.scene.world
    if world is None or world.node_tree is None:
        raise Exception("No world or node tree found in the scene.")
    sky_texture = None
    for node in world.node_tree.nodes:
        if node.type == 'TEX_SKY':
            sky_texture = node
            break
    if sky_texture is None:
        raise Exception("No sky texture node found in the world node tree.")
    
    with open(f"{timestamp}/labels.csv", "w") as file:
        file.write("frame, view_x, view_y, view_z, sun_elevation, sun_rotation\n")
    
    # Loop through frames
    for frame in range(NUMBER_OF_FRAMES):
        # Set frame number so output get different names
        bpy.context.scene.frame_set(frame)

        # Set up pivot rotations
        x_rot = np.random.uniform(-np.pi/2, np.pi/2)
        z_rot = np.random.uniform(-np.pi, np.pi)
        pivot.rotation_euler = mathutils.Euler((x_rot, 0, z_rot), 'XYZ')
        pivot_back.rotation_euler = pivot.rotation_euler
        pivot_vis.rotation_euler = pivot.rotation_euler

        # Calculate view vector (world coordinates)
        camera_pos = camera.matrix_world.to_translation()
        pivot_pos = pivot.matrix_world.to_translation()
        view = camera_pos - pivot_pos
        view.normalize()

        # Set up sky texture light direction
        sun_elevation = np.random.uniform()
        sun_rotation = np.random.uniform()
        # In range
        sky_texture.sun_elevation = np.deg2rad(LOWER_SUN_ELEVATION + sun_elevation * (UPPER_SUN_ELEVATION - LOWER_SUN_ELEVATION))
        sky_texture.sun_rotation = np.deg2rad(LOWER_SUN_ROTATION + sun_rotation * (UPPER_SUN_ROTATION - LOWER_SUN_ROTATION))

        pivot_sun_vis.rotation_euler = mathutils.Euler((
            sky_texture.sun_elevation,
            0,
            -sky_texture.sun_rotation
        ), 'XYZ')

        # Render
        bpy.ops.render.render(write_still=True)

        # Save labels
        with open(f"{timestamp}/labels.csv", "a") as file:
            file.write(f"{frame}, {view.x}, {view.y}, {view.z}, {sun_elevation}, {sun_rotation}\n")
        
        print(f"Rendered frame {frame}")


print(f"## Rendering {NUMBER_OF_FRAMES} frames ##")

start = time.time()

timestamp = get_timestamp()
os.makedirs(f"{timestamp}")
set_fileoutput(f"//{timestamp}")
render(timestamp)

print(f">> Rendering complete: {(time.time() - start) / 60.0:5.2f} minutes elapsed")
