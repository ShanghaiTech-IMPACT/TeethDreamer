"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

from mathutils import Vector, Matrix
import numpy as np

import bpy
from mathutils import Vector
import pickle
import shutil

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--target_dir", type=str, required=True)
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--num_images", type=int, default=16)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--device", type=str, default='GPU')

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

image_size = 512
render.resolution_x = image_size
render.resolution_y = image_size
render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_percentage = 100
render.threads_mode = 'FIXED'
render.threads = 6

scene.cycles.device = args.device
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"
bpy.context.scene.cycles.tile_size = 8192

# poses={'lower':{},'upper':{}}
# da=np.arange(4,dtype=np.float32)/4*np.pi/2
# de=np.linspace(0,90,6, endpoint=True)[1:-1]/180*np.pi
da = [np.pi / 6, np.pi / 3, np.pi / 3 * 2, np.pi / 6 * 5]
de = np.repeat(np.linspace(0, np.pi / 2, 4, endpoint=False)[None], 4, 0).T.flatten()
# da = [np.pi/4, np.pi/4*3, np.pi/4*5, np.pi/4*7]
# de = [np.pi/4]*4 + [0]*4 + [-np.pi/6]*4 + [-np.pi/3]*4
view16 = {}
view16['upper'] = np.array([da * 4, (-de).tolist(), [1.5] * 16], dtype=np.float32)
view16['lower'] = np.array([da * 4, de.tolist(), [1.5] * 16], dtype=np.float32)
cond = np.array([[0, 0, 1.5], [np.pi / 2, 0, 1.5], [np.pi, 0, 1.5], [np.pi / 2, -75 / 180 * np.pi, 1.5],
                 [np.pi / 2, 75 / 180 * np.pi, 1.5]], dtype=np.float32)
pose = np.array([-np.pi / 2, -np.pi, 0], dtype=np.float32)
names = ['left', 'front', 'right', 'down', 'up']


# poses['upper']['front']=np.array([-np.pi/2, -np.pi, 0], dtype=np.float32)
# poses['lower']['left']=np.array([-np.pi/2, -np.pi, 0], dtype=np.float32)
# poses['lower']['right']=np.array([-np.pi/2, -np.pi, 0], dtype=np.float32)
# poses['lower']['up']=np.array([-np.pi/2, -np.pi, 0], dtype=np.float32)
#
# poses['lower']['front']=np.array([-np.pi/2, -np.pi, 0], dtype=np.float32)
# poses['lower']['left']=np.array([-np.pi/2, -np.pi, 0], dtype=np.float32)
# poses['lower']['right']=np.array([-np.pi/2, -np.pi, 0], dtype=np.float32)
# poses['lower']['down']=np.array([-np.pi/2, -np.pi, 0], dtype=np.float32)

def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths) * np.cos(elevations)
    y = np.sin(azimuths) * np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x, y, z], -1)


def set_camera_location(cam_pt):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = cam_pt  # sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    return camera


def set_object_rotation(object_uid, rot_euler):
    obj = bpy.data.objects[object_uid]
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = rot_euler


def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.data.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera.data.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.asarray(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)), np.float32)
    return K


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    # for image in bpy.data.images:
    #     bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=object_path,
                              directory=os.path.dirname(object_path))
    elif object_path.endswith(".stl"):
        bpy.ops.import_mesh.stl(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

    obj = bpy.context.object
    obj.data.materials.append(bpy.data.materials.new(name="Material"))
    mat = obj.data.materials.get("Material")
    mat.use_nodes = True
    # print(mat.node_tree.nodes)
    if object_path.endswith(".ply"):
        vertex_color_node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        # print(mat.node_tree.nodes[0].inputs.keys())
        vertex_color_node.layer_name = "Col"
        mat.node_tree.links.new(vertex_color_node.outputs['Color'], mat.node_tree.nodes[0].inputs['Base Color'])
    else:
        mat.node_tree.nodes[0].inputs['Base Color'].default_value = [0.8, 0.4, 0.0, 1]
    # print(mat.node_tree.nodes[0].inputs['Roughness'])
    mat.node_tree.nodes[0].inputs['Roughness'].default_value = 0.25
    mat.node_tree.nodes[0].inputs['IOR'].default_value = 1.45
    mat.node_tree.nodes[0].inputs['Alpha'].default_value = 1
    mat.node_tree.nodes[0].inputs['Subsurface Weight'].default_value = 0.6
    # mat.node_tree.nodes[0].inputs["Subsurface"].default_value = 0.4
    mat.node_tree.nodes[0].inputs['Coat Weight'].default_value = 0.7
    mat.node_tree.nodes[0].inputs['Coat Roughness'].default_value = 0.2
    mat.node_tree.nodes[0].inputs['Coat IOR'].default_value = 1
    mat.node_tree.nodes[0].inputs['Coat Tint'].default_value = [0.6, 0.6, 0.3, 1.0]


def create_sun_light(loc=[0, 0, 0], rot=[0, 0, 0]):
    light_data = bpy.data.lights.new(name='Direct_Light', type='SUN')

    light_data.type = 'SUN'
    light_data.energy = 10

    light_object = bpy.data.objects.new(name='SunLight', object_data=light_data)
    bpy.context.scene.collection.objects.link(light_object)

    light_object.location = loc
    light_object.rotation_euler = rot
    return light_object


def remove_sun_light():
    for obj in bpy.data.objects:
        if obj.type in {"LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def set_sun_light(loc, rot):
    bpy.data.objects['SunLight'].location = loc
    bpy.data.objects['SunLight'].rotation_euler = rot
    return bpy.data.objects['SunLight']


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    # r = np.asarray(rotation.to_euler())
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)

    # print(np.concatenate([np.concatenate([R,t[:,None]],1),np.array([0,0,0,1])[None]],0)-np.asarray(cam.matrix_world))

    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t
    # R = cam_rec @ R
    # t = cam_rec @ t

    RT = np.concatenate([R_world2cv, t_world2cv[:, None]], 1)
    # RT = np.concatenate([R,t[:,None]],1)
    # RT = np.asarray(cam.matrix_world)
    return RT


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def render(azimuths, elevations, distances, output_dir, object_uid, num_images):
    cam_pts = az_el_to_points(azimuths, elevations) * distances[:, None]
    cam_poses = []
    num_images = range(num_images)
    # print(cam.data.angle_x)
    for i in num_images:
        # set camera
        camera = set_camera_location(cam_pts[i])
        RT = get_3x4_RT_matrix_from_blender(camera)
        name = os.path.join(object_uid, f"{i:03d}.png")
        cam_poses.append(RT)
        t, r = camera.matrix_world.decompose()[0:2]
        if 'input' in output_dir:
            set_sun_light(t, r.to_euler())

        render_path = os.path.join(output_dir, name)
        if os.path.exists(render_path): continue
        scene.render.filepath = os.path.abspath(render_path)
        bpy.ops.render.render(write_still=True)

    K = get_calibration_matrix_K_from_blender(camera)
    cam_poses = np.stack(cam_poses, 0)
    save_pickle([K, azimuths, elevations, distances, cam_poses],
                os.path.join(output_dir, object_uid, "meta.pkl"))

def save_cond_cams(object_file, output_dir):
    object_uid = os.path.basename(os.path.dirname(object_file)) + '_' + os.path.basename(object_file).split(".")[0]
    os.makedirs(output_dir, exist_ok=True)
    reset_scene()
    # load the object
    load_object(object_file)
    bpy.context.active_object.name = object_uid
    # object_uid = os.path.basename(object_file).split(".")[0]
    # normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    env_light = 1
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 3
    set_object_rotation(object_uid, pose)

    cam_pts = az_el_to_points(cond[:,0], cond[:,1]) * cond[:, [2]].repeat(3,1)
    cam_poses = []
    num_images = range(cond.shape[0])
    for i in num_images:
        camera = set_camera_location(cam_pts[i])
        RT = get_3x4_RT_matrix_from_blender(camera)
        name = f"{i:03d}.png"
        cam_poses.append(RT)
        render_path = os.path.join(output_dir, name)
        if os.path.exists(render_path): continue
        scene.render.filepath = os.path.abspath(render_path)
        bpy.ops.render.render(write_still=True)

    K = get_calibration_matrix_K_from_blender(camera)
    print(K)
    cam_poses = np.stack(cam_poses, 0)
    # save_pickle([K, cond[:,0], cond[:,1], cond[:,2], cam_poses],
    #             os.path.join(output_dir, "cond.pkl"))

def save_images(object_file, output_dir, num_images) -> None:
    object_uid = os.path.basename(os.path.dirname(object_file)) + '_' + os.path.basename(object_file).split(".")[0]
    # object_uid = os.path.basename(object_file).split(".")[0]
    os.makedirs(output_dir[0], exist_ok=True)
    os.makedirs(output_dir[1], exist_ok=True)

    reset_scene()
    # load the object
    load_object(object_file)
    bpy.context.active_object.name = object_uid
    # object_uid = os.path.basename(object_file).split(".")[0]
    # normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    env_light = 1
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 3

    # distances = np.asarray([1.5 for _ in range(args.num_images)])
    #    if camera_type=='fixed':
    #        distances = view16[2]
    #        azimuths = view16[0]
    #        elevations = view16[1]
    #        # azimuths = (np.arange(args.num_images)/args.num_images*np.pi*2).astype(np.float32)
    #        # elevations = np.deg2rad(np.asarray([args.elevation] * args.num_images).astype(np.float32))
    #    elif camera_type=='random':
    #        dazs = (np.random.rand(16)*20-10)/180*np.pi
    #        dels = (np.random.rand(16)*20-10)/180*np.pi
    #        dz = np.random.rand(16)*0.2
    #        azimuths = (dazs + cond[2][0]).astype(np.float32)
    #        elevations = (dels + cond[2][1]).astype(np.float32)
    #        distances = (dz + cond[2][2]).astype(np.float32)
    #    else:
    #        raise NotImplementedError

    set_object_rotation(object_uid, pose)
    if 'upper' in object_uid:
        views = view16['upper']
        view_ids = [0, 1, 2, 3]
    else:
        views = view16['lower']
        view_ids = [0, 1, 2, 4]

    # create sunlight for input views
    create_sun_light()

    azimuths, elevations, distances = views
    rendered = [os.listdir(output_dir[0]), os.listdir(output_dir[1])]
    for i in view_ids:
        (Path(output_dir[0]) / (object_uid + f'_{names[i]}')).mkdir(exist_ok=True, parents=True)
        (Path(output_dir[1]) / (object_uid + f'_{names[i]}')).mkdir(exist_ok=True, parents=True)
        if object_uid + f'_{names[i]}' in rendered[1]:
            continue
        dazs = (np.random.rand(num_images) * 20 - 10) / 180 * np.pi
        dels = (np.random.rand(num_images) * 20 - 10) / 180 * np.pi
        dz = np.random.rand(num_images) * 0.2
        azimuth, elevation, distance = cond[i]
        input_azimuths = (azimuth + dazs).astype(np.float32)
        input_elevations = (elevation + dels).astype(np.float32)
        input_distances = (distance + dz).astype(np.float32)
        render(input_azimuths, input_elevations, input_distances, output_dir[1], object_uid + f'_{names[i]}',
               num_images)

    remove_sun_light()
    obj = bpy.context.object
    mat = obj.data.materials.get("Material")
    mat.use_nodes = True
    mat.node_tree.nodes[0].inputs['Subsurface Weight'].default_value = 0.0

    if not object_uid + f'_{names[0]}' in rendered[0]:
        render(azimuths, elevations, distances, output_dir[0], object_uid + f'_{names[0]}', num_images)
    for i in os.listdir(os.path.join(output_dir[0], object_uid + f'_{names[0]}')):
        for j in view_ids:
            if j == 0 or object_uid + f'_{names[j]}' in rendered[0]:
                continue
            shutil.copy(os.path.join(output_dir[0], object_uid + f'_{names[0]}', i),
                        os.path.join(output_dir[0], object_uid + f'_{names[j]}'))


#    cam_pts = az_el_to_points(azimuths, elevations) * distances[:,None]
#    cam_poses = []
#    (Path(output_dir) / object_uid).mkdir(exist_ok=True, parents=True)
#    num_images=range(num_images)
#    # if 'lower' in object_file:
#    #     num_images=[4]
#    # else:
#    #     num_images = [12]
#    set_object_rotation(object_uid, pose)
#    print(cam.data.angle_x)
#    for i in num_images:
#        # set camera
#        camera = set_camera_location(cam_pts[i])
#        #print(object_uid)
#        RT = get_3x4_RT_matrix_from_blender(camera)
#        name = os.path.join(object_uid, f"{i:03d}.png")
#        # name = object_uid + '.png'
#        cam_poses.append(RT)
#        t, r = camera.matrix_world.decompose()[0:2]
#        # print(t,r.to_euler())
#        # print(camera.location,camera.rotation_euler)
#        # c2w = np.concatenate([np.asarray(t)[None],np.asarray(r.to_euler())[None]],axis=0)
#        # np.savetxt(os.path.join(r'D:\workspace\teeth_recon\original\scan\mesh',f"{i:03d}.txt"),c2w)
#        if camera_type == 'random':
#            # R = RT[:3,:3].T
#            # t = list((-R @ RT[:3,3])[:,0])
#            # r = Rotation.from_matrix(R).as_euler('xyz',degrees=False)
#            set_sun_light(t, r.to_euler())
#        #print(RT[:3,:3])
#        render_path = os.path.join(output_dir, name)
#        if os.path.exists(render_path): continue
#        scene.render.filepath = os.path.abspath(render_path)
#        bpy.ops.render.render(write_still=True)

#    K = get_calibration_matrix_K_from_blender(camera)
#    cam_poses = np.stack(cam_poses, 0)
#    if camera_type=='random':
#        save_pickle([K, azimuths, elevations, distances, cam_poses],
#                    os.path.join(output_dir, object_uid, "input.pkl"))
#    else:
#        save_pickle([K, azimuths, elevations, distances, cam_poses],
#                    os.path.join(output_dir, object_uid, "poses.pkl"))

if __name__ == "__main__":
    # from time import sleep
    # args.output_dir = r"D:\workspace\teeth_recon\datasets\flags"
    # for p in os.listdir(args.object_path):
    #     if p=='rendering' or p=='flags':
    #         continue
    #     for f in os.listdir(os.path.join(args.object_path, p)):
    #         # bpy.ops.world.new()
    #         save_images(os.path.join(args.object_path, p, f))
    #         # sleep(1)
    # object_file=r'D:\workspace\teeth_recon\datasets\whu_patient_6\norm_upper.ply'
    # object_file=r'D:\workspace\teeth_recon\datasets\ktj_patient_615\norm_lower.stl'
    # output_dir=[r'D:\workspace\teeth_recon\original\scan\rendering\target',r'D:\workspace\teeth_recon\original\scan\rendering\input']

    save_images(args.object_path, [args.target_dir,args.input_dir], args.num_images)

    # save_cond_cams(args.object_path, args.input_dir)

    # output_dir = [r'D:\workspace\teeth_recon\datasets\rendering\target',
    #               r'D:\workspace\teeth_recon\datasets\rendering\input']
    # num_images = 16
    # path = r'D:\workspace\teeth_recon\datasets'
    # fs = os.listdir(path)
    # flags = os.listdir(os.path.join(path, 'flags'))
    # for f in fs:
    #     if f == 'flags' or f == 'rendering':
    #         continue
    #     if 'norm_lower.stl' in os.listdir(os.path.join(path, f)):
    #         lower = os.path.join(path, f, 'norm_lower.stl')
    #     else:
    #         lower = os.path.join(path, f, 'norm_lower.ply')
    #     if 'norm_upper.stl' in os.listdir(os.path.join(path, f)):
    #         upper = os.path.join(path, f, 'norm_upper.stl')
    #     else:
    #         upper = os.path.join(path, f, 'norm_upper.ply')
    #     lower_flag = True if f + '_lower.png' in flags else False
    #     upper_flag = True if f + '_upper.png' in flags else False
    #     if lower_flag:
    #         save_images(lower, output_dir, num_images)
    #     if upper_flag:
    #         save_images(upper, output_dir, num_images)