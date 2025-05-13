import bpy
import struct
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.types import Context, Event, Operator, ActionFCurves, FCurve, Action
from bpy.props import CollectionProperty, StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
import ntpath
import mathutils
import math
import traceback
from .utils import Utils, CoordsSys
import xml.etree.ElementTree as ET
from mathutils import Vector, Quaternion, Matrix
from .bn_skeleton import SkeletonData
from pathlib import Path
import os

FRAME_SCALE = 160

class ImportAni(Operator, ImportHelper):
    bl_idname = "cbb.ani_import"
    bl_label = "Import ani"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".ANI"

    filter_glob: StringProperty(default="*.ANI", options={"HIDDEN"}) # type: ignore

    files: CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={"HIDDEN", "SKIP_SAVE"}
    ) # type: ignore

    directory: StringProperty(subtype="FILE_PATH") # type: ignore

    apply_to_selected_objects: BoolProperty(
        name="Apply to Selected Objects",
        description="Enabling this option will apply the animation to the currently selected objects. If false, a collection with the same base name as the animation is searched for and its objects used",
        default=False
    ) # type: ignore
    
    ignore_not_found: BoolProperty(
        name="Ignore Not Found Objects",
        description="Enabling this option will make the operator not raise an error in case an animated object or bone is not found, which is useful if the goal is to import the animation only to an armature and not to the whole mesh+skeleton package. It will still cause errors if the armature found is incompatible",
        default=True
    ) # type: ignore

    debug: BoolProperty(
        name="Debug",
        description="Enabling this option will print debug data to console",
        default=False
    ) # type: ignore

    def execute(self, context):
        return self.import_animations_from_files(context)

    def import_animations_from_files(self, context):
        for file in self.files:
            if file.name.casefold().endswith(".ani"):
                filepath: str = self.directory + file.name
                
                skeleton_data = None
                target_armature = None
                target_collection = None
                
                animated_object_count = 0
                animated_object_names: list[str] = []
                
                frame_amount = []
                frame_counts = []
                
                rotation_keyframe_counts = []
                rotation_frames: list[list[tuple[Quaternion, int]]] = []
                
                position_keyframe_counts = []
                position_frames: list[list[tuple[Vector, int]]] = []
                
                scale_keyframe_counts = []
                scale_frames: list[list[tuple[Vector, int]]] = []
                
                unknown_keyframe_counts = []
                unknown_frames: list[list[tuple[float, int]]] = []
                
                
                co_conv = Utils.CoordinatesConverter(CoordsSys._3DSMax, CoordsSys.Blender)
                
                file_base_name = Path(file.name).stem.split("_")[0]
                
                try:
                    with open(self.directory + file.name, "rb") as f:
                        reader = Utils.Serializer(f, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
                        animated_object_count = reader.read_ushort()
                        for i in range(animated_object_count):
                            animated_object_names.append(reader.read_fixed_string(100, "ascii"))
                            #If set to 0 animation is not considered
                            frame_amount.append(reader.read_ushort())
                            # if set higher than the amount keyframes are registered, it affects looping animations, which does indicate this is the maximum frame.
                            # if set lower, the highest keyframe set defines the maximum frame.
                            # actually, the value here is unknown in how it was stored. All I know is that each 160 values here mean one frame.
                            frame_counts.append(reader.read_ushort())
                            
                            f.seek(36,1)
                            
                            def __read_rotation_frames(reader: Utils.Serializer):
                                return (reader.read_converted_quaternion(), reader.read_uint())
                            
                            def __read_position_frames(reader: Utils.Serializer):
                                return (reader.read_converted_vector3f(), reader.read_uint())
                            
                            def __read_scale_frames(reader: Utils.Serializer):
                                return (reader.read_vector3f(), reader.read_uint())
                            
                            def __read_unknown_frames(reader: Utils.Serializer):
                                return (reader.read_float(), reader.read_uint())
                            
                            rotation_keyframe_counts.append(reader.read_ushort())
                            
                            rotation_frames.append([__read_rotation_frames(reader) for _ in range(rotation_keyframe_counts[i])])
                            
                            position_keyframe_counts.append(reader.read_ushort())
                            position_frames.append([__read_position_frames(reader) for _ in range(position_keyframe_counts[i])])
                            
                            scale_keyframe_counts.append(reader.read_ushort())
                            scale_frames.append([__read_scale_frames(reader) for _ in range(scale_keyframe_counts[i])])
                            
                            unknown_keyframe_counts.append(reader.read_ushort()) # What is being animated with a single float in the range of 0 to 1???
                            unknown_frames.append([__read_unknown_frames(reader) for _ in range(unknown_keyframe_counts[i])])

                except Exception as e:
                    self.report({"ERROR"}, f"Failed to read file at [{self.directory + file.name}]: {e}")
                    traceback.print_exc()
                    return {"CANCELLED"}

                found_objects: list[tuple[bpy.types.Object, int]] = []
                found_bones: list[tuple[str, int]] = []
                objects_collection = None
                
                if self.apply_to_selected_objects == True:
                    objects_collection = bpy.context.selected_objects
                else:
                    if file_base_name in bpy.data.collections:
                        Utils.debug_print(self.debug, f"Name [{file_base_name}] found within collections")
                        objects_collection = bpy.data.collections[file_base_name].objects
                    else:
                        self.report({"ERROR"}, f"No collection with the same base name [{file_base_name}] of the animation could be found in the scene.")
                        return {"CANCELLED"}
                    
                armature_objects = [obj for obj in objects_collection if obj.type == 'ARMATURE']
                if len(armature_objects) > 1:
                    self.report({"ERROR"}, "More than one armature is selected. Please select only one armature.")
                    return {"CANCELLED"}
                
                if armature_objects:
                    target_armature = armature_objects[0]
                    Utils.debug_print(self.debug, f"Target armature name: {target_armature.name}")
                    skeleton_data = SkeletonData(self.debug)
                    try:
                        skeleton_data.build_skeleton_from_armature(target_armature, False)
                    except Exception as e:
                        self.report({"ERROR"}, f"Armature [{target_armature}] which is the target of the imported animation has been found not valid. Aborting. Reason: {e}")
                        return {"CANCELLED"}
                
                for i, object_name in enumerate(animated_object_names):
                    found = False
                    
                    for obj in objects_collection:
                        if obj.name == object_name:
                            found_objects.append((obj, i))
                            found = True
                            break

                    if not found and target_armature:
                        for bone in target_armature.data.bones:
                            if bone.name == object_name:
                                found_bones.append((bone.name, i))
                                found = True
                                break

                    if not found and self.ignore_not_found == False:
                        self.report({"ERROR"}, f"Object or bone with name '{object_name}' not found in the selection.")
                        return {"CANCELLED"}

                Utils.debug_print(self.debug, f"Animation Data: ")
                Utils.debug_print(self.debug, f" Amount of animated objects: {animated_object_count}")
                Utils.debug_print(self.debug, f" Amount of frames: {frame_amount[0]}")
                
                try:
                    # Create animation action
                    action_name = Path(file.name).stem
                    action = bpy.data.actions.new(name=action_name)
                    highest_frame = 0
                    
                    for obj, index in found_objects:
                        obj.animation_data_create().action = action
                        
                        #obj.keyframe_insert(data_path="rotation_quaternion", frame=0)
                        for count in range(rotation_keyframe_counts[index]):
                            animated_rotation, scaled_frame = rotation_frames[index][count]
                            frame = scaled_frame / FRAME_SCALE

                            if highest_frame < frame:
                                highest_frame = frame

                            rot = animated_rotation.conjugated()
                            obj.rotation_quaternion = rot
                            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

                        #obj.keyframe_insert(data_path="location", frame=0)
                        for count in range(position_keyframe_counts[index]):
                            animated_position, scaled_frame = position_frames[index][count]
                            frame = scaled_frame / FRAME_SCALE

                            if highest_frame < frame:
                                highest_frame = frame

                            obj.location = animated_position
                            obj.keyframe_insert(data_path="location", frame=frame)
                        
                        #obj.keyframe_insert(data_path="scale", frame=0)
                        for count in range(scale_keyframe_counts[index]):
                            animated_scale, scaled_frame = scale_frames[index][count]
                            frame = scaled_frame / FRAME_SCALE

                            if highest_frame < frame:
                                highest_frame = frame

                            obj.scale = animated_scale
                            obj.keyframe_insert(data_path="scale", frame=frame)
                    
                    if target_armature and found_bones:
                        target_armature.animation_data_create().action = action
                        
                        for bone_name, index in found_bones:
                            bone_id = skeleton_data.bone_name_to_id[bone_name]
                            
                            if rotation_keyframe_counts[index] == 0:
                                target_armature.pose.bones[bone_name].rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="rotation_quaternion", frame=0)
                            for count in range(rotation_keyframe_counts[index]):
                                animated_rotation, scaled_frame = rotation_frames[index][count]
                                frame = scaled_frame/FRAME_SCALE
                                
                                if highest_frame < frame:
                                    highest_frame = frame
                                    
                                rot = Utils.get_local_rotation(skeleton_data.bone_local_rotations[bone_id], Quaternion((-animated_rotation.w, animated_rotation.x, animated_rotation.y, animated_rotation.z)))
                                
                                target_armature.pose.bones[bone_name].rotation_quaternion = rot
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="rotation_quaternion", frame=frame)
                                
                            if position_keyframe_counts[index] == 0:
                                target_armature.pose.bones[bone_name].location = Vector((0.0, 0.0, 0.0))
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="location", frame=0)
                            for count in range(position_keyframe_counts[index]):
                                animated_position, scaled_frame = position_frames[index][count]
                                frame = scaled_frame/FRAME_SCALE
                                
                                if highest_frame < frame:
                                    highest_frame = frame
                                
                                loc = Utils.get_local_position(skeleton_data.bone_local_positions[bone_id], skeleton_data.bone_local_rotations[bone_id], animated_position)

                                target_armature.pose.bones[bone_name].location = loc
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="location", frame=frame)
                            
                            if scale_keyframe_counts[index] == 0:
                                target_armature.pose.bones[bone_name].scale = Vector((1.0, 1.0, 1.0))
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="scale", frame=0)
                            for count in range(scale_keyframe_counts[index]):
                                animated_scale, scaled_frame = scale_frames[index][count]
                                frame = scaled_frame/FRAME_SCALE
                                
                                if highest_frame < frame:
                                    highest_frame = frame

                                target_armature.pose.bones[bone_name].scale = animated_scale
                                target_armature.pose.bones[bone_name].keyframe_insert(data_path="scale", frame=frame)

                    # Set animation frames range
                    action.frame_range = (0, highest_frame)

                except Exception as e:
                    animation_name = Path(file.name).stem
                    self.report({"ERROR"}, f"Failed to create animation {animation_name}: {e}")
                    traceback.print_exc()
                    return {"CANCELLED"}
                
        
        return {"FINISHED"}

    def invoke(self, context: Context, event: Event):
        if self.directory:
            return context.window_manager.invoke_props_dialog(self)
            # return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

class CBB_FH_ImportAni(bpy.types.FileHandler):
    bl_idname = "CBB_FH_ani_import"
    bl_label = "File handler for ani imports"
    bl_import_operator = ImportAni.bl_idname
    bl_file_extensions = ImportAni.filename_ext

    @classmethod
    def poll_drop(cls, context):
        return (context.area and context.area.type == "VIEW_3D")

class ExportAni(Operator, ExportHelper):
    bl_idname = "cbb.ani_export"
    bl_label = "Export ani"
    bl_options = {"PRESET"}

    filename_ext = ImportAni.filename_ext

    filter_glob: StringProperty(default="*.ANI", options={"HIDDEN"}) # type: ignore

    directory: StringProperty(subtype="FILE_PATH") # type: ignore
    
    action_export_option: bpy.props.EnumProperty(
        name="Action(s) to Export",
        description="Choose an option to define what action(s) will be exported",
        items=[
            ("ACTIVE_ACTION", "Active Action", "Export only the currently active action"),
            ("ACTIVE_COLLECTION_ACTIONS", "Active Collection Actions", "Export all actions among objects in the currently active collection"),
            ("SELECTED_OBJECT_ACTIONS", "Selected Objects Actions", "Export all actions among objects in the current selection"),
            ("ALL_ACTIONS", "All Actions", "Export All Scene Actions"),
        ],
        default='ALL_ACTIONS'
    ) # type: ignore

    debug: BoolProperty(
        name="Debug export",
        description="Enabling this option will make the exporter print debug data to console",
        default=False
    ) # type: ignore

    def execute(self, context):
        return self.export_animations(context, self.directory)
    
    def export_animations(self, context, directory):
        # Dictionary to hold actions and the associated objects
        actions_to_export = {}

        # ACTIVE_ACTION: Export only the currently active action
        if self.action_export_option == "ACTIVE_ACTION":
            active_object = context.object
            if active_object and active_object.animation_data and active_object.animation_data.action:
                active_action = active_object.animation_data.action
                actions_to_export[active_action] = [active_object]
        
        # ACTIVE_COLLECTION_ACTIONS: Export all actions among objects in the currently active collection
        elif self.action_export_option == "ACTIVE_COLLECTION_ACTIONS":
            active_collection = context.view_layer.active_layer_collection.collection
            if active_collection:
                for obj in active_collection.objects:
                    if obj.animation_data and obj.animation_data.action:
                        action = obj.animation_data.action
                        if action not in actions_to_export:
                            actions_to_export[action] = []
                        actions_to_export[action].append(obj)
                    
                    # Also check NLA strips for other linked actions
                    if obj.animation_data and obj.animation_data.nla_tracks:
                        nla_actions = Utils.get_actions_from_nla_tracks(obj)
                        for nla_action in nla_actions:
                            if nla_action not in actions_to_export:
                                actions_to_export[nla_action] = []
                            actions_to_export[nla_action].append(obj)
        
        # SELECTED_OBJECT_ACTIONS: Export all actions among objects in the current selection
        elif self.action_export_option == "SELECTED_OBJECT_ACTIONS":
            selected_objects = context.selected_objects
            for obj in selected_objects:
                if obj.animation_data and obj.animation_data.action:
                    action = obj.animation_data.action
                    if action not in actions_to_export:
                        actions_to_export[action] = []
                    actions_to_export[action].append(obj)
                
                # Also check NLA strips for other linked actions
                if obj.animation_data and obj.animation_data.nla_tracks:
                    nla_actions = Utils.get_actions_from_nla_tracks(obj)
                    for nla_action in nla_actions:
                        if nla_action not in actions_to_export:
                            actions_to_export[nla_action] = []
                        actions_to_export[nla_action].append(obj)

        # ALL_ACTIONS: Export all scene actions
        elif self.action_export_option == "ALL_ACTIONS":
            all_actions = bpy.data.actions
            for action in all_actions:
                # Find objects using this action
                linked_objects = []
                for obj in bpy.data.objects:
                    if obj.animation_data and (obj.animation_data.action == action):
                        linked_objects.append(obj)
                    elif obj.animation_data and obj.animation_data.nla_tracks:
                        nla_actions = Utils.get_actions_from_nla_tracks(obj)
                        if action in nla_actions:
                            linked_objects.append(obj)
                
                if linked_objects:
                    actions_to_export[action] = linked_objects

        # Now export all the collected actions
        for action, objects in actions_to_export.items():
            self.export_action(action, objects,  directory, self.filename_ext)

        return {"FINISHED"}
        
    def export_action(self, action: Action, action_objects: list[bpy.types.Object], directory: str, filename_ext: str):
        print(f"Exporting action {action.name}")
        
        old_active_object = bpy.context.view_layer.objects.active
        old_active_selected = bpy.context.view_layer.objects.active.select_get()
        old_active_mode = bpy.context.view_layer.objects.active.mode
        old_selection = [obj for obj in bpy.context.selected_objects]
        old_active_action = bpy.context.view_layer.objects.active.animation_data.action
        
        bpy.ops.object.mode_set()
        
        bpy.ops.object.select_all(action='DESELECT')
        
        
        bpy.context.view_layer.objects.active = action_objects[0]
        bpy.context.view_layer.objects.active.select_set(True)
        bpy.context.view_layer.objects.active.animation_data.action = action
        
        bpy.ops.nla.bake(
            frame_start=int(action.frame_range[0]),
            frame_end=int(action.frame_range[1]),
            only_selected=False,
            visual_keying=True,
            clear_constraints=False,
            use_current_action=False,
            bake_types={'POSE', 'OBJECT'}
        )
        
        baked_action = action_objects[0].animation_data.action
        
        filepath = bpy.path.ensure_ext(directory + "/" + action.name, filename_ext)
        
        initial_frame = baked_action.frame_range[0]
        last_frame = baked_action.frame_range[1]
        # +1 to include the last frame
        total_frames = int(last_frame+1 - initial_frame)
        total_export_frames = int(last_frame+1 - initial_frame)+1
        
        export_object_names = []
        export_unique_keyframe_counts = []
        export_maximum_frames = []
        export_rotation_keyframe_counts = []
        export_rotation_keyframes = {}
        export_position_keyframe_counts = []
        export_position_keyframes = {}
        export_scale_keyframe_counts = []
        export_scale_keyframes = {}
        export_unknown_keyframe_counts = []
        export_unknown_keyframes = {}
        
        Utils.debug_print(self.debug,f"Animation [{action.name}] frame range: {int(action.frame_range[0])} - {int(action.frame_range[1])}")
        
        for object in action_objects:
            
            if object.type in {"MESH", "EMPTY"}:
                def add_object_animation_data(_object_name, _total_frames, _total_export_frames, _export_rotation_keyframes, _export_position_keyframes, _export_scale_keyframes):
                    nonlocal export_object_names
                    nonlocal export_unique_keyframe_counts
                    nonlocal export_maximum_frames
                    nonlocal export_rotation_keyframe_counts
                    nonlocal export_rotation_keyframes
                    nonlocal export_position_keyframe_counts
                    nonlocal export_position_keyframes
                    nonlocal export_scale_keyframe_counts
                    nonlocal export_scale_keyframes
                    nonlocal export_unknown_keyframe_counts
                    nonlocal export_unknown_keyframes
                    
                    index = len(export_object_names)
                    
                    export_object_names.append(_object_name)
                    export_unique_keyframe_counts.append(_total_frames)
                    export_maximum_frames.append(_total_frames*FRAME_SCALE)
                    export_rotation_keyframe_counts.append(_total_export_frames)
                    export_position_keyframe_counts.append(_total_export_frames)
                    export_scale_keyframe_counts.append(_total_export_frames)
                    export_unknown_keyframe_counts.append(0)
                    
                    export_rotation_keyframes[index] = _export_rotation_keyframes
                    export_position_keyframes[index] = _export_position_keyframes
                    export_scale_keyframes[index] = _export_scale_keyframes
                
                object_name = object.name
                temp_rotation_keyframes = []
                temp_position_keyframes = []
                temp_scale_keyframes = []
                
                parent_matrix: Matrix = None
                if object.parent:
                    if object.parent_type == "BONE":
                        bone = object.parent.pose.bones[object.parent_bone]
                        # If the parent is a bone, object.parent refers to the armature itself, so we need to transform the bone matrix relative to the armature for the correct world matrix of the bone
                        parent_matrix = object.parent.matrix_basis @ bone.matrix
                    else:
                        parent_matrix = object.parent.matrix_basis
                else:
                    parent_matrix = object.matrix_basis
                
                object_local_matrix = parent_matrix.inverted() @ object.matrix_basis
                
                obj_local_pos = object_local_matrix.to_translation()
                obj_local_rot = object_local_matrix.to_quaternion()
                obj_local_scale = object_local_matrix.to_scale()
                temp_rotation_keyframes.append(Quaternion((-obj_local_rot.w, obj_local_rot.x, obj_local_rot.y, obj_local_rot.z)))
                temp_position_keyframes.append(obj_local_pos)
                temp_scale_keyframes.append(obj_local_scale)
                
                for frame in range(int(action.frame_range[0]), int(action.frame_range[1]+1)):
                    
                    obj_animated_rotation = Utils.get_object_rotation_at_frame_fcurves(baked_action, object.name, frame)
                    # Get the local rotation of the object, instead of the rest delta which is normal of Blender.
                    local_animated_rotation = Utils.get_world_rotation(obj_local_rot, obj_animated_rotation)
                    temp_rotation_keyframes.append(Quaternion((-local_animated_rotation.w, local_animated_rotation.x, local_animated_rotation.y, local_animated_rotation.z)))
                    
                    obj_animated_position = Utils.get_object_location_at_frame_fcurves(baked_action, object.name, frame)
                    local_animated_position = Utils.get_world_position(obj_local_pos, obj_local_rot, obj_animated_position)
                    temp_position_keyframes.append(local_animated_position)
                    
                    obj_animated_scale = Utils.get_object_scale_at_frame_fcurves(baked_action, object.name, frame)
                    temp_scale_keyframes.append(obj_animated_scale)
                    
                if object.type == "MESH":
                    mesh: bpy.types.Mesh = object.data
                    mesh_polygons = mesh.polygons
                    
                    amount_of_indices = sum((len(poly.loop_indices) - 2) * 3 for poly in mesh_polygons)
                    
                    if amount_of_indices <= 65535:
                            add_object_animation_data(object_name, total_frames, total_export_frames, temp_rotation_keyframes, temp_position_keyframes, temp_scale_keyframes)
                    else:
                        print("Object's animation has to be split in chunks.")
                
                        maximum_split_amount = math.ceil(amount_of_indices / 65535.0)
                        for object_number in range(0, maximum_split_amount):
                            print(f"Exporting split number: {object_number}")
                            split_object_name = f"{object_name}_{object_number}"
                            
                            add_object_animation_data(split_object_name, total_frames, total_export_frames, temp_rotation_keyframes, temp_position_keyframes, temp_scale_keyframes)
                    
            
            if object.type == "ARMATURE":
                skeleton_data = SkeletonData(self.debug)
                skeleton_data.build_skeleton_from_armature(object, False)
                
                for bone_name in skeleton_data.bone_names:
                    index = len(export_object_names)
                    export_object_names.append(bone_name)
                    export_unique_keyframe_counts.append(total_frames)
                    export_maximum_frames.append(total_frames*FRAME_SCALE)
                    export_rotation_keyframe_counts.append(total_export_frames)
                    export_position_keyframe_counts.append(total_export_frames)
                    export_scale_keyframe_counts.append(total_export_frames)
                    export_unknown_keyframe_counts.append(0)
                    
                    bone_id = skeleton_data.bone_name_to_id[bone_name]
                    pose_bone = object.pose.bones[bone_name]
                    
                    temp_rotation_keyframes = []
                    temp_position_keyframes = []
                    temp_scale_keyframes = []
                    
                    obj_local_rot = skeleton_data.bone_local_rotations[bone_id]
                    temp_rotation_keyframes.append(Quaternion((-obj_local_rot.w, obj_local_rot.x, obj_local_rot.y, obj_local_rot.z)))
                    temp_position_keyframes.append(skeleton_data.bone_local_positions[bone_id])
                    temp_scale_keyframes.append(skeleton_data.bone_absolute_scales[bone_id])
                    
                    for frame in range(int(action.frame_range[0]), int(action.frame_range[1]+1)):
                        
                        pose_bone_rotation = Utils.get_pose_bone_rotation_at_frame_fcurves(baked_action, pose_bone.name, frame)
                        # Get the local rotation of the bone, instead of the rest delta which is normal of Blender.
                        local_animated_rotation = Utils.get_world_rotation(skeleton_data.bone_local_rotations[bone_id], pose_bone_rotation)
                        temp_rotation_keyframes.append(Quaternion((-local_animated_rotation.w, local_animated_rotation.x, local_animated_rotation.y, local_animated_rotation.z)))
                        
                        pose_bone_position = Utils.get_pose_bone_location_at_frame_fcurves(baked_action, pose_bone.name, frame)
                        local_animated_position = Utils.get_world_position(skeleton_data.bone_local_positions[bone_id], skeleton_data.bone_local_rotations[bone_id], pose_bone_position)
                        temp_position_keyframes.append(local_animated_position)
                        
                        obj_animated_scale = Utils.get_pose_bone_scale_at_frame_fcurves(baked_action, pose_bone.name, frame)
                        temp_scale_keyframes.append(obj_animated_scale)
                        
                    export_rotation_keyframes[index] = temp_rotation_keyframes
                    export_position_keyframes[index] = temp_position_keyframes
                    export_scale_keyframes[index] = temp_scale_keyframes
                    
                        
        co_conv = Utils.CoordinatesConverter(CoordsSys.Blender, CoordsSys._3DSMax)
        with open(filepath, 'wb') as file:
            writer = Utils.Serializer(file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor, co_conv)
            writer.write_ushort(len(export_object_names))
            for index, object_name in enumerate(export_object_names):
                writer.write_fixed_string(100, "ascii", object_name)
                writer.write_ushort(export_unique_keyframe_counts[index])
                writer.write_ushort(export_maximum_frames[index])
                file.write(bytearray(36))
                
                exporting_rotation_keyframes = export_rotation_keyframes.get(index)
                writer.write_ushort(export_rotation_keyframe_counts[index])
                if exporting_rotation_keyframes is not None:
                    for frame_number, rotation in enumerate(exporting_rotation_keyframes):
                        writer.write_converted_quaternion(rotation)
                        writer.write_uint(frame_number*FRAME_SCALE)
                
                exporting_position_keyframes = export_position_keyframes.get(index)
                writer.write_ushort(export_position_keyframe_counts[index])
                if exporting_position_keyframes is not None:
                    for frame_number, position in enumerate(exporting_position_keyframes):
                        writer.write_converted_vector3f(position)
                        writer.write_uint(frame_number*FRAME_SCALE)
                
                exporting_scale_keyframes = export_scale_keyframes.get(index)
                writer.write_ushort(export_scale_keyframe_counts[index])
                if exporting_scale_keyframes is not None:
                    for frame_number, scale in enumerate(exporting_scale_keyframes):
                        writer.write_vector3f(scale)
                        writer.write_uint(frame_number*FRAME_SCALE)
                
                exporting_unknown_keyframes = export_unknown_keyframes.get(index)
                writer.write_ushort(export_unknown_keyframe_counts[index])
                if exporting_unknown_keyframes is not None:
                    for frame_number, unknown in enumerate(exporting_unknown_keyframes):
                        writer.write_float(unknown)
                        writer.write_uint(frame_number*FRAME_SCALE)
        
        
        bpy.context.view_layer.objects.active.animation_data.action = old_active_action
        
        
        
        bpy.data.actions.remove(baked_action, do_unlink=True)
        
        
        
        bpy.ops.object.select_all(action='DESELECT')
        
        bpy.context.view_layer.objects.active = old_active_object
        bpy.context.view_layer.objects.active.select_set(old_active_selected)
        
        for obj in old_selection:
            obj.select_set(True)
        if old_active_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=old_active_mode)
        
        

def menu_func_import(self, context):
    self.layout.operator(ImportAni.bl_idname, text="ANI (.ANI)")

def menu_func_export(self, context):
    self.layout.operator(ExportAni.bl_idname, text="ANI (.ANI)")



def register():
    bpy.utils.register_class(ImportAni)
    bpy.utils.register_class(CBB_FH_ImportAni)
    bpy.utils.register_class(ExportAni)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(ImportAni)
    bpy.utils.unregister_class(CBB_FH_ImportAni)
    bpy.utils.unregister_class(ExportAni)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
