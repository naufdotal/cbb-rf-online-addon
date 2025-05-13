from __future__ import annotations
import bpy
import struct
import traceback
from bpy.types import Operator, Context, Event
from mathutils import Vector, Quaternion, Matrix, Euler
import os
from enum import Enum
import io
from .utils import Vector3Int, Utils
import tempfile
from typing import Tuple
import numpy as np
from PIL import Image
from . import texture_utils

SCALE_FACTOR = 1

class RFShared(Operator):
    bl_idname = "cbb.rf_shared"
    bl_label = "RF Shared"
    bl_options = {'PRESET', 'UNDO'}
    
    @staticmethod
    def unlock_dds(buffer):
        password = bytes([
            0x2E, 0x80, 0x4D, 0x76, 0x2E, 0xF8, 0xD1, 0xF0, 0xBD, 0x3F, 0x86, 0x81, 0x58, 0x2C, 0x3F, 0x3F, 
            0x2E, 0x2E, 0x67, 0x6F, 0x3F, 0x40, 0x3F, 0x78, 0x3C, 0x3F, 0xF1, 0xC0, 0xA5, 0xF6, 0x3B, 0x9F, 
            0xC1, 0x20, 0x3F, 0xD7, 0xC8, 0xC1, 0xE9, 0x85, 0x86, 0xBD, 0xEF, 0x56, 0x3F, 0xA1, 0xFB, 0x2E, 
            0x87, 0x86, 0x61, 0x4C, 0x21, 0x3B, 0x4E, 0xB4, 0x78, 0x57, 0xAE, 0x97, 0x3F, 0x2E, 0x4A, 0x2E, 
            0x3F, 0x4C, 0x2E, 0x44, 0xCD, 0xC5, 0x5F, 0xE8, 0xE9, 0xEC, 0xEB, 0xBD, 0xBE, 0xBB, 0xF7, 0x6C, 
            0x2E, 0xF2, 0xE4, 0x2E, 0x3F, 0x3F, 0x97, 0x9F, 0x9D, 0xB3, 0x21, 0xB9, 0x76, 0x65, 0x54, 0x3F, 
            0xE6, 0xF6, 0xC6, 0xF0, 0x79, 0xDB, 0xE2, 0xB2, 0x4B, 0x2E, 0x2E, 0xEB, 0xD3, 0xD3, 0xCA, 0xAB, 
            0xEA, 0xC7, 0xED, 0x9C, 0xC7, 0xD9, 0xD0, 0x65, 0x48, 0xB4, 0xFA, 0x35, 0x2E, 0x2E, 0x6A, 0x9B, 
            #0xAF, 0x7E, 0xD6, 0xB7, 0x79, 
        ])
        password = struct.unpack('<32I', password)
        for i in range(32):
            buffer[i] ^= password[i]
        return buffer
    
    @staticmethod
    def get_materials_from_r3m_file(directory, file_name_stem)-> list[R3MMaterial]:
        """
        Returns a R3MMaterial collection with all the materials in the R3M file, given a directory and a file name to search for.
        """
        r3m_filepath = os.path.join(directory, file_name_stem + ".r3m")
        try:
            with open(r3m_filepath, 'rb') as r3m_file:
                return RFShared.get_materials_from_r3m_filestream(r3m_file)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"R3M file not found at: {r3m_filepath}")
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3M file at {r3m_filepath}") from e
        
    @staticmethod
    def get_materials_from_r3m_filestream(r3m_file: io.BufferedReader)-> list[R3MMaterial]:
        """
        Returns a R3MMaterial collection with all the materials in the R3M file from an already opened file.
        """
        try:
            reader = Utils.Serializer(r3m_file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor)
            version = reader.read_float()
            if abs(version - 1.1) > 0.01:
                print(f"Warning: R3M file version [{version}] is different than the version [1.1] this addon was built in mind with.")
                
            material_amount = reader.read_uint()
            materials: list[R3MMaterial] = []
            for a in range(material_amount):
                material = R3MMaterial()
                material.layer_num = reader.read_uint()
                material.flag = reader.read_uint()
                material.detail_surface = reader.read_int()
                material.detail_scale = reader.read_float()
                material.name = reader.read_fixed_string(128, "euc-kr")
                for b in range(material.layer_num):
                    layer = reader.read_values("h i I I I h h h h h h h h h h H h h h", 46)
                    texture_layer = TextureLayer.get_texture_layer_from_unpacked_bytes(layer)
                    material.texture_layers.append(texture_layer)
                materials.append(material)
            return materials
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3M file data: {e}\n{traceback.format_exc()}")
    
    @staticmethod
    def get_color_texture_dictionary_from_r3t_file(directory, file_name_stem) -> dict[int, str]:
        r3t_filepath = os.path.join(directory, file_name_stem + ".r3t")
        try:
            with open(r3t_filepath, 'rb') as r3t_file:
                return RFShared.get_color_texture_dictionary_from_r3t_filestream(r3t_file)
    
        except FileNotFoundError:
            raise FileNotFoundError(f"R3T file not found at: {r3t_filepath}")
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3T file at {r3t_filepath}") from e
        finally:
            print()
            
    @staticmethod
    def get_color_texture_dictionary_from_r3t_filestream(r3t_file: io.BufferedReader) -> dict[int, str]:
        try:
            texture_dictionary: dict = {}
            reader = Utils.Serializer(r3t_file, Utils.Serializer.Endianness.Little, Utils.Serializer.Quaternion_Order.XYZW, Utils.Serializer.Matrix_Order.ColumnMajor)
            version = reader.read_float()
            if abs(version - 1.2) > 0.01:
                print(f"Warning: R3T file version [{version}] is different than the version [1.2] this addon was built in mind with.")
                
            texture_amount = reader.read_uint()
            
            texture_paths: list[str] = []
            for _ in range(texture_amount):
                texture_paths.append(reader.read_fixed_string(128, "euc-kr"))
            
            # TextureLayer.texture_id starts at 1 instead of 0
            for texture_id, texture_path in enumerate(texture_paths, start=1):
                print(f"\rLoading textures {texture_id}/{texture_amount}", end="")
                texture_name = os.path.basename(texture_path)
                size = reader.read_uint()
                dds_header = bytearray(r3t_file.read(128))
                if not dds_header[:4] == b'DDS ':
                    dds_header = RFShared.unlock_dds(list(struct.unpack('<32I', dds_header)))
                    dds_header = struct.pack('<32I', *dds_header)
                texture_data = r3t_file.read(size - 128)

                temp_file = tempfile.NamedTemporaryFile(suffix=".dds", delete=False)
                temp_file.write(dds_header)
                temp_file.write(texture_data)
                temp_file.flush()
                blender_image = bpy.data.images.load(temp_file.name)
                blender_image.name = texture_name
                blender_image.pack()
                temp_file.close()
                os.remove(temp_file.name)

                texture_dictionary[texture_id] = texture_name
            print()
            return texture_dictionary
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3T file data: {e}\n{traceback.format_exc()}")
        finally:
            print()
            
    @staticmethod
    def get_light_texture_dictionary_from_r3t_file(directory, file_name_stem) -> dict[int, str]:
        r3t_filepath = os.path.join(directory, f"{file_name_stem}Lgt.r3t")
        try:
            with open(r3t_filepath, 'rb') as r3t_file:
                return RFShared.get_light_texture_dictionary_from_r3t_filestream(r3t_file)
    
        except FileNotFoundError:
            raise FileNotFoundError(f"R3T file not found at: {r3t_filepath}")
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3T file at {r3t_filepath}") from e
        finally:
            print()
            
    @staticmethod
    def get_light_texture_dictionary_from_r3t_filestream(r3t_file: io.BufferedReader) -> dict[int, str]:
        try:
            texture_dictionary: dict = {}
            reader = Utils.Serializer(
                r3t_file,
                Utils.Serializer.Endianness.Little,
                Utils.Serializer.Quaternion_Order.XYZW,
                Utils.Serializer.Matrix_Order.ColumnMajor,
            )
            version = reader.read_float()
            texture_amount = reader.read_uint()

            for texture_id in range(texture_amount):
                print(f"\rLoading textures {texture_id + 1}/{texture_amount}", end="")
                texture_name = f"LightTexture{texture_id}.dds"
                size = reader.read_uint()
                dds_header = bytearray(r3t_file.read(128))

                if not dds_header[:4] == b'DDS ':
                    dds_header = RFShared.unlock_dds(list(struct.unpack('<32I', dds_header)))
                    dds_header = struct.pack('<32I', *dds_header)

                texture_data = r3t_file.read(size - 128)
                width, height = struct.unpack('<II', dds_header[12:20])

                # Decode RGB565
                raw_data = np.frombuffer(texture_data, dtype=np.uint16).reshape((height, width))
                r = ((raw_data & 0xF800) >> 11) << 3
                g = ((raw_data & 0x07E0) >> 5) << 2
                b = (raw_data & 0x001F) << 3
                rgb_data = np.stack((r, g, b), axis=-1).astype(np.uint8)

                # Save as a new DDS
                pil_image = Image.fromarray(rgb_data, mode="RGB")
                with tempfile.NamedTemporaryFile(suffix=".dds", delete=False) as temp_file:
                    pil_image.save(temp_file.name, format="DDS")
                    blender_image = bpy.data.images.load(temp_file.name)
                    blender_image.name = texture_name
                    blender_image.pack()
                    temp_file.close()
                    os.remove(temp_file.name)

                texture_dictionary[texture_id] = texture_name
            print()
            return texture_dictionary
        except Exception as e:
            raise RuntimeError(f"Error while trying to read R3T file data: {e}\n{traceback.format_exc()}")
        finally:
            print()
    
    @staticmethod
    def process_texture_layers(r3m_material: R3MMaterial, material: bpy.types.Material, nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, bsdf: bpy.types.Node, texture_dictionary: dict[int, str], context: bpy.types.Context):
        """
        Given a R3MMaterial and other necessary inputs, organizes and adds the effects of each texture layer within the material.
        """
        specular_value = nodes.new(type="ShaderNodeValue")
        specular_value.label = "Specular"
        links.new(specular_value.outputs[0], bsdf.inputs[13])

        previous_color_output = None
        alpha_output = None
        bump_outputs = []
        uses_alpha = False
        
        texture_coordinates_node = nodes.new(type="ShaderNodeTexCoord")
        
        for texture_layer in r3m_material.texture_layers:
            uvs_output = texture_coordinates_node.outputs["UV"]
            separated_uvs_node = None
            
            image = bpy.data.images[texture_dictionary[texture_layer.texture_id]]
            tex_image_node = nodes.new('ShaderNodeTexImage')
            tex_image_node.image = image

            alpha_value = texture_layer.argb_color[0]
            tex_image_node[MaterialProperties.ARGB_ALPHA.value] = alpha_value
            alpha_direct_output = nodes.new(type="ShaderNodeValue")
            alpha_direct_output.label = f"ARGB Alpha [{image.name}]"
            Utils.create_driver_single(alpha_direct_output.outputs[0], "argb_alpha", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.ARGB_ALPHA.value}"]', "argb_alpha")
            alpha_direct_output.outputs[0].default_value = alpha_value
            
            alpha_output = alpha_direct_output.outputs[0]
            
            texture_is_bump = False
            
            if texture_layer.alpha_type == 1:
                multiply_node = nodes.new(type="ShaderNodeMath")
                multiply_node.operation = 'MULTIPLY'
                multiply_node.label = f"Alpha Multiply [{image.name}]"
                links.new(tex_image_node.outputs['Alpha'], multiply_node.inputs[0])
                links.new(alpha_output, multiply_node.inputs[1])
                alpha_output = multiply_node.outputs[0]
                uses_alpha = True
            elif texture_layer.alpha_type in (2, 3):
                color_ramp_node = nodes.new(type="ShaderNodeValToRGB")
                links.new(tex_image_node.outputs['Color'], color_ramp_node.inputs['Fac'])
                multiply_node = nodes.new(type="ShaderNodeMath")
                multiply_node.operation = 'MULTIPLY'
                multiply_node.label = f"Alpha Multiply [{image.name}]"
                links.new(color_ramp_node.outputs['Color'], multiply_node.inputs[0])
                links.new(alpha_output, multiply_node.inputs[1])
                alpha_output = multiply_node.outputs[0]
                uses_alpha = True
            
            if texture_layer.flags & int(LayerFlag._UV_ENV.value) or texture_layer.flags & int(LayerFlag._MAT_ENV_BUMP.value):
                
                tex_image_node[MaterialProperties.ENVIROMENT_MAT.value] = True
                
                tex_image_node.image.colorspace_settings.name = 'Non-Color'

                bump_multiply_node = nodes.new(type="ShaderNodeVectorMath")
                bump_multiply_node.operation = 'MULTIPLY'
                bump_multiply_node.label = f"Bump Multiply [{image.name}]"
                
                alpha_multiply_node = nodes.new(type="ShaderNodeMath")
                alpha_multiply_node.operation = 'MULTIPLY'
                alpha_multiply_node.label = f"Bump Alpha Multiply [{image.name}]"
                alpha_multiply_node.inputs[1].default_value = 0.0012
                links.new(alpha_output, alpha_multiply_node.inputs[0])
                
                
                links.new(tex_image_node.outputs["Color"], bump_multiply_node.inputs[0])
                links.new(alpha_multiply_node.outputs[0], bump_multiply_node.inputs[1])
                
                bump_outputs.append(bump_multiply_node.outputs[0])
                
                texture_is_bump = True
            
            if texture_layer.flags & int(LayerFlag._UV_METAL.value):
                tex_image_node[MaterialProperties.METAL_EFFECT_SIZE.value] = texture_layer.metal_effect_size
                metal_effect_size = nodes.new(type="ShaderNodeValue")
                metal_effect_size.label = "Metal Effect Size"
                metal_effect_size.label = f"Metal Effect Size [{image.name}]"
                Utils.create_driver_single(metal_effect_size.outputs[0], "metal_effect_size", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.METAL_EFFECT_SIZE.value}"]', "metal_effect_size/255.0")
                
                geometry_node = nodes.new(type="ShaderNodeNewGeometry")
                geometry_node.label = f"Geometry [{image.name}]"
                
                normal_node = nodes.new(type="ShaderNodeVectorMath")
                normal_node.operation = "NORMALIZE"
                normal_node.label = f"Normalize [{image.name}]"
                links.new(geometry_node.outputs["Normal"], normal_node.inputs["Vector"])
                
                normal_separated = nodes.new(type="ShaderNodeSeparateXYZ")
                normal_separated.label = f"Separated Normal [{image.name}]"
                links.new(normal_node.outputs["Vector"], normal_separated.inputs["Vector"])
                
                absolute_node = nodes.new(type="ShaderNodeMath")
                absolute_node.operation = "ABSOLUTE"
                absolute_node.label = f"Absolute [{image.name}]"
                links.new(normal_separated.outputs["Z"], absolute_node.inputs["Value"])
                
                #Greater Than V_S path
                greater_than_node = nodes.new(type="ShaderNodeMath")
                greater_than_node.operation = "GREATER_THAN"
                greater_than_node.label = f"Greater Than [{image.name}]"
                greater_than_node.inputs[1].default_value = 0.98
                links.new(absolute_node.outputs["Value"], greater_than_node.inputs["Value"])
                
                gt_combine_xyz = nodes.new(type="ShaderNodeCombineXYZ")
                gt_combine_xyz.label = f"Combine XYZ (GT) [{image.name}]"
                links.new(greater_than_node.outputs[0], gt_combine_xyz.inputs["X"])
                
                #Less Than V_S path
                less_than_node = nodes.new(type="ShaderNodeMath")
                less_than_node.operation = "LESS_THAN"
                less_than_node.label = f"Less Than [{image.name}]"
                less_than_node.inputs[1].default_value = 0.981
                links.new(absolute_node.outputs["Value"], less_than_node.inputs["Value"])
                
                y_normal_1 = nodes.new(type="ShaderNodeMath")
                y_normal_1.operation = "MULTIPLY"
                y_normal_1.label = f"Y Normal Multiply [{image.name}]"
                links.new(normal_separated.outputs["Y"], y_normal_1.inputs[0])
                links.new(less_than_node.outputs["Value"], y_normal_1.inputs[1])
                
                y_normal_1_flipped = nodes.new(type="ShaderNodeMath")
                y_normal_1_flipped.operation = "MULTIPLY"
                y_normal_1_flipped.label = f"Y Normal Flipped [{image.name}]"
                links.new(y_normal_1.outputs[0], y_normal_1_flipped.inputs[0])
                y_normal_1_flipped.inputs[1].default_value = -1.0
                
                lt_combine_xyz = nodes.new(type="ShaderNodeCombineXYZ")
                lt_combine_xyz.label = f"Combine XYZ (LT) [{image.name}]"
                links.new(y_normal_1_flipped.outputs[0], lt_combine_xyz.inputs["X"])
                
                lt_normalize = nodes.new(type="ShaderNodeVectorMath")
                lt_normalize.operation = "NORMALIZE"
                lt_normalize.label = f"Normalize (LT) [{image.name}]"
                links.new(lt_combine_xyz.outputs[0], lt_normalize.inputs[0])
                # Paths end
                
                v_s = nodes.new(type="ShaderNodeMix")
                v_s.data_type = "VECTOR"
                v_s.label = f"V_S [{image.name}]"
                links.new(gt_combine_xyz.outputs[0], v_s.inputs["A"])
                links.new(lt_normalize.outputs[0], v_s.inputs["B"])
                
                v_t = nodes.new(type="ShaderNodeVectorMath")
                v_t.operation = "CROSS_PRODUCT"
                v_t.label = f"V_T [{image.name}]"
                links.new(v_s.outputs["Result"], v_t.inputs[0])
                links.new(normal_node.outputs[0], v_t.inputs[1])
                
                camera_world_position = nodes.new(type="ShaderNodeVectorTransform")
                camera_world_position.label = f"Camera World Position [{image.name}]"
                camera_world_position.vector_type = "POINT"
                camera_world_position.convert_from = "CAMERA"
                camera_world_position.convert_to = "WORLD"
                camera_world_position.inputs[0].default_value = (0.0, 0.0, 0.0)
                
                #V_S for U operations
                
                v_s_dot_1 = nodes.new(type="ShaderNodeVectorMath")
                v_s_dot_1.operation = "DOT_PRODUCT"
                v_s_dot_1.label = f"V_S Dot 1 [{image.name}]"
                links.new(v_s.outputs["Result"], v_s_dot_1.inputs[0])
                links.new(geometry_node.outputs["Position"], v_s_dot_1.inputs[1])
                
                v_s_dot_2 = nodes.new(type="ShaderNodeVectorMath")
                v_s_dot_2.operation = "DOT_PRODUCT"
                v_s_dot_2.label = f"V_S Dot 2 [{image.name}]"
                links.new(v_s.outputs["Result"], v_s_dot_2.inputs[0])
                links.new(camera_world_position.outputs[0], v_s_dot_2.inputs[1])
                
                v_s_dot_mult = nodes.new(type="ShaderNodeMath")
                v_s_dot_mult.operation = "MULTIPLY"
                v_s_dot_mult.label = f"V_S Dot Multiply [{image.name}]"
                links.new(v_s_dot_2.outputs["Value"], v_s_dot_mult.inputs[0])
                v_s_dot_mult.inputs[1].default_value = 0.8
                
                v_s_dot_sub = nodes.new(type="ShaderNodeMath")
                v_s_dot_sub.operation = "SUBTRACT"
                v_s_dot_sub.label = f"V_S Dot Subtract [{image.name}]"
                links.new(v_s_dot_1.outputs["Value"], v_s_dot_sub.inputs[0])
                links.new(v_s_dot_mult.outputs[0], v_s_dot_sub.inputs[1])
                
                v_s_dot_div = nodes.new(type="ShaderNodeMath")
                v_s_dot_div.operation = "DIVIDE"
                v_s_dot_div.label = f"V_S Dot Divide [{image.name}]"
                links.new(v_s_dot_sub.outputs[0], v_s_dot_div.inputs[0])
                links.new(metal_effect_size.outputs[0], v_s_dot_div.inputs[1])
                
                # V_T for V operations
                v_t_dot_1 = nodes.new(type="ShaderNodeVectorMath")
                v_t_dot_1.operation = "DOT_PRODUCT"
                v_t_dot_1.label = f"V_T Dot 1 [{image.name}]"
                links.new(v_t.outputs[0], v_t_dot_1.inputs[0])
                links.new(geometry_node.outputs["Position"], v_t_dot_1.inputs[1])

                v_t_dot_2 = nodes.new(type="ShaderNodeVectorMath")
                v_t_dot_2.operation = "DOT_PRODUCT"
                v_t_dot_2.label = f"V_T Dot 2 [{image.name}]"
                links.new(v_t.outputs[0], v_t_dot_2.inputs[0])
                links.new(camera_world_position.outputs[0], v_t_dot_2.inputs[1])

                v_t_dot_mult = nodes.new(type="ShaderNodeMath")
                v_t_dot_mult.operation = "MULTIPLY"
                v_t_dot_mult.label = f"V_T Dot Multiply [{image.name}]"
                links.new(v_t_dot_2.outputs["Value"], v_t_dot_mult.inputs[0])
                v_t_dot_mult.inputs[1].default_value = 0.8

                v_t_dot_sub = nodes.new(type="ShaderNodeMath")
                v_t_dot_sub.operation = "SUBTRACT"
                v_t_dot_sub.label = f"V_T Dot Subtract [{image.name}]"
                links.new(v_t_dot_1.outputs["Value"], v_t_dot_sub.inputs[0])
                links.new(v_t_dot_mult.outputs[0], v_t_dot_sub.inputs[1])

                v_t_dot_div = nodes.new(type="ShaderNodeMath")
                v_t_dot_div.operation = "DIVIDE"
                v_t_dot_div.label = f"V_T Dot Divide [{image.name}]"
                links.new(v_t_dot_sub.outputs[0], v_t_dot_div.inputs[0])
                links.new(metal_effect_size.outputs[0], v_t_dot_div.inputs[1])

                final_metal_uv = nodes.new(type="ShaderNodeCombineXYZ")
                final_metal_uv.label = f"Final Metal UV [{image.name}]"
                links.new(v_s_dot_div.outputs[0], final_metal_uv.inputs["X"])
                links.new(v_t_dot_div.outputs[0], final_metal_uv.inputs["Y"])
                
                uvs_output = final_metal_uv.outputs[0]
                
            if texture_layer.flags & int(LayerFlag._UV_ROTATE.value):
                tex_image_node[MaterialProperties.UV_ROTATION.value] = texture_layer.uv_rotation
                uv_rotation = nodes.new(type="ShaderNodeValue")
                uv_rotation.label = f"UV Rotation [{image.name}]"
                Utils.create_driver_single(uv_rotation.outputs[0], "uv_rotation", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.UV_ROTATION.value}"]', "frame/60*(uv_rotation/255.0)")
                
                vec_rot_1 = nodes.new(type="ShaderNodeVectorRotate")
                vec_rot_1.label = f"Rotation Applier [{image.name}]"
                vec_rot_1.rotation_type = "Z_AXIS"
                links.new(uvs_output, vec_rot_1.inputs["Vector"])
                vec_rot_1.inputs["Center"].default_value = Vector((0.5, 0.5, 0.0))
                links.new(uv_rotation.outputs[0], vec_rot_1.inputs["Angle"])
                
                uvs_output = vec_rot_1.outputs[0]
            
            if texture_layer.flags & int(LayerFlag._UV_SCALE.value):
                tex_image_node[MaterialProperties.STARTING_SCALE.value] = texture_layer.uv_starting_scale
                tex_image_node[MaterialProperties.ENDING_SCALE.value] = texture_layer.uv_ending_scale
                tex_image_node[MaterialProperties.SCALE_SPEED.value] = texture_layer.uv_scale_speed
                
                starting_scale_var = f"(starting_scale/255.0)"
                ending_scale_var = f"(ending_scale/255.0)"
                scale_speed_var = f"(scale_speed/255.0)"
                
                
                scale_node = nodes.new(type="ShaderNodeValue")
                scale_node.label = f"Scale Rate [{image.name}]"
                Utils.create_driver_multiple(scale_node.outputs[0], ("starting_scale", "ending_scale", "scale_speed"), [material]*3, 
                                            (f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.STARTING_SCALE.value}"]', 
                                                f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.ENDING_SCALE.value}"]',
                                                f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.SCALE_SPEED.value}"]'), 
                                            f"({ending_scale_var}-{starting_scale_var})*(sin(((frame*40%(1.0/{scale_speed_var}*6000.0))/(1.0/{scale_speed_var}*6000.0))*2*pi)+1)*0.5+{starting_scale_var}")
                
                translation_1 = nodes.new(type="ShaderNodeVectorMath")
                translation_1.operation = 'ADD'
                translation_1.label = f"Translation 1 [{image.name}]"
                links.new(uvs_output, translation_1.inputs[0])
                translation_1.inputs[1].default_value = Vector((-0.5, -0.5, 0))
                
                mult_1 = nodes.new(type="ShaderNodeVectorMath")
                mult_1.operation = 'MULTIPLY'
                mult_1.label = f"Multiply 1 [{image.name}]"
                links.new(translation_1.outputs[0], mult_1.inputs[0])
                links.new(scale_node.outputs[0], mult_1.inputs[1])
                
                translation_2 = nodes.new(type="ShaderNodeVectorMath")
                translation_2.operation = 'ADD'
                translation_2.label = f"Translation 2 [{image.name}]"
                links.new(mult_1.outputs[0], translation_2.inputs[0])
                translation_2.inputs[1].default_value = Vector((0.5, 0.5, 0))
                
                uvs_output = translation_2.outputs[0]
            
            if texture_layer.flags & int(LayerFlag._UV_LAVA.value):
                tex_image_node[MaterialProperties.LAVA_WAVE_RATE.value] = texture_layer.lava_wave_effect_rate
                tex_image_node[MaterialProperties.LAVA_WAVE_SPEED.value] = texture_layer.lava_wave_effect_speed
                if separated_uvs_node is None:
                    separated_uvs_node = nodes.new(type="ShaderNodeSeparateXYZ")
                    separated_uvs_node.label = f"Separate UVs [{image.name}]"
                    links.new(uvs_output, separated_uvs_node.inputs[0])
                    
                combine_uvs_node = nodes.new(type="ShaderNodeCombineXYZ")
                combine_uvs_node.label = f"Combine UVs [{image.name}]"
                
                
                lava_wave_speed_node = nodes.new(type="ShaderNodeValue")
                lava_wave_speed_node.label = f"Lava Wave Speed [{image.name}]"
                Utils.create_driver_single(lava_wave_speed_node.outputs[0], "lava_wave_speed", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.LAVA_WAVE_RATE.value}"]', "frame/60*(lava_wave_speed/255.0)*3")
                lava_wave_rate_node = nodes.new(type="ShaderNodeValue")
                lava_wave_rate_node.label = f"Lava Wave Rate [{image.name}]"
                Utils.create_driver_single(lava_wave_rate_node.outputs[0], "lava_wave_rate", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.LAVA_WAVE_SPEED.value}"]', "(lava_wave_rate/255.0)/8.0")
                
                # U
                u_1 = nodes.new(type="ShaderNodeMath")
                u_1.label = "U_1"
                u_1.operation = 'MULTIPLY'
                links.new(separated_uvs_node.outputs["Y"], u_1.inputs[0])
                u_1.inputs[1].default_value = 20.0
                
                u_2 = nodes.new(type="ShaderNodeMath")
                u_2.label = "U_2"
                u_2.operation = 'ADD'
                links.new(lava_wave_speed_node.outputs[0], u_2.inputs[0])
                links.new(u_1.outputs[0], u_2.inputs[1])
                
                u_3 = nodes.new(type="ShaderNodeMath")
                u_3.label = "U_3"
                u_3.operation = 'SINE'
                links.new(u_2.outputs[0], u_3.inputs[0])
                
                u_4 = nodes.new(type="ShaderNodeMath")
                u_4.label = "U_4"
                u_4.operation = 'MULTIPLY'
                links.new(u_3.outputs[0], u_4.inputs[0])
                u_4.inputs[1].default_value = 0.15
                
                u_5 = nodes.new(type="ShaderNodeMath")
                u_5.label = "U_5"
                u_5.operation = 'MULTIPLY'
                links.new(u_4.outputs[0], u_5.inputs[0])
                links.new(lava_wave_rate_node.outputs[0], u_5.inputs[1])
                
                u_6 = nodes.new(type="ShaderNodeMath")
                u_6.label = "U_6"
                u_6.operation = 'ADD'
                links.new(u_5.outputs[0], u_6.inputs[0])
                links.new(separated_uvs_node.outputs["X"], u_6.inputs[1])
                
                links.new(u_6.outputs[0], combine_uvs_node.inputs["X"])
                
                # V
                v_1 = nodes.new(type="ShaderNodeMath")
                v_1.label = "V_1"
                v_1.operation = 'MULTIPLY'
                links.new(separated_uvs_node.outputs["X"], v_1.inputs[0])
                v_1.inputs[1].default_value = 20.0
                
                v_2 = nodes.new(type="ShaderNodeMath")
                v_2.label = "V_2"
                v_2.operation = 'ADD'
                links.new(lava_wave_speed_node.outputs[0], v_2.inputs[0])
                links.new(v_1.outputs[0], v_2.inputs[1])
                
                v_3 = nodes.new(type="ShaderNodeMath")
                v_3.label = "V_3"
                v_3.operation = 'COSINE'
                links.new(v_2.outputs[0], v_3.inputs[0])
                
                v_4 = nodes.new(type="ShaderNodeMath")
                v_4.label = "V_4"
                v_4.operation = 'MULTIPLY'
                links.new(v_3.outputs[0], v_4.inputs[0])
                v_4.inputs[1].default_value = 0.14
                
                v_5 = nodes.new(type="ShaderNodeMath")
                v_5.label = "V_5"
                v_5.operation = 'MULTIPLY'
                links.new(v_4.outputs[0], v_5.inputs[0])
                links.new(lava_wave_rate_node.outputs[0], v_5.inputs[1])
                
                v_6 = nodes.new(type="ShaderNodeMath")
                v_6.label = "V_6"
                v_6.operation = 'ADD'
                links.new(v_5.outputs[0], v_6.inputs[0])
                links.new(separated_uvs_node.outputs["Y"], v_6.inputs[1])
                
                links.new(v_6.outputs[0], combine_uvs_node.inputs["Y"])
                
                effect_adder_node = nodes.new(type="ShaderNodeVectorMath")
                effect_adder_node.label = f"Lava Wave Addition [{image.name}]"
                effect_adder_node.operation = 'ADD'
                links.new(uvs_output, effect_adder_node.inputs[0])
                links.new(combine_uvs_node.outputs['Vector'], effect_adder_node.inputs[1])
                uvs_output = effect_adder_node.outputs[0]
            
            if texture_layer.flags & int(LayerFlag._UV_SCROLL_U.value) or texture_layer.flags & int(LayerFlag._UV_SCROLL_V.value):
                combine_uvs_node = nodes.new(type="ShaderNodeCombineXYZ")
                combine_uvs_node.label = f"Combine UVs (Scroll) [{image.name}]"
                if texture_layer.flags & int(LayerFlag._UV_SCROLL_U.value):
                    tex_image_node[MaterialProperties.SCROLL_U.value] = texture_layer.scroll_u
                    scroll_u_node = nodes.new(type="ShaderNodeValue")
                    scroll_u_node.label = f"Scroll U [{image.name}]"
                    Utils.create_driver_single(scroll_u_node.outputs[0], "scroll_u", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.SCROLL_U.value}"]', "frame/60/4.0*(scroll_u/255.0)")
                    links.new(scroll_u_node.outputs[0], combine_uvs_node.inputs["X"])
                    
                if texture_layer.flags & int(LayerFlag._UV_SCROLL_V.value):
                    tex_image_node[MaterialProperties.SCROLL_V.value] = texture_layer.scroll_v
                    scroll_v_node = nodes.new(type="ShaderNodeValue")
                    scroll_v_node.label = f"Scroll V [{image.name}]"
                    Utils.create_driver_single(scroll_v_node.outputs[0], "scroll_v", material, f'node_tree.nodes["{tex_image_node.name}"]["{MaterialProperties.SCROLL_V.value}"]', "frame/60/4.0*(scroll_v/255.0)")
                    links.new(scroll_v_node.outputs[0], combine_uvs_node.inputs["Y"])
                
                effect_adder_node = nodes.new(type="ShaderNodeVectorMath")
                effect_adder_node.label = f"Scroll UV Addition [{image.name}]"
                effect_adder_node.operation = 'ADD'
                links.new(uvs_output, effect_adder_node.inputs[0])
                links.new(combine_uvs_node.outputs['Vector'], effect_adder_node.inputs[1])
                uvs_output = effect_adder_node.outputs[0]
            
            if uvs_output is not None:
                links.new(uvs_output, tex_image_node.inputs['Vector'])
            
            if texture_is_bump == False:
                if previous_color_output is None:
                    # If this is the first texture, connect it directly to the BSDF
                    links.new(tex_image_node.outputs['Color'], bsdf.inputs['Base Color'])
                    previous_color_output = tex_image_node.outputs['Color']
                    links.new(alpha_output, bsdf.inputs['Alpha'])
                else:
                    mix_rgb = nodes.new('ShaderNodeMixRGB')
                    mix_rgb.label = f"Mix RGB [{image.name}]"
                    mix_rgb.blend_type = 'MIX'
                    mix_rgb.inputs['Fac'].default_value = 0.5
                    
                    multiply_mix_alpha_node = nodes.new(type="ShaderNodeMath")
                    multiply_mix_alpha_node.operation = 'MULTIPLY'
                    multiply_mix_alpha_node.label = f"Mix Alpha Multiply [{image.name}]"
                    links.new(alpha_output, multiply_mix_alpha_node.inputs[0])
                    multiply_mix_alpha_node.inputs[1].default_value = 0.5
                    
                    links.new(mix_rgb.inputs[0], multiply_mix_alpha_node.outputs[0])
                    links.new(mix_rgb.inputs[1], previous_color_output)
                    links.new(mix_rgb.inputs[2], tex_image_node.outputs['Color'])
                    

                    previous_color_output = mix_rgb.outputs['Color']
                
                if previous_color_output is not None:
                    links.new(bsdf.inputs['Base Color'], previous_color_output)
        
        for bump_output in bump_outputs:
            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.image.colorspace_settings.name != 'Non-Color':
                    if node.inputs["Vector"].is_linked:
                        
                        previous_uv_input = node.inputs["Vector"].links[0].from_socket

                        mapping_node = nodes.new(type="ShaderNodeMapping")

                        links.new(previous_uv_input, mapping_node.inputs["Vector"])

                        links.new(bump_output, mapping_node.inputs["Rotation"])

                        links.new(mapping_node.outputs["Vector"], node.inputs["Vector"])
        
        organizer = Utils.NodeOrganizer()
        organizer.arrange_nodes(context, material.node_tree, 300, 300, True)
        
        if uses_alpha == True:
            material.blend_method = 'BLEND'
    
    @staticmethod
    def convert_vector3s_to_f(vector3s_collection, scale: float, position: Vector)-> Vector:
        """
        Converts a given vector3short to a float vector.
        """
        return Vector(((vector3s_collection[0]/32767.0)*scale+position[0], (vector3s_collection[1]/32767.0)*scale+position[1], (vector3s_collection[2]/32767.0)*scale+position[2]))
    
    @staticmethod
    def convert_vector3c_to_f(vector3c_collection, scale: float, position: Vector)-> Vector:
        """
        Converts a given vector3char to a float vector.
        """
        return Vector(((vector3c_collection[0]/127.0)*scale+position[0], (vector3c_collection[1]/127.0)*scale+position[1], (vector3c_collection[2]/127.0)*scale+position[2]))
    
    @staticmethod
    def arrange_nodes(node_tree, start_x=0, start_y=0, x_offset=200, y_offset=200):
        """
        Arrange nodes in the node tree with specified offsets.
        
        :param node_tree: The node tree containing the nodes.
        :param start_x: The starting X position for the first node.
        :param start_y: The starting Y position for the first node.
        :param x_offset: The horizontal space between nodes.
        :param y_offset: The vertical space between nodes.
        """
        x = start_x
        y = start_y

        for node in node_tree.nodes:
            node.location = (x, y)
            y -= y_offset

            # If the node has an output, place the connected nodes next to it
            for output in node.outputs:
                if output.is_linked:
                    for link in output.links:
                        linked_node = link.to_node
                        linked_node.location = (x + x_offset, y)
                        y -= y_offset

    
class R3MMaterial:
    """
    k
    """
    def __init__(self):
        self.layer_num: int = 0
        self.flag: int = 0
        self.detail_surface: int = 0
        self.detail_scale: float = 0.0
        self.name: str = ""
        self.texture_layers: list[TextureLayer] = []

class MaterialGroup:
    """
    k
    """
    def __init__(
        self,
        attribute: int = 0,
        number_of_faces: int = 0,
        starting_face_id: int = 0,
        material_id: int = 0,
        light_id: int = 0,
        bounding_box_min: Vector3Int = Vector3Int(0, 0, 0),
        bounding_box_max: Vector3Int = Vector3Int(0, 0, 0),
        position: Vector = Vector((0.0, 0.0, 0.0)),
        scale: float = 1.0,
        animated_object_id: int = 0
    ):
        self.attribute: int = attribute
        self.number_of_faces: int = number_of_faces
        self.starting_face_id: int = starting_face_id
        self.material_id: int = material_id
        self.light_id: int = light_id
        self.bounding_box_min: Vector3Int = bounding_box_min
        self.bounding_box_max: Vector3Int = bounding_box_max
        self.position: Vector = position
        self.scale: float = scale
        self.animated_object_id: int = animated_object_id
    
    @staticmethod
    def bsp_material_from_unpacked_bytes(unpacked_material_group) -> "MaterialGroup":
        material_group = MaterialGroup()
        material_group.attribute = unpacked_material_group[0]
        material_group.number_of_faces = unpacked_material_group[1]
        material_group.starting_face_id = unpacked_material_group[2]
        material_group.material_id = unpacked_material_group[3]
        material_group.light_id = unpacked_material_group[4]
        material_group.bounding_box_min = Vector3Int(unpacked_material_group[5], unpacked_material_group[6], unpacked_material_group[7])
        material_group.bounding_box_max = Vector3Int(unpacked_material_group[8], unpacked_material_group[9], unpacked_material_group[10])
        material_group.position = Vector((unpacked_material_group[11], unpacked_material_group[12], unpacked_material_group[13]))
        material_group.scale = unpacked_material_group[14]
        material_group.animated_object_id = unpacked_material_group[15]
        return material_group
    
    @staticmethod
    def r3e_material_from_unpacked_bytes(unpacked_material_group) -> "MaterialGroup":
        material_group = MaterialGroup()
        material_group.number_of_faces = unpacked_material_group[0]
        material_group.starting_face_id = unpacked_material_group[1]
        material_group.material_id = unpacked_material_group[2]
        material_group.animated_object_id = unpacked_material_group[3]
        material_group.bounding_box_min = Vector((unpacked_material_group[4], unpacked_material_group[5], unpacked_material_group[6]))
        material_group.bounding_box_max = Vector((unpacked_material_group[7], unpacked_material_group[8], unpacked_material_group[9]))
        return material_group
      
        
class TextureLayer:
    """
    Class that holds data for each texture a material has.
    """
    """
    # also, each layer signifies a new texture for a certain material.
    # dwFlag for layers flags:
    
    

    dwAlphaType flags:
    blend_none: 0 opaque
    group_type_reflect: 0x4000 reflective
    """
    def __init__(self):
        self.iTileAniTexNum: int = 0
        self.texture_id: int = 0
        self.alpha_type: int = 0
        self.argb_color: Vector = Vector((0.0, 0.0, 0.0, 0.0))
        self.flags: int = 0
        self.lava_wave_effect_rate: int = 0
        self.lava_wave_effect_speed: int = 0
        self.scroll_u: int = 0
        self.scroll_v: int = 0
        self.uv_rotation: int = 0
        self.uv_starting_scale: int = 0
        self.uv_ending_scale: int = 0
        self.uv_scale_speed: int = 0
        self.metal_effect_size: int = 0
        self.alpha_flicker_rate: int = 0
        self.alpha_flicker_animation: int = 0
        self.animated_texture_frame: int = 0
        self.animated_texture_speed: int = 0
        self.gradient_alpha: int = 0

    def get_texture_layer_from_unpacked_bytes(unpacked_layer) -> "TextureLayer":
        layer = TextureLayer()
        layer.iTileAniTexNum = unpacked_layer[0]
        layer.texture_id = unpacked_layer[1]
        layer.alpha_type = unpacked_layer[2]
        argb = unpacked_layer[3]
        a = ((argb >> 24) & 0xFF)/255.0
        r = ((argb >> 16) & 0xFF)/255.0
        g = ((argb >> 8) & 0xFF)/255.0
        b = (argb & 0xFF)/255.0
        layer.argb_color = Vector((a, r, g, b))
        layer.flags = unpacked_layer[4]
        layer.lava_wave_effect_rate = unpacked_layer[5]
        layer.lava_wave_effect_speed = unpacked_layer[6]
        layer.scroll_u = unpacked_layer[7]
        # Different coordinates system
        layer.scroll_v = -unpacked_layer[8]
        layer.uv_rotation = -unpacked_layer[9]
        layer.uv_starting_scale = unpacked_layer[10]
        layer.uv_ending_scale = unpacked_layer[11]
        layer.uv_scale_speed = unpacked_layer[12]
        layer.metal_effect_size = unpacked_layer[13]
        layer.alpha_flicker_rate = unpacked_layer[14]
        layer.alpha_flicker_animation = unpacked_layer[15]
        layer.animated_texture_frame = unpacked_layer[16]
        layer.animated_texture_speed = unpacked_layer[17]
        layer.gradient_alpha = unpacked_layer[18]
        return layer

class AnimatedObject:
    """
    Class that holds data for materials that have animations.
    """
    def __init__(self):
        self.flag: int = 0
        self.parent: int = 0
        self.frames: int = 0
        self.pos_count: int = 0
        self.rot_count: int = 0
        self.scale_count: int = 0
        self.scale: Vector = Vector((1.0, 1.0, 1.0))
        self.scale_quat: Quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        self.pos: Vector = Vector((0.0, 0.0, 0.0))
        self.quat: Quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        self.pos_offset: int = 0
        self.rot_offset: int = 0
        self.scale_offset: int = 0

class ReadFaceStruct:
    def __init__(self, vertex_amount, vertex_start_id, material_id = -1):
        self.vertex_amount:int = vertex_amount
        self.vertex_start_id:int = vertex_start_id
        self.material_id:int = material_id
        
class ReadEntityStruct:
    def __init__(self, id, scale, position, rot_x, rot_y, bb_min, bb_max):
        self.id: int = id
        self.scale: float = scale
        self.position: Vector = position
        self.rot_x: float = rot_x
        self.rot_y: float = rot_y
        self.bb_min: Vector = bb_min
        self.bb_max: Vector = bb_max
        
class EntityStruct:
    def __init__(self, is_particle, is_file_exists, file_path, fade_start, fade_end, flag, shader_id, factors):
        self.is_particle: bool = True if is_particle != 0 else False
        self.is_file_exists: bool = True if is_file_exists != 0 else False
        self.file_path: str = file_path
        self.fade_start: float = fade_start
        self.fade_end: float = fade_end
        self.flag: int = flag
        self.shader_id: int = shader_id
        self.factors: Tuple[int, int] = factors
        
class EntityRPKIndices:
    def __init__(self, key_name, r3e_index = None, r3m_index = None, r3t_index = None):
        self.key_name = key_name
        self.r3e_index = r3e_index
        self.r3m_index = r3m_index
        self.r3t_index = r3t_index
        
    def is_valid(self) -> bool:
        if self.r3e_index is not None and self.r3m_index is not None and self.r3t_index is not None:
            return True
        return False

class MaterialProperties(Enum):
    ARGB_ALPHA = "argb_alpha"
    METAL_EFFECT_SIZE = "metal_effect_size"
    ENVIROMENT_MAT = "enviroment_mat"
    UV_ROTATION = "uv_rotation"
    STARTING_SCALE = "starting_scale"
    ENDING_SCALE = "ending_scale"
    SCALE_SPEED = "scale_speed"
    LAVA_WAVE_RATE = "lava_wave_rate"
    LAVA_WAVE_SPEED = "lava_wave_speed"
    SCROLL_U = "scroll_u"
    SCROLL_V = "scroll_v"

class LayerFlag(Enum):
    ALPHA_SORT_ON		= 0x80000000


    _UV_ENV					= 0x00000001	
    #//_UV_WATER				= 0x00000002
    _UV_LAVA				= 0x00000004
    _UV_METAL_FLOOR			= 0x00000002 # makes object flicker
    _UV_METAL_WALL			= 0x00000008
    _UV_METAL				= 0x0000000a#	//= 0x2 or = 0x8
    _UV_SCROLL_U			= 0x00000010
    _UV_SCROLL_V			= 0x00000020
    _UV_ROTATE				= 0x00000040
    _UV_SCALE				= 0x00000080#	//The scale itself is anime.
    # the = 0x300 below is simply a combination of U and V. If a flag has either ALPHA U or V and you use & you will see if the mat has alpha gradient
    _UV_GRADIENT_ALPHA_UV	= 0x00000300#	//Gladiating the alpha.
    _UV_GRADIENT_ALPHA_U	= 0x00000100#	//Gladiating the alpha.
    _UV_GRADIENT_ALPHA_V	= 0x00000200#	//Gladiating the alpha.

    _ANI_ALPHA_FLICKER		= 0x00000400#	//alpha flashing, literally.
    _ANI_TEXTURE			= 0x00000800#  //Texture animation...
    _ANI_TILE_TEXTURE		= 0x00001000#  //Tile texture animation...

    _MAT_ZWRITE				= 0x00002000#	//Z light the material. The default is not to do it.
    _MAT_ENV_BUMP			= 0x00008000#	//Env bump mapping.(texture is treated as bump)
    #//_MAT_VOLUME_FOG		= 0x00008000#	//The material is volume fog.
    _MAT_TEX_CLAMP			= 0x00010000#	//Texture clamp option.


    _MAT_NO_COLLISON		= 0x00020000#	//Polygons using this material do not perform any collision checks. //Use layer flag 0.
    _MAT_WATER				= 0x00040000#	//Polygons using this material have a water effect.

class AlphaType(Enum):
    NONE	= 0
    DIRECT = 1
    BLACK_ALPHA = 2
    METAL_ALPHA = 3

def register():
    bpy.utils.register_class(RFShared)

def unregister():
    bpy.utils.unregister_class(RFShared)

if __name__ == "__main__":
    register()
