import os
import subprocess
import tempfile
from typing import Optional, Tuple, List
import bpy
import numpy as np
import struct
from enum import Enum

class TextureProcessingError(Exception):
    """Custom exception for texture processing errors"""
    pass

def check_imagemagick() -> bool:
    """Check if ImageMagick is installed and accessible"""
    try:
        from wand.image import Image
        with Image() as img:
            return True
    except Exception:
        return False

            
def ensure_dependencies() -> bool:
    if not check_imagemagick():
        raise TextureProcessingError(
            "ImageMagick is not installed or not accessible.\n"
            "Please install ImageMagick from https://imagemagick.org/script/download.php\n"
            "Make sure to check 'Install legacy utilities' during installation."
        )

    return True
    

def get_dxt_format(image: bpy.types.Image) -> str:
    """Determine the best DXT format for the image"""
    # Check if image has alpha
    has_alpha = image.channels == 4
    
    # Check if it's a normal map (based on name convention)
    is_normal = any(x in image.name.lower() for x in ['normal', 'nrm', 'norm'])
    
    if is_normal:
        return 'dxt5'  # Best for normal maps
    elif has_alpha:
        return 'dxt3'  # For images with alpha
    else:
        return 'dxt1'  # For RGB images

def convert_to_dds(image: bpy.types.Image) -> Optional[bytes]:
    """Convert an image to DDS format with appropriate compression"""
    from wand.image import Image
    import numpy as np
    
    if image.size[0] == 0 or image.size[1] == 0:
        raise TextureProcessingError(f"Invalid image size for {image.name}")
    
    # Create temporary files for conversion
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_in, \
         tempfile.NamedTemporaryFile(suffix='.dds', delete=False) as temp_out:
        
        # Get image data
        pixels = np.array(image.pixels[:])
        width, height = image.size
        rgba = (pixels.reshape(height, width, 4) * 255).astype(np.uint8)
        
        # Save as PNG first (Wand works better with files)
        with Image.from_array(rgba) as img:
            img.save(filename=temp_in.name)
        
        # Convert to DDS using ImageMagick command-line
        dxt_format = get_dxt_format(image)
        convert_cmd = [
            'magick',
            'convert',
            temp_in.name,
            '-quality', '100',
            '-depth', '8',
            '-flip',
            '-define',
            f'dds:compression={dxt_format}',
            '-define',
            'dds:mipmaps=6',  # Disable mipmaps
            temp_out.name
        ]
        
        try:
            subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"ImageMagick conversion failed: {e.stderr}")
            return None
        
        # Read the resulting DDS file
        try:
            with open(temp_out.name, 'rb') as f:
                dds_data = f.read()
        except IOError as e:
            print(f"Failed to read converted DDS file: {e}")
            return None
        finally:
            # Cleanup temporary files
            try:
                os.unlink(temp_in.name)
                os.unlink(temp_out.name)
            except OSError:
                pass
        
        return dds_data

class D3DFormat(Enum):
    R5G6B5 = "D3DFMT_R5G6B5"
    A8R8G8B8 = "D3DFMT_A8R8G8B8"

def convert_to_dds_with_format(image: bpy.types.Image, format: D3DFormat = D3DFormat.R5G6B5) -> Optional[bytes]:
    """
    Convert an image to DDS format with specified compression.
    Supports D3DFMT_R5G6B5 and D3DFMT_A8R8G8B8.
    
    :param image: The Blender image to convert.
    :param format: The target format (D3DFMT_R5G6B5 or D3DFMT_A8R8G8B8).
    :return: The DDS file data as bytes.
    """
    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Invalid image size for {image.name}")
    
    width, height = image.size
    pixels = np.array(image.pixels[:])
    rgba = (pixels.reshape(height, width, 4) * 255).astype(np.uint8)

    # Flip the Y-axis of the image
    rgba = rgba[::-1, ...]

    
    if format == D3DFormat.R5G6B5:
        # Convert RGBA to RGB565
        r = (rgba[..., 0] >> 3).astype(np.uint16) << 11
        g = (rgba[..., 1] >> 2).astype(np.uint16) << 5
        b = (rgba[..., 2] >> 3).astype(np.uint16)
        rgb565 = (r | g | b).flatten().tobytes()
    elif format == D3DFormat.A8R8G8B8:
        # Convert RGBA to A8R8G8B8 (direct mapping)
        argb = np.zeros((height, width, 4), dtype=np.uint8)
        argb[..., 0] = rgba[..., 3]  # Alpha
        argb[..., 1] = rgba[..., 0]  # Red
        argb[..., 2] = rgba[..., 1]  # Green
        argb[..., 3] = rgba[..., 2]  # Blue
        rgb565 = argb.flatten().tobytes()
    else:
        raise ValueError(f"Unsupported format: {format}")

    dds_pixel_format = (
        32,                      # Size of Pixel Format
        0x40 if format == D3DFormat.R5G6B5 else 0x1,  # Flags (DDPF_RGB | DDPF_ALPHAPIXELS)
        b'\0\0\0\0' if format == D3DFormat.R5G6B5 else b'8888',  # FourCC for uncompressed
        16 if format == D3DFormat.R5G6B5 else 32,      # RGBBitCount
        0xF800 if format == D3DFormat.R5G6B5 else 0xFF0000,  # RBitMask
        0x07E0 if format == D3DFormat.R5G6B5 else 0xFF00,    # GBitMask
        0x001F if format == D3DFormat.R5G6B5 else 0xFF,      # BBitMask
        0x0000 if format == D3DFormat.R5G6B5 else 0xFF000000  # AlphaBitMask (only for A8R8G8B8)
    )
    dds_pixel_format_packed = struct.pack('<2I4s5I', *dds_pixel_format)
    print(f"Managed to pack the dds pixel format, error is after.")
    # Create DDS header
    dds_header = struct.pack(
        '<4s18I32s5I',
        b'DDS ',                     # Magic number
        124,                         # Size of header
        0xA1007,                     # Flags (DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT)
        height,                      # Height
        width,                       # Width
        width * height // (2 if format == D3DFormat.R5G6B5 else 4),  # Pitch or linear size
        0,                           # Depth
        1,                           # MipMapCount
        *[0] * 11,                   # Reserved
        dds_pixel_format_packed,                           # Pixel Format
        0x401008,                      # Caps (DDSCAPS_TEXTURE)
        0, 0, 0,                      # Reserved Caps
        0
    )

    # Combine header and pixel data
    dds_data = dds_header + rgb565

    return dds_data

def convert_bytes_to_dds(image_data: bytes, width: int, height: int, dxt_format: str = "dxt1") -> Optional[bytes]:
    """Convert raw image bytes to DDS format with appropriate compression."""
    from wand.image import Image
    import numpy as np

    if width == 0 or height == 0:
        raise TextureProcessingError(f"Invalid image dimensions: {width}x{height}")
    
    # Create temporary files for conversion
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_in, \
         tempfile.NamedTemporaryFile(suffix='.dds', delete=False) as temp_out:
        
        # Convert raw image bytes to PNG format using PIL or numpy
        try:
            rgba = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 4)
        except ValueError as e:
            raise TextureProcessingError(f"Invalid image data format: {e}")
        
        # Save as PNG first (ImageMagick works better with files)
        with Image.from_array(rgba) as img:
            img.save(filename=temp_in.name)
        
        # Convert to DDS using ImageMagick command-line
        convert_cmd = [
            'magick',
            'convert',
            temp_in.name,
            '-quality', '100',
            '-depth', '8',
            '-flip',  # Flip vertically to handle orientation issues
            '-define',
            f'dds:compression={dxt_format}',
            '-define',
            'dds:mipmaps=6',  # Generate mipmaps
            temp_out.name
        ]
        
        try:
            subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"ImageMagick conversion failed: {e.stderr}")
            return None
        
        # Read the resulting DDS file
        try:
            with open(temp_out.name, 'rb') as f:
                dds_data = f.read()
        except IOError as e:
            print(f"Failed to read converted DDS file: {e}")
            return None
        finally:
            # Cleanup temporary files
            try:
                os.unlink(temp_in.name)
                os.unlink(temp_out.name)
            except OSError:
                pass
        
        return dds_data
