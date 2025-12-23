"""
Vision preprocessing for Qwen2-VL models.
Handles image loading, resizing, and conversion to visual tokens.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import requests
import torch
from PIL import Image
from torchvision import transforms


class VisionProcessor:
    """
    Processes images for Qwen2-VL models.
    
    Handles:
    - Loading images from URLs, file paths, or base64 strings
    - Smart resizing to fit within pixel constraints while maintaining aspect ratio
    - Normalization and conversion to tensors
    """
    
    def __init__(
        self,
        min_pixels: int = 256 * 28 * 28,  # ~200K pixels
        max_pixels: int = 1280 * 28 * 28,  # ~1M pixels
        patch_size: int = 14,  # Qwen2-VL uses 14x14 patches
    ):
        """
        Initialize vision processor.
        
        Args:
            min_pixels: Minimum number of pixels for resizing
            max_pixels: Maximum number of pixels for resizing
            patch_size: Size of vision patches (must match model architecture)
        """
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        
        # Standard ImageNet normalization used by Qwen2-VL
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
    def load_image(self, image_input: Union[str, Path, bytes]) -> Image.Image:
        """
        Load image from various sources.
        
        Args:
            image_input: Can be:
                - HTTP/HTTPS URL
                - File path (with or without file:// prefix)
                - Base64 encoded string (data:image/...)
                - Raw bytes
        
        Returns:
            PIL Image in RGB mode
        """
        if isinstance(image_input, (str, Path)):
            image_str = str(image_input)
            
            if image_str.startswith('http://') or image_str.startswith('https://'):
                # Load from URL
                try:
                    response = requests.get(image_str, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                except Exception as e:
                    raise ValueError(f"Failed to load image from URL {image_str}: {e}")
                    
            elif image_str.startswith('data:image'):
                # Base64 encoded image
                try:
                    # Format: data:image/jpeg;base64,<base64_data>
                    base64_data = image_str.split(',', 1)[1]
                    image_data = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(image_data))
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 image: {e}")
                    
            elif image_str.startswith('file://'):
                # File path with file:// prefix
                file_path = image_str[7:]
                try:
                    image = Image.open(file_path)
                except Exception as e:
                    raise ValueError(f"Failed to load image from {file_path}: {e}")
            else:
                # Regular file path
                try:
                    image = Image.open(image_str)
                except Exception as e:
                    raise ValueError(f"Failed to load image from {image_str}: {e}")
        else:
            # Raw bytes
            try:
                image = Image.open(BytesIO(image_input))
            except Exception as e:
                raise ValueError(f"Failed to load image from bytes: {e}")
        
        # Convert to RGB (handle RGBA, grayscale, etc.)
        return image.convert('RGB')
    
    def smart_resize(
        self,
        image: Image.Image,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ) -> Image.Image:
        """
        Resize image intelligently to fit within constraints.
        
        Two modes:
        1. Dynamic: Resize to fit within min/max pixels while maintaining aspect ratio
        2. Fixed: Resize to exact dimensions (rounded to patch_size multiples)
        
        Args:
            image: Input PIL image
            min_pixels: Minimum total pixels (overrides instance default)
            max_pixels: Maximum total pixels (overrides instance default)
            resized_height: Specific height to resize to (optional)
            resized_width: Specific width to resize to (optional)
        
        Returns:
            Resized PIL image with dimensions as multiples of patch_size
        """
        # Mode 2: Fixed dimensions
        if resized_height is not None and resized_width is not None:
            # Round to nearest patch_size multiple
            new_height = (resized_height // self.patch_size) * self.patch_size
            new_width = (resized_width // self.patch_size) * self.patch_size
            
            # Ensure at least one patch
            new_height = max(new_height, self.patch_size)
            new_width = max(new_width, self.patch_size)
            
            if (new_width, new_height) != image.size:
                image = image.resize((new_width, new_height), Image.BICUBIC)
            
            return image
        
        # Mode 1: Dynamic resizing
        min_pixels = min_pixels or self.min_pixels
        max_pixels = max_pixels or self.max_pixels
        
        width, height = image.size
        current_pixels = width * height
        
        # Calculate scaling factor
        if current_pixels > max_pixels:
            scale = (max_pixels / current_pixels) ** 0.5
        elif current_pixels < min_pixels:
            scale = (min_pixels / current_pixels) ** 0.5
        else:
            scale = 1.0
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Round to patch_size multiples
        new_width = (new_width // self.patch_size) * self.patch_size
        new_height = (new_height // self.patch_size) * self.patch_size
        
        # Ensure at least one patch in each dimension
        new_width = max(new_width, self.patch_size)
        new_height = max(new_height, self.patch_size)
        
        # Resize if dimensions changed
        if (new_width, new_height) != (width, height):
            image = image.resize((new_width, new_height), Image.BICUBIC)
        
        return image
    
    def preprocess_image(
        self,
        image: Image.Image,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Convert PIL image to normalized tensor.
        
        Args:
            image: PIL Image in RGB mode
            device: Target device for tensor
            dtype: Data type for tensor
        
        Returns:
            Tensor of shape [1, 3, height, width] with normalized values
        """
        # Convert to tensor [3, H, W] with values in [0, 1]
        tensor = transforms.ToTensor()(image)
        
        # Normalize using ImageNet statistics
        tensor = self.normalize(tensor)
        
        # Add batch dimension [1, 3, H, W]
        tensor = tensor.unsqueeze(0)
        
        # Move to device and convert dtype
        tensor = tensor.to(device=device, dtype=dtype)
        
        return tensor
    
    def process_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        End-to-end image processing: load, resize, and convert to tensor.
        
        Args:
            image_input: Image source (URL, path, bytes, or PIL Image)
            device: Target device
            dtype: Tensor data type
            min_pixels: Minimum pixels for resizing
            max_pixels: Maximum pixels for resizing
            resized_height: Fixed height (optional)
            resized_width: Fixed width (optional)
        
        Returns:
            Tuple of (tensor, (height, width))
        """
        # Load image if not already a PIL Image
        if isinstance(image_input, Image.Image):
            image = image_input
        else:
            image = self.load_image(image_input)
        
        # Resize
        image = self.smart_resize(
            image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            resized_height=resized_height,
            resized_width=resized_width,
        )
        
        # Get dimensions after resize
        width, height = image.size
        
        # Convert to tensor
        tensor = self.preprocess_image(image, device, dtype)
        
        return tensor, (height, width)
    
    def get_num_patches(self, height: int, width: int) -> int:
        """
        Calculate number of patches for given image dimensions.
        
        Args:
            height: Image height (should be multiple of patch_size)
            width: Image width (should be multiple of patch_size)
        
        Returns:
            Total number of patches
        """
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        return num_patches_h * num_patches_w


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    processor = VisionProcessor()
    
    # Test with a sample image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a public test image
        image_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    print(f"Loading image from: {image_path}")
    
    # Load image
    image = processor.load_image(image_path)
    print(f"Original size: {image.size}")
    
    # Resize
    resized = processor.smart_resize(image)
    print(f"Resized to: {resized.size}")
    
    # Calculate patches
    num_patches = processor.get_num_patches(resized.size[1], resized.size[0])
    print(f"Number of patches: {num_patches}")
    
    # Convert to tensor
    tensor, (h, w) = processor.process_image(
        image_path,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
