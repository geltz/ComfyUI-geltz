import torch
import numpy as np

class StructuredLatent:

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "width": ("INT", {
          "default": 1024,
          "min": 64,
          "max": 8192,
          "step": 64
        }),
        "height": ("INT", {
          "default": 1024,
          "min": 64,
          "max": 8192,
          "step": 64
        }),
        "initialization": (
          [
            'perlin_noise',
            'low_frequency',
            'gaussian_pyramid',
            'fractal_noise',
            'smooth_gradient',
            'random_standard'
          ],
          {
            "default": 'perlin_noise'
          }),
        "channels": ("INT", {
          "default": 4,
          "min": 1,
          "max": 16,
          "step": 1
        }),
        "noise_scale": ("FLOAT", {
          "default": 1.0,
          "min": 0.1,
          "max": 2.0,
          "step": 0.1
        }),
        "clip_scale": ("FLOAT", {
          "default": 2.0,
          "min": 1.0,
          "max": 10.0,
          "step": 0.5
        }),
        "batch_size": ("INT", {
          "default": 1,
          "min": 1,
          "max": 64
        }),
        "seed": ("INT", {
          "default": 0,
          "min": 0,
          "max": 0xffffffffffffffff
        }),
      }
    }

  RETURN_TYPES = ("LATENT", "INT", "INT")
  RETURN_NAMES = ("LATENT", "CLIP_WIDTH", "CLIP_HEIGHT")
  FUNCTION = "generate"

  def generate_perlin_noise(self, shape, scale=10):
    """Generate Perlin-like noise using interpolated random gradients"""
    def interpolate(a, b, w):
      return (b - a) * (3.0 - w * 2.0) * w * w + a
    
    batch, channels, height, width = shape
    noise = torch.zeros(shape)
    
    for b in range(batch):
      for c in range(channels):
        # Generate gradient grid
        grid_h, grid_w = height // scale + 2, width // scale + 2
        gradients = torch.randn(grid_h, grid_w, 2)
        
        # Interpolate across the image
        for y in range(height):
          for x in range(width):
            # Grid coordinates
            gx, gy = x / scale, y / scale
            gx0, gy0 = int(gx), int(gy)
            gx1, gy1 = gx0 + 1, gy0 + 1
            
            # Interpolation weights
            sx, sy = gx - gx0, gy - gy0
            sx = 3 * sx * sx - 2 * sx * sx * sx
            sy = 3 * sy * sy - 2 * sy * sy * sy
            
            # Sample gradients
            n00 = torch.dot(gradients[gy0, gx0], torch.tensor([gx - gx0, gy - gy0]))
            n10 = torch.dot(gradients[gy0, gx1], torch.tensor([gx - gx1, gy - gy0]))
            n01 = torch.dot(gradients[gy1, gx0], torch.tensor([gx - gx0, gy - gy1]))
            n11 = torch.dot(gradients[gy1, gx1], torch.tensor([gx - gx1, gy - gy1]))
            
            # Bilinear interpolation
            nx0 = interpolate(n00, n10, sx)
            nx1 = interpolate(n01, n11, sx)
            noise[b, c, y, x] = interpolate(nx0, nx1, sy)
    
    return noise

  def generate_low_frequency(self, shape):
    """Generate low-frequency noise by upsampling small random tensor"""
    batch, channels, height, width = shape
    # Start with very small resolution
    small_h, small_w = max(4, height // 16), max(4, width // 16)
    small_noise = torch.randn(batch, channels, small_h, small_w)
    # Upsample with bilinear interpolation
    noise = torch.nn.functional.interpolate(
      small_noise, 
      size=(height, width), 
      mode='bilinear', 
      align_corners=False
    )
    return noise

  def generate_gaussian_pyramid(self, shape):
    """Generate noise using multiple frequency bands"""
    batch, channels, height, width = shape
    noise = torch.zeros(shape)
    
    # Add multiple octaves of noise
    scales = [1, 2, 4, 8]
    weights = [0.5, 0.25, 0.15, 0.1]
    
    for scale, weight in zip(scales, weights):
      small_h = max(4, height // scale)
      small_w = max(4, width // scale)
      octave = torch.randn(batch, channels, small_h, small_w)
      octave = torch.nn.functional.interpolate(
        octave,
        size=(height, width),
        mode='bilinear',
        align_corners=False
      )
      noise += octave * weight
    
    return noise

  def generate_fractal_noise(self, shape):
    """Generate fractal noise with multiple octaves"""
    batch, channels, height, width = shape
    noise = torch.zeros(shape)
    
    amplitude = 1.0
    frequency = 1.0
    
    for _ in range(5):  # 5 octaves
      octave_h = max(4, int(height / frequency))
      octave_w = max(4, int(width / frequency))
      
      octave = torch.randn(batch, channels, octave_h, octave_w)
      octave = torch.nn.functional.interpolate(
        octave,
        size=(height, width),
        mode='bilinear',
        align_corners=False
      )
      
      noise += octave * amplitude
      amplitude *= 0.5
      frequency *= 2.0
    
    return noise

  def generate_smooth_gradient(self, shape):
    """Generate smooth gradient-based initialization"""
    batch, channels, height, width = shape
    noise = torch.zeros(shape)
    
    for b in range(batch):
      for c in range(channels):
        # Create smooth gradients
        y_grad = torch.linspace(-1, 1, height).unsqueeze(1).expand(height, width)
        x_grad = torch.linspace(-1, 1, width).unsqueeze(0).expand(height, width)
        
        # Combine with random weights
        weight_y, weight_x = torch.randn(2)
        gradient = weight_y * y_grad + weight_x * x_grad
        
        # Add some high-frequency detail
        detail = torch.randn(height, width) * 0.2
        noise[b, c] = gradient + detail
    
    return noise

  def generate(self, width, height, initialization, channels, noise_scale, clip_scale, batch_size, seed):
    """Generates structured latent and exposes clip dimensions"""
    
    # Calculate latent dimensions (using 8x downsampling, standard for SD/SDXL)
    latent_width = width // 8
    latent_height = height // 8
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Shape: [batch, channels, height, width]
    shape = (batch_size, channels, latent_height, latent_width)
    
    # Generate structured noise based on initialization type
    if initialization == 'perlin_noise':
      samples = self.generate_perlin_noise(shape, scale=8)
    elif initialization == 'low_frequency':
      samples = self.generate_low_frequency(shape)
    elif initialization == 'gaussian_pyramid':
      samples = self.generate_gaussian_pyramid(shape)
    elif initialization == 'fractal_noise':
      samples = self.generate_fractal_noise(shape)
    elif initialization == 'smooth_gradient':
      samples = self.generate_smooth_gradient(shape)
    else:  # random_standard
      samples = torch.randn(shape)
    
    # Apply noise scale
    samples = samples * noise_scale
    
    # Create latent dict in ComfyUI format
    latent = {"samples": samples}
    
    return (
      latent,
      int(width * clip_scale),
      int(height * clip_scale),
    )
    
NODE_CLASS_MAPPINGS = {"StructuredLatent": StructuredLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"StructuredLatent": "Structured Latent"}