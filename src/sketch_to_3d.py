import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
import os
import tempfile
import subprocess
import sys

class SketchConverter:
    def __init__(self):
        self.current_model = None
        self.models_dir = "../models"
        
        # Make sure the models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
    
    def process_sketch(self, sketch_image, description="", is_improved=False):
        """
        Turn a 2D sketch into a simple 3D model
        
        Args:
            sketch_image: The drawing canvas (original or improved)
            description: Text description of the sketch
            is_improved: Whether the sketch is already improved
        """
        # Convert to grayscale if needed
        if len(sketch_image.shape) == 3:
            if is_improved:
                # Handle colored regions in improved sketches
                hsv = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2HSV)
                sat = hsv[:, :, 1]
                val = hsv[:, :, 2]
                combined = cv2.addWeighted(sat, 0.5, val, 0.5, 0)
                gray_sketch = combined
            else:
                gray_sketch = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_sketch = sketch_image.copy()
        
        # Threshold to make a binary image
        _, binary = cv2.threshold(gray_sketch, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No sketch contours found")
            return None
        
        # Get the largest contour (main sketch)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask from the contour
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [main_contour], 0, 255, -1)
        
        # Process based on whether it's an improved sketch
        if is_improved:
            return self._process_improved_sketch(sketch_image, description)
        else:
            return self._process_basic_sketch(mask, description)
    
    def _process_basic_sketch(self, mask, description=""):
        """Process a basic sketch using distance transform"""
        # Create a height map using distance transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        height_map = dist_transform / dist_transform.max()
        
        # Create a simple 3D model using the height map
        x, y = np.meshgrid(np.arange(height_map.shape[1]), np.arange(height_map.shape[0]))
        z = height_map * 20  # Scale height for better visualization
        
        try:
            # Create a volume from the height map
            volume = np.zeros((height_map.shape[0], height_map.shape[1], 30))
            
            # Fill volume based on height map
            for i in range(volume.shape[0]):
                for j in range(volume.shape[1]):
                    if height_map[i, j] > 0:
                        max_height = int(height_map[i, j] * 20)
                        volume[i, j, 0:max_height] = 1
            
            # Run marching cubes to get a mesh
            verts, faces, _, _ = measure.marching_cubes(volume, 0.5)
            
            # Create a trimesh object
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            
            # Save the mesh
            file_base = "sketch_model"
            if description:
                file_base = "_".join(description.lower().split()[:3])
                
            file_path = os.path.join(self.models_dir, f"{file_base}.obj")
            mesh.export(file_path)
            print(f"Model saved to {file_path}")
            
            # Store the current model
            self.current_model = {
                'mesh': mesh,
                'path': file_path,
                'description': description
            }
            
            return self.current_model
            
        except Exception as e:
            print(f"Error creating 3D model: {e}")
            return None
    
    def _process_improved_sketch(self, colored_sketch, description=""):
        """Process an improved sketch to create a more detailed 3D model"""
        try:
            # Convert to HSV for better color separation
            hsv = cv2.cvtColor(colored_sketch, cv2.COLOR_BGR2HSV)
            
            # Create a combined height map
            height_map = np.zeros(colored_sketch.shape[:2], dtype=np.float32)
            
            # Detect key regions by color
            red_mask = colored_sketch[:,:,2] > 150
            green_mask = colored_sketch[:,:,1] > 150
            blue_mask = colored_sketch[:,:,0] > 150
            
            # Assign different heights to different regions
            height_map[red_mask] = 0.8  # Tall elements
            height_map[green_mask & ~red_mask] = 0.4  # Medium elements
            height_map[blue_mask & ~red_mask & ~green_mask] = 0.1  # Low elements
            
            # Ensure areas with actual drawing have some height
            gray = cv2.cvtColor(colored_sketch, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            height_map[binary > 0] = np.maximum(height_map[binary > 0], 0.2)
            
            # Smooth the height map
            height_map = cv2.GaussianBlur(height_map, (7, 7), 0)
            
            # Create a volume from the height map
            volume = np.zeros((height_map.shape[0], height_map.shape[1], 40))
            
            # Fill volume based on height map
            for i in range(volume.shape[0]):
                for j in range(volume.shape[1]):
                    if height_map[i, j] > 0:
                        max_height = int(height_map[i, j] * 30)
                        volume[i, j, 0:max_height] = 1
            
            # Run marching cubes to get a mesh
            verts, faces, _, _ = measure.marching_cubes(volume, 0.5)
            
            # Create a trimesh object
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            
            # Save the mesh with a different name
            file_base = "improved_model"
            if description:
                file_base = "_".join(description.lower().split()[:3]) + "_improved"
                
            file_path = os.path.join(self.models_dir, f"{file_base}.obj")
            mesh.export(file_path)
            print(f"Improved 3D model saved to {file_path}")
            
            # Store the current model
            self.current_model = {
                'mesh': mesh,
                'path': file_path,
                'description': description,
                'improved': True
            }
            
            return self.current_model
            
        except Exception as e:
            print(f"Error creating improved 3D model: {e}")
            # Fall back to basic approach if improved fails
            gray = cv2.cvtColor(colored_sketch, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            return self._process_basic_sketch(binary, description)
    
    def visualize_model(self, model=None):
        """Show a visualization of the 3D model using a separate process"""
        if model is None:
            model = self.current_model
            
        if model is None:
            print("No model to visualize")
            return
        
        try:
            # Save the model to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                f.write("""
import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the model
mesh = trimesh.load_mesh(sys.argv[1])

# Create a figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh
verts = mesh.vertices
faces = mesh.faces

# Plot as a trisurf
ax.plot_trisurf(
    verts[:, 0], verts[:, 1], verts[:, 2],
    triangles=faces,
    cmap='viridis',
    linewidth=0.2,
    antialiased=True
)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set title
ax.set_title(sys.argv[2])

# Show the plot
plt.tight_layout()
plt.show()
                """)
                script_path = f.name
            
            # Get the model path
            model_path = model['path']
            description = model.get('description', 'Model')
            
            # Run the script in a separate process
            subprocess.Popen([sys.executable, script_path, model_path, description])
            print(f"Visualization launched in separate window")
            
        except Exception as e:
            print(f"Error visualizing 3D model: {e}")
    
    def get_current_model(self):
        """Return the current model"""
        return self.current_model