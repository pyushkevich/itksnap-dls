from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
import torch
import SimpleITK as sitk

class SegmentSession:
    
    def __init__(self, model_path):
        
        # Create an interactive session
        self.session = nnInteractiveInferenceSession(
            device=torch.device("cuda"),  
            use_torch_compile=False,
            verbose=False,
            torch_n_threads=2,
            do_autozoom=True,
            use_pinned_memory=True
        )
        
        # Load the model
        self.session.initialize_from_trained_model_folder(model_path)
        print(f'MODEL INITIALIZED')
        
    def set_image(self, sitk_image):
        
        # Read the image
        self.input_image = sitk_image
        img = sitk.GetArrayFromImage(self.input_image)[None]  # Ensure shape (1, x, y, z)
        
        # Validate input dimensions
        if img.ndim != 4:
            raise ValueError("Input image must be 4D with shape (1, x, y, z)")

        # Set the image for this session
        self.session.set_image(img)
        print(f'Image set of size {img.shape}')
        self.target_tensor = torch.zeros(img.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
        self.session.set_target_buffer(self.target_tensor)
        
    def add_point_interaction(self, index_itk, include_interaction):        
        self.session.add_point_interaction(tuple(index_itk[::-1]), 
                                           include_interaction=include_interaction)
    
    def add_scribble_interaction(self, sitk_image, include_interaction):  
        img = sitk.GetArrayFromImage(sitk_image)      
        self.session.add_scribble_interaction(img, include_interaction=include_interaction)
    
    def reset_interactions(self):
        self.target_tensor = torch.zeros(self.target_tensor.shape, dtype=torch.uint8)
        self.session.set_target_buffer(self.target_tensor)
        self.session.reset_interactions()
        
    def get_result(self):
        result = sitk.GetImageFromArray(self.target_tensor)
        result.CopyInformation(self.input_image)
        return result
