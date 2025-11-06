import huggingface_hub as hf
import torch
import SimpleITK as sitk
import os
import requests

# Server configuration
class SegmentServerConfig:
    hf_models_path: str = None
    device: str = None
    n_cpu_threads = 2
    https_verify = True

# Global config
global_config = SegmentServerConfig()

# Configure the HTTP backend to use requests with custom settings
def config_hf_backend():
    if hasattr(hf, 'configure_http_backend'):
        import requests
        def backend_factory_requests() -> requests.Session:
            session = requests.Session()
            session.verify = not global_config.https_verify
            return session
        hf.configure_http_backend(backend_factory=backend_factory_requests)
    elif hasattr(hf, 'set_client_factory'):
        import httpx
        hf.set_client_factory(lambda : httpx.Client(verify=global_config.https_verify))
        

class SegmentSession:
    
    NNINTERACTIVE_REPO_ID = "nnInteractive/nnInteractive"
    NNINTERACTIVE_MODEL_NAME = "nnInteractive_v1.0"  
    
    def __init__(self, config: SegmentServerConfig = global_config):
        
        # Set the environment variables so that nnUnet does not complain
        os.environ['nnUNet_raw'] = '/nnUNet_raw'
        os.environ['nnUNet_preprocessed'] = '/nnUNet_preprocessed'
        os.environ['nnUNet_results'] = '/nnUNet_results'
        
        # Import nnInteractiveInferenceSession here to prevent slow startup
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
        
        # Create an interactive session
        self.session = nnInteractiveInferenceSession(
            device=torch.device(config.device),
            use_torch_compile=False,
            verbose=False,
            torch_n_threads=config.n_cpu_threads,
            do_autozoom=True,
            use_pinned_memory=True
        )
        
        # Set it as the default session factory - to allow -k flag
        config_hf_backend()

        # Download the model, optionally
        self.model_path = hf.snapshot_download(
            repo_id=self.NNINTERACTIVE_REPO_ID,
            allow_patterns=[f"{self.NNINTERACTIVE_MODEL_NAME}/*"],
            local_dir=config.hf_models_path)
        
        # Append the model name
        self.model_path = os.path.join(self.model_path, self.NNINTERACTIVE_MODEL_NAME)

        # Print where the model was downloaded to
        print(f'nnInteractive model available in {self.model_path}')
        
        # Load the model
        self.session.initialize_from_trained_model_folder(self.model_path)

        
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
    
    def add_lasso_interaction(self, sitk_image, include_interaction):  
        img = sitk.GetArrayFromImage(sitk_image)      
        self.session.add_lasso_interaction(img, include_interaction=include_interaction)
    
    def reset_interactions(self):
        self.target_tensor = torch.zeros(self.target_tensor.shape, dtype=torch.uint8)
        self.session.set_target_buffer(self.target_tensor)
        self.session.reset_interactions()
        
    def get_result(self):
        result = sitk.GetImageFromArray(self.target_tensor)
        result.CopyInformation(self.input_image)
        return result
