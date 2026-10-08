import huggingface_hub as hf
import torch
import SimpleITK as sitk
import os
import requests
import typing
import json
import numpy as np

# Server configuration
class SegmentServerConfig:
    hf_models_path: str = None
    device: str = None
    n_cpu_threads = 2
    https_verify = True
    https_enabled = True
    mock_models = False

# Global config
global_config = SegmentServerConfig()

# Configure the HTTP backend to use requests with custom settings
def config_hf_backend():
    import urllib3
    print(f'Configuring Huggingface HTTP backend, verify={global_config.https_verify}')
    if hasattr(hf, 'configure_http_backend'):
        import requests
        def backend_factory_requests() -> requests.Session:
            session = requests.Session()
            session.verify = global_config.https_verify
            return session
        hf.configure_http_backend(backend_factory=backend_factory_requests)
    elif hasattr(hf, 'set_client_factory'):
        import httpx
        print(f'Configuring Huggingface HTTP client, verify={global_config.https_verify}')
        hf.set_client_factory(lambda : httpx.Client(verify=global_config.https_verify))
    if not global_config.https_verify:
        urllib3.disable_warnings()
        print('*** Warning: HTTPS verification is disabled! ***')
        
        
class ModelWrapper:
    # Properties supported by this model
    ID: str = ""
    
    # Dimensionality of the model (2 or 3)
    DIMENSIONS: int = 0
    
    # Supported number of channels in the model (e.g., 1 or 3); list of integers, empty means any
    CHANNELS: list[int] = [1]
    
    # Supported interaction types
    INTERACTIONS: list[str] = []
    
    def __init__(self):
        pass
        

class nnInteractiveWrapper(ModelWrapper):
    # Huggingface properties
    HF_REPO_ID = "MIC-DKFZ/nnInteractive"
    HF_MODEL_NAME = "nnInteractive_v1.0" 
    
    # Model descriptor
    ID = "nnInteractive"
    DIMENSIONS = 3
    CHANNELS = [1]
    INTERACTIONS = [ "point", "box", "scribble", "lasso" ]
    
    def __init__(self, config: SegmentServerConfig = global_config):
        super().__init__()

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
            do_autozoom=True
        )
        
        # Set it as the default session factory - to allow -k flag
        config_hf_backend()

        # Download the model, optionally
        self.model_path = hf.snapshot_download(
            repo_id=self.HF_REPO_ID,
            allow_patterns=[f"{self.HF_MODEL_NAME}/*"],
            local_dir=config.hf_models_path)
        
        # Append the model name
        self.model_path = os.path.join(self.model_path, self.HF_MODEL_NAME)

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

class SAM2Wrapper(ModelWrapper):
    # Huggingface properties
    HF_REPO_ID = "facebook/sam2.1-hiera-large"

    # Model descriptor    
    DIMENSIONS = 2
    CHANNELS = [1,3]
    INTERACTIONS = [ "point" ]
    ID = "SAM2"
    
    def __init__(self, config: SegmentServerConfig = global_config):
        super().__init__()
        self.config = config

        # Import transformers here to prevent slow startup
        from transformers import Sam2Processor, Sam2Model

        # Set it as the default session factory - to allow -k flag
        config_hf_backend()
        
        lfo = not config.https_enabled
        self.model = Sam2Model.from_pretrained(self.HF_REPO_ID,local_files_only=lfo).to(self.config.device)
        self.processor = Sam2Processor.from_pretrained(self.HF_REPO_ID, local_files_only=lfo)
        
    def set_image(self, sitk_image: sitk.Image):
        
        # Get the image header information for returning masks later
        self.image_itk = sitk_image

        # Read image and validate input dimensions
        if sitk_image.GetNumberOfComponentsPerPixel() == 3:
            self.image_arr = sitk.GetArrayFromImage(sitk_image)[None, :, :, :]  # Add batch dimension
        elif sitk_image.GetNumberOfComponentsPerPixel() == 1:
            self.image_arr = sitk.GetArrayFromImage(sitk_image)[None, :, :, None]  # Add batch and channel dimension
        else:
            raise ValueError("Input image must have 1 or 3 components per pixel")
        
        # Reset the mask and the embeddings
        self.mask_pt = None
        self.image_embeddings_pt = None
        self.image_sizes = None
        self.all_points = None
        self.all_labels = None
        
    def add_point_interaction(self, index_itk: list[int], include_interaction: bool):
        
        # Map the ITK index to expected format
        input_points = torch.tensor([[[[index_itk[0], index_itk[1]]]]], dtype=torch.float32)
        input_labels = torch.tensor([[[1 if include_interaction else 0]]])
        
        # Append these to the existing interactions
        self.all_points = input_points if self.all_points is None else torch.cat([self.all_points, input_points], dim=-2)
        self.all_labels = input_labels if self.all_labels is None else torch.cat([self.all_labels, input_labels], dim=-1)
        
        # Prepare inputs
        print(f'Interactions {self.all_points.detach().cpu().numpy().squeeze()} with labels {self.all_labels.detach().cpu().numpy().squeeze()}')
        inputs = self.processor(
            images=self.image_arr if self.image_embeddings_pt is None else None,
            original_sizes=self.image_sizes if self.image_embeddings_pt is not None else None, 
            input_points=self.all_points, 
            input_labels=self.all_labels, 
            return_tensors="pt").to(self.model.device)
    
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs, 
                                 multimask_output=False,
                                 # input_masks=self.mask_pt,
                                 image_embeddings=self.image_embeddings_pt)
            
        # DEBUG: store the mask in raw form
        sitk.WriteImage(sitk.GetImageFromArray(outputs.pred_masks.squeeze().detach().cpu().numpy()), '/tmp/masksam.nii.gz')
        with open('/tmp/sam_inputs.json', 'wt') as f:
            json.dump({ 'points': self.all_points.detach().cpu().numpy().tolist(),
                        'labels': self.all_labels.detach().cpu().numpy().tolist() }, f, indent=2)

        # Get and store the best mask
        # self.mask_pt = outputs.pred_masks[:,0,:,:]
        
        # Store the image embeddings for next interaction
        self.image_embeddings_pt = outputs.image_embeddings
        self.image_sizes = inputs["original_sizes"]

        # Resize the mask to original image size and store as numpy array
        m = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"])
        self.mask_arr = np.array(m[0][0,0,:,:])
    
    def reset_interactions(self):
        
        # Reset the mask - the segmentation will start anew
        self.mask_pt = None
        self.all_points = None
        self.all_labels = None
    
    def get_result(self) -> sitk.Image:
        # Generate an ITK image for the mask
        mask_itk = sitk.GetImageFromArray(self.mask_arr.astype(np.uint8))
        mask_itk.CopyInformation(self.image_itk)
        return mask_itk

class MockModelWrapper(ModelWrapper):
    """
    Fast stand-in for a deep learning model, used to test the server and the
    ITK-SNAP client without a GPU. No inference is performed: point clicks paint
    a ball around the point, scribbles are dilated, and lassos are used as-is.
    Foreground interactions add to the mask, background interactions erase it.
    """
    POINT_RADIUS = 5
    SCRIBBLE_RADIUS = 2

    def __init__(self, config: SegmentServerConfig = global_config):
        super().__init__()
        print(f'Mock {self.ID} model created, no deep learning inference will be performed')

    def set_image(self, sitk_image: sitk.Image):
        self.input_image = sitk_image
        self.mask = np.zeros(sitk_image.GetSize()[::-1], dtype=np.uint8)
        print(f'Image set of size {self.mask.shape}')

    def _apply(self, region, include_interaction):
        if include_interaction:
            self.mask[region] = 1
        else:
            self.mask[region] = 0

    def _get_interaction_array(self, sitk_image):
        arr = sitk.GetArrayFromImage(sitk_image) > 0
        if arr.shape != self.mask.shape:
            raise ValueError(f"Interaction image shape {arr.shape} does not match image shape {self.mask.shape}")
        return arr

    def add_point_interaction(self, index_itk, include_interaction):
        # Paint a ball within the bounding box of the point, in numpy index order
        center = np.array(index_itk[:self.mask.ndim][::-1])
        lo = np.maximum(center - self.POINT_RADIUS, 0)
        hi = np.minimum(center + self.POINT_RADIUS + 1, self.mask.shape)
        if np.any(lo >= hi):
            return
        box = tuple(slice(l, h) for l, h in zip(lo, hi))
        grid = np.ogrid[box]
        ball = sum((g - c) ** 2 for g, c in zip(grid, center)) <= self.POINT_RADIUS ** 2
        self.mask[box][ball] = 1 if include_interaction else 0

    def add_scribble_interaction(self, sitk_image, include_interaction):
        arr = self._get_interaction_array(sitk_image)
        dilated = sitk.BinaryDilate(sitk.GetImageFromArray(arr.astype(np.uint8)),
                                    [self.SCRIBBLE_RADIUS] * arr.ndim)
        self._apply(sitk.GetArrayFromImage(dilated) > 0, include_interaction)

    def add_lasso_interaction(self, sitk_image, include_interaction):
        self._apply(self._get_interaction_array(sitk_image), include_interaction)

    def reset_interactions(self):
        self.mask[:] = 0

    def get_result(self) -> sitk.Image:
        result = sitk.GetImageFromArray(self.mask)
        result.CopyInformation(self.input_image)
        return result

class nnInteractiveMockWrapper(MockModelWrapper):
    ID = nnInteractiveWrapper.ID
    DIMENSIONS = nnInteractiveWrapper.DIMENSIONS
    CHANNELS = nnInteractiveWrapper.CHANNELS
    INTERACTIONS = nnInteractiveWrapper.INTERACTIONS

class SAM2MockWrapper(MockModelWrapper):
    ID = SAM2Wrapper.ID
    DIMENSIONS = SAM2Wrapper.DIMENSIONS
    CHANNELS = SAM2Wrapper.CHANNELS
    INTERACTIONS = SAM2Wrapper.INTERACTIONS

def get_model_classes(config: SegmentServerConfig = global_config) -> list[type[ModelWrapper]]:
    """Return the model wrapper classes, or their mock stand-ins in mock mode."""
    if config.mock_models:
        return [ nnInteractiveMockWrapper, SAM2MockWrapper ]
    return [ nnInteractiveWrapper, SAM2Wrapper ]

def get_model_listing():
    """Return a list of available models and their capabilities."""
    model_list = []
    for model in get_model_classes():
        model_info = {
            "id": model.ID,
            "channels": model.CHANNELS,
            "dimensions": model.DIMENSIONS,
            "interactions": model.INTERACTIONS
        }
        model_list.append(model_info)   
    return model_list


def instantiate_model_wrapper(repo_id: str, config: SegmentServerConfig = global_config) -> ModelWrapper:
    """Instantiate a model wrapper based on the given repo ID."""
    for model in get_model_classes(config):
        if repo_id == model.ID:
            return model(config)
    raise ValueError(f"Unknown model repo ID: {repo_id}")
