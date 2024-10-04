import os, sys, gdown, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

# Path to the MEDIAR directory
mediar_path = os.path.join(os.path.dirname(__file__), 'resources', 'MEDIAR')

# Temporarily create __init__.py to make MEDIAR a package
init_file = os.path.join(mediar_path, '__init__.py')
if not os.path.exists(init_file):
    with open(init_file, 'w'):  # Create the __init__.py file
        pass

# Add MEDIAR to sys.path
sys.path.insert(0, mediar_path)

try:
    # Now import the dependencies from MEDIAR
    from core.MEDIAR import Predictor, EnsemblePredictor
    from train_tools.models import MEDIARFormer
finally:
    # Remove the temporary __init__.py file after the import
    if os.path.exists(init_file):
        os.remove(init_file)  # Remove the __init__.py file

def display_imgs_in_list(lists_of_imgs, cmaps=None):
    """
    Displays images from multiple lists side by side. 
    Each row will display one image from each list (lists_of_imgs[i][j] is the j-th image in the i-th list).
    
    :param lists_of_imgs: A list of lists, where each inner list contains images.
    :param cmaps: List of colormaps to use for each list (optional). If not provided, defaults to 'gray' for all lists.
    """
    num_lists = len(lists_of_imgs)
    num_images = len(lists_of_imgs[0])
    
    # Ensure that all lists have the same number of images
    for img_list in lists_of_imgs:
        assert len(img_list) == num_images, "All inner lists must have the same number of images"

    # Use 'gray' as the default colormap if cmaps are not provided
    if cmaps is None:
        cmaps = ['gray'] * num_lists
    else:
        assert len(cmaps) == num_lists, "The number of colormaps must match the number of lists"

    plt.figure(figsize=(15, 5 * num_images))
    
    for j in range(num_images):
        for i, img_list in enumerate(lists_of_imgs):
            img = img_list[j]
            plt.subplot(num_images, num_lists, j * num_lists + i + 1)
            
            if len(img.shape) == 2:  # Grayscale image
                plt.imshow(img, cmap=cmaps[i])
            elif len(img.shape) == 3 and img.shape[0] == 3:  # 3-channel image (C, H, W)
                plt.imshow(img.transpose(1, 2, 0))  # Change shape to (H, W, C) for displaying
            else:
                plt.imshow(img)
            
            plt.axis('off')
            plt.title(f'Image {j+1} from list {i+1}')

    plt.tight_layout()
    plt.show()

def get_weights(finetuned_weights=False):
    if finetuned_weights:
        model_path1 = os.path.join(os.path.dirname(__file__), 'resources', 'MEDIAR_weights', 'from_phase1.pth')
        if not os.path.exists(model_path1):
            print("Downloading finetuned model 1...")
            gdown.download('https://drive.google.com/uc?id=1JJ2-QKTCk-G7sp5ddkqcifMxgnyOrXjx', model_path1, quiet=False)
    else:
        model_path1 = os.path.join(os.path.dirname(__file__), 'resources', 'MEDIAR_weights', 'phase1.pth')
        if not os.path.exists(model_path1):
            print("Downloading model 1...")
            gdown.download('https://drive.google.com/uc?id=1v5tYYJDqiwTn_mV0KyX5UEonlViSNx4i', model_path1, quiet=False)
            
    if finetuned_weights:
        model_path2 = os.path.join(os.path.dirname(__file__), 'resources', 'MEDIAR_weights', 'from_phase2.pth')
        if not os.path.exists(model_path2):
            print("Downloading finetuned model 2...")
            gdown.download('https://drive.google.com/uc?id=168MtudjTMLoq9YGTyoD2Rjl_d3Gy6c_L', model_path2, quiet=False)
    else:
        model_path2 = os.path.join(os.path.dirname(__file__), 'resources', 'MEDIAR_weights', 'phase2.pth')
        if not os.path.exists(model_path2):
            print("Downloading model 2...")
            gdown.download('https://drive.google.com/uc?id=1NHDaYvsYz3G0OCqzegT-bkNcly2clPGR', model_path2, quiet=False)
    
    return model_path1, model_path2

def normalize_image(image, lower_percentile=0.0, upper_percentile=99.5):
    """
    Normalize an image based on the 0.0 and 99.5 percentiles.
    
    :param image: Input image (numpy array).
    :param lower_percentile: Lower percentile (default is 0.0).
    :param upper_percentile: Upper percentile (default is 99.5).
    :return: Normalized image (numpy array).
    """
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    
    # Clip image values to the calculated percentiles
    image = np.clip(image, lower_bound, upper_bound)
    
    # Normalize to [0, 1]
    image = (image - lower_bound) / (upper_bound - lower_bound + 1e-5)  # Add small epsilon to avoid division by zero
    
    return image

class MEDIARPredictor:
    def __init__(self, input_path=None, output_path=None, device=None, model="ensemble", roi_size=512, overlap=0.6, finetuned_weights=False, test=False, use_tta=False, normalize=True, quantiles=[0.0, 99.5]):
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.test = test
        self.model = model
        self.normalize = normalize
        self.quantiles = quantiles

        # Paths to model weights
        self.model1_path, self.model2_path = get_weights(finetuned_weights)

        # Load main models
        self.model1 = self.load_model(self.model1_path, device=self.device)
        self.model2 = self.load_model(self.model2_path, device=self.device) if model == "ensemble" or model == "model2" else None
        if self.test:
            # Define input and output paths for running test
            self.input_path = os.path.join(os.path.dirname(__file__), 'resources/images')
            self.output_path = os.path.join(os.path.dirname(__file__), 'resources/MEDIAR/results')
        else:
            self.input_path = input_path
            self.output_path = output_path

        # If using a single model
        if self.model == "model1":
            self.predictor = Predictor(
                model=self.model1,
                device=self.device,
                input_path=self.input_path,
                output_path=self.output_path,
                algo_params={"use_tta": use_tta}
            )

        # If using a single model
        if self.model == "model2":
            self.predictor = Predictor(
                model=self.model2,
                device=self.device,
                input_path=self.input_path,
                output_path=self.output_path,
                algo_params={"use_tta": use_tta}
            )

        # If using two models
        elif self.model == "ensemble":
            self.predictor = EnsemblePredictor(
                model=self.model1,  # Pass model1 as model
                model_aux=self.model2,  # Pass model2 as model_aux
                device=self.device,
                input_path=self.input_path,
                output_path=self.output_path,
                algo_params={"use_tta": use_tta}
            )

        if self.test:
            self.run_test()

        if not self.model in ["model1", "model2", "ensemble"]:
            raise ValueError("Invalid model type. Choose from 'model1', 'model2', or 'ensemble'.")

    def load_model(self, model_path, device):
        model_args = {
            "classes": 3,
            "decoder_channels": [1024, 512, 256, 128, 64],
            "decoder_pab_channels": 256,
            "encoder_name": 'mit_b5',
            "in_channels": 3
        }
        model = MEDIARFormer(**model_args)
        weights = torch.load(model_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        model.to(device)
        model.eval()
        return model
    
    def display_image_and_mask(self, img, mask):

        from .plot import generate_mask_random_cmap
        """
        Displays the normalized input image and the predicted mask side by side.
        """
        # If img is a tensor, convert it to NumPy for display
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        # If mask is a tensor, convert it to NumPy for display
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # Transpose the image to have (H, W, C) format for display if needed
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        # Scale the normalized image back to [0, 255] for proper display
        img_display = (img * 255).astype(np.uint8)

        plt.figure(figsize=(10, 5))

        # Display normalized image
        plt.subplot(1, 2, 1)
        plt.imshow(img_display)
        plt.title("Normalized Image")
        plt.axis("off")

        r_cmap = generate_mask_random_cmap(mask)

        # Display predicted mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap=r_cmap)
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def predict_batch(self, imgs):
        """
        Predict masks for a batch of images.
        
        :param imgs: List of input images as NumPy arrays (each in (H, W, C) format).
        :return: List of predicted masks as NumPy arrays.
        """
        processed_imgs = []

        # Preprocess and normalize each image
        for img in imgs:
            if self.normalize:
                # Normalize the image using the specified quantiles
                img_normalized = normalize_image(img, lower_percentile=self.quantiles[0], upper_percentile=self.quantiles[1])
            else:
                img_normalized = img

            # Convert image to tensor and send to device
            img_tensor = torch.tensor(img_normalized.astype(np.float32).transpose(2, 0, 1)).to(self.device)  # (C, H, W)
            processed_imgs.append(img_tensor)

        # Stack all processed images into a batch tensor
        batch_tensor = torch.stack(processed_imgs)

        # Run inference to get predicted masks
        pred_masks = self.predictor._inference(batch_tensor)

        # Ensure pred_masks is always treated as a batch
        if len(pred_masks.shape) == 3:  # If single image, add batch dimension
            pred_masks = pred_masks.unsqueeze(0)

        # Convert predictions to NumPy arrays and post-process each mask
        predicted_masks = []
        for pred_mask in pred_masks:
            pred_mask_np = pred_mask.cpu().numpy()

            # Extract dP and cellprob from pred_mask
            dP = pred_mask_np[:2]  # First two channels as dP (displacement field)
            cellprob = pred_mask_np[2]  # Third channel as cell probability

            # Concatenate dP and cellprob along axis 0 to pass a single array
            combined_pred_mask = np.concatenate([dP, np.expand_dims(cellprob, axis=0)], axis=0)

            # Post-process the predicted mask
            mask = self.predictor._post_process(combined_pred_mask)

            # Append the processed mask to the list
            predicted_masks.append(mask.astype(np.uint16))

        return predicted_masks

    def run_test(self):
        """
        Run the model on test images if the test flag is True.
        """
        # List of input images
        imgs = []
        img_names = []
        
        for img_file in os.listdir(self.input_path):
            img_path = os.path.join(self.input_path, img_file)
            img = io.imread(img_path)
            
            # Check if the image is grayscale (2D) or RGB (3D), and convert grayscale to RGB
            if len(img.shape) == 2:  # Grayscale image (H, W)
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)  # Convert grayscale to RGB

            # Normalize the image if the normalize flag is True
            if self.normalize:
                img_normalized = normalize_image(img, lower_percentile=self.quantiles[0], upper_percentile=self.quantiles[1])
            else:
                img_normalized = img
            
            # Convert image to tensor and send directly to device
            img_tensor = torch.tensor(img_normalized.astype(np.float32).transpose(2, 0, 1)).to(self.device)  # (C, H, W)

            imgs.append(img_tensor)
            img_names.append(os.path.splitext(img_file)[0])

        # Stack all images into a batch (ensure it's always treated as a batch)
        batch_tensor = torch.stack(imgs)

        # Predict using the predictor (or ensemble predictor)
        pred_masks = self.predictor._inference(batch_tensor)

        # Ensure pred_masks is always treated as a batch
        if len(pred_masks.shape) == 3:  # If single image, add batch dimension
            pred_masks = pred_masks.unsqueeze(0)

        # Convert predictions to NumPy arrays and post-process each mask
        for i, pred_mask in enumerate(pred_masks):
            # Ensure the dimensions of pred_mask remain consistent
            pred_mask_np = pred_mask.cpu().numpy()

            # Extract dP and cellprob from pred_mask
            dP = pred_mask_np[:2]  # First two channels as dP (displacement field)
            cellprob = pred_mask_np[2]  # Third channel as cell probability

            # Concatenate dP and cellprob along axis 0 to pass a single array
            combined_pred_mask = np.concatenate([dP, np.expand_dims(cellprob, axis=0)], axis=0)

            # Post-process the predicted mask
            mask = self.predictor._post_process(combined_pred_mask)

            # Convert the mask to 16-bit format (ensure values fit into 16-bit range)
            mask_to_save = mask.astype(np.uint16)

            # Save the post-processed mask as a .tif file using cv2
            mask_output_path = os.path.join(self.output_path, f"{img_names[i]}_mask.tiff")
            cv2.imwrite(mask_output_path, mask_to_save)

            print(f"Predicted mask saved at: {mask_output_path}")

            self.display_image_and_mask(imgs[i].cpu().numpy(), mask)

        print(f"Test predictions saved in {self.output_path}")

    def preprocess_image(self, img):
        """
        Preprocess input image (numpy array) for compatibility with the model.
        """
        if isinstance(img, np.ndarray):  # Check if the input is a numpy array
            if len(img.shape) == 2:  # Grayscale image (H, W)
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

            elif img.shape[2] == 1:  # Single channel grayscale (H, W, 1)
                img = np.repeat(img, 3, axis=2)  # Convert to 3-channel RGB

            img_tensor = torch.tensor(img.astype(np.float32).transpose(2, 0, 1))  # Change shape to (C, H, W)
        else:
            img_tensor = img  # If it's already a tensor, assume it's in (C, H, W) format

        return img_tensor.float()