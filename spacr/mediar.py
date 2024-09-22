import os, sys, gdown
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

# Path to the MEDIAR directory
mediar_path = os.path.join(os.path.dirname(__file__), 'resources', 'MEDIAR')

print('mediar path', mediar_path)

# Temporarily create __init__.py to make MEDIAR a package
init_file = os.path.join(mediar_path, '__init__.py')
if not os.path.exists(init_file):
    with open(init_file, 'w'):  # Create the __init__.py file
        pass

# Add MEDIAR to sys.path
sys.path.insert(0, mediar_path)

try:
    # Now import the dependencies from MEDIAR
    from core.MEDIAR import Predictor
    from train_tools.models import MEDIARFormer

    print("Imports successful.")
finally:
    # Remove the temporary __init__.py file after the import
    if os.path.exists(init_file):
        os.remove(init_file)  # Remove the __init__.py file

# Import using the exact structure within the MEDIAR submodule
from core.MEDIAR import Predictor
from train_tools.models import MEDIARFormer


def display_imgs_in_list(imgs, cmap='gray'):
    plt.figure(figsize=(15, 10))
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i + 1)
        
        if len(img.shape) == 2:  # Grayscale image
            plt.imshow(img, cmap=cmap)
        elif len(img.shape) == 3 and img.shape[0] == 3:  # 3-channel image (C, H, W)
            plt.imshow(img.transpose(1, 2, 0))  # Change shape to (H, W, C) for displaying
        else:
            plt.imshow(img)
        plt.axis('off')
        if cmap == 'viridis':
            cell_count = len(np.unique(img)) - 1  # exclude the background
            print(f"\n{cell_count} objects detected!")
    plt.show()

def get_weights(finetuned_weights=False):
    if model_path1 is None:
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
                
    if model_path2 is None:
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


class MEDIARPredictor:
    def __init__(self, additional_models=None, device=None, model="ensemble", batch_size=None, roi_size=512, overlap=0.6, finetuned_weights=False, test=False):
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.test = test
        self.model = model
        self.additional_models = additional_models if additional_models is not None else []
        self.batch_size = batch_size
        self.roi_size = roi_size
        self.overlap = overlap
        
        self.model1_path, self.model2_path = get_weights(finetuned_weights)

        # Verify that the weights exist
        if not os.path.exists(self.model1_path) or not os.path.exists(self.model2_path):
            raise FileNotFoundError("Weights not found in the specified location.")

        # Load additional models for ensemble (if any)
        self.additional_model_objects = []
        for model_path in self.additional_models:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Additional model not found: {model_path}")
            additional_model = self.load_model(model_path, device=self.device)  # Default to 3 channels
            self.additional_model_objects.append(additional_model)

        # Run test if specified
        if self.test:
            self.run_test()

    def load_model(self, model_path, device):
        # In_channels must be 3 for MixVisionTransformer
        model_args = {
            "classes": 3,
            "decoder_channels": [1024, 512, 256, 128, 64],
            "decoder_pab_channels": 256,
            "encoder_name": 'mit_b5',
            "in_channels": 3  # Ensure this is always set to 3
        }
        model = MEDIARFormer(**model_args)
        
        # Load weights and then load into the model
        weights = torch.load(model_path, map_location=device)
        model.load_state_dict(weights, strict=False)

        model.to(device)
        model.eval()
        return model

    def preprocess_image(self, img):
        """
        Preprocesses the input image to ensure compatibility with the model.
        Converts grayscale images to 3-channel RGB by repeating the grayscale values.
        Ensures the image is in float32 format.

        :param img: Input image as a numpy array or PyTorch tensor.
        :return: Preprocessed image tensor.
        """
        if isinstance(img, np.ndarray):  # Check if the input is a numpy array
            if len(img.shape) == 2:  # Grayscale image (H, W)
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

            elif img.shape[2] == 1:  # Single channel grayscale (H, W, 1)
                img = np.repeat(img, 3, axis=2)  # Convert to 3-channel RGB

            # Convert image to float32 and normalize to [0, 1]
            img_tensor = torch.tensor(img.astype(np.float32).transpose(2, 0, 1))  # Change shape to (C, H, W)
        else:
            img_tensor = img  # If it's already a tensor, assume it's in (C, H, W) format

        #return img_tensor.float() / 255.0  # Normalize to range [0, 1]
        return img_tensor.float()

    def ensemble_predict(self, batch):
        models = [self.model1, self.model2] + self.additional_model_objects

        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = sliding_window_inference(
                    batch,
                    roi_size=self.roi_size,
                    sw_batch_size=self.batch_size or 1,  # Default to batch size 1 if None
                    predictor=model,
                    overlap=self.overlap
                )
                predictions.append(pred)
        
        # Ensemble sum (like in MEDIAR paper)
        ensemble_pred = torch.sum(torch.stack(predictions), dim=0)

        return ensemble_pred

    def predict(self, batch):
        processed_batch = []
        for img in batch:
            processed_img = self.preprocess_image(img)
            processed_batch.append(processed_img)
        
        batch_tensor = torch.stack(processed_batch).to(self.device)
        # Load models
        self.model1 = self.load_model(self.model1_path, self.device)
        self.model2 = self.load_model(self.model2_path, self.device)

        if self.model == "ensemble":
            return self.ensemble_predict(batch_tensor)
        elif self.model == "model1":
            self.model1.eval()
            with torch.no_grad():
                return sliding_window_inference(
                    batch_tensor,
                    roi_size=self.roi_size,
                    sw_batch_size=self.batch_size or 1,  # Default to batch size 1 if None
                    predictor=self.model1,
                    overlap=self.overlap
                )
            
        elif self.model == "model2":
            self.model2.eval()
            with torch.no_grad():
                return sliding_window_inference(
                    batch_tensor,
                    roi_size=self.roi_size,
                    sw_batch_size=self.batch_size or 1,  # Default to batch size 1 if None
                    predictor=self.model2,
                    overlap=self.overlap
                )
        else:
            raise ValueError("Invalid model option. Choose 'ensemble', 'model1', or 'model2'.")

    def run_test(self):
        """
        Run the model on test images if test flag is True.
        """
        import skimage.io as io

        input_path = os.path.join(os.path.dirname(__file__), 'resources/MEDIAR/image/examples')
        input_path_extra = os.path.join(os.path.dirname(__file__), 'resources/image')
        output_path = os.path.join(os.path.dirname(__file__), 'resources/MEDIAR/results')

        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Load test images from both input paths
        imgs = []
        img_names = []
        for dir_path in [input_path, input_path_extra]:
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                img = io.imread(file_path)
                imgs.append(img)
                img_names.append(file_name)
                break
            break

        # Display the input images
        display_imgs_in_list(imgs, cmap='gray')

        # Perform predictions, display the predicted masks, and save them
        masks = []
        for img, img_name in zip(imgs, img_names):
            #img_tensor = self.preprocess_image(img).unsqueeze(0).to(self.device)
            img_tensor = self.preprocess_image(img).to(self.device)

            # Perform prediction
            pred = self.predict([img_tensor])
            pred_img = pred.squeeze().cpu().numpy()

            masks.append(pred_img)

            # Save the predicted mask to output_path
            mask_output_path = os.path.join(output_path, f"{os.path.splitext(img_name)[0]}_mask.tiff")
            io.imsave(mask_output_path, pred_img.astype(np.uint16))  # Save as 16-bit TIFF

        # Display the predicted masks
        display_imgs_in_list(masks, cmap='viridis')