import cv2
import torch
import warnings

class DepthEstimator:
    def __init__(self):
        """
        Initializes the MiDaS depth estimator.
        Downloads the MiDaS_small model from torch hub if not found locally.
        Runs on GPU if CUDA is available, otherwise falls back to CPU.
        """
        # Suppress warnings from torch hub downloading
        warnings.filterwarnings("ignore", category=UserWarning)

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DepthEstimator] Using device: {self.device}")

        # Use MiDaS_small for real-time performance
        model_type = "MiDaS_small"
        print(f"[DepthEstimator] Loading model {model_type}...")
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()

        # Load transforms to resize and normalize the image exactly how MiDaS wants
        #preprocessing 
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def estimate(self, frame):
        """
        goruntu isleme ve Tahmin Alma
        """
        # RGB ye cevirme
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # tensore cevirme
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            # derinlik cikarimi
            prediction = self.midas(input_batch)

            # Resize the prediction to match original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Output contains relative depth values
        # Since it is a GPU tensor, move to CPU and convert to numpy
        depth_map = prediction.cpu().numpy()
        return depth_map
