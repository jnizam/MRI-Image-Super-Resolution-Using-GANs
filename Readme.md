# Project Readme

## Prerequisites

Python 3.11.6
Up-to-date pip installer
CUDA support for GPU (required for SRGAN, optional for other models)
## Installation

cd Final_Project
Install dependencies:
```
pip install -r requirements.txt
```

Note: If using CPU only, modify the requirements.txt file to remove version constraints and '+cu' for PyTorch and torchvision.

For SRGAN, ensure CUDA support is available for GPU. If using CPU only, SRGAN cannot be run.

## Performance Metric Plots

To generate performance metric plots for the study:
```
python Evaluation/main.py
```

## EDSR

Place images to upscale in EDSR/test folder.
Navigate to EDSR/src and run the following command in the terminal:
```
sh demo.sh
```

For non-Linux systems (e.g., Windows), you can use a Bash emulator like Git Bash or run the equivalent commands in a command prompt:
```
bash demo.sh
```

Note: You can change the upscale factor inside demo.sh on the first line of the scale argument.

```
python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results
```
Results will be in EDSR/experiment/test/results-Demo.

## Real-ESRGAN

Manually update settings in Real-ESRGAN/main.py:
```
def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=8)
    model.load_weights('/path/to/weights/RealESRGAN_x8.pth', download=True)
    image = Image.open("/path/to/input/image.png").convert('RGB')
    sr_image = model.predict(image)
    sr_image.save('/path/to/output/image.png')

```

Change scale to the desired upscale factor.
Update load_weights with the path to the desired weights file.
Update Image.open with the path to the input image.
Update sr_image.save with the desired output path.
Run the modified main.py script.
Results will be in Real-ESRGAN/results.

## SRGAN

Place input images in SRGAN/data folder (study images are already included).
Run the following command in the terminal:
```
python3 test.py --config_path ./configs/test/SRGAN_x4-SRGAN_ImageNet-Set5.yaml
```

Change the upscale factor in the code to either 'x2', 'x4', or 'x8.
Note: The provided yaml files in SRGAN/configs/test assume a specific image setup.
Results will be in SRGAN/results. Update yaml files for custom image setups. If using your own images, modify the yaml files to match your directory setup