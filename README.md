# Smart Image Watermarker

Smart Image Watermarker is a Python application designed to automatically add watermarks to images with intelligent positioning and scaling. It ensures the watermark does not obstruct important parts of the image by analyzing the image's content for low-activity areas. This application is perfect for photographers, graphic designers, and content creators looking to protect their work while maintaining its aesthetic quality.

## Features

- **Automatic Image Detection**: Select either specific image files or an entire directory to process all images within.
- **Intelligent Watermark Positioning**: Analyzes images to find the best spot for the watermark, avoiding main subjects or important details.
- **Adaptive Watermark Scaling**: Scales the watermark according to the image size, ensuring it is neither too large nor too small.
- **Watermark Transparency Control**: Allows customization of the watermark's transparency level for subtlety.
- **Background Contrast Adjustment**: Automatically adjusts the watermark color based on the image's background to ensure visibility.
- **Graphical Progress Indicator**: Includes a progress bar to visually track the watermarking process.

## Installation

To use Smart Image Watermarker, you need to have Python installed on your computer along with the following libraries:
- OpenCV
- NumPy
- scikit-image
- Tkinter

You can install the required libraries using pip:
```
pip install opencv-python-headless numpy scikit-image
```
Note: `opencv-python-headless` is used instead of `opencv-python` to avoid unnecessary GUI dependencies. Tkinter is typically included with Python, but if it's missing, refer to [Tkinter installation](https://tkdocs.com/tutorial/install.html) for guidance.

## Usage

1. Clone this repository or download the source code.
2. Navigate to the downloaded directory.
3. Run the script using Python: `python main.py`
4. Follow the GUI prompts to select images, a watermark, and set the desired transparency.

## Contributing

Contributions to Smart Image Watermarker are welcome! Feel free to fork the repository, make your changes, and submit a pull request.