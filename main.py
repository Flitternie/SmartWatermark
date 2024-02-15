import os
import cv2
import numpy as np
from tkinter import Tk, filedialog, simpledialog, messagebox
from tkinter import ttk 
from skimage import io

def select_images():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilenames(title='Select Image Files', 
                                            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if not file_path:
        messagebox.showinfo("Info", "No file selected.")
        return -1
    return root.tk.splitlist(file_path)

def select_watermark():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title='Select Watermark Image', 
                                           filetypes=(("PNG files", "*.png"), ("All files", "*.*")))
    if not file_path:
        messagebox.showinfo("Info", "No watermark selected.")
        return -1
    return file_path

def find_low_activity_area_for_watermark(image, watermark_shape):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Get the dimensions of the watermark
    wm_h, wm_w = watermark_shape[:2]

    # Get the dimensions of the image
    img_h, img_w = edges.shape

    # Define regions to check based on the watermark size
    regions = {
        'bottom_middle': edges[img_h-wm_h:, img_w//2-wm_w//2:img_w//2+wm_w//2],
        'bottom_left': edges[img_h-wm_h:, :wm_w],
        'bottom_right': edges[img_h-wm_h:, img_w-wm_w:],
        # 'top_right': edges[:wm_h, img_w-wm_w:],
        # 'top_middle': edges[:wm_h, img_w//2-wm_w//2:img_w//2+wm_w//2],
        # 'top_left': edges[:wm_h, :wm_w]
    }

    # Initialize the minimum edge sum to a high value and the selected region
    min_edges_sum = float('inf')
    selected_region = 'bottom_right'

    # Iterate over regions to find the one with the least edge sum
    for region, region_edges in regions.items():
        edge_sum = np.sum(region_edges)
        if edge_sum < min_edges_sum:
            min_edges_sum = edge_sum
            selected_region = region

    return selected_region

def add_watermark(image_path, watermark_path, transparency):
    image = io.imread(image_path)
    watermark = io.imread(watermark_path)

    # Ensure image is in RGB
    if image.ndim == 2:  # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] == 4:  # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Handling watermark with alpha channel
    if watermark.shape[-1] == 4:
        alpha_channel = watermark[:, :, 3] / 255.0  # Normalize alpha values
        watermark_rgb = watermark[:, :, :3]
    else:
        # If no alpha channel, create a uniform one with full opacity
        watermark_rgb = watermark
        alpha_channel = np.ones((watermark.shape[0], watermark.shape[1]))
    
    h_img, w_img, _ = image.shape
    h_wm, w_wm, _ = watermark_rgb.shape

    selected_region = find_low_activity_area_for_watermark(image, watermark_rgb.shape)


    # Scale watermark
    if w_img > h_img:  # Landscape
        scale_factor = min(1.0, min(h_img, w_img) / (4 * max(h_wm, w_wm)))
    else:  # Portrait
        scale_factor = min(1.2, min(h_img, w_img) / (4 * max(h_wm, w_wm)))
    resized_watermark = cv2.resize(watermark_rgb, (0, 0), fx=scale_factor, fy=scale_factor)
    resized_alpha_channel = cv2.resize(alpha_channel, (resized_watermark.shape[1], resized_watermark.shape[0]))

    # Choose location
    locations = [
        (h_img - resized_watermark.shape[0], w_img - resized_watermark.shape[1]),  # bottom right
        (h_img - resized_watermark.shape[0], 0),  # bottom left
        (h_img - resized_watermark.shape[0], (w_img - resized_watermark.shape[1]) // 2),  # bottom middle
        (0, w_img - resized_watermark.shape[1]),  # top right
        (0, (w_img - resized_watermark.shape[1]) // 2),  # top middle
        (0, 0)  # top left
    ]
        # Placeholder for actual position based on selected_region
    if selected_region == 'bottom_right':
        pos_y, pos_x = locations[0]
    elif selected_region == 'bottom_left':
        pos_y, pos_x = locations[1]
    elif selected_region == 'bottom_middle':
        pos_y, pos_x = locations[2]
    elif selected_region == 'top_right':
        pos_y, pos_x = locations[3]
    elif selected_region == 'top_middle':
        pos_y, pos_x = locations[4]
    elif selected_region == 'top_left':
        pos_y, pos_x = locations[5]

    # Check background brightness at the selected location
    background_slice = image[pos_y:pos_y+resized_watermark.shape[0], pos_x:pos_x+resized_watermark.shape[1]]
    if np.mean(background_slice) < 127:  # Dark background, adjust watermark color
        # Invert watermark color for visibility
        resized_watermark = 255 - resized_watermark

    # Blend watermark using its alpha channel
    for c in range(0, 3):
        image[pos_y:pos_y+resized_watermark.shape[0], pos_x:pos_x+resized_watermark.shape[1], c] = \
            resized_alpha_channel * resized_watermark[:, :, c] * transparency + \
            image[pos_y:pos_y+resized_watermark.shape[0], pos_x:pos_x+resized_watermark.shape[1], c] * (1 - resized_alpha_channel * transparency)

    return image



def save_image(image, original_path):
    # save based on original file extension
    save_path = os.path.splitext(original_path)[0] + "_watermarked" + os.path.splitext(original_path)[1]
    io.imsave(save_path, image)
    return save_path

def main():
    # Select images and watermark
    images = select_images()
    if not images:
        exit()
    
    watermark = select_watermark()
    if not watermark:
        exit()

    transparency = simpledialog.askfloat("Transparency", "Enter watermark transparency (0.0 - 1.0):", initialvalue=0.5, minvalue=0.0, maxvalue=1.0)
    if transparency is None:
        messagebox.showinfo("Info", "No transparency selected.")
        exit()

    # Configure the progress bar
    root = Tk()
    root.title("Watermarking Progress")

    # Set up the progress bar
    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=20)
    progress['maximum'] = len(images)

    for index, image_path in enumerate(images):
        if "watermarked" in image_path:
                continue
        try:
            watermarked_image = add_watermark(image_path, watermark, transparency)
            save_path = save_image(watermarked_image, image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add watermark. Error: {e}")
            break
        progress['value'] = index + 1
        root.update_idletasks()  # Update the progress bar

    messagebox.showinfo("Success", "Watermark added successfully!")
    root.destroy()

if __name__ == "__main__":
    main()

