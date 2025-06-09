import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import cv2
import numpy as np
import json
import os
import platform

# --- Configuration ---
# IMPORTANT: User should verify this mapping matches their labels.
CLASS_MAPPING = {
    "background": 0,
    "lines": 1,
    "object": 2,
    "person": 3,
    "vehicle": 4,
    "animal": 5,
    "marker": 6,
    "path": 7
    # Add any other custom classes here
}
DEFAULT_LINE_THICKNESS = 5


def create_mask_from_annotations(annotation_data,
                                 image_filename,
                                 class_to_id_mapping,
                                 line_thickness=DEFAULT_LINE_THICKNESS,
                                 image_dir=None,
                                 log_callback=None):
    """
    Generates a segmentation mask for a single image from its annotation data.
    The annotation_data is the loaded content of a single JSON file.
    """
    def log(message):
        if log_callback: log_callback(message)
        else: print(message)

    img_info = annotation_data
    
    # Try to get image size from the annotation data first
    original_size = None
    if "original_size" in img_info and isinstance(img_info["original_size"], list) and len(img_info["original_size"]) == 2:
        original_size = img_info["original_size"]
    
    # If not in JSON, try to get from the actual image file
    if original_size is None:
        if image_dir:
            original_image_path = os.path.join(image_dir, image_filename)
            if os.path.exists(original_image_path):
                try:
                    img_cv = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
                    if img_cv is not None:
                        height, width = img_cv.shape[:2]
                        original_size = [width, height]
                        log(f"Info: 'original_size' missing for '{image_filename}'. Used dimensions from file: {original_size}")
                    else:
                        log(f"Error: Could not read original image '{original_image_path}'.")
                        return None, None
                except Exception as e:
                    log(f"Error reading original image '{original_image_path}': {e}")
                    return None, None
            else:
                log(f"Error: 'original_size' missing in JSON and original image not found at '{original_image_path}'.")
                return None, None
        else:
            log(f"Error: 'original_size' missing for '{image_filename}' and no image_dir provided to find the file.")
            return None, None

    width, height = int(original_size[0]), int(original_size[1])
    if width <= 0 or height <= 0:
        log(f"Error: Invalid original_size {original_size} for '{image_filename}'.")
        return None, None

    mask = np.zeros((height, width), dtype=np.uint8)

    if "annotations" not in img_info or not img_info["annotations"]:
        log(f"Info: No annotations to draw for '{image_filename}'. Returning empty mask.")
        return mask, (width, height)

    for ann_idx, ann in enumerate(img_info["annotations"]):
        label = ann.get("label")
        ann_type = ann.get("type")
        coords_original = ann.get("coordinates_original")

        if not all([label, ann_type, coords_original is not None]):
            log(f"Warning: Incomplete annotation (idx {ann_idx}) for '{image_filename}'. Skipping.")
            continue

        class_id = class_to_id_mapping.get(label)
        if class_id is None:
            log(f"Warning: Label '{label}' for '{image_filename}' not in CLASS_MAPPING. Skipping.")
            continue

        try:
            if ann_type == "rectangle":
                if len(coords_original) == 4:
                    x1, y1, x2, y2 = map(int, map(round, coords_original))
                    pt1 = (min(x1, x2), min(y1, y2))
                    pt2 = (max(x1, x2), max(y1, y2))
                    cv2.rectangle(mask, pt1, pt2, color=int(class_id), thickness=cv2.FILLED)
                else: log(f"Warning: Rectangle (ann idx {ann_idx}) for '{image_filename}' has incorrect coords.")
            
            elif ann_type == "line":
                if len(coords_original) == 4:
                    x1, y1, x2, y2 = map(int, map(round, coords_original))
                    cv2.line(mask, (x1, y1), (x2, y2), color=int(class_id), thickness=line_thickness)
                else: log(f"Warning: Line (ann idx {ann_idx}) for '{image_filename}' has incorrect coords.")

            elif ann_type == "freehand":
                if isinstance(coords_original, list) and len(coords_original) >= 2:
                    points_int = np.array(coords_original, dtype=np.float32).round().astype(np.int32)
                    cv2.polylines(mask, [points_int], isClosed=False, color=int(class_id), thickness=line_thickness)
                else: log(f"Warning: Freehand (ann idx {ann_idx}) for '{image_filename}' has too few points.")
            else:
                log(f"Warning: Unknown annotation type '{ann_type}' for '{image_filename}'.")
        except Exception as e:
            log(f"Error drawing annotation (idx {ann_idx}, type {ann_type}) for '{image_filename}': {e}")
            
    return mask, (width, height)


class MaskGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Segmentation Mask Generator")
        
        window_width = 700
        window_height = 550
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        master.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        master.minsize(600, 450)

        # Apply a ttk theme first for better styling control
        self.style = ttk.Style(master)
        available_themes = self.style.theme_names()
        if 'clam' in available_themes: self.style.theme_use('clam')
        elif 'alt' in available_themes: self.style.theme_use('alt')
        else: self.style.theme_use(available_themes[0] if available_themes else 'default')

        # Variables for paths and options
        self.json_dir_path = tk.StringVar()
        self.image_dir_path = tk.StringVar()
        self.mask_output_dir_path = tk.StringVar()
        self.line_thickness_var = tk.StringVar(value=str(DEFAULT_LINE_THICKNESS))

        # --- Main Frame ---
        main_frame = ttk.Frame(master, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)

        # --- Path Selection UI ---
        ttk.Label(config_frame, text="Annotation Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        json_entry = ttk.Entry(config_frame, textvariable=self.json_dir_path, width=50)
        json_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(config_frame, text="Browse...", command=self.select_json_dir).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(config_frame, text="Original Images Directory:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        img_dir_entry = ttk.Entry(config_frame, textvariable=self.image_dir_path, width=50)
        img_dir_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(config_frame, text="Browse...", command=self.select_image_dir).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(config_frame, text="Mask Output Directory:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        mask_dir_entry = ttk.Entry(config_frame, textvariable=self.mask_output_dir_path, width=50)
        mask_dir_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(config_frame, text="Browse...", command=self.select_mask_output_dir).grid(row=2, column=2, padx=5, pady=5)

        # --- Options Frame ---
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0,10))

        ttk.Label(options_frame, text="Line/Freehand Thickness:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(options_frame, textvariable=self.line_thickness_var, width=5).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # --- Action Button ---
        self.generate_button = ttk.Button(main_frame, text="Generate Masks", command=self.start_mask_generation, style="Accent.TButton")
        self.generate_button.pack(pady=10, ipady=5)

        # --- Log Area ---
        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, font=("Consolas", 9) if platform.system() == "Windows" else ("Monaco", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state='disabled')

        self.apply_styles() # Apply custom styles

    def apply_styles(self):
        default_font = ('Segoe UI', 10) if platform.system() == "Windows" else ('Helvetica', 10)
        bold_font = (default_font[0], default_font[1], 'bold')
        
        self.style.configure('TButton', font=default_font, padding=(6,4))
        
        # --- MODIFIED: This is the style for your blue button ---
        accent_bg = '#007bff' # A standard blue color
        accent_fg = 'white'
        self.style.configure('Accent.TButton', font=bold_font, padding=(8,6), background=accent_bg, foreground=accent_fg)
        self.style.map('Accent.TButton', 
                       background=[('active', '#0056b3'), ('pressed', '#004085')], # Darker blue on hover/press
                       relief=[('pressed', 'sunken'), ('!pressed', 'raised')])

        self.style.configure('TLabelFrame.Label', font=(default_font[0], default_font[1], 'bold'))

    def log_message(self, message):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        self.master.update_idletasks()

    def select_json_dir(self):
        dirpath = filedialog.askdirectory(title="Select Annotation Directory")
        if dirpath: self.json_dir_path.set(dirpath)

    def select_image_dir(self):
        dirpath = filedialog.askdirectory(title="Select Original Images Directory")
        if dirpath: self.image_dir_path.set(dirpath)

    def select_mask_output_dir(self):
        dirpath = filedialog.askdirectory(title="Select Mask Output Directory")
        if dirpath: self.mask_output_dir_path.set(dirpath)

    def find_matching_image(self, image_dir, base_name):
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(path):
                return base_name + ext
        return None

    def start_mask_generation(self):
        json_dir = self.json_dir_path.get()
        image_dir = self.image_dir_path.get()
        mask_output_dir = self.mask_output_dir_path.get()
        
        try:
            line_thickness = int(self.line_thickness_var.get())
            if line_thickness <= 0: raise ValueError("Line thickness must be positive.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Line thickness must be a positive integer.")
            return

        if not all([json_dir, image_dir, mask_output_dir]):
            messagebox.showerror("Missing Paths", "Please specify all input and output directories.")
            return
        if not os.path.isdir(json_dir):
            messagebox.showerror("Directory Not Found", f"Annotation directory not found:\n{json_dir}")
            return
        if not os.path.isdir(image_dir):
            messagebox.showerror("Directory Not Found", f"Image directory not found:\n{image_dir}")
            return
        
        os.makedirs(mask_output_dir, exist_ok=True)
        
        self.log_text.configure(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.configure(state='disabled')

        self.log_message("--- Starting Mask Generation ---")
        self.log_message(f"Annotation Directory: {json_dir}")
        self.log_message(f"Image Directory: {image_dir}")
        self.log_message(f"Mask Output Directory: {mask_output_dir}")
        self.log_message(f"Line Thickness: {line_thickness}")
        self.log_message(f"Class Mapping Used: {CLASS_MAPPING}")

        try:
            json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
            if not json_files:
                self.log_message("Error: No JSON files found in the annotation directory.")
                messagebox.showwarning("No Files Found", "No .json files were found in the specified annotation directory.")
                return
        except Exception as e:
            self.log_message(f"Error reading annotation directory '{json_dir}': {e}"); return
            
        self.log_message(f"Found {len(json_files)} annotation files to process.")

        generated_mask_count = 0; failed_mask_count = 0
        
        for json_filename in json_files:
            self.log_message(f"\nProcessing: {json_filename}")
            json_base_name, _ = os.path.splitext(json_filename)
            
            actual_image_filename = self.find_matching_image(image_dir, json_base_name)
            
            if not actual_image_filename:
                self.log_message(f"  Warning: No matching image found for '{json_filename}' in '{image_dir}'. Skipping.")
                failed_mask_count +=1
                continue

            self.log_message(f"  Matching image found: {actual_image_filename}")
            
            full_json_path = os.path.join(json_dir, json_filename)
            try:
                with open(full_json_path, 'r') as f:
                    single_annotation_data = json.load(f)
            except Exception as e:
                self.log_message(f"  Error loading '{json_filename}': {e}. Skipping.")
                failed_mask_count += 1
                continue
                
            generated_mask_raw, _ = create_mask_from_annotations(
                single_annotation_data, actual_image_filename, CLASS_MAPPING,
                line_thickness=line_thickness, image_dir=image_dir,
                log_callback=self.log_message
            )
            
            if generated_mask_raw is not None:
                # --- THIS IS THE KEY CHANGE FOR VISIBILITY ---
                # Scale class IDs to be visible for the saved PNG mask
                # Pixels with value 1 will become 255 (white), 2 -> some gray, etc.
                visual_mask = np.zeros_like(generated_mask_raw, dtype=np.uint8)
                num_classes = len(CLASS_MAPPING) - 1 # Exclude background
                scale_factor = 255 / num_classes if num_classes > 0 else 255
                
                for class_name, class_id in CLASS_MAPPING.items():
                    if class_id != 0:
                        visual_mask[generated_mask_raw == class_id] = int(class_id * scale_factor)

                mask_filename = f"{json_base_name}_mask.png"
                mask_save_path = os.path.join(mask_output_dir, mask_filename)
                
                try:
                    cv2.imwrite(mask_save_path, visual_mask)
                    self.log_message(f"  Successfully saved visually distinct mask to: {mask_save_path}")
                    generated_mask_count += 1
                except Exception as e:
                    self.log_message(f"  Error saving mask for '{actual_image_filename}' to '{mask_save_path}': {e}")
                    failed_mask_count +=1
            else:
                self.log_message(f"  Failed to generate mask for '{json_filename}'.")
                failed_mask_count +=1

        summary_message = f"\n--- Mask Generation Complete ---\nGenerated: {generated_mask_count} masks.\nFailed/Skipped: {failed_mask_count} masks."
        self.log_message(summary_message)
        messagebox.showinfo("Process Complete", summary_message.replace("\n--- ", "\n").strip())


if __name__ == "__main__":
    root = tk.Tk()
    app = MaskGeneratorApp(root)
    root.mainloop()
