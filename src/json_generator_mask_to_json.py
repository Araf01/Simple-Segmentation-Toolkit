import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import cv2
import json
import os
import platform

def convert_mask_to_json_data(mask_image, class_mapping, log_callback=None):
    """
    Converts mask images into a JSON data structure compatible with the labeling app(label.py).

    :param mask_image: The mask image loaded as a NumPy array (grayscale).
    :param class_mapping: A dictionary mapping {pixel_value: "class_label"}.
    :param log_callback: A function to send log messages to.
    :return: A dictionary containing the formatted annotation data.
    """
    def log(message):
        if log_callback: log_callback(message)
        else: print(message)

    height, width = mask_image.shape[:2]
    json_data = {
        "annotations": [],
        "original_size": [width, height]
    }

    # Iterate through each class we need to find in the mask
    for pixel_value, label in class_mapping.items():
        # Create a binary image containing only the pixels for the current class
        # This isolates one class at a time for contour finding
        binary_mask = cv2.inRange(mask_image, pixel_value, pixel_value)

        # Find the contours of the shapes for the current class
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        log(f"  Found {len(contours)} contour(s) for class '{label}' (pixel value {pixel_value})")

        for contour in contours:
            # Ignore tiny contours that are likely just noise
            if cv2.contourArea(contour) < 4:
                continue

            # The contour is a list of points. The labeling app uses "freehand" for this.
            # Convert contour from numpy format to a simple list of [x, y] pairs.
            # .squeeze() removes unnecessary dimensions from the array.
            coordinates = contour.squeeze().tolist()

            # Ensure coordinates are in the correct list-of-lists format even if it's a short line
            if not isinstance(coordinates[0], list):
                coordinates = [coordinates]

            annotation = {
                "label": label,
                "type": "freehand",  # All detected shapes are treated as 'freehand' polygons
                "coordinates_original": coordinates
            }
            json_data["annotations"].append(annotation)

    return json_data


class MaskConverterApp:
    def __init__(self, master):
        self.master = master
        master.title("Mask to JSON Converter")
        
        # UI Setup
        window_width = 700
        window_height = 600
        center_x = int(master.winfo_screenwidth() / 2 - window_width / 2)
        center_y = int(master.winfo_screenheight() / 2 - window_height / 2)
        master.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        master.minsize(650, 550)

        self.style = ttk.Style(master)
        if 'clam' in self.style.theme_names(): self.style.theme_use('clam')

        self.mask_dir_path = tk.StringVar()
        self.json_output_dir_path = tk.StringVar()

        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input/Output Paths ---
        io_frame = ttk.LabelFrame(main_frame, text="Input & Output", padding="10")
        io_frame.pack(fill=tk.X, pady=(0, 10))
        io_frame.columnconfigure(1, weight=1)

        ttk.Label(io_frame, text="Input Masks Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(io_frame, textvariable=self.mask_dir_path).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(io_frame, text="Browse...", command=self.select_mask_dir).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(io_frame, text="Base Output Directory:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(io_frame, textvariable=self.json_output_dir_path).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(io_frame, text="Browse...", command=self.select_json_output_dir).grid(row=1, column=2, padx=5, pady=5)

        # --- Class Mapping Configuration ---
        mapping_frame = ttk.LabelFrame(main_frame, text="Class Mapping", padding="10")
        mapping_frame.pack(fill=tk.X, pady=(0, 10))

        mapping_info = "Enter one mapping per line.\nFormat: pixel_value, class_label\nExample:\n255, lines\n170, object"
        ttk.Label(mapping_frame, text=mapping_info, justify=tk.LEFT).pack(anchor="w", pady=(0, 5))

        self.mapping_text = tk.Text(mapping_frame, height=5, width=40, font=("Consolas", 10))
        self.mapping_text.pack(fill=tk.X, expand=True)
        self.mapping_text.insert("1.0", "255, lines\n170, person\n85, vehicle") # Example mapping

        # --- Actions and Logging ---
        ttk.Button(main_frame, text="Convert Masks to JSON", command=self.start_conversion, style="Accent.TButton").pack(pady=10, ipady=5)

        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, font=("Consolas", 9))
        self.log_text_widget.pack(fill=tk.BOTH, expand=True)
        self.log_text_widget.configure(state='disabled')
        
        self.apply_styles()

    def apply_styles(self):
        default_font = ('Segoe UI', 10) if platform.system() == "Windows" else ('Helvetica', 10)
        bold_font = (default_font[0], default_font[1], 'bold')
        self.style.configure('TButton', font=default_font, padding=(6,4))
        self.style.configure('Accent.TButton', font=bold_font, padding=(8,6), background='#007bff', foreground='white')
        self.style.map('Accent.TButton', background=[('active', '#0056b3')])
        self.style.configure('TLabelFrame.Label', font=(default_font[0], default_font[1], 'bold'))

    def log_message(self, message):
        self.log_text_widget.configure(state='normal')
        self.log_text_widget.insert(tk.END, message + "\n")
        self.log_text_widget.see(tk.END)
        self.log_text_widget.configure(state='disabled')
        self.master.update_idletasks()

    def select_mask_dir(self):
        dirpath = filedialog.askdirectory(title="Select Input Masks Directory")
        if dirpath: self.mask_dir_path.set(dirpath)

    def select_json_output_dir(self):
        dirpath = filedialog.askdirectory(title="Select Base Output Directory")
        if dirpath: self.json_output_dir_path.set(dirpath)
        
    def parse_class_mapping(self):
        """Parses the text from the mapping widget into a dictionary."""
        mapping = {}
        text = self.mapping_text.get("1.0", tk.END)
        lines = text.strip().split("\n")
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 2:
                raise ValueError(f"Invalid format on line {i+1}: '{line}'. Expected 'value, label'.")
            
            try:
                pixel_value = int(parts[0])
            except ValueError:
                raise ValueError(f"Invalid pixel value on line {i+1}: '{parts[0]}'. Must be an integer.")

            label = parts[1]
            if not label:
                raise ValueError(f"Empty label on line {i+1}.")
            
            mapping[pixel_value] = label
        return mapping

    def start_conversion(self):
        mask_dir = self.mask_dir_path.get()
        base_output_dir = self.json_output_dir_path.get()

        # Validate paths
        if not all([mask_dir, base_output_dir]):
            messagebox.showerror("Missing Paths", "Please specify both input and base output directories.")
            return
        if not os.path.isdir(mask_dir):
            messagebox.showerror("Directory Not Found", f"Input mask directory not found:\n{mask_dir}")
            return
            
        # *** MODIFICATION: Define and create the specific subfolder for JSON output ***
        json_output_dir = os.path.join(base_output_dir, "json_data_from_masks")
        os.makedirs(json_output_dir, exist_ok=True)

        # Validate class mapping
        try:
            class_mapping = self.parse_class_mapping()
            if not class_mapping:
                messagebox.showerror("Invalid Input", "Class mapping cannot be empty.")
                return
        except ValueError as e:
            messagebox.showerror("Invalid Class Mapping", str(e))
            return
        
        # Clear log and start process
        self.log_text_widget.configure(state='normal'); self.log_text_widget.delete('1.0', tk.END); self.log_text_widget.configure(state='disabled')
        self.log_message("--- Starting Mask to JSON Conversion ---")
        self.log_message(f"Input Directory: {mask_dir}")
        self.log_message(f"Output Directory: {json_output_dir}") # Log the actual final directory
        self.log_message(f"Class Mapping Parsed: {class_mapping}")

        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if not mask_files:
            self.log_message("Error: No mask images found in the input directory.")
            messagebox.showwarning("No Files Found", "No compatible image files were found in the selected input directory.")
            return
            
        converted_count = 0
        for mask_filename in mask_files:
            self.log_message(f"\nProcessing: {mask_filename}")
            full_mask_path = os.path.join(mask_dir, mask_filename)
            
            try:
                # Load the mask as a grayscale image
                mask_image = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_image is None:
                    self.log_message("  Error: Could not read image file. Skipping.")
                    continue
                
                # Perform the conversion
                json_data = convert_mask_to_json_data(mask_image, class_mapping, self.log_message)
                
                # If no annotations were found, don't save an empty file
                if not json_data["annotations"]:
                    self.log_message("  Warning: No contours found for the given classes. No JSON file will be created.")
                    continue

                # Save the resulting JSON file
                json_base_name = os.path.splitext(mask_filename)[0].replace('_mask', '')
                json_filename = f"{json_base_name}.json"
                # *** MODIFICATION: Use the new output directory for saving ***
                json_save_path = os.path.join(json_output_dir, json_filename)
                
                with open(json_save_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
                
                self.log_message(f"  Successfully saved JSON to: {json_save_path}")
                converted_count += 1

            except Exception as e:
                self.log_message(f"  FATAL ERROR during processing: {e}")
        
        summary = f"\n--- Conversion Complete ---\nSuccessfully converted {converted_count} mask(s)."
        self.log_message(summary)
        messagebox.showinfo("Process Complete", summary.strip())

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskConverterApp(root)
    root.mainloop()