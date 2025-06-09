import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import json
import math
import platform

# --- Configuration ---
INITIAL_MAGNIFICATION = 1.0
DEFAULT_CLASSES_LIST = ["object", "lines","person", "vehicle", "animal", "marker", "path"] 
MIN_ZOOM_LEVEL = 0.02
MAX_ZOOM_LEVEL = 100.0
ZOOM_BUTTON_FACTOR = 1.20
ZOOM_WHEEL_FACTOR = 1.04

PAN_BUTTON = "<ButtonPress-3>"
PAN_MOTION = "<B3-Motion>"
PAN_RELEASE = "<ButtonRelease-3>"

FINAL_RESIZE_RESAMPLE_METHOD = Image.Resampling.BILINEAR
JSON_SUBDIR_NAME = "json_data"


class LabelingApp:
    DEBUG_VIEW_OPS = False

    def __init__(self, master):
        self.master = master
        master.title("Image Labeling App - Feature Rich")
        
        # --- Fullscreen/Maximized Startup ---
        try:
            master.state('zoomed')
        except tk.TclError:
            try:
                master.attributes('-zoomed', True)
            except tk.TclError:
                screen_w = master.winfo_screenwidth()
                screen_h = master.winfo_screenheight()
                master.geometry(f"{screen_w}x{screen_h}+0+0")
        
        # --- Apply a ttk theme ---
        self.style = ttk.Style(master)
        try:
            if 'clam' in self.style.theme_names():
                self.style.theme_use('clam')
            elif 'alt' in self.style.theme_names():
                self.style.theme_use('alt')
        except tk.TclError:
            if self.DEBUG_VIEW_OPS: print("Could not apply 'clam' or 'alt' ttk theme.")
        
        # --- App State Variables ---
        self.current_image_path = None
        self.original_pil_image = None
        self.display_tk_image = None
        self.image_files = []
        self.current_image_index = -1
        self.annotations = {} 
        self.available_classes = list(DEFAULT_CLASSES_LIST)
        self.has_unsaved_changes = False # Flag to track unsaved work

        # --- View State Variables ---
        self.base_scale_factor = 1.0
        self.zoom_level = 1.0
        self.effective_scale = 1.0
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0
        self.last_crop_box_original_coords = None

        self.pan_start_x_view_root = 0
        self.pan_start_y_view_root = 0
        self.pan_initial_offset_x = 0.0
        self.pan_initial_offset_y = 0.0
        self.is_panning = False

        self.start_x_canvas = None
        self.start_y_canvas = None
        self.current_shape_id = None
        self.current_freehand_points_canvas = []
        self._resize_after_id = None
        
        self.hovered_ann_tag = None

        # --- UI Layout ---
        main_frame = ttk.Frame(master, padding=10)
        main_frame.pack(fill="both", expand=True)

        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        controls_frame.pack(side="left", fill="y", padx=(0,10))

        self.canvas_frame = ttk.LabelFrame(main_frame, text="Image Display", padding=5)
        self.canvas_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg="dim gray")
        self.canvas.pack(fill="both", expand=True)

        # --- Control Widgets ---
        open_folder_button = ttk.Button(controls_frame, text="Open Image Folder", command=self.open_folder)
        open_folder_button.pack(fill="x", pady=(0,10))
        
        # --- Image Navigation ---
        nav_frame = ttk.LabelFrame(controls_frame, text="Navigation", padding=5)
        nav_frame.pack(fill="x", pady=5)

        prev_next_frame = ttk.Frame(nav_frame)
        prev_next_frame.pack(fill="x")
        ttk.Button(prev_next_frame, text="<< Prev", command=self.prev_image).pack(side="left", expand=True, fill="x", padx=(0,1))
        ttk.Button(prev_next_frame, text="Next >>", command=self.next_image).pack(side="right", expand=True, fill="x", padx=(1,0))

        jump_frame = ttk.Frame(nav_frame)
        jump_frame.pack(fill="x", pady=(5,0))
        ttk.Label(jump_frame, text="Go to Img #:").pack(side="left", padx=(0,2))
        self.jump_image_var = tk.StringVar()
        self.jump_image_entry = ttk.Entry(jump_frame, textvariable=self.jump_image_var, width=5)
        self.jump_image_entry.pack(side="left", padx=(0,3))
        ttk.Button(jump_frame, text="Go", command=self.jump_to_image, width=4).pack(side="left")
        self.jump_image_entry.bind("<Return>", self.jump_to_image)

        # --- Info Labels ---
        self.image_info_label = ttk.Label(controls_frame, text="No image loaded.", anchor="center")
        self.image_info_label.pack(fill="x", pady=5)
        self.mouse_pos_label = ttk.Label(controls_frame, text="Canvas: (-, -) | Orig: (-, -)", anchor="center")
        self.mouse_pos_label.pack(fill="x", pady=2)
        self.zoom_label = ttk.Label(controls_frame, text=f"Zoom: {self.zoom_level * 100:.0f}%", anchor="center")
        self.zoom_label.pack(fill="x", pady=2)

        # --- Class and Tool Selection ---
        class_tools_frame = ttk.LabelFrame(controls_frame, text="Classes & Tools", padding=10)
        class_tools_frame.pack(fill="x", pady=5, ipady=5)
        
        ttk.Label(class_tools_frame, text="Current Class:").pack(anchor="w", padx=5)
        self.current_class_var = tk.StringVar(master)
        self.class_dropdown = ttk.Combobox(
            class_tools_frame,
            textvariable=self.current_class_var,
            values=self.available_classes,
            state="readonly",
        )
        if self.available_classes: self.current_class_var.set(self.available_classes[0])
        self.class_dropdown.pack(fill="x", pady=(0,5), padx=5)

        # Add new class UI
        add_class_frame = ttk.Frame(class_tools_frame)
        add_class_frame.pack(fill="x", padx=5, pady=(0,5))
        self.new_class_var = tk.StringVar()
        ttk.Entry(add_class_frame, textvariable=self.new_class_var, width=15).pack(side="left", expand=True, fill="x", padx=(0,3))
        ttk.Button(add_class_frame, text="Add Class", command=self.add_new_class).pack(side="left")
        
        ttk.Label(class_tools_frame, text="Drawing Tool:").pack(anchor="w", padx=5)
        self.drawing_tool_var = tk.StringVar(master, "rectangle")
        ttk.Radiobutton(class_tools_frame, text="Rectangle", variable=self.drawing_tool_var, value="rectangle").pack(anchor="w", padx=10)
        ttk.Radiobutton(class_tools_frame, text="Line", variable=self.drawing_tool_var, value="line").pack(anchor="w", padx=10)
        ttk.Radiobutton(class_tools_frame, text="Freehand", variable=self.drawing_tool_var, value="freehand").pack(anchor="w", padx=10)
        ttk.Radiobutton(class_tools_frame, text="Select & Delete", variable=self.drawing_tool_var, value="select").pack(anchor="w", padx=10)
        self.drawing_tool_var.trace_add("write", self.update_canvas_cursor)


        # --- Zoom Controls ---
        zoom_control_frame = ttk.LabelFrame(controls_frame, text="View Controls", padding=5)
        zoom_control_frame.pack(fill="x", pady=5)
        zoom_buttons_frame = ttk.Frame(zoom_control_frame)
        zoom_buttons_frame.pack(fill="x")
        ttk.Button(zoom_buttons_frame,text="Zoom In (+)",command=lambda: self.adjust_zoom(ZOOM_BUTTON_FACTOR)).pack(side="left", expand=True, fill="x", padx=(0,1))
        ttk.Button(zoom_buttons_frame,text="Zoom Out (-)",command=lambda: self.adjust_zoom(1 / ZOOM_BUTTON_FACTOR)).pack(side="right", expand=True, fill="x", padx=(1,0))
        ttk.Button(zoom_control_frame, text="Reset View", command=self.reset_view).pack(fill="x", pady=(5,0))

        # --- Annotation Actions ---
        action_frame = ttk.LabelFrame(controls_frame, text="Annotation Actions", padding=5)
        action_frame.pack(fill="x", pady=10)
        ttk.Button(action_frame, text="Save Annotations", command=self.save_annotations, style="Accent.TButton").pack(fill="x", pady=(0,5), ipady=4)
        ttk.Button(action_frame, text="Clear Current Image Annotations", command=self.clear_current_annotations).pack(fill="x")
        
        self.apply_styles()

        # --- Event Bindings ---
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Motion>", self.on_canvas_mouse_move)
        self.canvas.bind(PAN_BUTTON, self.on_pan_press)
        self.canvas.bind(PAN_MOTION, self.on_pan_drag)
        self.canvas.bind(PAN_RELEASE, self.on_pan_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel_zoom)
        self.canvas.bind("<Button-4>", self.on_scroll_zoom)
        self.canvas.bind("<Button-5>", self.on_scroll_zoom)
        self.master.bind("<Configure>", self.on_window_resize_debounced)
        
        # --- MODIFIED: Intercept window close event ---
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def apply_styles(self):
        default_font = ('Segoe UI', 10) if platform.system() == "Windows" else ('Helvetica', 10)
        bold_font = (default_font[0], default_font[1], 'bold')
        title_font = (default_font[0], default_font[1] + 1, 'bold')

        self.style.configure('.', font=default_font, padding=3)
        self.style.configure('TButton', font=default_font, padding=(6, 4))
        self.style.map('TButton',
                  background=[('active', '#e0e0e0'), ('pressed', '#c0c0c0')],
                  relief=[('pressed', 'sunken'), ('!pressed', 'raised')])

        accent_bg = '#007bff'
        accent_fg = 'white'
        self.style.configure('Accent.TButton', font=bold_font, background=accent_bg, foreground=accent_fg, padding=(10,6))
        self.style.map('Accent.TButton',
                  background=[('active', '#0069d9'), ('pressed', '#005cbf')])

        self.style.configure('TLabel', padding=2)
        self.style.configure('TLabelframe', padding=8, borderwidth=1, relief='groove')
        self.style.configure('TLabelframe.Label', font=title_font, padding=(0,0,0,4))
        self.style.configure('TCombobox', padding=4, font=default_font)
        self.style.map('TCombobox', fieldbackground=[('readonly','white')])
        self.style.configure('TRadiobutton', font=default_font, padding=(5, 2))
        self.style.map('TRadiobutton',
                  indicatorbackground=[('selected', accent_bg), ('!selected', '#f0f0f0')],
                  indicatorforeground=[('selected', accent_fg)])
        self.style.configure('TEntry', padding=(4,3), font=default_font)


    def add_new_class(self):
        new_class = self.new_class_var.get().strip()
        if not new_class:
            messagebox.showwarning("Empty Class", "Class name cannot be empty.")
            return
        if new_class in self.available_classes:
            messagebox.showinfo("Class Exists", f"Class '{new_class}' already exists.")
        else:
            self.available_classes.append(new_class)
            self.class_dropdown['values'] = self.available_classes
            self.current_class_var.set(new_class)
            self.new_class_var.set("")
            if self.DEBUG_VIEW_OPS: print(f"Added new class: {new_class}. Available: {self.available_classes}")
            messagebox.showinfo("Class Added", f"Class '{new_class}' added.")

    def jump_to_image(self, event=None):
        if not self.image_files:
            messagebox.showinfo("No Images", "Please open a folder with images first.")
            return
        try:
            img_num_str = self.jump_image_var.get()
            if not img_num_str: return

            img_num = int(img_num_str)
            if 1 <= img_num <= len(self.image_files):
                self.current_image_index = img_num - 1
                self.load_image_and_annotations()
                self.jump_image_var.set("")
            else:
                messagebox.showerror("Invalid Number", f"Please enter a number between 1 and {len(self.image_files)}.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid image number.")
        except Exception as e:
            messagebox.showerror("Error Jumping", f"Could not jump to image: {e}")
            if self.DEBUG_VIEW_OPS: print(f"Error jumping to image: {e}")


    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        
        current_folder = os.path.dirname(self.image_files[0]) if self.image_files else None
        if folder_path != current_folder:
            self.annotations = {}
            self.current_image_index = -1
            self.has_unsaved_changes = False

        self.image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'))])
        
        if self.image_files:
            if self.current_image_index == -1:
                self.current_image_index = 0
            self.load_image_and_annotations()
        else:
            messagebox.showinfo("No Images", "No compatible images found in the selected folder.")
            self.image_info_label.config(text="No images found.")
            self.original_pil_image = None; self.display_tk_image = None; self.last_crop_box_original_coords = None
            self.canvas.delete("all")
            self.annotations = {}

    def save_annotations(self):
        if not self.image_files:
             messagebox.showwarning("Cannot Save", "Open an image folder first.")
             return
        if not self.annotations:
            messagebox.showinfo("Save Annotations", "There are no new or modified annotations in this session to save.")
            return

        saved_count = 0
        deleted_count = 0
        error_count = 0
        
        base_dir = os.path.dirname(self.image_files[0])
        json_dir = os.path.join(base_dir, JSON_SUBDIR_NAME)
        
        try:
            os.makedirs(json_dir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Directory Error", f"Could not create directory for annotations:\n{json_dir}\n\nError: {e}")
            return

        for img_basename, ann_data in self.annotations.items():
            json_basename = os.path.splitext(img_basename)[0] + ".json"
            save_path = os.path.join(json_dir, json_basename)
            
            try:
                if not ann_data.get("annotations"):
                    if os.path.exists(save_path):
                        os.remove(save_path)
                        deleted_count += 1
                else:
                    with open(save_path, 'w') as f:
                        json.dump(ann_data, f, indent=4)
                    saved_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error processing annotations for {img_basename}: {e}")

        # --- MODIFIED: Update unsaved changes flag and message ---
        self.has_unsaved_changes = False
        
        msg_parts = []
        if saved_count > 0:
            msg_parts.append(f"Successfully saved annotations for {saved_count} image(s).")
        if deleted_count > 0:
            msg_parts.append(f"Removed {deleted_count} empty annotation file(s).")
        
        final_message = "\n".join(msg_parts)

        if error_count > 0:
            messagebox.showerror("Save Complete with Errors", f"{final_message}\nFailed operations for {error_count} file(s). See console for details.")
        elif final_message:
            messagebox.showinfo("Save Complete", final_message)
        else:
            messagebox.showinfo("Save Annotations", "No changes required saving.")


    # --- View Manipulation Methods ---
    def on_window_resize_debounced(self, event):
        if event.widget == self.master:
            if self._resize_after_id:
                self.master.after_cancel(self._resize_after_id)
            self._resize_after_id = self.master.after(300, self.handle_resize)

    def handle_resize(self):
        if self.is_panning or not self.original_pil_image: return
        self.master.update_idletasks()
        canvas_width_before_resize = self.canvas.winfo_width()
        canvas_height_before_resize = self.canvas.winfo_height()
        current_eff_scale_before_resize = self.base_scale_factor * self.zoom_level
        if canvas_width_before_resize <= 1 or canvas_height_before_resize <= 1 or abs(current_eff_scale_before_resize) < 1e-9:
            self.reset_view(); return
        center_canvas_x_before = canvas_width_before_resize / 2
        center_canvas_y_before = canvas_height_before_resize / 2
        orig_center_x = (center_canvas_x_before - self.view_offset_x) / current_eff_scale_before_resize
        orig_center_y = (center_canvas_y_before - self.view_offset_y) / current_eff_scale_before_resize
        canvas_width_after_resize = self.canvas.winfo_width()
        canvas_height_after_resize = self.canvas.winfo_height()
        if canvas_width_after_resize <= 1 or canvas_height_after_resize <= 1: return
        original_width_px, original_height_px = self.original_pil_image.size
        if original_width_px == 0 or original_height_px == 0: return
        ratio_w = canvas_width_after_resize / original_width_px
        ratio_h = canvas_height_after_resize / original_height_px
        self.base_scale_factor = min(ratio_w, ratio_h) * INITIAL_MAGNIFICATION
        self.effective_scale = self.base_scale_factor * self.zoom_level
        if abs(self.effective_scale) < 1e-9:
            self.effective_scale = 0.01
            self.zoom_level = self.effective_scale / self.base_scale_factor if abs(self.base_scale_factor) > 1e-9 else MIN_ZOOM_LEVEL
        new_center_canvas_x = canvas_width_after_resize / 2
        new_center_canvas_y = canvas_height_after_resize / 2
        self.view_offset_x = new_center_canvas_x - (orig_center_x * self.effective_scale)
        self.view_offset_y = new_center_canvas_y - (orig_center_y * self.effective_scale)
        self.refresh_display()

    def _calculate_base_scale_and_initial_offsets(self):
        if not self.original_pil_image: return False
        self.master.update_idletasks()
        original_width, original_height = self.original_pil_image.size
        if original_width == 0 or original_height == 0: return False
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return False
        ratio_w = canvas_width / original_width
        ratio_h = canvas_height / original_height
        self.base_scale_factor = min(ratio_w, ratio_h) * INITIAL_MAGNIFICATION
        self.effective_scale = self.base_scale_factor * self.zoom_level
        if abs(self.effective_scale) < 1e-9: return False
        full_effective_width = original_width * self.effective_scale
        full_effective_height = original_height * self.effective_scale
        self.view_offset_x = (canvas_width - full_effective_width) / 2
        self.view_offset_y = (canvas_height - full_effective_height) / 2
        return True

    def reset_view(self):
        self.zoom_level = 1.0
        if self._calculate_base_scale_and_initial_offsets():
            self.zoom_label.config(text=f"Zoom: {self.zoom_level * 100:.0f}% (Fit)")
        else:
            self.zoom_label.config(text=f"Zoom: {self.zoom_level * 100:.0f}%")
        self.refresh_display()

    def adjust_zoom(self, factor, canvas_x_root=None, canvas_y_root=None):
        if not self.original_pil_image or abs(self.base_scale_factor) < 1e-9: return
        current_effective_scale = self.base_scale_factor * self.zoom_level
        if abs(current_effective_scale) < 1e-9: return
        current_canvas_width = self.canvas.winfo_width()
        current_canvas_height = self.canvas.winfo_height()
        if current_canvas_width <= 1 or current_canvas_height <= 1: return
        if canvas_x_root is None: canvas_x_root = current_canvas_width / 2
        if canvas_y_root is None: canvas_y_root = current_canvas_height / 2
        canvas_center_x = self.canvas.canvasx(canvas_x_root)
        canvas_center_y = self.canvas.canvasy(canvas_y_root)
        orig_x_at_zoom_center = (canvas_center_x - self.view_offset_x) / current_effective_scale
        orig_y_at_zoom_center = (canvas_center_y - self.view_offset_y) / current_effective_scale
        old_zoom_level = self.zoom_level
        self.zoom_level = max(MIN_ZOOM_LEVEL, min(self.zoom_level * factor, MAX_ZOOM_LEVEL))
        if abs(self.zoom_level - old_zoom_level) < 1e-5: return
        self.effective_scale = self.base_scale_factor * self.zoom_level
        self.view_offset_x = canvas_center_x - (orig_x_at_zoom_center * self.effective_scale)
        self.view_offset_y = canvas_center_y - (orig_y_at_zoom_center * self.effective_scale)
        self.zoom_label.config(text=f"Zoom: {self.zoom_level * 100:.0f}%")
        self.refresh_display()

    def on_mouse_wheel_zoom(self, event):
        if self.is_panning: return "break"
        factor = ZOOM_WHEEL_FACTOR if event.delta > 0 else 1 / ZOOM_WHEEL_FACTOR
        self.adjust_zoom(factor, event.x, event.y)
        return "break"

    def on_scroll_zoom(self, event):
        if self.is_panning: return "break"
        factor = ZOOM_WHEEL_FACTOR if event.num == 4 else 1 / ZOOM_WHEEL_FACTOR
        self.adjust_zoom(factor, event.x, event.y)
        return "break"

    def on_pan_press(self, event):
        if not self.original_pil_image: return
        self.is_panning = True
        self.pan_start_x_view_root = event.x_root
        self.pan_start_y_view_root = event.y_root
        self.pan_initial_offset_x = self.view_offset_x
        self.pan_initial_offset_y = self.view_offset_y
        self.canvas.config(cursor="fleur")

    def on_pan_drag(self, event):
        if not self.is_panning or not self.original_pil_image: return
        current_effective_scale_for_pan = self.base_scale_factor * self.zoom_level
        if abs(current_effective_scale_for_pan) < 1e-9: return
        delta_x = event.x_root - self.pan_start_x_view_root
        delta_y = event.y_root - self.pan_start_y_view_root
        new_view_offset_x = self.pan_initial_offset_x + delta_x
        new_view_offset_y = self.pan_initial_offset_y + delta_y
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 0 or canvas_height <= 0: return
        full_eff_img_width = self.original_pil_image.width * current_effective_scale_for_pan
        full_eff_img_height = self.original_pil_image.height * current_effective_scale_for_pan
        limit1_x = 0.0; limit2_x = canvas_width - full_eff_img_width
        self.view_offset_x = max(min(limit1_x, limit2_x), min(max(limit1_x, limit2_x), new_view_offset_x))
        limit1_y = 0.0; limit2_y = canvas_height - full_eff_img_height
        self.view_offset_y = max(min(limit1_y, limit2_y), min(max(limit1_y, limit2_y), new_view_offset_y))
        self.refresh_display()

    def on_pan_release(self, event):
        if not self.is_panning: return
        self.is_panning = False
        self.update_canvas_cursor()

    def refresh_display(self):
        if not self.original_pil_image:
            self.canvas.delete("all"); self.display_tk_image = None; self.last_crop_box_original_coords = None; return
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
        current_display_effective_scale = self.effective_scale
        if canvas_width <= 1 or canvas_height <= 1 or abs(current_display_effective_scale) < 1e-9:
            self.display_tk_image = None; self.last_crop_box_original_coords = None
            if abs(current_display_effective_scale) < 1e-9: self.canvas.create_text(canvas_width/2, canvas_height/2, text="<Zoom Error: Scale is zero>", fill="red")
            return
        img_w, img_h = self.original_pil_image.size
        target_crop_w_orig_f = canvas_width / current_display_effective_scale
        target_crop_h_orig_f = canvas_height / current_display_effective_scale
        crop_orig_x1_f = (-self.view_offset_x) / current_display_effective_scale
        crop_orig_y1_f = (-self.view_offset_y) / current_display_effective_scale
        crop_orig_x1_int = int(round(crop_orig_x1_f)); crop_orig_y1_int = int(round(crop_orig_y1_f))
        crop_w_int = max(1, int(round(target_crop_w_orig_f))); crop_h_int = max(1, int(round(target_crop_h_orig_f)))
        crop_orig_x2_int = crop_orig_x1_int + crop_w_int; crop_orig_y2_int = crop_orig_y1_int + crop_h_int
        final_crop_x1 = max(0, min(img_w, crop_orig_x1_int)); final_crop_y1 = max(0, min(img_h, crop_orig_y1_int))
        final_crop_x2 = max(0, min(img_w, crop_orig_x2_int)); final_crop_y2 = max(0, min(img_h, crop_orig_y2_int))
        if final_crop_x1 >= final_crop_x2: final_crop_x2 = final_crop_x1 + 1
        if final_crop_y1 >= final_crop_y2: final_crop_y2 = final_crop_y1 + 1
        final_crop_x2 = min(img_w, final_crop_x2); final_crop_y2 = min(img_h, final_crop_y2)
        if final_crop_x2 > 0: final_crop_x1 = min(final_crop_x1, final_crop_x2 - 1)
        if final_crop_y2 > 0: final_crop_y1 = min(final_crop_y1, final_crop_y2 - 1)
        final_crop_x1 = max(0, final_crop_x1); final_crop_y1 = max(0, final_crop_y1)
        self.last_crop_box_original_coords = [final_crop_x1, final_crop_y1, final_crop_x2, final_crop_y2]
        current_crop_width = final_crop_x2 - final_crop_x1; current_crop_height = final_crop_y2 - final_crop_y1
        if current_crop_width <= 0 or current_crop_height <= 0:
            self.display_tk_image = None
            self.draw_existing_annotations(); return
        try:
            cropped_pil_image_part = self.original_pil_image.crop(self.last_crop_box_original_coords)
            final_display_pil = cropped_pil_image_part.resize((max(1, canvas_width), max(1, canvas_height)), FINAL_RESIZE_RESAMPLE_METHOD)
            self.display_tk_image = ImageTk.PhotoImage(final_display_pil)
            self.canvas.create_image(0, 0, anchor="nw", image=self.display_tk_image, tags="bg_image")
        except Exception as e:
            if self.DEBUG_VIEW_OPS: print(f"Error in refresh_display: {e}")
            self.display_tk_image = None
        self.draw_existing_annotations()

    def view_to_original_coords(self, canvas_x, canvas_y):
        if not self.original_pil_image or not self.last_crop_box_original_coords: return canvas_x, canvas_y
        if abs(self.effective_scale) < 1e-9: return ((canvas_x - self.view_offset_x) / 1e-9, (canvas_y - self.view_offset_y) / 1e-9)
        canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
        crop_ox1, crop_oy1, crop_ox2, crop_oy2 = self.last_crop_box_original_coords
        cropped_width_orig = crop_ox2 - crop_ox1; cropped_height_orig = crop_oy2 - crop_oy1
        if canvas_width <= 0 or canvas_height <= 0 or cropped_width_orig <= 0 or cropped_height_orig <= 0:
            return ((canvas_x - self.view_offset_x) / self.effective_scale, (canvas_y - self.view_offset_y) / self.effective_scale)
        prop_x = canvas_x / canvas_width; prop_y = canvas_y / canvas_height
        original_x = crop_ox1 + (prop_x * cropped_width_orig); original_y = crop_oy1 + (prop_y * cropped_height_orig)
        return original_x, original_y

    def original_to_view_coords(self, original_x, original_y):
        if not self.original_pil_image or not self.last_crop_box_original_coords or abs(self.effective_scale) < 1e-9:
             return None, None
        canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
        crop_ox1, crop_oy1, crop_ox2, crop_oy2 = self.last_crop_box_original_coords
        cropped_width_orig = crop_ox2 - crop_ox1; cropped_height_orig = crop_oy2 - crop_oy1
        if cropped_width_orig <= 0 or cropped_height_orig <= 0 or canvas_width <= 0 or canvas_height <= 0: return None, None
        epsilon = 1e-9
        if not (crop_ox1 - epsilon <= original_x <= crop_ox2 + epsilon and crop_oy1 - epsilon <= original_y <= crop_oy2 + epsilon): return None, None
        relative_x_in_crop = original_x - crop_ox1; relative_y_in_crop = original_y - crop_oy1
        prop_x = relative_x_in_crop / cropped_width_orig if cropped_width_orig > 0 else 0
        prop_y = relative_y_in_crop / cropped_height_orig if cropped_height_orig > 0 else 0
        canvas_x = prop_x * canvas_width; canvas_y = prop_y * canvas_height
        return canvas_x, canvas_y

    def update_canvas_cursor(self, *args):
        tool = self.drawing_tool_var.get()
        if self.is_panning: self.canvas.config(cursor="fleur")
        elif tool == 'select': self.canvas.config(cursor="hand2")
        else: self.canvas.config(cursor="cross")

    def on_mouse_press(self, event):
        if self.drawing_tool_var.get() == 'select':
            self.handle_selection_click(event)
            return
        
        if not self.original_pil_image or self.is_panning: return
        self.start_x_canvas = self.canvas.canvasx(event.x); self.start_y_canvas = self.canvas.canvasy(event.y)
        tool = self.drawing_tool_var.get()
        color = {"rectangle": "red", "line": "blue", "freehand": "green"}.get(tool, "red")
        
        if tool == "rectangle": self.current_shape_id = self.canvas.create_rectangle(self.start_x_canvas, self.start_y_canvas, self.start_x_canvas, self.start_y_canvas, outline=color, width=2, tags="current_shape")
        elif tool == "line": self.current_shape_id = self.canvas.create_line(self.start_x_canvas, self.start_y_canvas, self.start_x_canvas, self.start_y_canvas, fill=color, width=2, tags="current_shape")
        elif tool == "freehand":
            self.current_freehand_points_canvas = [(self.start_x_canvas, self.start_y_canvas)]
            self.current_shape_id = self.canvas.create_line(self.current_freehand_points_canvas, fill=color, width=2, tags="current_shape", smooth=tk.TRUE, capstyle=tk.ROUND)

    def on_mouse_drag(self, event):
        if self.is_panning or self.drawing_tool_var.get() == 'select' or not self.start_x_canvas: return
        cur_x_canvas = self.canvas.canvasx(event.x); cur_y_canvas = self.canvas.canvasy(event.y)
        tool = self.drawing_tool_var.get()
        if tool in ["rectangle", "line"]:
            if self.current_shape_id: self.canvas.coords(self.current_shape_id, self.start_x_canvas, self.start_y_canvas, cur_x_canvas, cur_y_canvas)
        elif tool == "freehand":
            self.current_freehand_points_canvas.append((cur_x_canvas, cur_y_canvas))
            if self.current_shape_id: self.canvas.coords(self.current_shape_id, [c for p in self.current_freehand_points_canvas for c in p])
        self.update_mouse_pos_label(event.x, event.y)

    def on_mouse_release(self, event):
        if self.is_panning or self.drawing_tool_var.get() == 'select' or not self.start_x_canvas:
            self.reset_drawing_state()
            return
            
        end_x_canvas = self.canvas.canvasx(event.x); end_y_canvas = self.canvas.canvasy(event.y)
        tool = self.drawing_tool_var.get(); label = self.current_class_var.get()
        if not label: messagebox.showwarning("No Class", "Please select a class name."); self.reset_drawing_state(); return
        
        coords_orig = []
        if tool == "rectangle":
            orig_x1, orig_y1 = self.view_to_original_coords(self.start_x_canvas, self.start_y_canvas)
            orig_x2, orig_y2 = self.view_to_original_coords(end_x_canvas, end_y_canvas)
            if abs(orig_x1 - orig_x2) >= 1 and abs(orig_y1 - orig_y2) >= 1:
                coords_orig = [min(orig_x1, orig_x2), min(orig_y1, orig_y2), max(orig_x1, orig_x2), max(orig_y1, orig_y2)]
        elif tool == "line":
            orig_x1, orig_y1 = self.view_to_original_coords(self.start_x_canvas, self.start_y_canvas)
            orig_x2, orig_y2 = self.view_to_original_coords(end_x_canvas, end_y_canvas)
            if math.hypot(orig_x2 - orig_x1, orig_y2 - orig_y1) >= 1.0:
                coords_orig = [orig_x1, orig_y1, orig_x2, orig_y2]
        elif tool == "freehand":
            if len(self.current_freehand_points_canvas) >= 2:
                coords_orig = [self.view_to_original_coords(vx, vy) for vx, vy in self.current_freehand_points_canvas]

        if not coords_orig:
            self.reset_drawing_state()
            return

        img_name = os.path.basename(self.current_image_path)
        if img_name not in self.annotations:
            self.annotations[img_name] = {"annotations": [], "original_size": self.original_pil_image.size}
        self.annotations[img_name]["annotations"].append({"label": label, "type": tool, "coordinates_original": coords_orig})
        
        self.has_unsaved_changes = True # Mark that we have unsaved work
        self.draw_existing_annotations()
        self.reset_drawing_state()

    def reset_drawing_state(self):
        if self.current_shape_id: self.canvas.delete(self.current_shape_id)
        self.start_x_canvas = None; self.start_y_canvas = None; self.current_shape_id = None; self.current_freehand_points_canvas = []

    def draw_existing_annotations(self):
        self.canvas.delete("annotation_tag")
        if not self.original_pil_image or not self.last_crop_box_original_coords: return
        img_name = os.path.basename(self.current_image_path)
        if img_name in self.annotations:
            for ann_idx, ann in enumerate(self.annotations[img_name].get("annotations", [])):
                unique_tag = f"ann_idx_{ann_idx}"
                tags_tuple = ("annotation_tag", unique_tag)
                coords_orig_list = ann["coordinates_original"]; label = ann["label"]; ann_type = ann["type"]
                color = {"rectangle": "red", "line": "blue", "freehand": "green"}.get(ann_type, "orange")
                
                view_coords_list = []
                text_pos_v = None
                try:
                    if ann_type in ["rectangle", "line"]:
                        if len(coords_orig_list) != 4: continue
                        p1_orig_x, p1_orig_y, p2_orig_x, p2_orig_y = coords_orig_list
                        v_x1, v_y1 = self.original_to_view_coords(p1_orig_x, p1_orig_y)
                        v_x2, v_y2 = self.original_to_view_coords(p2_orig_x, p2_orig_y)
                        if v_x1 is None or v_x2 is None: continue
                        view_coords_list = [v_x1, v_y1, v_x2, v_y2]
                        if ann_type == "rectangle": text_pos_v = (min(v_x1, v_x2) + 3, min(v_y1, v_y2) + 3)
                        else: text_pos_v = ((v_x1 + v_x2) / 2 + 3, (v_y1 + v_y2) / 2 + 3)
                    elif ann_type == "freehand":
                        if len(coords_orig_list) < 2: continue
                        temp_view_points = [c for p in coords_orig_list for c in self.original_to_view_coords(*p) if c is not None]
                        if len(temp_view_points) < len(coords_orig_list) * 2: continue
                        view_coords_list = temp_view_points
                        text_pos_v = (view_coords_list[0] + 3, view_coords_list[1] + 3)

                    if view_coords_list:
                        if ann_type == "rectangle": self.canvas.create_rectangle(*view_coords_list, outline=color, width=2, tags=tags_tuple)
                        elif ann_type == "line": self.canvas.create_line(*view_coords_list, fill=color, width=2, tags=tags_tuple)
                        elif ann_type == "freehand": self.canvas.create_line(view_coords_list, fill=color, width=2, tags=tags_tuple, smooth=tk.TRUE, capstyle=tk.ROUND)
                        if text_pos_v: self.canvas.create_text(*text_pos_v, text=label, anchor="nw", fill=color, tags=tags_tuple, font=("Arial", 9, "bold"))
                except Exception as e:
                    if self.DEBUG_VIEW_OPS: print(f"Error drawing annotation index {ann_idx}: {e}")

    def on_canvas_mouse_move(self, event):
        if not self.original_pil_image: return
        self.update_mouse_pos_label(event.x, event.y)
        self.handle_annotation_hover(event)


    def update_mouse_pos_label(self, event_x_widget, event_y_widget):
        canvas_x = self.canvas.canvasx(event_x_widget); canvas_y = self.canvas.canvasy(event_y_widget)
        if self.original_pil_image and self.last_crop_box_original_coords:
            orig_x, orig_y = self.view_to_original_coords(canvas_x, canvas_y)
            self.mouse_pos_label.config(text=f"Canvas:({int(canvas_x)},{int(canvas_y)}) | Orig:({int(orig_x)},{int(orig_y)})")
        else: self.mouse_pos_label.config(text=f"Canvas:({int(canvas_x)},{int(canvas_y)}) | Orig: (No View)")


    def load_image_and_annotations(self):
        if not self.image_files or not (0 <= self.current_image_index < len(self.image_files)): return
        
        self.current_image_path = self.image_files[self.current_image_index]
        img_name = os.path.basename(self.current_image_path)
        if self.DEBUG_VIEW_OPS: print(f"Loading image: {img_name}")

        try:
            with Image.open(self.current_image_path) as img: self.original_pil_image = img.convert("RGB")
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Could not load image: {self.current_image_path}\n{e}")
            self.original_pil_image = None
            self.refresh_display(); return

        if img_name not in self.annotations:
            img_dir = os.path.dirname(self.current_image_path)
            json_basename = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(img_dir, JSON_SUBDIR_NAME, json_basename)
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        self.annotations[img_name] = json.load(f)
                        if self.DEBUG_VIEW_OPS: print(f"Loaded annotations for {img_name} from {json_path}")
                except Exception as e:
                    messagebox.showwarning("Annotation Load Error", f"Could not load or parse annotation file:\n{os.path.basename(json_path)}\n\nError: {e}")

        self.reset_view()
        self.master.title(f"Labeling App - {img_name} [{self.current_image_index+1}/{len(self.image_files)}]")
        self.image_info_label.config(text=f"{img_name} ({self.original_pil_image.size[0]}x{self.original_pil_image.size[1]})")
        self.jump_image_entry.delete(0, tk.END)
        self.jump_image_entry.insert(0, str(self.current_image_index + 1))


    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1; self.load_image_and_annotations()

    def prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1; self.load_image_and_annotations()

    def clear_current_annotations(self):
        if not self.current_image_path: return
        img_name = os.path.basename(self.current_image_path)
        
        if messagebox.askyesno("Confirm Clear", f"Clear all annotations for image: {img_name}?\n(This will be made permanent when you save)"):
            if img_name not in self.annotations:
                self.annotations[img_name] = {"annotations": [], "original_size": self.original_pil_image.size}
            else:
                self.annotations[img_name]["annotations"] = []
            self.has_unsaved_changes = True # Mark that we have unsaved work
            self.refresh_display()

    # --- Methods for Selection and Deletion ---

    def handle_annotation_hover(self, event):
        tool = self.drawing_tool_var.get()
        new_hover_tag = None
        if tool == 'select':
            item_ids = self.canvas.find_withtag(tk.CURRENT)
            if item_ids:
                for tag in self.canvas.gettags(item_ids[0]):
                    if tag.startswith("ann_idx_"):
                        new_hover_tag = tag
                        break
        if self.hovered_ann_tag != new_hover_tag:
            if self.hovered_ann_tag and self.canvas.find_withtag(self.hovered_ann_tag):
                self.canvas.itemconfig(self.hovered_ann_tag, width=2)
            if new_hover_tag and self.canvas.find_withtag(new_hover_tag):
                self.canvas.itemconfig(new_hover_tag, width=4)
            self.hovered_ann_tag = new_hover_tag
    
    def handle_selection_click(self, event):
        if not self.hovered_ann_tag: return
        try:
            ann_idx_to_delete = int(self.hovered_ann_tag.split('_')[-1])
            self.delete_annotation(ann_idx_to_delete)
        except (ValueError, IndexError) as e:
            if self.DEBUG_VIEW_OPS: print(f"Error parsing tag: {self.hovered_ann_tag}. Error: {e}")
            
    def delete_annotation(self, index_to_delete):
        if not self.current_image_path: return
        img_name = os.path.basename(self.current_image_path)
        if img_name in self.annotations and "annotations" in self.annotations[img_name]:
            annotations_list = self.annotations[img_name]["annotations"]
            if 0 <= index_to_delete < len(annotations_list):
                label_to_delete = annotations_list[index_to_delete].get('label', 'N/A')
                if messagebox.askyesno("Confirm Delete", f"Delete the selected '{label_to_delete}' annotation?"):
                    del annotations_list[index_to_delete]
                    self.has_unsaved_changes = True # Mark that we have unsaved work
                    self.hovered_ann_tag = None
                    self.refresh_display()

    # --- NEW METHOD for handling window close ---
    def on_closing(self):
        """Handles the window close event, prompting to save if necessary."""
        if self.has_unsaved_changes:
            response = messagebox.askyesnocancel("Exit", "You have unsaved changes. Do you want to save before exiting?")
            
            if response is True:  # Yes
                self.save_annotations()
                self.master.destroy()
            elif response is False:  # No
                self.master.destroy()
            else:  # Cancel
                return
        else:
            self.master.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()