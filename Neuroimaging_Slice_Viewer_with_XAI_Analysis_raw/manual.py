# manual.py
"""
Neuroimaging Slice Viewer
Copyright (c) 2024 Falah Sheikh, ADSA Lab, University of Calgary
Licensed under CC BY-NC 4.0 - see LICENSE file for details.

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 
International License. Commercial use is prohibited without explicit permission.

Developed at the Advanced Database Systems and Applications (ADSA) Lab,
University of Calgary, with funding from Alberta Innovates Summer Research Studentship.

Author: Falah Sheikh (https://github.com/falahsheikh/Lightweight_MRI_EAD_Detection)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import SimpleITK as sitk
import os
import re
from concurrent.futures import ThreadPoolExecutor
import skimage.measure
from functools import partial
from datetime import datetime
import json
import io
from PIL import Image, ImageTk
from loading_window import LoadingWindow
import time

try:
    import cv2
    import scipy.ndimage
    import tensorflow as tf
    from tensorflow.keras.models import Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow or OpenCV not found. XAI functionality will be disabled.")
    TENSORFLOW_AVAILABLE = False

AXIAL, CORONAL, SAGITTAL = 0, 1, 2
STATE_NONE, STATE_SLICE, STATE_WL, STATE_ZOOM, STATE_PAN, STATE_SELECT_ZOOM, STATE_DRAW = 0, 1, 2, 3, 4, 5, 6

class XAIProcessor:

    def __init__(self, model_path=None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for XAI functionality.")

        self.model = None
        self.model_path = model_path
        self.IMG_SIZE = (224, 224)
        self.CLASS_NAMES = ['CN', 'EMCI', 'LMCI']

        self.PREPROCESS_INPUT = tf.keras.applications.efficientnet_v2.preprocess_input

        if self.model_path:
            self.load_model(self.model_path)

        self.pan_mode = tk.BooleanVar(value=False)
        self.pan_mode.trace_add('write', self._on_pan_toggle)
        self.show_axis_scales = tk.BooleanVar(value=True) 

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.model_path = model_path

    def _prepare_image(self, slice_data):
        img_min, img_max = np.min(slice_data), np.max(slice_data)
        if img_max > img_min:
            slice_data = 255 * (slice_data - img_min) / (img_max - img_min)
        else:
            slice_data = np.zeros_like(slice_data) # Handle flat images
        slice_data = slice_data.astype(np.uint8)

        rgb_img = cv2.cvtColor(slice_data, cv2.COLOR_GRAY2RGB)
        resized_img = cv2.resize(rgb_img, self.IMG_SIZE, interpolation=cv2.INTER_AREA)

        array = tf.keras.preprocessing.image.img_to_array(resized_img)
        original_img = array.copy()
        array_preprocessed = self.PREPROCESS_INPUT(np.expand_dims(array, axis=0))
        return original_img, array_preprocessed

    def _find_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found in the model.")

    def _make_gradcam_plus_plus(self, img_tensor, class_idx):
        last_conv_layer_name = self._find_last_conv_layer()
        grad_model = Model(inputs=self.model.inputs, outputs=[self.model.get_layer(last_conv_layer_name).output, self.model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            class_output = predictions[:, class_idx]

        grads = tape.gradient(class_output, conv_outputs)
        conv_outputs, grads = conv_outputs[0], grads[0]
        grads_2 = tf.square(grads)
        grads_3 = tf.pow(grads, 3)
        alpha_denom = 2.0 * grads_2 + tf.reduce_sum(conv_outputs * grads_3, axis=(0, 1), keepdims=True) + 1e-7
        alphas = grads_2 / alpha_denom
        weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))
        heatmap = tf.reduce_sum(weights * conv_outputs, axis=2)
        heatmap = tf.nn.relu(heatmap)
        if tf.reduce_max(heatmap) > 0:
            heatmap /= tf.reduce_max(heatmap)
        return tf.image.resize(heatmap[..., tf.newaxis], self.IMG_SIZE, method='bilinear').numpy().squeeze()

    def generate_visualization(self, coronal_slice_data, slice_index):
        if self.model is None: return None

        original_img, img_array = self._prepare_image(coronal_slice_data)
        predictions = self.model.predict(img_array, verbose=0)[0]
        pred_idx = np.argmax(predictions)
        pred_label = self.CLASS_NAMES[pred_idx]
        confidence = predictions[pred_idx]

        heatmap = self._make_gradcam_plus_plus(img_array, pred_idx)

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(original_img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, heatmap_colored, 0.5, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(12, 4), facecolor='#1E1E1E')
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.1)
        fig.suptitle(f"XAI for Coronal Slice {slice_index} | Prediction: {pred_label} ({confidence:.2f})",
                         fontsize=12, fontweight='bold', color='white')
        
        titles = ['Original Slice', 'Grad-CAM++ Heatmap', 'Overlay']
        images = [original_img.astype('uint8'), heatmap, overlay]
        cmaps = [None, 'jet', None]
        
        for i, (title, img, cmap) in enumerate(zip(titles, images, cmaps)):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=10, color='white')
            ax.set_xticks([]); ax.set_yticks([])
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        return fig

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class SlicerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Neuroimaging Slice Viewer")
        self.root.geometry("1400x1000") 

        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        self._configure_styles() 

        self.volume_data = None
        self.dimensions = [0, 0, 0]
        self.spacing = [1.0, 1.0, 1.0]
        self.origin = [0.0, 0.0, 0.0]
        self.current_slices = [0, 0, 0]
        self.patient_info = {"name": "N/A", "id": "N/A"}
        self.patient_info_label = None
        self.filepath = ""

        self.window = 2000
        self.level = 1000
        self.vmin = 0
        self.vmax = 0
        self.colormap = tk.StringVar(value='gray')

        self.mouse_state = STATE_NONE
        self.last_mouse_pos = {'x': 0, 'y': 0}
        
        self.crosshair_enabled = tk.BooleanVar(value=True)
        self.show_overlays = tk.BooleanVar(value=True)
        self.show_overlays.trace_add('write', lambda *args: self.update_2d_views())
        
        self.pan_mode = tk.BooleanVar(value=False)
        self.pan_mode.trace_add('write', self._on_pan_toggle)
        self.show_axis_scales = tk.BooleanVar(value=True)

        self.zoom_factors = {'axial': 1.0, 'coronal': 1.0, 'sagittal': 1.0}
        self.zoom_centers = {'axial': (0.5, 0.5), 'coronal': (0.5, 0.5), 'sagittal': (0.5, 0.5)}
        self.zoom_selection = {'active': False, 'start': None, 'end': None, 'view': None, 'rect_patch': None}
        self.zoom_select_mode = tk.BooleanVar(value=False)
        self.zoom_select_mode.trace_add('write', self._on_zoom_select_toggle)

        self.mesh_verts = None
        self.mesh_faces = None
        self.mesh_color = tk.StringVar(value='ivory')
        self.mesh_opacity = tk.DoubleVar(value=0.7)
        self.iso_value = tk.DoubleVar(value=300)
        self.show_model = tk.BooleanVar(value=True)
        self.show_3d_axes = tk.BooleanVar(value=True)
        self.show_axial_plane = tk.BooleanVar(value=True)
        self.show_coronal_plane = tk.BooleanVar(value=True)
        self.show_sagittal_plane = tk.BooleanVar(value=True)

        self.drawings = {}
        self.measurements = []
        self.last_drawn_stroke = None
        self.last_measurement = None
        self.highlighted_item = {'type': None, 'id': None}
        self.annotation_id_counter = 0
        self.measurement_id_counter = 0
        self.temp_measure_point = None
        
        self.draw_opacity = tk.DoubleVar(value=0.8)
        self.draw_color = tk.StringVar(value='yellow')
        self.draw_size = tk.DoubleVar(value=2.0)
        
        self.measurement_mode = tk.BooleanVar(value=False)
        self.measurement_mode.trace_add('write', self._on_measure_toggle)
        self.draw_mode = tk.BooleanVar(value=False)
        self.draw_mode.trace_add('write', self._on_draw_toggle)

        self.history_stack = []
        self.redo_stack = []

        self.panes = {}
        self.figs = {}
        self.axes = {}
        self.canvases = {}
        self.maximize_buttons = {}
        self.maximized_view = None
        self.active_view = None
        self.view_states = {}
        
        self.scroll_timer = None
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.root.configure(bg=self.style.lookup('TFrame', 'background'))

        try:
            self.root.state('zoomed')
        except tk.TclError:
            # If zoomed doesn't work, just maximize normally
            self.root.attributes('-zoomed', True)
        except:
            # If that doesn't work either, just set a large size
            self.root.geometry("1400x1000")

        self._create_widgets()
        
        self.update_all_views()
        
        # Set up layout after a short delay
        self.root.after(100, self.reset_layout)
        
        # Set up close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self._update_history_buttons()

    def save_session(self):
        if self.volume_data is None:
            messagebox.showerror("Error", "No volume loaded to save session for.", parent=self.root)
            return 

        pat_id_safe = re.sub(r'[\W_]+', '', self.patient_info.get("id", "NA")) or "NA"
        pat_name_safe = re.sub(r'[\W_]+', '', self.patient_info.get("name", "Patient")) or "Patient"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_filename = f"Session_{pat_id_safe}_{pat_name_safe}_{timestamp}.json"
        
        filepath = filedialog.asksaveasfilename(
            initialfile=initial_filename,
            defaultextension=".json",
            filetypes=[("Session files", "*.json"), ("All files", "*.*")],
            title="Save Session",
            parent=self.root
        )
        
        if not filepath:
            return
        
        self.status_label.config(text="Saving session...")
        self.progress.pack(side=tk.RIGHT, padx=5)
        self.progress.start()
        self.root.update_idletasks()
        
        try:
            session_data = {
                "session_info": {
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "volume_file": self.filepath,
                    "volume_checksum": self._calculate_volume_checksum() if self.volume_data is not None else None
                },
                "patient_info": {
                    "name": self.patient_info.get("name", "N/A"),
                    "id": self.patient_info.get("id", "N/A")
                },
                "view_state": {
                    "current_slices": self.current_slices.copy(),
                    "window": self.window,
                    "level": self.level,
                    "zoom_factors": self.zoom_factors.copy(),
                    "zoom_centers": self.zoom_centers.copy(),
                    "colormap": self.colormap.get(),
                    "crosshair_enabled": self.crosshair_enabled.get(),
                    "show_overlays": self.show_overlays.get()
                },
                "drawings": self._serialize_drawings(),
                "measurements": self._serialize_measurements(),
                "tool_state": {
                    "draw_color": self.draw_color.get(),
                    "draw_size": self.draw_size.get(),
                    "draw_opacity": self.draw_opacity.get(),
                    "measurement_mode": self.measurement_mode.get(),
                    "draw_mode": self.draw_mode.get()
                },
                "3d_settings": {
                    "iso_value": self.iso_value.get(),
                    "mesh_color": self.mesh_color.get(),
                    "mesh_opacity": self.mesh_opacity.get(),
                    "show_model": self.show_model.get(),
                    "show_3d_axes": self.show_3d_axes.get()
                }
            }   

            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=4, cls=NumpyEncoder)
            
            self.status_label.config(text="Session saved successfully.")
            messagebox.showinfo("Success", f"Session saved to:\n{os.path.basename(filepath)}", parent=self.root)
            
        except Exception as e:
            self.status_label.config(text="Session save failed.")
            messagebox.showerror("Error", f"Failed to save session:\n{str(e)}", parent=self.root)
        
        finally:
            self.progress.stop()
            self.progress.pack_forget()

    def load_session(self):
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("Session files", "*.json"), ("All files", "*.*")],
            title="Load Session",
            parent=self.root
        )
        
        if not filepath:
            return
        
        self.status_label.config(text="Loading session...")
        self.progress.pack(side=tk.RIGHT, padx=5)
        self.progress.start()
        self.root.update_idletasks()
        
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            if not self._validate_session_data(session_data):
                raise ValueError("Invalid session file format")
            
            volume_file = session_data["session_info"]["volume_file"]
            if not os.path.exists(volume_file):
                new_volume_file = self._prompt_for_volume_file(volume_file)
                if not new_volume_file:
                    return
                volume_file = new_volume_file   

            self._load_volume_from_path(volume_file)           

            if session_data["session_info"]["volume_checksum"]:
                current_checksum = self._calculate_volume_checksum()
                if current_checksum != session_data["session_info"]["volume_checksum"]:
                    if not messagebox.askyesno("Volume Mismatch", 
                        "The loaded volume appears to be different from the one used when the session was saved. "
                        "Annotations may not align correctly. Continue loading session?", parent=self.root):
                        return
            
            if "patient_info" in session_data:
                self.patient_info = {
                    "name": session_data["patient_info"].get("name", "N/A"),
                    "id": session_data["patient_info"].get("id", "N/A")
                }
            else:

                self.patient_info = {"name": "N/A", "id": "N/A"}           

            self._update_patient_info_display()
            
            view_state = session_data["view_state"]
            self.current_slices = view_state["current_slices"].copy()
            self.window = view_state["window"]
            self.level = view_state["level"]
            self.zoom_factors = view_state["zoom_factors"].copy()
            self.zoom_centers = view_state["zoom_centers"].copy()
            self.colormap.set(view_state.get("colormap", "gray"))
            self.crosshair_enabled.set(view_state.get("crosshair_enabled", True))
            self.show_overlays.set(view_state.get("show_overlays", True))
            
            self._deserialize_drawings(session_data["drawings"])
            self._deserialize_measurements(session_data["measurements"])
            
            tool_state = session_data.get("tool_state", {})
            self.draw_color.set(tool_state.get("draw_color", "yellow"))
            self.draw_size.set(tool_state.get("draw_size", 2.0))
            self.draw_opacity.set(tool_state.get("draw_opacity", 0.8))
            self.measurement_mode.set(tool_state.get("measurement_mode", False))
            self.draw_mode.set(tool_state.get("draw_mode", False))
            
            settings_3d = session_data.get("3d_settings", {})
            self.iso_value.set(settings_3d.get("iso_value", 300))
            self.mesh_color.set(settings_3d.get("mesh_color", "ivory"))
            self.mesh_opacity.set(settings_3d.get("mesh_opacity", 0.7))
            self.show_model.set(settings_3d.get("show_model", True))
            self.show_3d_axes.set(settings_3d.get("show_3d_axes", True))
            
            self._update_wl_vars()
            self._update_wl_controls()
            self._update_slice_sliders()
            self._update_slice_labels()
            self._populate_annotation_trees()
            
            self.update_all_views()
            
            self.status_label.config(text="Session loaded successfully.")
            messagebox.showinfo("Success", f"Session loaded from:\n{os.path.basename(filepath)}", parent=self.root)
            
        except Exception as e:
            self.status_label.config(text="Session load failed.")
            messagebox.showerror("Error", f"Failed to load session:\n{str(e)}", parent=self.root)
        
        finally:
            self.progress.stop()
            self.progress.pack_forget()

    def _calculate_volume_checksum(self):
        if self.volume_data is None:
            return None

        shape_str = str(self.volume_data.shape)
        sample_indices = [
            (0, 0, 0),
            (self.volume_data.shape[0]//2, self.volume_data.shape[1]//2, self.volume_data.shape[2]//2),
            (-1, -1, -1)
        ]
        sample_values = [str(self.volume_data[idx]) for idx in sample_indices]
        checksum_str = shape_str + ''.join(sample_values)
        return str(hash(checksum_str))

    def _validate_session_data(self, session_data):
        required_keys = ["session_info", "patient_info", "view_state", "drawings", "measurements"]
        return all(key in session_data for key in required_keys)

    def _prompt_for_volume_file(self, original_path): 
        result = messagebox.askyesnocancel(
            "Volume File Not Found",
            f"The original volume file was not found:\n{original_path}\n\n"
            "Would you like to locate the volume file manually?",
            parent=self.root
        )
        
        if result is True:  # Yes - browse for file
            return filedialog.askopenfilename(
                title="Locate Volume File",
                filetypes=[
                    ("All supported", "*.nii *.nii.gz *.mhd"),
                    ("NIfTI files", "*.nii *.nii.gz"),
                    ("MetaImage", "*.mhd")
                ],
                parent=self.root
            )
        elif result is False:  # No - continue without volume
            return None
        else:  # Cancel
            return None

    def _load_volume_from_path(self, filepath):
        try:
            image = sitk.ReadImage(filepath)
            self.volume_data = sitk.GetArrayFromImage(image).astype(np.float32)
            self.dimensions = list(self.volume_data.shape)
            self.spacing = list(reversed(image.GetSpacing()))
            self.filepath = filepath
            

            self._configure_slice_sliders()
            
        except Exception as e:
            raise Exception(f"Failed to load volume from {filepath}: {str(e)}")

    def _serialize_drawings(self):
        serialized = {}
        for plane, slices in self.drawings.items():
            serialized[str(plane)] = {}
            for slice_idx, strokes in slices.items():
                serialized[str(plane)][str(slice_idx)] = []
                for stroke in strokes:
                    stroke_data = {
                        'id': stroke['id'],
                        'plane': stroke['plane'],
                        'slice': stroke['slice'],
                        'points': stroke['points'],
                        'color': stroke['color'],
                        'size': stroke['size'],
                        'opacity': stroke.get('opacity', 0.8)
                    }
                    if 'comment' in stroke:
                        stroke_data['comment'] = stroke['comment']
                    serialized[str(plane)][str(slice_idx)].append(stroke_data)
        return serialized

    def _deserialize_drawings(self, serialized_drawings):
        self.drawings.clear()
        self.annotation_id_counter = 0
        
        for plane_str, slices in serialized_drawings.items():
            plane = int(plane_str)
            self.drawings[plane] = {}
            
            for slice_str, strokes in slices.items():
                slice_idx = int(slice_str)
                self.drawings[plane][slice_idx] = []
                
                for stroke_data in strokes:
                    stroke = {
                        'id': stroke_data['id'],
                        'plane': stroke_data['plane'],
                        'slice': stroke_data['slice'],
                        'points': stroke_data['points'],
                        'color': stroke_data['color'],
                        'size': stroke_data['size'],
                        'opacity': stroke_data.get('opacity', 0.8)
                    }
                    if 'comment' in stroke_data:
                        stroke['comment'] = stroke_data['comment']
                    
                    self.drawings[plane][slice_idx].append(stroke)
                    

                    if stroke['id'] > self.annotation_id_counter:
                        self.annotation_id_counter = stroke['id']
                        self.last_drawn_stroke = stroke

    def _serialize_measurements(self):
        serialized = []
        for measure in self.measurements:
            measure_data = {
                'id': measure['id'],
                'p1': measure['p1'],
                'p2': measure['p2'],
                'plane': measure['plane'],
                'slice': measure['slice'],
                'color': measure['color'],
                'dist_mm': measure['dist_mm']
            }
            if 'comment' in measure:
                measure_data['comment'] = measure['comment']
            serialized.append(measure_data)
        return serialized

    def _deserialize_measurements(self, serialized_measurements):
        self.measurements.clear()
        self.measurement_id_counter = 0
        
        for measure_data in serialized_measurements:
            measure = {
                'id': measure_data['id'],
                'p1': tuple(measure_data['p1']),
                'p2': tuple(measure_data['p2']),
                'plane': measure_data['plane'],
                'slice': measure_data['slice'],
                'color': measure_data['color'],
                'dist_mm': measure_data['dist_mm']
            }
            if 'comment' in measure_data:
                measure['comment'] = measure_data['comment']
            
            self.measurements.append(measure)
            

            if measure['id'] > self.measurement_id_counter:
                self.measurement_id_counter = measure['id']
                self.last_measurement = measure

    def _populate_annotation_trees(self):
        for item in self.drawings_tree.get_children():
            self.drawings_tree.delete(item)
        for item in self.measurements_tree.get_children():
            self.measurements_tree.delete(item)

        for plane, slices in self.drawings.items():
            for slice_idx, strokes in slices.items():
                for stroke in strokes:
                    if 'comment' in stroke:
                        self._add_item_to_tree(self.drawings_tree, stroke, "Drawing")
        
        for measure in self.measurements:
            if 'comment' in measure:
                self._add_item_to_tree(self.measurements_tree, measure, "Measure")

    def _configure_styles(self):
        base02 = '#073642'; base2 = '#eee8d5'; base3 = '#fdf6e3'; blue = '#000080'
        self.root.configure(bg=base3)
        self.style.configure('.', background=base3, foreground=base02, font=('Segoe UI', 9), borderwidth=0)
        self.style.configure('TFrame', background=base3)
        self.style.configure('TLabel', background=base3, foreground=base02)
        self.style.configure('Title.TLabel', font=('Segoe UI', 11, 'bold'), foreground=base02)
        self.style.configure('Info.TLabel', foreground='#586e75', font=('Segoe UI', 8))
        self.style.configure('TNotebook', background=base3, borderwidth=1)
        self.style.configure('TNotebook.Tab', padding=[10, 5], font=('Segoe UI', 9), background=base2, foreground=base02)
        self.style.map('TNotebook.Tab', background=[('selected', blue), ('active', base3)], foreground=[('selected', 'white')])
        self.style.configure('TButton', padding=5, font=('Segoe UI', 9), background=base2, foreground=base02, borderwidth=1, relief='raised')
        self.style.map('TButton', background=[('active', base3), ('pressed', base3)], relief=[('pressed', 'sunken')])
        self.style.configure('Toolbutton.TButton', padding=0, relief='flat', background='black', foreground='white')
        self.style.map('Toolbutton.TButton', background=[('active', '#333333')])
        self.style.configure('TLabelFrame', relief='solid', borderwidth=1, bordercolor='#93a1a1', padding=10, background=base2)
        self.style.configure('TLabelFrame.Label', font=('Segoe UI', 9, 'bold'), background=base2, foreground=base02)
        self.style.configure('TCheckbutton', indicatorrelief='flat', background=base3, foreground=base02)
        self.style.configure('TRadiobutton', background=base3, foreground=base02)
        self.style.configure('TScale', troughcolor=base2, background=blue)
        self.style.configure("Treeview", background=base2, foreground=base02, fieldbackground=base2, rowheight=25)
        self.style.map("Treeview", background=[('selected', blue)], foreground=[('selected', 'white')])
        self.style.configure("Treeview.Heading", font=('Segoe UI', 9, 'bold'), background=base3)

    def _create_widgets(self):
        self._create_menu()
        self._create_toolbar()
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.main_pane = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)
        
        control_frame = self._create_control_panel()
        self.main_pane.add(control_frame, weight=0)
        
        vis_frame = self._create_viewer_panel()
        self.main_pane.add(vis_frame, weight=1)
        
        self._create_status_bar()

    def _create_menu(self):
        
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Volume", command=self.load_volume, accelerator="Ctrl+O")
        file_menu.add_command(label="Edit Patient Info", command=self._prompt_for_patient_info)
        file_menu.add_separator()

        file_menu.add_command(label="Save Session...", command=self.save_session, accelerator="Ctrl+Shift+S")
        file_menu.add_command(label="Load Session...", command=self.load_session, accelerator="Ctrl+Shift+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save Report...", command=self.save_report, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo_action, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo_action, accelerator="Ctrl+Y")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Overlays", variable=self.show_overlays)
        view_menu.add_separator()
        view_menu.add_command(label="Reset Layout", command=self.reset_layout, accelerator="Ctrl+R")
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom, accelerator="Ctrl+0")

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_checkbutton(label="Crosshairs", variable=self.crosshair_enabled, command=self.update_2d_views)
        tools_menu.add_checkbutton(label="Measurements", variable=self.measurement_mode)
        tools_menu.add_command(label="Clear All Measurements", command=self.clear_measurements)
        tools_menu.add_checkbutton(label="Draw Mode", variable=self.draw_mode)
        tools_menu.add_command(label="Clear All Drawings", command=self.clear_drawings)

        self.root.bind('<Control-o>', lambda e: self.load_volume())
        self.root.bind('<Control-s>', lambda e: self.save_report())
        self.root.bind('<Control-Shift-S>', lambda e: self.save_session())
        self.root.bind('<Control-Shift-O>', lambda e: self.load_session())
        self.root.bind('<Control-r>', lambda e: self.reset_layout())
        self.root.bind('<Control-0>', lambda e: self.reset_zoom())
        self.root.bind('<Control-z>', lambda e: self.undo_action())
        self.root.bind('<Control-y>', lambda e: self.redo_action())

    def _create_toolbar(self):
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(2, 5))
        ttk.Button(toolbar_frame, text="Load Volume", command=self.load_volume).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Patient Info", command=self._prompt_for_patient_info).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5, pady=2)
        ttk.Button(toolbar_frame, text="Save Session", command=self.save_session).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Load Session", command=self.load_session).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5, pady=2)
        self.undo_button = ttk.Button(toolbar_frame, text="Undo", command=self.undo_action)
        self.undo_button.pack(side=tk.LEFT, padx=2)
        self.redo_button = ttk.Button(toolbar_frame, text="Redo", command=self.redo_action)
        self.redo_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5, pady=2)
        ttk.Button(toolbar_frame, text="Reset Layout", command=self.reset_layout).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5, pady=2)
        ttk.Checkbutton(toolbar_frame, text="Crosshairs", variable=self.crosshair_enabled, command=self.update_2d_views).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar_frame, text="Measure", variable=self.measurement_mode).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar_frame, text="Draw", variable=self.draw_mode).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar_frame, text="Zoom Select", variable=self.zoom_select_mode).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar_frame, text="Pan", variable=self.pan_mode).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar_frame, text="Overlays", variable=self.show_overlays).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5, pady=2)

        ttk.Separator(toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5, pady=2)
        ttk.Button(toolbar_frame, text="Launch Analysis Tool", command=self.launch_explainability).pack(side=tk.LEFT, padx=2)

    def _on_pan_toggle(self, *args):
        if self.pan_mode.get():
            # Disable other interaction modes when pan is active
            self.measurement_mode.set(False)
            self.draw_mode.set(False)
            self.zoom_select_mode.set(False)
            self.status_label.config(text="Pan Mode: Click and drag to pan the view.")
            
            # Set pan cursor for 2D views
            for name, canvas in self.canvases.items():
                if name != '3d' and canvas: 
                    canvas.get_tk_widget().config(cursor="fleur")
        else:
            if not any([self.measurement_mode.get(), self.draw_mode.get(), self.zoom_select_mode.get()]):
                self.status_label.config(text="Ready")
                self._reset_cursors()

    def _on_zoom_select_toggle(self, *args): 
        if self.zoom_select_mode.get():
            self.measurement_mode.set(False)
            self.draw_mode.set(False)
            self.pan_mode.set(False)  
            self.status_label.config(text="Zoom Select Mode: Click and drag on a 2D view to zoom to selection.")
            for name, canvas in self.canvases.items():
                if name != '3d' and canvas: 
                    canvas.get_tk_widget().config(cursor="crosshair")
        else:
            self._reset_zoom_selection()
            self.update_2d_views()
            if not any([self.measurement_mode.get(), self.draw_mode.get(), self.pan_mode.get()]):
                self.status_label.config(text="Ready")
                self._reset_cursors()

    def _create_status_bar(self):
        self.status_frame = ttk.Frame(self.root, padding=(5, 2), style='TFrame')
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(self.status_frame, text="Ready", style='Info.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=5)
        ttk.Separator(self.status_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.view_info_label = ttk.Label(self.status_frame, text="View: ---", style='Info.TLabel', width=15)
        self.view_info_label.pack(side=tk.LEFT, padx=5)
        ttk.Separator(self.status_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.position_label = ttk.Label(self.status_frame, text="Position: ---", style='Info.TLabel', width=25)
        self.position_label.pack(side=tk.LEFT, padx=5)
        self.intensity_label = ttk.Label(self.status_frame, text="Intensity: ---", style='Info.TLabel', width=20)
        self.intensity_label.pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate', length=150)

    def _create_control_panel(self):
        control_frame = ttk.Frame(self.main_pane, width=350)
        control_frame.pack_propagate(False)  # Prevent shrinking

        header_frame = ttk.Frame(control_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        title_label = ttk.Label(header_frame, text="Controls & Analysis", style='Title.TLabel')
        title_label.pack()

        # Patient info frame
        patient_frame = ttk.Frame(header_frame)
        patient_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.patient_info_label = ttk.Label(
            patient_frame,
            text="Patient: N/A (ID: N/A)",
            style='Info.TLabel',
            wraplength=320
        )
        self.patient_info_label.pack(side=tk.LEFT)
        
        edit_button = ttk.Button(
            patient_frame,
            text="Edit",
            command=self._prompt_for_patient_info,
            width=6
        )
        edit_button.pack(side=tk.RIGHT)

        # Create notebook for tabs
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._create_slice_tab(notebook)
        self._create_3d_tab(notebook)
        self._create_analysis_tab(notebook)
        
        return control_frame

    def _create_slice_tab(self, notebook):
        slice_tab = ttk.Frame(notebook)
        notebook.add(slice_tab, text="Navigation")
        
        wl_frame = ttk.LabelFrame(slice_tab, text="Window & Level")
        wl_frame.pack(fill=tk.X, pady=10, padx=5)
        ttk.Label(wl_frame, text="Window:", style='Info.TLabel').grid(row=0, column=0, sticky='w', padx=5)
        self.window_label = ttk.Label(wl_frame, text="2000", style='Info.TLabel')
        self.window_label.grid(row=0, column=1, sticky='e', padx=5)
        self.window_slider = ttk.Scale(wl_frame, from_=1, to=4096, command=self.set_window)
        self.window_slider.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5)
        ttk.Label(wl_frame, text="Level:", style='Info.TLabel').grid(row=2, column=0, sticky='w', padx=5)
        self.level_label = ttk.Label(wl_frame, text="1000", style='Info.TLabel')
        self.level_label.grid(row=2, column=1, sticky='e', padx=5)
        self.level_slider = ttk.Scale(wl_frame, from_=-1024, to=3071, command=self.set_level)
        self.level_slider.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5)
        wl_frame.columnconfigure(0, weight=1)

        pos_frame = ttk.LabelFrame(slice_tab, text="Slice Position")
        pos_frame.pack(fill=tk.X, pady=10, padx=5)
        planes = [('Axial (Z)', "#00ffff", AXIAL), ('Coronal (Y)', "#00ff6e", CORONAL), ('Sagittal (X)', "#ff0000", SAGITTAL)]
        for i, (name, color, idx) in enumerate(planes):
            label = ttk.Label(pos_frame, text=name, foreground=color, font=('Segoe UI', 9, 'bold'))
            label.grid(row=i, column=0, sticky='w', padx=5, pady=(5,0))
            slider = ttk.Scale(pos_frame, from_=0, to=1, command=lambda v, p=idx: self.update_slice(p, v))
            slider.grid(row=i, column=1, sticky='ew', padx=5, pady=2)
            pos_label = ttk.Label(pos_frame, text="0/0", style='Info.TLabel', width=8)
            pos_label.grid(row=i, column=2, padx=5, pady=2)
            if idx == AXIAL: self.axial_slider, self.axial_label = slider, pos_label
            elif idx == CORONAL: self.coronal_slider, self.coronal_label = slider, pos_label
            else: self.sagittal_slider, self.sagittal_label = slider, pos_label
            slider.bind("<ButtonRelease-1>", self._on_slider_release)
        pos_frame.columnconfigure(1, weight=1)

    def _create_3d_tab(self, notebook):
        render_tab = ttk.Frame(notebook)
        
        model_frame = ttk.LabelFrame(render_tab, text="3D Surface Model")
        model_frame.pack(fill=tk.X, pady=10, padx=5)

        iso_frame = ttk.Frame(model_frame, style='TFrame')
        iso_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(iso_frame, text="ISO Threshold:", style='Info.TLabel').pack(side=tk.LEFT)
        self.iso_label = ttk.Label(iso_frame, text=f"{self.iso_value.get():.0f}", style='Info.TLabel')
        self.iso_label.pack(side=tk.RIGHT)
        self.iso_slider = ttk.Scale(model_frame, variable=self.iso_value, command=lambda v: self.iso_label.config(text=f"{float(v):.0f}"))
        self.iso_slider.pack(fill=tk.X, padx=5, pady=(0,5))
        
        btn_frame = ttk.Frame(model_frame, style='TFrame')
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.generate_model_button = ttk.Button(btn_frame, text="Generate 3D Model", command=self.generate_3d_model)
        self.generate_model_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self.clear_model_button = ttk.Button(btn_frame, text="Clear 3D Model", command=self.clear_3d_model)
        self.clear_model_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        options_frame = ttk.LabelFrame(render_tab, text="3D View Options")
        options_frame.pack(fill=tk.X, pady=10, padx=5)
        ttk.Checkbutton(options_frame, text="Show 3D Axes/Scale", variable=self.show_3d_axes, command=self.update_3d_view).pack(anchor='w')

    def _create_analysis_tab(self, notebook):
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Annotations")
        
        draw_frame = ttk.LabelFrame(analysis_tab, text="Drawing Tool")
        draw_frame.pack(fill=tk.X, pady=10, padx=5)

        color_ctrl_frame = ttk.Frame(draw_frame)
        color_ctrl_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(color_ctrl_frame, text="Color:").pack(side=tk.LEFT)
        self.draw_color_combo = ttk.Combobox(
            color_ctrl_frame, 
            textvariable=self.draw_color,
            values=['yellow', 'red', 'lime', 'cyan', 'magenta', 'orange', 'white'],
            state='readonly',
            width=10
        )
        self.draw_color_combo.pack(side=tk.RIGHT)
        self.draw_color_combo.set('yellow')

        size_ctrl_frame = ttk.Frame(draw_frame)
        size_ctrl_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(size_ctrl_frame, text="Brush Size (px):").pack(side=tk.LEFT)
        self.draw_size_spinbox = ttk.Spinbox(
            size_ctrl_frame,
            from_=1,
            to=20,
            increment=1,
            textvariable=self.draw_size,
            width=10
        )
        self.draw_size_spinbox.pack(side=tk.RIGHT)

        opacity_ctrl_frame = ttk.Frame(draw_frame)
        opacity_ctrl_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(opacity_ctrl_frame, text="Opacity:").pack(side=tk.LEFT)
        opacity_scale = ttk.Scale(
            opacity_ctrl_frame,
            from_=0.1,
            to=1.0,
            variable=self.draw_opacity,
            orient=tk.HORIZONTAL,
            length=100
        )
        opacity_scale.pack(side=tk.RIGHT)
        self.opacity_label = ttk.Label(opacity_ctrl_frame, text="0.8")
        self.opacity_label.pack(side=tk.RIGHT, padx=(5, 10))
        self.draw_opacity.trace_add(
            'write',
            lambda *args: self.opacity_label.config(text=f"{self.draw_opacity.get():.1f}")
        )

        results_notebook = ttk.Notebook(analysis_tab)
        results_notebook.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        draw_tab = ttk.Frame(results_notebook)
        results_notebook.add(draw_tab, text="Drawings")

        draw_btn_frame = ttk.Frame(draw_tab)
        draw_btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(
            draw_btn_frame,
            text="Add Comment",
            command=self.add_drawing_comment,
            style='Toolbutton.TButton'
        ).pack(side=tk.LEFT, padx=2, expand=True)

        ttk.Button(
            draw_btn_frame,
            text="Edit Comment",
            command=self.edit_annotation_comment,
            style='Toolbutton.TButton'
        ).pack(side=tk.LEFT, padx=2, expand=True)

        ttk.Button(
            draw_btn_frame,
            text="Delete",
            command=self.delete_selected_annotation,
            style='Toolbutton.TButton'
        ).pack(side=tk.LEFT, padx=2, expand=True)

        ttk.Button(
            draw_btn_frame,
            text="Clear All",
            command=lambda: self.clear_drawings(confirm=True),
            style='Toolbutton.TButton'
        ).pack(side=tk.LEFT, padx=2, expand=True)

        self.drawings_tree = ttk.Treeview(
            draw_tab,
            columns=('info'),
            show='headings',
            selectmode='browse',
            height=8
        )
        self.drawings_tree.heading('info', text='Drawing Annotations')
        self.drawings_tree.column('info', width=250)
        
        scroll_y = ttk.Scrollbar(draw_tab, orient="vertical", command=self.drawings_tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.drawings_tree.configure(yscrollcommand=scroll_y.set)
        
        self.drawings_tree.pack(fill=tk.BOTH, expand=True)
        self.drawings_tree.bind('<<TreeviewSelect>>', self.on_drawing_select)

        measure_tab = ttk.Frame(results_notebook)
        results_notebook.add(measure_tab, text="Measurements")

        measure_btn_frame = ttk.Frame(measure_tab)
        measure_btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(
            measure_btn_frame,
            text="Add Comment",
            command=self.add_measurement_comment,
            style='Toolbutton.TButton'
        ).pack(side=tk.LEFT, padx=2, expand=True)

        ttk.Button(
            measure_btn_frame,
            text="Edit Comment",
            command=self.edit_annotation_comment,
            style='Toolbutton.TButton'
        ).pack(side=tk.LEFT, padx=2, expand=True)

        ttk.Button(
            measure_btn_frame,
            text="Delete",
            command=self.delete_selected_annotation,
            style='Toolbutton.TButton'
        ).pack(side=tk.LEFT, padx=2, expand=True)

        ttk.Button(
            measure_btn_frame,
            text="Clear All",
            command=lambda: self.clear_measurements(confirm=True),
            style='Toolbutton.TButton'
        ).pack(side=tk.LEFT, padx=2, expand=True)

        self.measurements_tree = ttk.Treeview(
            measure_tab,
            columns=('info'),
            show='headings',
            selectmode='browse',
            height=8
        )
        self.measurements_tree.heading('info', text='Measurement Annotations')
        self.measurements_tree.column('info', width=250)
        
        scroll_y = ttk.Scrollbar(measure_tab, orient="vertical", command=self.measurements_tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.measurements_tree.configure(yscrollcommand=scroll_y.set)
        
        self.measurements_tree.pack(fill=tk.BOTH, expand=True)
        self.measurements_tree.bind('<<TreeviewSelect>>', self.on_measurement_select)

        status_frame = ttk.Frame(analysis_tab)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.annotation_status = ttk.Label(
            status_frame,
            text="Select an annotation to edit or delete",
            style='Info.TLabel'
        )
        self.annotation_status.pack(side=tk.LEFT)

    def _create_annotation_tree(self, parent, select_cmd):
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        tree = ttk.Treeview(tree_frame, columns=('info'), show='headings')
        tree.heading('info', text='Annotation')
        tree.column('info', width=250)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scroll.set)
        tree.bind('<<TreeviewSelect>>', select_cmd)
        return tree

    def _create_viewer_panel(self):
        vis_container = ttk.Frame(self.main_pane, style='TFrame')
        
        # Create notebook for different views
        self.view_notebook = ttk.Notebook(vis_container)
        self.view_notebook.pack(fill=tk.BOTH, expand=True)

        # Create main view tab
        main_view_tab = ttk.Frame(self.view_notebook)
        self.view_notebook.add(main_view_tab, text="Image Viewer")
        
        self.panes['main_h'] = ttk.PanedWindow(main_view_tab, orient=tk.HORIZONTAL)
        self.panes['main_h'].pack(fill=tk.BOTH, expand=True)
        
        left_v_pane = ttk.PanedWindow(self.panes['main_h'], orient=tk.VERTICAL)
        right_v_pane = ttk.PanedWindow(self.panes['main_h'], orient=tk.VERTICAL)
        
        self.panes['main_h'].add(left_v_pane, weight=1)
        self.panes['main_h'].add(right_v_pane, weight=1)
        
        self.panes['left_v'] = left_v_pane
        self.panes['right_v'] = right_v_pane

        view_config = {
            'axial': {'parent': left_v_pane, 'title': 'Axial View'},
            'sagittal': {'parent': left_v_pane, 'title': 'Sagittal View'},
            'coronal': {'parent': right_v_pane, 'title': 'Coronal View'}
        }

        ordered_views = ['axial', 'sagittal', 'coronal']

        for name in ordered_views:
            config = view_config[name]
            
            frame = ttk.Frame(config['parent'], style='TFrame')
            config['parent'].add(frame, weight=1)
            
            fig = plt.figure(facecolor='black', figsize=(6, 6))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = fig.add_subplot(111, facecolor='black')
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            control_overlay = tk.Frame(frame, bg='black', bd=0, highlightthickness=0)
            control_overlay.place(relx=1.0, rely=0, anchor='ne', x=-5, y=5)
            
            max_btn = ttk.Button(
                control_overlay, 
                text="❐", 
                width=2, 
                style='Toolbutton.TButton',
                command=lambda v=name: self.toggle_maximize_view(v)
            )
            max_btn.pack()
            
            self.figs[name] = fig
            self.axes[name] = ax
            self.canvases[name] = canvas
            self.maximize_buttons[name] = max_btn

            canvas.mpl_connect('scroll_event', partial(self.on_scroll, view=name))
            canvas.mpl_connect('button_press_event', partial(self.on_button_press, view=name))
            canvas.mpl_connect('button_release_event', partial(self.on_button_release, view=name))
            canvas.mpl_connect('motion_notify_event', partial(self.on_mouse_move, view=name))
            canvas.mpl_connect('figure_enter_event', partial(self._on_view_enter, view=name))
            canvas.mpl_connect('figure_leave_event', partial(self._on_view_leave))

        return vis_container

    def update_all_views(self):
        self.update_2d_views()

    def update_2d_views(self, draw_3d=False):
        view_map = { 'axial': "#00ffff", 'coronal': "#00ff6e", 'sagittal': "#ff0000" }
        for name, color in view_map.items():
            if name in self.axes:
                self._draw_2d_view(self.axes[name], self.canvases[name], self._get_plane_idx(name), color, name)

    def _draw_2d_view(self, ax, canvas, plane_idx, color, view_name):
        ax.clear()
        ax.set_facecolor('black')

        if self.volume_data is None:
            ax.text(0.5, 0.5, 'No Volume Loaded', color='grey', ha='center', va='center', transform=ax.transAxes)
            for spine in ax.spines.values(): 
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return

        slice_data = self._get_slice_data(plane_idx)
        if slice_data is None: 
            return
            
        h, w = slice_data.shape
        
        # Display the image
        ax.imshow(slice_data, cmap=self.colormap.get(), vmin=self.vmin, vmax=self.vmax, 
                origin='lower', extent=(0, w, 0, h), interpolation='bilinear')
        
        aspect_map = {
            AXIAL: self.spacing[1]/self.spacing[0], 
            CORONAL: self.spacing[0]/self.spacing[2], 
            SAGITTAL: self.spacing[1]/self.spacing[2]
        }
        ax.set_aspect(aspect_map.get(plane_idx, 1.0))

        zoom = self.zoom_factors[view_name]
        center_x_ratio, center_y_ratio = self.zoom_centers[view_name]
        view_w, view_h = w / zoom, h / zoom
        center_x, center_y = center_x_ratio * w, center_y_ratio * h
        
        xlim_min = max(0, center_x - view_w / 2)
        xlim_max = min(w, center_x + view_w / 2)
        ylim_min = max(0, center_y - view_h / 2)
        ylim_max = min(h, center_y + view_h / 2)
        
        if xlim_max - xlim_min < view_w:
            if xlim_min == 0:
                xlim_max = min(w, xlim_min + view_w)
            elif xlim_max == w:
                xlim_min = max(0, xlim_max - view_w)
        
        if ylim_max - ylim_min < view_h:
            if ylim_min == 0:
                ylim_max = min(h, ylim_min + view_h)
            elif ylim_max == h:
                ylim_min = max(0, ylim_max - view_h)
        
        is_maximized = (self.maximized_view == view_name)
        
        if is_maximized and self.show_axis_scales.get():
            # Add 15% padding on each side for axis labels
            padding_x = (xlim_max - xlim_min) * 0.05
            padding_y = (ylim_max - ylim_min) * 0.08
            ax.figure.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.97)
            ax.set_xlim(xlim_min - padding_x, xlim_max + padding_x)
            ax.set_ylim(ylim_min - padding_y, ylim_max + padding_y)
            
            self._configure_axis_scales(ax, plane_idx, view_name, 
                                    xlim_min - padding_x, xlim_max + padding_x, 
                                    ylim_min - padding_y, ylim_max + padding_y)
        else:
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_ylim(ylim_min, ylim_max)
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values(): 
                spine.set_visible(False)

        # Draw crosshairs if enabled
        if self.crosshair_enabled.get():
            pos_map = {
                AXIAL: (self.current_slices[SAGITTAL], self.current_slices[CORONAL]), 
                CORONAL: (self.current_slices[SAGITTAL], self.current_slices[AXIAL]), 
                SAGITTAL: (self.current_slices[CORONAL], self.current_slices[AXIAL])
            }
            colors_map = {
                AXIAL: ("#ff0000", "#00ff6e"), 
                CORONAL: ("#ff0000", "#00ffff"), 
                SAGITTAL: ("#00ff6e", "#00ffff")
            }
            x_pos, y_pos = pos_map[plane_idx]
            x_color, y_color = colors_map[plane_idx]
            
            # Only draw crosshairs if they're within view
            if xlim_min <= x_pos <= xlim_max:
                ax.axvline(x_pos, color=x_color, lw=0.7, alpha=0.9, linestyle='-')
            if ylim_min <= y_pos <= ylim_max:
                ax.axhline(y_pos, color=y_color, lw=0.7, alpha=0.9, linestyle='-')

        # Draw annotations
        self._draw_slice_measurements(ax, plane_idx, self.current_slices[plane_idx])
        self._draw_slice_drawings(ax, plane_idx, self.current_slices[plane_idx])

        if self.show_overlays.get():
            slice_pos_mm = self.current_slices[plane_idx] * self.spacing[2-plane_idx] if plane_idx < 3 and len(self.spacing) == 3 else 0
            
            info_text = (f"{view_name.upper()}\n"
                        f"Slice: {self.current_slices[plane_idx]}/{self.dimensions[plane_idx]-1} ({slice_pos_mm:.1f} mm)\n"
                        f"W: {self.window:.0f} L: {self.level:.0f} | Z: {zoom:.1f}x")
            
            ax.text(0.02, 0.98, info_text, color=color, fontsize=7, ha='left', va='top', 
                    transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
            
            pat_text = f"{os.path.basename(self.filepath)}"
            ax.text(0.98, 0.98, pat_text, color='gray', fontsize=6, ha='right', va='top', 
                    transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))

        # Zoom selection rectangle
        if self.zoom_selection['rect_patch'] and self.zoom_selection['view'] == view_name:
            ax.add_patch(self.zoom_selection['rect_patch'])
        
        canvas.draw_idle()


    def _round_to_nice_number(self, value):
        return self._round_to_nice_measurement(value)

    def _configure_axis_scales(self, ax, plane_idx, view_name, xlim_min, xlim_max, ylim_min, ylim_max):
        spacing_map = {
            AXIAL: (self.spacing[0], self.spacing[1]),    
            CORONAL: (self.spacing[0], self.spacing[2]),   
            SAGITTAL: (self.spacing[1], self.spacing[2]) 
        }
        sp_x, sp_y = spacing_map.get(plane_idx, (1.0, 1.0))
        
        # Calculate view size in pixels and real-world units
        view_width_pixels = xlim_max - xlim_min
        view_height_pixels = ylim_max - ylim_min
        view_width_mm = view_width_pixels * sp_x
        view_height_mm = view_height_pixels * sp_y
        
        zoom = self.zoom_factors[view_name]
        if zoom > 10:
            target_ticks = 12  # More ticks when highly zoomed
        elif zoom > 5:
            target_ticks = 10
        elif zoom > 2:
            target_ticks = 8
        else:
            target_ticks = 6
        
        target_spacing_x_mm = view_width_mm / target_ticks
        target_spacing_y_mm = view_height_mm / target_ticks
        
        nice_spacing_x_mm = self._round_to_nice_measurement(target_spacing_x_mm)
        nice_spacing_y_mm = self._round_to_nice_measurement(target_spacing_y_mm)
        
        x_tick_spacing_pixels = nice_spacing_x_mm / sp_x
        y_tick_spacing_pixels = nice_spacing_y_mm / sp_y
        
        x_ticks = []
        y_ticks = []
        
        x_start_mm = (xlim_min * sp_x // nice_spacing_x_mm) * nice_spacing_x_mm
        x_start_pixels = x_start_mm / sp_x
        x_pos = x_start_pixels
        
        while x_pos <= xlim_max and len(x_ticks) < 15:  # Max 15 ticks
            if x_pos >= xlim_min:
                x_ticks.append(x_pos)
            x_pos += x_tick_spacing_pixels
        
        y_start_mm = (ylim_min * sp_y // nice_spacing_y_mm) * nice_spacing_y_mm
        y_start_pixels = y_start_mm / sp_y
        y_pos = y_start_pixels
        
        while y_pos <= ylim_max and len(y_ticks) < 15:  # Max 15 ticks
            if y_pos >= ylim_min:
                y_ticks.append(y_pos)
            y_pos += y_tick_spacing_pixels
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        x_labels = []
        for x in x_ticks:
            mm_pos = x * sp_x
            x_labels.append(self._format_measurement(mm_pos))
        
        y_labels = []
        for y in y_ticks:
            mm_pos = y * sp_y
            y_labels.append(self._format_measurement(mm_pos))
        
        font_size = max(7, min(10, 60 // len(x_ticks)))  # Adaptive font size
        ax.set_xticklabels(x_labels, fontsize=font_size, color='white', rotation=0)
        ax.set_yticklabels(y_labels, fontsize=font_size, color='white')
        
        for spine in ax.spines.values():
            spine.set_color('gray')
            spine.set_linewidth(0.5)
            spine.set_visible(True)
            spine.set_alpha(0.8)
        
        ax.tick_params(colors='white', length=4, width=0.5, labelsize=font_size, pad=3)
        
        axis_labels = {
            AXIAL: ("Right → Left", "Anterior → Posterior"),
            CORONAL: ("Right → Left", "Inferior → Superior"),
            SAGITTAL: ("Anterior → Posterior", "Inferior → Superior")
        }
        
        if plane_idx in axis_labels:
            xlabel, ylabel = axis_labels[plane_idx]
            
            x_unit = self._get_display_unit(view_width_mm)
            y_unit = self._get_display_unit(view_height_mm)
            
            ax.set_xlabel(f"{xlabel} ({x_unit})", fontsize=font_size+1, color='lightgray', labelpad=5)
            ax.set_ylabel(f"{ylabel} ({y_unit})", fontsize=font_size+1, color='lightgray', labelpad=5)

    def _get_display_unit(self, range_mm):
        return "mm"

    def _format_measurement(self, value_mm):
        abs_value = abs(value_mm)
        
        if abs_value == 0:
            return "0"
        elif abs_value < 0.001: 
            return f"{value_mm:.4f}"
        elif abs_value < 0.01:  # 0.001-0.01 mm
            return f"{value_mm:.3f}"
        elif abs_value < 0.1:  # 0.01-0.1 mm
            return f"{value_mm:.2f}"
        elif abs_value < 1:  # 0.1-1 mm
            return f"{value_mm:.1f}"
        elif abs_value < 10:  # 1-10 mm
            return f"{value_mm:.1f}"
        elif abs_value < 100:  # 10-100 mm
            return f"{value_mm:.0f}"
        else:  # > 100 mm
            return f"{value_mm:.0f}"

    def _round_to_nice_measurement(self, value_mm):
        if value_mm <= 0:
            return 0.001  # 1 micrometer minimum
        
        nice_values = [
            # Micrometers (to mm)
            0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            # Sub-millimeter
            0.1, 0.2, 0.25, 0.5,
            # Millimeters
            1, 2, 2.5, 5, 10, 20, 25, 50,
            # Centimeters
            100, 200, 250, 500,
            # Larger measurements
            1000, 2000, 5000, 10000
        ]
        
        # Find the closest nice value that's >= target
        for nice in nice_values:
            if nice >= value_mm:
                return nice
        
        # If value is very large, use scientific notation base
        magnitude = 10 ** int(np.log10(value_mm))
        normalized = value_mm / magnitude
        
        if normalized <= 2:
            return 2 * magnitude
        elif normalized <= 5:
            return 5 * magnitude
        else:
            return 10 * magnitude

    def _draw_slice_measurements(self, ax, plane_idx, slice_idx):
        for measure in self.measurements:
            if measure['plane'] != plane_idx or measure['slice'] != slice_idx:
                continue
                
            p1, p2 = measure['p1'], measure['p2']
            color = measure.get('color', 'yellow')
            
            is_highlighted = (self.highlighted_item.get('type') == 'Measure' and 
                            self.highlighted_item.get('id') == measure.get('id'))      

            linewidth = 2.5 if is_highlighted else 1.2
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                    color=color, linewidth=linewidth, alpha=0.9)
            
            marker_size = 6 if is_highlighted else 4
            ax.plot(p1[0], p1[1], 'o', color=color, markersize=marker_size, alpha=0.9)
            ax.plot(p2[0], p2[1], 'o', color=color, markersize=marker_size, alpha=0.9)
            
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            dist_text = f"{measure['dist_mm']:.1f}mm"
            
            font_size = 7 if is_highlighted else 6
            text_bg = 'white' if is_highlighted else 'black'
            text_color = 'black' if is_highlighted else color
            alpha = 0.9 if is_highlighted else 0.7
            
            ax.text(mid_x, mid_y, dist_text, 
                    color=text_color, fontsize=font_size, fontweight='normal',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', 
                            facecolor=text_bg, 
                            alpha=alpha, 
                            edgecolor=color,
                            linewidth=0.5))
        
        if (self.temp_measure_point and 
            self.temp_measure_point['plane'] == plane_idx and 
            self.temp_measure_point['slice'] == slice_idx):
            
            p1 = self.temp_measure_point['p1']
            ax.plot(p1[0], p1[1], 'o', color='yellow', markersize=6, alpha=0.9)
            ax.text(p1[0], p1[1], "Start", 
                    color='yellow', fontsize=6, fontweight='normal',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15', 
                            facecolor='black', 
                            alpha=0.6, 
                            edgecolor='none'))
    
    def _draw_slice_drawings(self, ax, plane_idx, slice_idx): 
        if plane_idx not in self.drawings or slice_idx not in self.drawings[plane_idx]:
            return
        
        strokes = self.drawings[plane_idx][slice_idx]
        for stroke in strokes:
            if len(stroke['points']) < 2:
                continue
                
            points = np.array(stroke['points'])
            color = stroke['color']
            size = stroke.get('size', 2.0)
            opacity = stroke.get('opacity', 0.8)
            
            is_highlighted = (self.highlighted_item.get('type') == 'Drawing' and 
                            self.highlighted_item.get('id') == stroke.get('id'))
            
            if is_highlighted:

                ax.plot(points[:, 0], points[:, 1], color='white', linewidth=size + 4, 
                    solid_capstyle='round', solid_joinstyle='round', alpha=0.9)
            
            ax.plot(points[:, 0], points[:, 1], color=color, linewidth=size, 
                solid_capstyle='round', solid_joinstyle='round', alpha=opacity)

    def update_3d_view(self):
        ax = self.axes.get('3d')
        if not ax: return
        canvas = self.canvases['3d']
        current_cam = (ax.elev, ax.azim) if hasattr(ax, 'elev') and ax.elev is not None else (20, 45)
        ax.clear()
        ax.set_facecolor('black')
        ax.set_axis_off()

        if self.volume_data is None:
            ax.text(0.5, 0.5, 0.5, 'No Volume Loaded', color='grey', ha='center', va='center', transform=ax.transAxes)
            canvas.draw_idle()
            return

        sz, sy, sx = self.dimensions
        sp_z, sp_y, sp_x = self.spacing
        
        width_mm = sx * sp_x
        height_mm = sy * sp_y
        depth_mm = sz * sp_z
        
        self._draw_coordinate_frame(ax, width_mm, height_mm, depth_mm)
        
        if self.show_model.get() and self.mesh_verts is not None:
            ax.plot_trisurf(self.mesh_verts[:, 0], self.mesh_verts[:, 1], self.mesh_verts[:, 2],
                                triangles=self.mesh_faces, color=self.mesh_color.get(), alpha=self.mesh_opacity.get(),
                                linewidth=0, antialiased=True)
        
        ax.set_box_aspect((width_mm, height_mm, depth_mm))
        ax.invert_zaxis()
        ax.invert_yaxis()
        ax.view_init(elev=current_cam[0], azim=current_cam[1])
        canvas.draw_idle()

    def _draw_coordinate_frame(self, ax, width_mm, height_mm, depth_mm):       
        corners = [
            [0, 0, 0], [width_mm, 0, 0], [width_mm, height_mm, 0], [0, height_mm, 0],  # bottom face
            [0, 0, depth_mm], [width_mm, 0, depth_mm], [width_mm, height_mm, depth_mm], [0, height_mm, depth_mm]  # top face
        ]
    
        edges = [

            [0, 1], [1, 2], [2, 3], [3, 0],

            [4, 5], [5, 6], [6, 7], [7, 4],

            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for edge in edges:
            start, end = corners[edge[0]], corners[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    color='magenta', linewidth=1.5, alpha=0.8)
        
        label_offset = max(width_mm, height_mm, depth_mm) * 0.05
        
        ax.text(0, 0, depth_mm + label_offset, 'S', color='magenta', fontsize=12, fontweight='bold')       
        ax.text(0, 0, -label_offset, 'I', color='magenta', fontsize=12, fontweight='bold')       
        ax.text(0, -label_offset, 0, 'A', color='magenta', fontsize=12, fontweight='bold')
        ax.text(0, height_mm + label_offset, 0, 'P', color='magenta', fontsize=12, fontweight='bold')
        ax.text(width_mm + label_offset, 0, 0, 'L', color='magenta', fontsize=12, fontweight='bold')
        ax.text(-label_offset, 0, 0, 'R', color='magenta', fontsize=12, fontweight='bold')
        padding = max(width_mm, height_mm, depth_mm) * 0.1
        ax.set_xlim(-padding, width_mm + padding)
        ax.set_ylim(-padding, height_mm + padding) 
        ax.set_zlim(-padding, depth_mm + padding)

    def clear_3d_model(self):
        if self.mesh_verts is None and self.mesh_faces is None:
            messagebox.showinfo("Info", "3D model is already cleared.", parent=self.root)
            return
        self.mesh_verts = None
        self.mesh_faces = None
        self.show_model.set(False)
        self.status_label.config(text="3D model cleared from memory.")
        self.update_3d_view()

    def save_report(self):
        if self.volume_data is None:
            messagebox.showerror("Error", "No data loaded to report on.", parent=self.root)
            return
        report_dir = filedialog.askdirectory(title="Select Directory to Save Report", parent=self.root)
        if not report_dir: return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pat_id_safe = re.sub(r'[\W_]+', '', self.patient_info.get("id", "NA")) or "NA"
        report_sub_dir = os.path.join(report_dir, f"Report_{pat_id_safe}_{timestamp}")
        
        try:
            os.makedirs(report_sub_dir, exist_ok=True)
            images_dir = os.path.join(report_sub_dir, "snapshots")
            os.makedirs(images_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not create report directory:\n{e}", parent=self.root)
            return

        self.status_label.config(text="Generating report..."); self.progress.pack(side=tk.RIGHT, padx=5); self.progress.start(); self.root.update_idletasks()
        
        report_data = {
            "patient_info": self.patient_info, "source_file": self.filepath, "report_date": datetime.now().isoformat(),
            "view_settings": {"window": self.window, "level": self.level, "current_slices": self.current_slices, "zoom_factors": self.zoom_factors},
            "drawings": [], "measurements": []
        }
        
        for plane, slices in self.drawings.items():
            for s_idx, strokes in slices.items():
                for stroke in (s for s in strokes if 'comment' in s):
                    filename = f"drawing_{stroke['id']}.png"
                    filepath = os.path.join(images_dir, filename)
                    success = self._generate_annotation_snapshot(stroke, filepath)
                    report_data["drawings"].append({**stroke, "image_file": os.path.join("snapshots", filename) if success else None})

        for m in (m for m in self.measurements if 'comment' in m):
            filename = f"measurement_{m['id']}.png"
            filepath = os.path.join(images_dir, filename)
            success = self._generate_measurement_snapshot(m, filepath)
            report_data["measurements"].append({**m, "image_file": os.path.join("snapshots", filename) if success else None})

        json_filepath = os.path.join(report_sub_dir, "report.json")
        try:
            with open(json_filepath, 'w') as f:
                json.dump(report_data, f, indent=4, cls=NumpyEncoder)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report data file:\n{e}", parent=self.root)
            self.progress.stop(); self.progress.pack_forget(); return
            
        self.progress.stop(); self.progress.pack_forget()
        self.status_label.config(text="Report saved successfully.")
        messagebox.showinfo("Success", f"Report saved to:\n{report_sub_dir}", parent=self.root)

    def _generate_annotation_snapshot(self, stroke_data, output_path):
        fig, ax = self._setup_snapshot_figure(stroke_data['plane'], stroke_data['slice'])
        if fig is None: return False
        
        points = np.array(stroke_data['points'])
        ax.plot(points[:, 0], points[:, 1], color='white', lw=stroke_data['size'] + 2, solid_capstyle='round', alpha=0.9)
        ax.plot(points[:, 0], points[:, 1], color=stroke_data['color'], lw=stroke_data['size'], solid_capstyle='round', alpha=0.8)
        
        return self._save_snapshot_figure(fig, ax, points, stroke_data['comment'], output_path)

    def _generate_measurement_snapshot(self, measure_data, output_path):
        fig, ax = self._setup_snapshot_figure(measure_data['plane'], measure_data['slice'])
        if fig is None: return False

        p1, p2 = measure_data['p1'], measure_data['p2']
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white', lw=3, marker='o', markersize=6)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=measure_data['color'], lw=1.5, marker='o', markersize=4)

        text = f"{measure_data['comment']}\n{measure_data['dist_mm']:.2f} mm"
        points = np.array([p1, p2])
        return self._save_snapshot_figure(fig, ax, points, text, output_path)

    def _setup_snapshot_figure(self, plane_idx, slice_idx):
        original_slice_num = self.current_slices[plane_idx]
        self.current_slices[plane_idx] = slice_idx
        slice_data = self._get_slice_data(plane_idx)
        self.current_slices[plane_idx] = original_slice_num
        if slice_data is None: return None, None
            
        fig = plt.figure(figsize=(8, 8), dpi=150, facecolor='black')
        ax = fig.add_subplot(111, facecolor='black')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        h, w = slice_data.shape
        ax.imshow(slice_data, cmap=self.colormap.get(), vmin=self.vmin, vmax=self.vmax, origin='lower', extent=(0, w, 0, h))
        aspect_map = {AXIAL: self.spacing[1]/self.spacing[0], CORONAL: self.spacing[0]/self.spacing[2], SAGITTAL: self.spacing[1]/self.spacing[2]}
        ax.set_aspect(aspect_map.get(plane_idx, 1))
        return fig, ax
        
    def _save_snapshot_figure(self, fig, ax, points, text, output_path):
        mean_x, mean_y = np.mean(points[:, 0]), np.mean(points[:, 1])
        x_range, y_range = np.ptp(points[:, 0]), np.ptp(points[:, 1])
        buffer = max(x_range, y_range, 50) * 1.5
        ax.set_xlim(mean_x - buffer, mean_x + buffer); ax.set_ylim(mean_y - buffer, mean_y + buffer)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.text(mean_x, mean_y, text, color='yellow', fontsize=10, ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))

        try:
            fig.savefig(output_path, facecolor='black', bbox_inches='tight', pad_inches=0.05)
            plt.close(fig); return True
        except Exception as e:
            print(f"Error saving snapshot: {e}"); plt.close(fig); return False

    def clear_drawings(self, confirm=True):     
        if not self.drawings:
            if confirm:
                messagebox.showinfo("Info", "No drawings to clear.", parent=self.root)
            return
            
        if confirm and not messagebox.askyesno("Confirm", "Clear all drawings?", parent=self.root):
            return
        
        drawings_backup = {}
        for plane, slices in self.drawings.items():
            drawings_backup[plane] = {}
            for slice_idx, strokes in slices.items():
                drawings_backup[plane][slice_idx] = [stroke.copy() for stroke in strokes]
        
        self._add_to_history({
            'type': 'clear_drawings',
            'data': drawings_backup
        })
        
        self.drawings.clear()
        self.last_drawn_stroke = None
        
        for item in self.drawings_tree.get_children():
            self.drawings_tree.delete(item)
        
        self.update_2d_views()
        self.status_label.config(text="All drawings cleared")
    
    def clear_measurements(self, confirm=True):      
        if not self.measurements:
            if confirm:
                messagebox.showinfo("Info", "No measurements to clear.", parent=self.root)
            return
            
        if confirm and not messagebox.askyesno("Confirm", "Clear all measurements?", parent=self.root):
            return
        
        measurements_backup = [measure.copy() for measure in self.measurements]
        self._add_to_history({
            'type': 'clear_measurements',
            'data': measurements_backup
        })
        
        self.measurements.clear()
        self.last_measurement = None
        
        for item in self.measurements_tree.get_children():
            self.measurements_tree.delete(item)
        
        self.update_2d_views()
        self.status_label.config(text="All measurements cleared")

    def add_drawing_comment(self):       
        if not self.last_drawn_stroke: 
            messagebox.showwarning("Warning", "Please make a drawing first.", parent=self.root)
            return
        if 'comment' in self.last_drawn_stroke: 
            messagebox.showinfo("Info", "This drawing already has a comment.", parent=self.root)
            return
        
        comment = simpledialog.askstring("Input", "Enter comment for the last drawing:", parent=self.root)
        if comment:
            stroke = self.last_drawn_stroke
            stroke['comment'] = comment
            self._add_item_to_tree(self.drawings_tree, stroke, "Drawing")
            
            self._add_to_history({
                'type': 'add_draw_comment',
                'data': {
                    'id': stroke['id'],
                    'comment': comment
                }
            })

    def add_measurement_comment(self):       
        if not self.last_measurement: 
            messagebox.showwarning("Warning", "Please make a measurement first.", parent=self.root)
            return
        if 'comment' in self.last_measurement: 
            messagebox.showinfo("Info", "This measurement already has a comment.", parent=self.root)
            return
        
        comment = simpledialog.askstring("Input", "Enter comment for the last measurement:", parent=self.root)
        if comment:
            measure = self.last_measurement
            measure['comment'] = comment
            self._add_item_to_tree(self.measurements_tree, measure, "Measure")
            
            self._add_to_history({
                'type': 'add_measure_comment',
                'data': {
                    'id': measure['id'],
                    'comment': comment
                }
            })
    
    def _add_item_to_tree(self, tree, item_data, item_type):
        plane_name = {AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"}[item_data['plane']]
        comment = item_data.get('comment', '')
        text = f"({plane_name} {item_data['slice']}) {comment}"
        tree.insert('', tk.END, text=text, values=(text,), iid=f"{item_type}_{item_data['id']}")

    def on_drawing_select(self, event): self._on_annotation_select_handler(self.drawings_tree, 'Drawing')

    def on_measurement_select(self, event): self._on_annotation_select_handler(self.measurements_tree, 'Measure')

    def _on_annotation_select_handler(self, tree, item_type):
        selected_items = tree.selection()
        if not selected_items:
            self.highlighted_item = {'type': None, 'id': None}
            self.update_2d_views()
            return

        iid = selected_items[0]
        item_id = int(iid.split('_')[1])
        self.highlighted_item = {'type': item_type, 'id': item_id}
        
        collection = self.drawings if item_type == 'Drawing' else self.measurements
        found_item = None
        if item_type == 'Drawing':
            found_item = next((s for p, slices in collection.items() for i, strokes in slices.items() for s in strokes if s.get('id') == item_id), None)
        else: # Measurement
            found_item = next((m for m in collection if m.get('id') == item_id), None)
            
        if found_item:
            self.current_slices[found_item['plane']] = found_item['slice']
            self._update_slice_sliders(); self._update_slice_labels(); self.update_all_views()
            self.root.after(1500, self.clear_highlight)

    def clear_highlight(self):
        self.highlighted_item = {'type': None, 'id': None}
        self.update_2d_views()
    
    def _prompt_for_patient_info(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Patient Information")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.style.lookup('TFrame', 'background'))
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        title_label = ttk.Label(main_frame, text="Patient Information", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky='w')

        ttk.Label(main_frame, text="Patient Name:", style='TLabel').grid(row=1, column=0, padx=(0, 10), pady=5, sticky='w')
        name_var = tk.StringVar(value=self.patient_info.get("name", ""))
        name_entry = ttk.Entry(main_frame, textvariable=name_var, width=25)
        name_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        ttk.Label(main_frame, text="Patient ID:", style='TLabel').grid(row=2, column=0, padx=(0, 10), pady=5, sticky='w')
        id_var = tk.StringVar(value=self.patient_info.get("id", ""))
        id_entry = ttk.Entry(main_frame, textvariable=id_var, width=25)
        id_entry.grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        
        main_frame.columnconfigure(1, weight=1)
        
        def on_ok():
            name = name_var.get().strip()
            patient_id = id_var.get().strip()           

            self.patient_info["name"] = name if name else "N/A"
            self.patient_info["id"] = patient_id if patient_id else "N/A"
            
            self._update_patient_info_display()
            
            dialog.destroy()
            
            self.status_label.config(text="Patient information updated")
            
        def on_cancel():
            dialog.destroy()
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(15, 0))
        
        ok_button = ttk.Button(btn_frame, text="OK", command=on_ok)
        ok_button.pack(side=tk.LEFT, padx=(0, 5))
        
        cancel_button = ttk.Button(btn_frame, text="Cancel", command=on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=(5, 0))
        

        name_entry.focus_set()
        if name_var.get():
            name_entry.select_range(0, tk.END)
        
        dialog.bind('<Return>', lambda e: on_ok())
        dialog.bind('<Escape>', lambda e: on_cancel())
        
        dialog.wait_window()

    def _update_patient_info_display(self):
        name = self.patient_info.get("name", "N/A")
        patient_id = self.patient_info.get("id", "N/A")
        
        if hasattr(self, 'patient_info_label') and self.patient_info_label:
            self.patient_info_label.config(text=f"Patient: {name} (ID: {patient_id})")
        
        self.update_2d_views()

    def reset_layout(self):
        if self.maximized_view: self.toggle_maximize_view(self.maximized_view)
        if hasattr(self, 'main_pane') and self.main_pane.winfo_exists():
            w, h = self.root.winfo_width(), self.root.winfo_height()
            if w > 1 and h > 1:
                self.main_pane.sashpos(0, 350)
                vis_w = w - 350
                if vis_w > 1 and h > 1:
                    self.panes['main_h'].sashpos(0, vis_w // 2)
                    self.panes['left_v'].sashpos(0, (h-50) // 2)

    def reset_zoom(self, event=None):
        for view in ['axial', 'coronal', 'sagittal']:
            self.zoom_factors[view] = 1.0; self.zoom_centers[view] = (0.5, 0.5)
        self.update_all_views()

    def toggle_maximize_view(self, view_name):
        btn = self.maximize_buttons[view_name]
        if self.maximized_view == view_name:
            self.restore_view()
            btn.config(text="❐")
            for other_btn in self.maximize_buttons.values(): 
                other_btn.config(state=tk.NORMAL)
        else:
            if self.maximized_view: 
                self.toggle_maximize_view(self.maximized_view)
            self.maximize_view(view_name)
            btn.config(text="▣")
            for name, other_btn in self.maximize_buttons.items():
                if name != view_name: 
                    other_btn.config(state=tk.DISABLED)

    def maximize_view(self, view_name):
        self.maximized_view = view_name
        
        self.view_states['sash_main_h'] = self.panes['main_h'].sashpos(0)
        self.view_states['sash_left_v'] = self.panes['left_v'].sashpos(0)
        
        for v_name in ['axial', 'coronal', 'sagittal']:
            if v_name != view_name:
                widget = self.canvases[v_name].get_tk_widget().master
                try: 
                    widget.master.forget(widget)
                except tk.TclError: 
                    pass
        
        total_width = self.panes['main_h'].winfo_width()
        if total_width > 1:
            if view_name in ['axial', 'sagittal']:
                self.panes['main_h'].sashpos(0, total_width - 5)
            else:  
                self.panes['main_h'].sashpos(0, 5)
        
        self.root.update_idletasks()
        self.update_2d_views()

    def restore_view(self):
        if not self.maximized_view: 
            return
            
        view_config = {
            'axial': self.panes['left_v'], 
            'sagittal': self.panes['left_v'], 
            'coronal': self.panes['right_v']
        }
        
        # Restore all views
        for v_name in ['axial', 'sagittal', 'coronal']:
            widget = self.canvases[v_name].get_tk_widget().master
            parent_pane = view_config[v_name]
            try: 
                parent_pane.add(widget, weight=1)
            except tk.TclError: 
                pass
        
        self.maximized_view = None
        self.root.after(50, self.reset_layout)
        self.root.after(100, self.update_2d_views)

    def activate_zoom_select(self):
        self.mouse_state = STATE_SELECT_ZOOM
        self.status_label.config(text="Zoom Select: Click and drag on a 2D view.")
        for name, canvas in self.canvases.items():
            if name != '3d': canvas.get_tk_widget().config(cursor="crosshair")

    def on_scroll(self, event, view):
        if self.volume_data is None or event.xdata is None: return
        if self.scroll_timer:
            self.root.after_cancel(self.scroll_timer)

        if event.key == 'control':
            base_step = -1 if event.button == 'up' else 1
            zoom_factor = 1.2 if base_step < 0 else 1/1.2
            self.zoom_factors[view] *= zoom_factor
            plane_idx = self._get_plane_idx(view)
            if plane_idx is not None and (slice_data := self._get_slice_data(plane_idx)) is not None:
                h, w = slice_data.shape
                self.zoom_centers[view] = (event.xdata / w, event.ydata / h)
        else:
            base_step = -1 if event.button == 'up' else 1
            plane_idx = self._get_plane_idx(view)
            new_slice = self.current_slices[plane_idx] + base_step
            self.current_slices[plane_idx] = np.clip(new_slice, 0, self.dimensions[plane_idx] - 1)
            self._update_slice_labels()
            self._update_slice_sliders()
        
        self.update_2d_views()

    def _get_plane_idx(self, view_name): return {'axial': AXIAL, 'coronal': CORONAL, 'sagittal': SAGITTAL}.get(view_name)

    def _handle_measurement_click(self, event, view):
        plane_idx = self._get_plane_idx(view)
        if plane_idx is None:
            return
        
        slice_idx = self.current_slices[plane_idx]
        x, y = event.xdata, event.ydata
        
        if self.temp_measure_point is None:

            self.temp_measure_point = {
                'p1': (x, y),
                'plane': plane_idx,
                'slice': slice_idx
            }
            self.status_label.config(text="Measure Mode: Click to set end point (ESC to cancel)")
        else:

            if (self.temp_measure_point['plane'] == plane_idx and 
                self.temp_measure_point['slice'] == slice_idx):
                
                p1 = self.temp_measure_point['p1']
                p2 = (x, y)
                

                spacing_map = {
                    AXIAL: (self.spacing[0], self.spacing[1]),  # X, Y spacing
                    CORONAL: (self.spacing[0], self.spacing[2]),  # X, Z spacing
                    SAGITTAL: (self.spacing[1], self.spacing[2])  # Y, Z spacing
                }
                sp_x, sp_y = spacing_map.get(plane_idx, (1.0, 1.0))
                
                dx = (p2[0] - p1[0]) * sp_x
                dy = (p2[1] - p1[1]) * sp_y
                dist_mm = np.sqrt(dx**2 + dy**2)
                
                self.measurement_id_counter += 1
                new_measurement = {
                    'id': self.measurement_id_counter,
                    'p1': p1,
                    'p2': p2,
                    'plane': plane_idx,
                    'slice': slice_idx,
                    'color': self.draw_color.get(),
                    'dist_mm': dist_mm
                }
                
                self.measurements.append(new_measurement)
                self.last_measurement = new_measurement
                self._add_to_history({'type': 'measure', 'data': new_measurement.copy()})
                self.temp_measure_point = None
                self.status_label.config(text=f"Measurement created: {dist_mm:.2f} mm")
            else:

                self.temp_measure_point = {
                    'p1': (x, y),
                    'plane': plane_idx,
                    'slice': slice_idx
                }
                self.status_label.config(text="Measure Mode: Click to set end point (ESC to cancel)")
        
        self.update_2d_views()

    def _cancel_measurement(self, event=None):       
        if self.temp_measure_point:
            self.temp_measure_point = None
            self.status_label.config(text="Measurement cancelled")
            self.update_2d_views()

    def on_button_press(self, event, view):
        if event.inaxes is None or event.xdata is None or event.ydata is None: 
            return
            
        self.last_mouse_pos = {'x': event.x, 'y': event.y, 'data_x': event.xdata, 'data_y': event.ydata}
        plane_idx = self._get_plane_idx(view)
        if plane_idx is None: 
            return
        slice_idx = self.current_slices[plane_idx]

        if self.draw_mode.get() and event.button == 1:
            self.mouse_state = STATE_DRAW
            self.annotation_id_counter += 1
            new_stroke = {
                'id': self.annotation_id_counter, 
                'plane': plane_idx, 
                'slice': slice_idx, 
                'points': [(event.xdata, event.ydata)], 
                'color': self.draw_color.get(), 
                'size': self.draw_size.get(),
                'opacity': self.draw_opacity.get()
            }
            
            if plane_idx not in self.drawings: 
                self.drawings[plane_idx] = {}
            if slice_idx not in self.drawings[plane_idx]: 
                self.drawings[plane_idx][slice_idx] = []
                
            self.drawings[plane_idx][slice_idx].append(new_stroke)
            self.last_drawn_stroke = new_stroke
            self._add_to_history({'type': 'draw', 'data': new_stroke})
            self.update_2d_views()
            return

        if self.measurement_mode.get() and event.button == 1: 
            self._handle_measurement_click(event, view)
            return
            
        if self.zoom_select_mode.get() and event.button == 1:
            self.mouse_state = STATE_SELECT_ZOOM
            self.zoom_selection.update({
                'view': view, 
                'start': (event.xdata, event.ydata), 
                'rect_patch': patches.Rectangle(
                    (event.xdata, event.ydata), 0, 0, 
                    linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--'
                )
            })
            self.update_2d_views()
            return

        if self.pan_mode.get() and event.button == 1:
            self.mouse_state = STATE_PAN
            return

        if event.button == 1: 
            self.mouse_state = STATE_SLICE
            self._update_slices_from_click(event, view)
        elif event.button == 2: 
            self.mouse_state = STATE_PAN
        elif event.button == 3: 
            self.mouse_state = STATE_WL

    def on_button_release(self, event, view):
        if self.mouse_state == STATE_SELECT_ZOOM: 
            self._finalize_zoom_selection(view)

            if not self.zoom_select_mode.get():
                self.mouse_state = STATE_NONE
            else:
                self.mouse_state = STATE_NONE  
        else:
            self.mouse_state = STATE_NONE
        

        if not self.zoom_select_mode.get():
            self._reset_cursors()

    def on_mouse_move(self, event, view):
        if event.inaxes is None or event.xdata is None or event.ydata is None: 
            self._update_status_bar(None, None)
            return
        
        self._update_status_bar(event, view)
        if self.mouse_state == STATE_NONE: 
            return
            
        plane_idx = self._get_plane_idx(view)
        
        if self.mouse_state == STATE_DRAW and plane_idx is not None:
            slice_idx = self.current_slices[plane_idx]
            if (plane_idx in self.drawings and 
                slice_idx in self.drawings[plane_idx] and 
                self.drawings[plane_idx][slice_idx]):
                
                current_stroke = self.drawings[plane_idx][slice_idx][-1]
                last_point = current_stroke['points'][-1]
                new_point = (event.xdata, event.ydata)
                
                distance = np.sqrt((new_point[0] - last_point[0])**2 + (new_point[1] - last_point[1])**2)
                if distance > 1.0:  # Minimum distance threshold
                    current_stroke['points'].append(new_point)
                    self.update_2d_views()
                    
        elif self.mouse_state == STATE_SELECT_ZOOM and self.zoom_selection.get('start'):
            start_x, start_y = self.zoom_selection['start']
            self.zoom_selection['rect_patch'].set_width(event.xdata - start_x)
            self.zoom_selection['rect_patch'].set_height(event.ydata - start_y)
            self.update_2d_views()
            
        elif self.mouse_state == STATE_PAN:
            if 'data_x' in self.last_mouse_pos and 'data_y' in self.last_mouse_pos:
                dx_data = event.xdata - self.last_mouse_pos['data_x']
                dy_data = event.ydata - self.last_mouse_pos['data_y']
                
                slice_data = self._get_slice_data(plane_idx)
                if slice_data is not None:
                    h, w = slice_data.shape
                    zoom = self.zoom_factors[view]
                    
                    center_x_ratio, center_y_ratio = self.zoom_centers[view]
                    
                    dx_ratio = -dx_data / w  # Negative for natural panning
                    dy_ratio = -dy_data / h
                    
                    new_center_x = np.clip(center_x_ratio + dx_ratio, 0, 1)
                    new_center_y = np.clip(center_y_ratio + dy_ratio, 0, 1)
                    
                    self.zoom_centers[view] = (new_center_x, new_center_y)
                    self.update_2d_views()
            
            self.last_mouse_pos.update({'data_x': event.xdata, 'data_y': event.ydata})
            
        elif self.mouse_state == STATE_WL:
            dx, dy = event.x - self.last_mouse_pos['x'], event.y - self.last_mouse_pos['y']
            self.window += dx * (self.window_slider.cget('to') / 200)
            level_range = self.level_slider.cget('to') - self.level_slider.cget('from')
            self.level -= dy * (level_range / 200)
            self.window = max(1, self.window)
            self._update_wl_vars()
            self._update_wl_controls()
            self.update_2d_views()
        
        self.last_mouse_pos.update({'x': event.x, 'y': event.y})

    def _finalize_zoom_selection(self, view):
        if not (self.zoom_selection.get('start') and self.zoom_selection.get('rect_patch')): return
        rect = self.zoom_selection['rect_patch']; x, y, w, h = rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()
        if abs(w) < 2 or abs(h) < 2: self._reset_zoom_selection(); self.update_2d_views(); return
        
        new_center_x, new_center_y = x + w / 2, y + h / 2
        if (slice_data := self._get_slice_data(self._get_plane_idx(view))) is not None:
            total_h, total_w = slice_data.shape; self.zoom_centers[view] = (new_center_x / total_w, new_center_y / total_h)
        
        ax = self.axes[view]; ax_w = ax.get_xlim()[1] - ax.get_xlim()[0]; ax_h = ax.get_ylim()[1] - ax.get_ylim()[0]
        if abs(w) > 0 and abs(h) > 0: self.zoom_factors[view] *= min(ax_w / abs(w), ax_h / abs(h))
        
        self._reset_zoom_selection(); self.update_all_views()

    def _reset_zoom_selection(self):
        self.zoom_selection = {
            'active': False, 
            'start': None, 
            'end': None, 
            'view': None, 
            'rect_patch': None
        }

    def reset_zoom(self, event=None):
        for view in ['axial', 'coronal', 'sagittal']:
            self.zoom_factors[view] = 1.0
            self.zoom_centers[view] = (0.5, 0.5)
        self.update_all_views()

    def _on_slider_release(self, event): self.update_all_views()

    def _on_measure_toggle(self, *args):
        if self.measurement_mode.get():
            self.draw_mode.set(False)
            self.zoom_select_mode.set(False)
            self.pan_mode.set(False) 
            self.status_label.config(text="Measure Mode: Click to set start point, click again to set end point.")
            for name, canvas in self.canvases.items():
                if name != '3d' and canvas: 
                    canvas.get_tk_widget().config(cursor="cross")
        else:
            self.temp_measure_point = None
            if not any([self.draw_mode.get(), self.zoom_select_mode.get(), self.pan_mode.get()]):
                self.status_label.config(text="Ready")
                self._reset_cursors()
        self.update_2d_views()

    def _on_draw_toggle(self, *args):
        if self.draw_mode.get():
            self.measurement_mode.set(False)
            self.zoom_select_mode.set(False)
            self.pan_mode.set(False)  
            self.status_label.config(text="Draw Mode: Click and drag to draw.")
            for name, canvas in self.canvases.items():
                if name != '3d' and canvas: 
                    canvas.get_tk_widget().config(cursor="cross")
        else:
            if not any([self.measurement_mode.get(), self.zoom_select_mode.get(), self.pan_mode.get()]):
                self.status_label.config(text="Ready")
                self._reset_cursors()
    
    def _reset_cursors(self):
        if not any([self.draw_mode.get(), self.measurement_mode.get(), 
                    self.zoom_select_mode.get(), self.pan_mode.get()]):
            for canvas in self.canvases.values():
                if canvas: 
                    canvas.get_tk_widget().config(cursor="")

    def _on_view_enter(self, event, view): self.active_view = view; self.view_info_label.config(text=f"View: {view.capitalize()}")
    def _on_view_leave(self, event): self.active_view = None; self.view_info_label.config(text="View: ---")
    
    def _get_sanitized_filename(self, base_name):
        pat_id = re.sub(r'[\W_]+', '', self.patient_info.get("id", "NA"))
        pat_name = re.sub(r'[\W_]+', '', self.patient_info.get("name", "Patient"))
        
        if not pat_id:
            pat_id = "NA"
        if not pat_name:
            pat_name = "Patient"
        
        return f"{pat_id}_{pat_name}_{base_name}"

    def _update_status_bar(self, event, data_view):
        if self.volume_data is None or event is None: self.position_label.config(text="Position: ---"); self.intensity_label.config(text="Intensity: ---"); return
        x_coord, y_coord = int(round(event.xdata)), int(round(event.ydata)); pos = list(self.current_slices)
        plane_map = {'axial': (SAGITTAL, CORONAL), 'coronal': (SAGITTAL, AXIAL), 'sagittal': (CORONAL, AXIAL)}
        if data_view in plane_map: idx1, idx2 = plane_map[data_view]; pos[idx1], pos[idx2] = x_coord, y_coord
        pos = [np.clip(p, 0, d - 1) for p, d in zip(pos, self.dimensions)]; self.position_label.config(text=f"Position (X,Y,Z): {pos[SAGITTAL]}, {pos[CORONAL]}, {pos[AXIAL]}")
        try: self.intensity_label.config(text=f"Intensity: {self.volume_data[pos[AXIAL], pos[CORONAL], pos[SAGITTAL]]:.2f} HU")
        except IndexError: self.intensity_label.config(text="Intensity: ---")

    def load_volume(self):
        filepath = filedialog.askopenfilename(title="Select Medical Volume File", filetypes=[("All supported", "*.nii *.nii.gz *.mhd"), ("NIfTI files", "*.nii *.nii.gz"), ("MetaImage", "*.mhd")], parent=self.root)
        if not filepath: return
        self.filepath = filepath
        self.status_label.config(text=f"Loading {os.path.basename(filepath)}..."); self.progress.pack(side=tk.RIGHT, padx=5); self.progress.start(); self.root.update_idletasks()
        try:

            image = sitk.ReadImage(filepath)
            self.volume_data = sitk.GetArrayFromImage(image).astype(np.float32)
            self.dimensions = list(self.volume_data.shape)
            self.spacing = list(reversed(image.GetSpacing()))
            
            self.patient_info = {"name": "N/A", "id": "N/A"}
            self._update_patient_info_display()
            
            self.clear_drawings(confirm=False)
            self.clear_measurements(confirm=False)
            self.history_stack.clear()
            self.redo_stack.clear()
            self._update_history_buttons()

            min_val, max_val = np.min(self.volume_data), np.max(self.volume_data)
            self.level = (max_val + min_val) / 2
            self.window = max_val - min_val if max_val > min_val else 1
            self.window_slider.config(from_=1, to=max(self.window * 2, 1))
            self.level_slider.config(from_=min_val, to=max_val)
            self.iso_slider.config(from_=min_val, to=max_val)
            self.iso_value.set(np.mean([min_val, max_val]) + np.std(self.volume_data))
            
            self._update_wl_vars()
            self._update_wl_controls()
            self.current_slices = [d // 2 for d in self.dimensions]
            self._configure_slice_sliders()
            self._update_slice_labels()
            self.reset_zoom()
            self.update_all_views()
            self.status_label.config(text="Volume loaded successfully")

        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load volume:\n{str(e)}", parent=self.root); self.status_label.config(text="Load failed")
        finally:
            self.progress.stop(); self.progress.pack_forget()

    def _update_slices_from_click(self, event, view):
        if self.volume_data is None: return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        plane_map = {'axial': (SAGITTAL, CORONAL), 'coronal': (SAGITTAL, AXIAL), 'sagittal': (CORONAL, AXIAL)}
        if view in plane_map:
            idx1, idx2 = plane_map[view]
            self.current_slices[idx1] = np.clip(x, 0, self.dimensions[idx1]-1)
            self.current_slices[idx2] = np.clip(y, 0, self.dimensions[idx2]-1)
        self._update_slice_labels(); self._update_slice_sliders(); self.update_all_views()

    def _get_slice_data(self, plane_idx):
        if self.volume_data is None: return None
        s = self.current_slices[plane_idx]
        d = self.dimensions[plane_idx]
        if not (0 <= s < d): return np.zeros((100,100)) 
        
        if plane_idx == AXIAL: return self.volume_data[s, :, :]
        elif plane_idx == CORONAL: return self.volume_data[:, s, :]
        elif plane_idx == SAGITTAL: return self.volume_data[:, :, s]
        return None

    def update_slice(self, plane_idx, value):
        if self.volume_data is None: return
        self.current_slices[plane_idx] = int(float(value)); self._update_slice_labels(); self.update_2d_views()

    def set_window(self, value): self.window = float(value); self._update_wl_vars(); self.window_label.config(text=f"{int(self.window)}"); self.update_2d_views()

    def set_level(self, value): self.level = float(value); self._update_wl_vars(); self.level_label.config(text=f"{int(self.level)}"); self.update_2d_views()

    def _update_wl_vars(self): self.vmin = self.level - self.window/2; self.vmax = self.level + self.window/2

    def _update_wl_controls(self): self.window_slider.set(self.window); self.level_slider.set(self.level); self.window_label.config(text=f"{int(self.window)}"); self.level_label.config(text=f"{int(self.level)}")

    def _configure_slice_sliders(self):
        if self.volume_data is None: return
        self.axial_slider.config(to=self.dimensions[AXIAL] - 1); self.coronal_slider.config(to=self.dimensions[CORONAL] - 1); self.sagittal_slider.config(to=self.dimensions[SAGITTAL] - 1)
        self._update_slice_sliders()

    def _update_slice_sliders(self):
        if self.volume_data is not None: self.axial_slider.set(self.current_slices[AXIAL]); self.coronal_slider.set(self.current_slices[CORONAL]); self.sagittal_slider.set(self.current_slices[SAGITTAL])
        
    def _update_slice_labels(self):
        if self.volume_data is not None:
            self.axial_label.config(text=f"{self.current_slices[AXIAL]}/{self.dimensions[AXIAL]-1}"); self.coronal_label.config(text=f"{self.current_slices[CORONAL]}/{self.dimensions[CORONAL]-1}"); self.sagittal_label.config(text=f"{self.current_slices[SAGITTAL]}/{self.dimensions[SAGITTAL]-1}")
        else: self.axial_label.config(text="0/0"); self.coronal_label.config(text="0/0"); self.sagittal_label.config(text="0/0")

    def generate_3d_model(self):
        if self.volume_data is None: return
        self.status_label.config(text="Generating 3D model..."); self.progress.pack(side=tk.RIGHT, padx=5); self.progress.start(); self.root.update_idletasks()
        future = self.executor.submit(skimage.measure.marching_cubes, self.volume_data, level=self.iso_value.get(), spacing=(self.spacing[2], self.spacing[1], self.spacing[0]))
        future.add_done_callback(self._on_3d_model_generated)

    def _on_3d_model_generated(self, future):
        self.progress.stop(); self.progress.pack_forget()
        try:
            verts, faces, _, _ = future.result()
            self.mesh_verts, self.mesh_faces = verts, faces; self.show_model.set(True)
            self.status_label.config(text=f"3D Model generated ({len(verts)} vertices)."); self.update_3d_view()
        except Exception as e:
            self.status_label.config(text="3D Model generation failed."); messagebox.showerror("3D Model Error", f"Failed to generate 3D model:\n{e}", parent=self.root)

    def export_model(self):
        if not (self.mesh_verts is not None and self.mesh_faces is not None): messagebox.showinfo("Export Error", "No 3D model to export. Please generate one first.", parent=self.root); return
        initial_filename = self._get_sanitized_filename("3D_Model") + ".obj"
        filepath = filedialog.asksaveasfilename(initialfile=initial_filename, defaultextension=".obj", filetypes=[("OBJ file", "*.obj")], parent=self.root)
        if not filepath: return
        try:
            with open(filepath, 'w') as f:
                f.write("# 3D Model exported from Medical Image Viewer\n")
                for v in self.mesh_verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in self.mesh_faces: f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            messagebox.showinfo("Export Successful", f"Model saved to {os.path.basename(filepath)}", parent=self.root)
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save model:\n{e}", parent=self.root)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?", parent=self.root):
            self.executor.shutdown(wait=False, cancel_futures=True)
            plt.close('all')
            self.root.destroy()

    def launch_explainability(self):
        # Launch the Analysis Tool as separate process
        try:
            import subprocess
            import sys
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "exnModel", "explainability_visuals.py")
            
            subprocess.Popen([sys.executable, script_path])
            self.status_label.config(text="Analysis Tool launched successfully")
            
        except Exception as e:
            self.status_label.config(text="Failed to launch Analysis Tool")
            messagebox.showerror("Error", f"Failed to launch Analysis Tool:\n{str(e)}")

    def _add_to_history(self, action):
        self.history_stack.append(action)
        self.redo_stack.clear()
        self._update_history_buttons()

    def undo_action(self):
        if not self.history_stack:
            self.status_label.config(text="Nothing to undo")
            return
            
        last_action = self.history_stack.pop()
        self.redo_stack.append(last_action)
        
        if last_action['type'] == 'draw':

            stroke = last_action['data']
            plane, slice_idx = stroke['plane'], stroke['slice']
            if plane in self.drawings and slice_idx in self.drawings[plane]:
                self.drawings[plane][slice_idx] = [s for s in self.drawings[plane][slice_idx] 
                                                if s['id'] != stroke['id']]

                if not self.drawings[plane][slice_idx]:
                    del self.drawings[plane][slice_idx]

                if self.last_drawn_stroke and self.last_drawn_stroke['id'] == stroke['id']:
                    self.last_drawn_stroke = self._find_most_recent_drawing()

                try:
                    self.drawings_tree.delete(f"Drawing_{stroke['id']}")
                except tk.TclError:
                    pass  # Item wasn't in tree (no comment)
                self.status_label.config(text=f"Undo: Removed drawing {stroke['id']}")
                    
        elif last_action['type'] == 'measure':

            measure = last_action['data']
            self.measurements = [m for m in self.measurements if m['id'] != measure['id']]

            if self.last_measurement and self.last_measurement['id'] == measure['id']:
                self.last_measurement = self._find_most_recent_measurement()

            try:
                self.measurements_tree.delete(f"Measure_{measure['id']}")
            except tk.TclError:
                pass  # Item wasn't in tree (no comment)
            self.status_label.config(text=f"Undo: Removed measurement {measure['id']}")
        
        elif last_action['type'] == 'delete_draw':

            stroke = last_action['data']
            plane, slice_idx = last_action['plane'], last_action['slice']
            
            if plane not in self.drawings:
                self.drawings[plane] = {}
            if slice_idx not in self.drawings[plane]:
                self.drawings[plane][slice_idx] = []
                
            self.drawings[plane][slice_idx].append(stroke)

            if 'comment' in stroke:
                self._add_item_to_tree(self.drawings_tree, stroke, "Drawing")
            self.status_label.config(text=f"Undo: Restored drawing {stroke['id']}")
                
        elif last_action['type'] == 'delete_measure':

            measure = last_action['data']
            self.measurements.append(measure)

            if 'comment' in measure:
                self._add_item_to_tree(self.measurements_tree, measure, "Measure")
            self.status_label.config(text=f"Undo: Restored measurement {measure['id']}")
        
        elif last_action['type'] == 'edit_draw_comment':

            data = last_action['data']
            for plane, slices in self.drawings.items():
                for slice_idx, strokes in slices.items():
                    for stroke in strokes:
                        if stroke.get('id') == data['id']:
                            old_comment = data['old_comment']
                            if old_comment:
                                stroke['comment'] = old_comment
                                self._update_annotation_in_tree(
                                    self.drawings_tree,
                                    f"Drawing_{data['id']}",
                                    stroke
                                )
                            else:

                                if 'comment' in stroke:
                                    del stroke['comment']
                                try:
                                    self.drawings_tree.delete(f"Drawing_{data['id']}")
                                except tk.TclError:
                                    pass
                            self.status_label.config(text=f"Undo: Reverted comment on drawing {data['id']}")
                            break
        
        elif last_action['type'] == 'edit_measure_comment':
            data = last_action['data']
            for measure in self.measurements:
                if measure.get('id') == data['id']:
                    old_comment = data['old_comment']
                    if old_comment:
                        measure['comment'] = old_comment
                        self._update_annotation_in_tree(
                            self.measurements_tree,
                            f"Measure_{data['id']}",
                            measure
                        )
                    else:
                        if 'comment' in measure:
                            del measure['comment']
                        try:
                            self.measurements_tree.delete(f"Measure_{data['id']}")
                        except tk.TclError:
                            pass
                    self.status_label.config(text=f"Undo: Reverted comment on measurement {data['id']}")
                    break
        
        elif last_action['type'] == 'add_draw_comment':
            data = last_action['data']
            for plane, slices in self.drawings.items():
                for slice_idx, strokes in slices.items():
                    for stroke in strokes:
                        if stroke.get('id') == data['id']:
                            if 'comment' in stroke:
                                del stroke['comment']
                            try:
                                self.drawings_tree.delete(f"Drawing_{data['id']}")
                            except tk.TclError:
                                pass
                            self.status_label.config(text=f"Undo: Removed comment from drawing {data['id']}")
                            break
        
        elif last_action['type'] == 'add_measure_comment':
            data = last_action['data']
            for measure in self.measurements:
                if measure.get('id') == data['id']:
                    if 'comment' in measure:
                        del measure['comment']
                    try:
                        self.measurements_tree.delete(f"Measure_{data['id']}")
                    except tk.TclError:
                        pass
                    self.status_label.config(text=f"Undo: Removed comment from measurement {data['id']}")
                    break
        
        self._update_history_buttons()
        self.update_all_views()

    def redo_action(self):     
        if not self.redo_stack:
            self.status_label.config(text="Nothing to redo")
            return
            
        redo_action = self.redo_stack.pop()
        self.history_stack.append(redo_action)
        
        if redo_action['type'] == 'draw':

            stroke = redo_action['data']
            plane, slice_idx = stroke['plane'], stroke['slice']
            
            if plane not in self.drawings:
                self.drawings[plane] = {}
            if slice_idx not in self.drawings[plane]:
                self.drawings[plane][slice_idx] = []
                
            self.drawings[plane][slice_idx].append(stroke)

            if 'comment' in stroke:
                self._add_item_to_tree(self.drawings_tree, stroke, "Drawing")
            self.last_drawn_stroke = stroke
            self.status_label.config(text=f"Redo: Restored drawing {stroke['id']}")
            
        elif redo_action['type'] == 'measure':

            measure = redo_action['data']
            self.measurements.append(measure)

            if 'comment' in measure:
                self._add_item_to_tree(self.measurements_tree, measure, "Measure")
            self.last_measurement = measure
            self.status_label.config(text=f"Redo: Restored measurement {measure['id']}")
        
        elif redo_action['type'] == 'delete_draw':

            stroke = redo_action['data']
            plane, slice_idx = stroke['plane'], stroke['slice']
            if plane in self.drawings and slice_idx in self.drawings[plane]:
                self.drawings[plane][slice_idx] = [s for s in self.drawings[plane][slice_idx] 
                                                if s['id'] != stroke['id']]

                if not self.drawings[plane][slice_idx]:
                    del self.drawings[plane][slice_idx]

                try:
                    self.drawings_tree.delete(f"Drawing_{stroke['id']}")
                except tk.TclError:
                    pass
                self.status_label.config(text=f"Redo: Deleted drawing {stroke['id']}")
        
        elif redo_action['type'] == 'delete_measure':

            measure = redo_action['data']
            self.measurements = [m for m in self.measurements if m['id'] != measure['id']]

            try:
                self.measurements_tree.delete(f"Measure_{measure['id']}")
            except tk.TclError:
                pass
            self.status_label.config(text=f"Redo: Deleted measurement {measure['id']}")
        
        elif redo_action['type'] == 'edit_draw_comment':

            data = redo_action['data']
            for plane, slices in self.drawings.items():
                for slice_idx, strokes in slices.items():
                    for stroke in strokes:
                        if stroke.get('id') == data['id']:
                            stroke['comment'] = data['new_comment']

                            item_id = f"Drawing_{data['id']}"
                            if not self.drawings_tree.exists(item_id):
                                self._add_item_to_tree(self.drawings_tree, stroke, "Drawing")
                            else:
                                self._update_annotation_in_tree(self.drawings_tree, item_id, stroke)
                            self.status_label.config(text=f"Redo: Updated comment on drawing {data['id']}")
                            break
        
        elif redo_action['type'] == 'edit_measure_comment':

            data = redo_action['data']
            for measure in self.measurements:
                if measure.get('id') == data['id']:
                    measure['comment'] = data['new_comment']

                    item_id = f"Measure_{data['id']}"
                    if not self.measurements_tree.exists(item_id):
                        self._add_item_to_tree(self.measurements_tree, measure, "Measure")
                    else:
                        self._update_annotation_in_tree(self.measurements_tree, item_id, measure)
                    self.status_label.config(text=f"Redo: Updated comment on measurement {data['id']}")
                    break
        
        elif redo_action['type'] == 'add_draw_comment':

            data = redo_action['data']
            for plane, slices in self.drawings.items():
                for slice_idx, strokes in slices.items():
                    for stroke in strokes:
                        if stroke.get('id') == data['id']:
                            stroke['comment'] = data['comment']
                            self._add_item_to_tree(self.drawings_tree, stroke, "Drawing")
                            self.status_label.config(text=f"Redo: Added comment to drawing {data['id']}")
                            break
        
        elif redo_action['type'] == 'add_measure_comment':
            data = redo_action['data']
            for measure in self.measurements:
                if measure.get('id') == data['id']:
                    measure['comment'] = data['comment']
                    self._add_item_to_tree(self.measurements_tree, measure, "Measure")
                    self.status_label.config(text=f"Redo: Added comment to measurement {data['id']}")
                    break
        
        self._update_history_buttons()
        self.update_all_views()
        
        self._update_history_buttons()
        self.update_all_views()

    def _find_most_recent_drawing(self): 
        most_recent = None
        highest_id = -1  
        for plane, slices in self.drawings.items():
            for slice_idx, strokes in slices.items():
                for stroke in strokes:
                    if stroke.get('id', 0) > highest_id:
                        highest_id = stroke['id']
                        most_recent = stroke 
        return most_recent
    
    def _find_most_recent_measurement(self):
        if not self.measurements:
            return None
        
        return max(self.measurements, key=lambda m: m.get('id', 0))
    
    def _update_history_buttons(self): 
        self.undo_button.config(state=tk.NORMAL if self.history_stack else tk.DISABLED)
        self.redo_button.config(state=tk.NORMAL if self.redo_stack else tk.DISABLED)

    def delete_selected_annotation(self):
        selected_drawing = self.drawings_tree.selection()
        selected_measure = self.measurements_tree.selection()
        
        if not selected_drawing and not selected_measure:
            messagebox.showwarning("No Selection", "Please select an annotation to delete.", parent=self.root)
            return
        
        if selected_drawing:
            item_id = selected_drawing[0]
            item_type = 'Drawing'
            item_num = int(item_id.split('_')[1])
        else:
            item_id = selected_measure[0]
            item_type = 'Measure'
            item_num = int(item_id.split('_')[1])
        
        if not messagebox.askyesno(
            "Confirm Deletion", 
            f"Delete this {item_type.lower()}?",
            parent=self.root
        ):
            return
        
        if item_type == 'Drawing':

            for plane, slices in self.drawings.items():
                for slice_idx, strokes in slices.items():
                    for i, stroke in enumerate(strokes):
                        if stroke.get('id') == item_num:

                            self._add_to_history({
                                'type': 'delete_draw',
                                'data': stroke.copy(),
                                'plane': plane,
                                'slice': slice_idx
                            })
                            
                            del self.drawings[plane][slice_idx][i]
                            
                            if not self.drawings[plane][slice_idx]:
                                del self.drawings[plane][slice_idx]
                            
                            self.drawings_tree.delete(item_id)
                            
                            if (self.highlighted_item.get('type') == 'Drawing' and 
                                self.highlighted_item.get('id') == item_num):
                                self.highlighted_item = {'type': None, 'id': None}
                            
                            self.update_2d_views()
                            self.status_label.config(text=f"Deleted drawing {item_num}")
                            return
                            
        elif item_type == 'Measure':
            for i, measure in enumerate(self.measurements):
                if measure.get('id') == item_num:

                    self._add_to_history({
                        'type': 'delete_measure',
                        'data': measure.copy()
                    })                    

                    del self.measurements[i]
                    
                    self.measurements_tree.delete(item_id)
                    
                    if (self.highlighted_item.get('type') == 'Measure' and 
                        self.highlighted_item.get('id') == item_num):
                        self.highlighted_item = {'type': None, 'id': None}
                    
                    self.update_2d_views()
                    self.status_label.config(text=f"Deleted measurement {item_num}")
                    return

    def edit_annotation_comment(self):
        selected_drawing = self.drawings_tree.selection()
        selected_measure = self.measurements_tree.selection()
        
        if not selected_drawing and not selected_measure:
            messagebox.showwarning("No Selection", "Please select an annotation to edit.", parent=self.root)
            return
        
        if selected_drawing:
            item_id = selected_drawing[0]
            item_type = 'Drawing'
            item_num = int(item_id.split('_')[1])
            tree = self.drawings_tree
        else:
            item_id = selected_measure[0]
            item_type = 'Measure'
            item_num = int(item_id.split('_')[1])
            tree = self.measurements_tree
        
        if item_type == 'Drawing':
            for plane, slices in self.drawings.items():
                for slice_idx, strokes in slices.items():
                    for stroke in strokes:
                        if stroke.get('id') == item_num:
                            old_comment = stroke.get('comment', '')
                            new_comment = simpledialog.askstring(
                                "Edit Drawing Comment", 
                                "Edit drawing comment:",
                                initialvalue=old_comment,
                                parent=self.root
                            )
                            if new_comment is not None and new_comment != old_comment:

                                self._add_to_history({
                                    'type': 'edit_draw_comment',
                                    'data': {
                                        'id': item_num,
                                        'old_comment': old_comment,
                                        'new_comment': new_comment,
                                        'plane': plane,
                                        'slice': slice_idx
                                    }
                                })                                

                                stroke['comment'] = new_comment
                                

                                plane_name = {AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"}[stroke['plane']]
                                text = f"({plane_name} {stroke['slice']}) {new_comment}"
                                tree.item(item_id, text=text, values=(text,))
                                
                                self.status_label.config(text=f"Updated drawing {item_num}")
                            return
                            
        elif item_type == 'Measure':
            for measure in self.measurements:
                if measure.get('id') == item_num:
                    old_comment = measure.get('comment', '')
                    new_comment = simpledialog.askstring(
                        "Edit Measurement Comment", 
                        "Edit measurement comment:",
                        initialvalue=old_comment,
                        parent=self.root
                    )
                    if new_comment is not None and new_comment != old_comment:

                        self._add_to_history({
                            'type': 'edit_measure_comment',
                            'data': {
                                'id': item_num,
                                'old_comment': old_comment,
                                'new_comment': new_comment
                            }
                        })
                        
                        measure['comment'] = new_comment
                        
                        plane_name = {AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"}[measure['plane']]
                        text = f"({plane_name} {measure['slice']}) {new_comment}"
                        tree.item(item_id, text=text, values=(text,))
                        
                        self.status_label.config(text=f"Updated measurement {item_num}")
                    return

    def _update_annotation_in_tree(self, tree, item_id, annotation_data):
        plane_name = {AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"}[annotation_data['plane']]
        text = f"({plane_name} {annotation_data['slice']}) {annotation_data.get('comment', '')}"
        tree.item(item_id, text=text, values=(text,))
            
if __name__ == "__main__":

    def start_main_app(): 
        root = tk.Tk()
        root.withdraw()  
        
        if not TENSORFLOW_AVAILABLE:
            messagebox.showwarning("Missing Dependencies", "TensorFlow and/or OpenCV could not be imported.\nThe Analysis functionality will be disabled. Please install them to enable all features:\n\n`pip install tensorflow opencv-python Pillow`")
        
        app = SlicerApp(root)       

        root.after(100, lambda: root.deiconify())
        root.mainloop()
    
    try:
        from loading_window import LoadingWindow

        loading = LoadingWindow("Neuroimaging Slice Viewer", "Loading Neuroimaging Slice Viewer...")
        loading.show_and_animate(2.5, start_main_app) 
        
    except ImportError:
        print("Loading window not available, starting directly...")
        start_main_app()