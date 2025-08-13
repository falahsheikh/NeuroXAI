# explainability_visuals.py
"""
XAI Analysis Tool for Early Alzheimer's Disease Detection
Copyright (c) 2024 Falah Sheikh, ADSA Lab, University of Calgary
Licensed under CC BY-NC 4.0 - see LICENSE file for details.

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 
International License. Commercial use is prohibited without explicit permission.

Developed at the Advanced Database Systems and Applications (ADSA) Lab,
University of Calgary, with funding from Alberta Innovates Summer Research Studentship.

Author: Falah Sheikh (https://github.com/falahsheikh/Lightweight_MRI_EAD_Detection)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Frame, Toplevel
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import cv2
import scipy.ndimage
from tensorflow.keras.models import Model
from datetime import datetime
import json
import scipy.stats
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loading_window import LoadingWindow
import time
from PIL import ImageFilter
from scipy.ndimage import zoom
from skimage import measure, morphology
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Frame, Toplevel

IMG_SIZE = (224, 224)
DEFAULT_CLASS_NAMES = ['CN', 'EMCI', 'LMCI']
DEFAULT_CLASS_DESCRIPTIONS = {
    'CN': 'Cognitively Normal',
    'EMCI': 'Early Mild Cognitive Impairment',
    'LMCI': 'Late Mild Cognitive Impairment'
}
CLASS_NAMES = DEFAULT_CLASS_NAMES.copy()
CLASS_DESCRIPTIONS = DEFAULT_CLASS_DESCRIPTIONS.copy()

class MedicalImageProcessor:
    
    @staticmethod
    def find_brain_bbox(slice_data, threshold_percentile=5):
        if np.max(slice_data) == 0:
            return 0, slice_data.shape[0], 0, slice_data.shape[1]
            
        threshold = np.percentile(slice_data[slice_data > 0], threshold_percentile)
        binary_mask = slice_data > threshold
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=256)
        binary_mask = morphology.binary_closing(binary_mask, morphology.disk(5))
        
        labeled_mask = measure.label(binary_mask)
        if labeled_mask.max() == 0:
            return 0, slice_data.shape[0], 0, slice_data.shape[1]
            
        props = measure.regionprops(labeled_mask)
        largest_region = max(props, key=lambda x: x.area)
        min_row, min_col, max_row, max_col = largest_region.bbox
        
        padding = 10
        return (max(0, min_row - padding), min(slice_data.shape[0], max_row + padding),
                max(0, min_col - padding), min(slice_data.shape[1], max_col + padding))
    
    @staticmethod
    def crop_and_resize_slice(slice_data, target_size=(224, 224)):
        min_row, max_row, min_col, max_col = MedicalImageProcessor.find_brain_bbox(slice_data)
        cropped = slice_data[min_row:max_row, min_col:max_col]
        
        if cropped.size == 0:
            return np.zeros(target_size, dtype=slice_data.dtype)
            
        scale = min(target_size[0] / cropped.shape[0], target_size[1] / cropped.shape[1])
        new_shape = (int(cropped.shape[0] * scale), int(cropped.shape[1] * scale))
        resized = zoom(cropped, (new_shape[0] / cropped.shape[0], new_shape[1] / cropped.shape[1]), order=1)
        
        final_image = np.zeros(target_size, dtype=resized.dtype)
        start_h = (target_size[0] - new_shape[0]) // 2
        start_w = (target_size[1] - new_shape[1]) // 2
        final_image[start_h:start_h + new_shape[0], start_w:start_w + new_shape[1]] = resized
        
        return final_image

class ContentWindow(Frame):

    def __init__(self, parent, title, app):
        super().__init__(parent, relief=tk.SUNKEN, bd=1, bg='white')
        self.title = title
        self.app = app  
        self.is_maximized = False
        self.is_blurred = True 
        self.original_figure = None

        self.title_bar = Frame(self, bg='#000080', height=28, relief=tk.RAISED, bd=1)
        self.title_bar.pack(fill=tk.X)
        self.title_bar.pack_propagate(False)

        self.title_label = tk.Label(self.title_bar, text=title, fg='white', bg='#000080',
                                    font=('MS Sans Serif', 8, 'bold'), anchor='w')
        self.title_label.pack(side=tk.LEFT, padx=8, pady=4, fill=tk.X, expand=True)

        self.controls_frame = Frame(self.title_bar, bg='#2E3B55')
        self.controls_frame.pack(side=tk.RIGHT, padx=4, pady=2)

        self.maximize_btn = tk.Button(
            self.controls_frame, 
            text="Max", 
            command=self.toggle_maximize,
            font=('MS Sans Serif', 7, 'normal'), 
            fg='black', 
            bg='#C0C0C0',
            activebackground='#E0E0E0',
            activeforeground='black',
            relief=tk.RAISED,
            bd=2,
            width=4,
            height=1,
            cursor='hand2'
        )
        self.maximize_btn.pack(side=tk.RIGHT, padx=1)

        self.content_frame = Frame(self, bg='#eee8d5')
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.figure = None
        self.canvas = None
        self.toolbar = None

        self.bind('<Configure>', self.on_resize)

    def on_resize(self, event):
        if event.widget == self:
            self._last_size = (self.winfo_width(), self.winfo_height())

    def toggle_maximize(self):
        self.is_maximized = not self.is_maximized

        if self.is_maximized:
            self.maximize_btn.config(text="Min", bg='#A0A0A0')

            if self.is_blurred:
                self.remove_blur()
        else:
            self.maximize_btn.config(text="Max", bg='#C0C0C0')

            if not self.is_blurred and self.should_be_blurred():
                self.apply_blur()
        
        self.app.toggle_maximize_pane(self)

    def _delayed_blur(self):
        if self.should_be_blurred() and not self.is_blurred:
            self._last_size = (self.winfo_width(), self.winfo_height())
            self.apply_blur()
            self._blur_applied = True

    def apply_blur(self):
        if not self.figure or not self.canvas:
            return
            
        try:
            if self.original_figure is None:

                import io
                import pickle
                buf = io.BytesIO()
                pickle.dump(self.figure, buf)
                buf.seek(0)
                self.original_figure = pickle.load(buf)
            
            self.canvas.get_tk_widget().destroy()
            if hasattr(self, 'toolbar') and self.toolbar:
                self.toolbar.destroy()            

            blur_fig = Figure(figsize=self.figure.get_size_inches(), dpi=self.figure.dpi)
            blur_fig.patch.set_facecolor('lightgray')
            
            ax = blur_fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='white', alpha=0.7))
            ax.text(0.5, 0.5, 'Maximize to View\nChart Content', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.8", facecolor="beige", alpha=0.9))
            
            self.canvas = FigureCanvasTkAgg(blur_fig, self.content_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.toolbar = EnhancedMedicalNavigationToolbar(self.canvas, self.content_frame, self.title)
            self.toolbar.update()
            
            if self.figure != self.original_figure:
                plt.close(self.figure)
            self.figure = blur_fig
            
            self.is_blurred = True
            
        except Exception as e:
            print(f"Error applying blur: {e}")
            import traceback
            traceback.print_exc()

    def remove_blur(self):
        if not self.is_blurred or not self.original_figure:
            return
            
        try:
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            if hasattr(self, 'toolbar') and self.toolbar:
                self.toolbar.destroy()
            
            if self.figure and self.figure != self.original_figure:
                plt.close(self.figure)
            
            self.figure = self.original_figure
            
            self.canvas = FigureCanvasTkAgg(self.figure, self.content_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.toolbar = EnhancedMedicalNavigationToolbar(self.canvas, self.content_frame, self.title)
            self.toolbar.update()
            
            self.is_blurred = False
            
        except Exception as e:
            print(f"Error removing blur: {e}")
            import traceback
            traceback.print_exc()

    def set_content(self, figure):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        if self.figure and self.figure != figure:
            plt.close(self.figure)
        if self.original_figure and self.original_figure != figure:
            plt.close(self.original_figure)
        
        self.figure = figure
        self.original_figure = None  
        self.is_blurred = False
        self._blur_applied = False 

        self.canvas = FigureCanvasTkAgg(figure, self.content_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = EnhancedMedicalNavigationToolbar(self.canvas, self.content_frame, self.title)
        self.toolbar.update()
        
        if self.should_be_blurred() and not self.is_maximized:

            self.content_frame.after(500, self._delayed_blur)

    def should_be_blurred(self):
        no_blur_windows = [
            "Original MRI", 
            "Grad-CAM++ Raw", 
            "Grad-CAM++ Masked", 
            "Grad-CAM++ Overlay", 
            "Guided Grad-CAM++ Raw",
            "Guided Grad-CAM++ Masked", 
            "Guided Grad-CAM++ Overlay", 
            "Consensus Raw", 
            "Consensus Masked", 
            "Consensus Overlay"
        ]
        return self.title not in no_blur_windows

class EnhancedMedicalNavigationToolbar(NavigationToolbar2Tk):

    def __init__(self, canvas, parent, window_title=""):
        super().__init__(canvas, parent)
        self.window_title = window_title
        unwanted_buttons = ['!button5', '!button4'] 
        for btn_name in unwanted_buttons:
            if btn_name in self.children:
                self.children[btn_name].destroy()
        
        if self.needs_medical_tools():
            self.add_medical_tools()
    
    def needs_medical_tools(self):
        no_tools_windows = [
            "Guided Stats", "Grad-CAM++ Stats", "Class Probabilities", 
            "Confidence Map", "Prediction Analysis", "Medical Report"
        ]
        return self.window_title not in no_tools_windows

    def add_medical_tools(self):
        ttk.Separator(self, orient='vertical').pack(side=tk.LEFT, padx=2, pady=2, fill=tk.Y)

class XAIProcessor:
        
    def __init__(self, model):
        self.model = model
        self.num_classes = self._get_num_classes()
        self._update_class_info()

        self.conv_layers = self._get_conv_layers()
        self.selected_layers = {
            'gradcam': self.conv_layers[-1] if self.conv_layers else None,  # Default to last conv layer
            'guided_gradcam': self.conv_layers[-1] if self.conv_layers else None,
            'consensus': self.conv_layers[-1] if self.conv_layers else None
        }

    def _get_conv_layers(self):
        """Extract all convolutional layer names from the model"""
        conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer.name)
        return conv_layers
    
    def get_available_layers(self):
        """Return list of available convolutional layers"""
        return self.conv_layers.copy()
    
    def set_layer_for_method(self, method, layer_name):
        """Set the layer to use for a specific XAI method"""
        if layer_name not in self.conv_layers:
            raise ValueError(f"Layer {layer_name} not found in model. Available layers: {self.conv_layers}")
        
        valid_methods = ['gradcam', 'guided_gradcam', 'consensus']
        if method not in valid_methods:
            raise ValueError(f"Method {method} not valid. Valid methods: {valid_methods}")
        
        self.selected_layers[method] = layer_name
        print(f"Set {method} to use layer: {layer_name}")

    def get_layer_for_method(self, method):
        """Get the currently selected layer for a method"""
        return self.selected_layers.get(method, self.conv_layers[-1] if self.conv_layers else None)
    
    def _get_num_classes(self):
        try:
            output_shape = self.model.output_shape
            if isinstance(output_shape, tuple): 
                return output_shape[-1]
            if isinstance(output_shape, list): 
                return output_shape[0][-1]
            return 3
        except: 
            return 3
            
    def _update_class_info(self):
        global CLASS_NAMES, CLASS_DESCRIPTIONS
        if self.num_classes == 1:
            CLASS_NAMES, CLASS_DESCRIPTIONS = ['Binary'], {'Binary': 'Binary Classification Output'}
        elif self.num_classes == 2:
            CLASS_NAMES, CLASS_DESCRIPTIONS = ['Class_0', 'Class_1'], {'Class_0': 'Class 0 (e.g., Normal)', 'Class_1': 'Class 1 (e.g., Abnormal)'}
        elif self.num_classes == 3:
            CLASS_NAMES, CLASS_DESCRIPTIONS = DEFAULT_CLASS_NAMES.copy(), DEFAULT_CLASS_DESCRIPTIONS.copy()
        else:
            CLASS_NAMES = [f'Class_{i}' for i in range(self.num_classes)]
            CLASS_DESCRIPTIONS = {name: f'Class {i}' for i, name in enumerate(CLASS_NAMES)}
        print(f"Model detected with {self.num_classes} classes: {CLASS_NAMES}")
        
    def validate_class_index(self, class_idx):
        return min(class_idx, self.num_classes - 1)
        
    def find_last_conv_layer(self):
        """Deprecated: Use get_layer_for_method instead"""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D): 
                return layer.name
        raise ValueError("No Conv2D layer found in the model.")
        
    def apply_brain_mask(self, heatmap, original_img_array):
        if original_img_array.dtype != np.uint8:
            original_img_array = np.clip(original_img_array, 0, 255).astype(np.uint8)
        gray_img = cv2.cvtColor(original_img_array, cv2.COLOR_RGB2GRAY) if len(original_img_array.shape) == 3 else original_img_array
        _, binary_mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        mask = (binary_mask > 0).astype(np.float32)
        heatmap = heatmap.numpy() if hasattr(heatmap, 'numpy') else heatmap
        return heatmap * mask
        
    def make_gradcam_plus_plus(self, img_array, target_class_idx, layer_name=None):
        """Generate Grad-CAM++ using specified or default layer"""
        target_class_idx = self.validate_class_index(target_class_idx)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32) if isinstance(img_array, np.ndarray) else img_array
        
        # Use specified layer or default for gradcam
        if layer_name is None:
            layer_name = self.get_layer_for_method('gradcam')
        
        if layer_name is None:
            raise ValueError("No convolutional layer available for Grad-CAM++")
        
        try:
            grad_model = Model(inputs=self.model.input, outputs=[self.model.get_layer(layer_name).output, self.model.output])
        except Exception as e:
            print(f"Error creating gradient model with layer {layer_name}: {e}")
            return np.zeros(IMG_SIZE)
            
        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                predictions = tf.convert_to_tensor(predictions) if not isinstance(predictions, tf.Tensor) else predictions
                if len(predictions.shape) == 1: 
                    predictions = tf.expand_dims(predictions, 0)
                elif len(predictions.shape) == 3 and predictions.shape[0] == 1: 
                    predictions = tf.squeeze(predictions, axis=1)
                actual_num_classes = predictions.shape[-1]
                if target_class_idx >= actual_num_classes: 
                    target_class_idx = actual_num_classes - 1
                class_output = predictions[:, target_class_idx]
                
            grads = tape.gradient(class_output, conv_outputs)
            if grads is None: 
                return np.zeros(IMG_SIZE)
                
            conv_outputs, grads = conv_outputs[0], grads[0]
            alpha_denom = 2.0 * tf.square(grads) + tf.reduce_sum(conv_outputs * tf.pow(grads, 3), axis=[0, 1], keepdims=True) + 1e-7
            alphas = tf.square(grads) / alpha_denom
            weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=[0, 1])
            heatmap = tf.nn.relu(tf.reduce_sum(weights * conv_outputs, axis=2))
            heatmap_max = tf.reduce_max(heatmap)
            if heatmap_max > 0: 
                heatmap /= heatmap_max
            return tf.squeeze(tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE, method='bilinear')).numpy()
            
        except Exception as e:
            print(f"Error in grad-cam computation with layer {layer_name}: {e}")
            return np.zeros(IMG_SIZE)
            
    def make_guided_backprop(self, img_array, target_class_idx):
        """Guided backprop doesn't use specific conv layers, works on input"""
        target_class_idx = self.validate_class_index(target_class_idx)
        try:
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32) if isinstance(img_array, np.ndarray) else img_array
            img_tensor = tf.Variable(img_tensor, trainable=True)
            
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                predictions = self.model(img_tensor)
                predictions = tf.convert_to_tensor(predictions) if not isinstance(predictions, tf.Tensor) else predictions
                if len(predictions.shape) == 1: 
                    predictions = tf.expand_dims(predictions, 0)
                elif len(predictions.shape) == 3 and predictions.shape[0] == 1: 
                    predictions = tf.squeeze(predictions, axis=1)
                actual_num_classes = predictions.shape[-1]
                if target_class_idx >= actual_num_classes: 
                    target_class_idx = actual_num_classes - 1
                pred_class = predictions[:, target_class_idx]
                
            grads = tape.gradient(pred_class, img_tensor)
            if grads is None: 
                return np.zeros(IMG_SIZE)
            
            input_positive = tf.cast(img_tensor > 0, tf.float32)
            grad_positive = tf.cast(grads > 0, tf.float32)
            guided_grads = grads * input_positive * grad_positive
            
            guided_grads = tf.abs(guided_grads[0])
            attribution_map = tf.reduce_max(guided_grads, axis=-1)
            
            return attribution_map.numpy()
            
        except Exception as e:
            print(f"Error in guided backprop: {e}")
            return np.zeros(IMG_SIZE)

    def make_guided_gradcam_plus_plus(self, img_array, target_class_idx, layer_name=None):
        """Generate Guided Grad-CAM++ using specified or default layer"""
        try:
            # Use specified layer or default for guided_gradcam
            if layer_name is None:
                layer_name = self.get_layer_for_method('guided_gradcam')
            
            gradcam_heatmap = self.make_gradcam_plus_plus(img_array, target_class_idx, layer_name)
            guided_backprop = self.make_guided_backprop(img_array, target_class_idx)

            if gradcam_heatmap.shape != guided_backprop.shape:
                print(f"Warning: Shape mismatch - GradCAM: {gradcam_heatmap.shape}, Guided: {guided_backprop.shape}")
                if len(guided_backprop.shape) == 2:
                    guided_backprop = cv2.resize(guided_backprop, (gradcam_heatmap.shape[1], gradcam_heatmap.shape[0]))
            
            guided_gradcam = gradcam_heatmap * guided_backprop           
            guided_gradcam_max = np.max(guided_gradcam)
            if guided_gradcam_max > 0:
                guided_gradcam = guided_gradcam / guided_gradcam_max
                
        except Exception as e:
            print(f"Warning: Guided backprop failed ({str(e)}), using Grad-CAM++ only.")
            if layer_name is None:
                layer_name = self.get_layer_for_method('guided_gradcam')
            guided_gradcam = self.make_gradcam_plus_plus(img_array, target_class_idx, layer_name)        

        smoothed_guided = scipy.ndimage.gaussian_filter(guided_gradcam, sigma=0.8)
        return smoothed_guided
        
    def create_consensus_map(self, heatmaps, weights=None):
        """Create consensus map from multiple heatmaps"""
        if not heatmaps: 
            return None
        weights = [1.0] * len(heatmaps) if weights is None else weights
        
        normalized_heatmaps = []
        for h in heatmaps:
            h_max, h_min = np.max(h), np.min(h)
            h_norm = (h - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(h)
            normalized_heatmaps.append(h_norm)
            
        if not normalized_heatmaps: 
            return np.zeros_like(heatmaps[0])
        return np.average(np.array(normalized_heatmaps), axis=0, weights=weights)
        
    def create_overlay(self, heatmap, original_img_rgb, alpha=0.6, colormap=cv2.COLORMAP_JET):
        """Create overlay of heatmap on original image"""
        heatmap = heatmap.numpy() if hasattr(heatmap, 'numpy') else heatmap
        original_img_rgb = original_img_rgb.numpy() if hasattr(original_img_rgb, 'numpy') else original_img_rgb
        
        heatmap_max = np.max(heatmap)
        heatmap_normalized = np.clip(heatmap / heatmap_max if heatmap_max > 0 else heatmap, 0, 1)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), colormap)
        original_img_bgr = cv2.cvtColor(np.clip(original_img_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
class LayerSelectionDialog:

    def __init__(self, parent, xai_processor):
        self.parent = parent
        self.xai_processor = xai_processor
        self.result = None
        
    def show(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Configure XAI Layers")
        self.dialog.geometry("650x800")
        self.dialog.resizable(True, True)
        
        self.dialog.transient(self.parent)
        self.dialog.lift()
        self.dialog.focus_force()
        
        self.dialog.geometry("+%d+%d" % (
            self.parent.winfo_rootx() + 100,
            self.parent.winfo_rooty() + 50
        ))
        
        self.dialog.configure(bg='white')
        
        self.create_widgets()
        
        self.dialog.focus_set()
        
        self.dialog.wait_window()
        return self.result
    
    def create_widgets(self):
        main_container = Frame(self.dialog, bg='white')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_container, bg='white', highlightthickness=0)
        scrollbar_main = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview, 
                                     bg='#E0E0E0', troughcolor='#F5F5DC', activebackground='#D0D0D0')
        scrollable_frame = Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_main.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_main.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main_frame = Frame(scrollable_frame, bg='white', padx=25, pady=25)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(
            main_frame, 
            text="XAI Layer Configuration",
            font=('SF Pro Text', 14, 'bold') if self.is_macos() else ('MS Sans Serif', 14, 'bold'),
            bg='white',
            fg='black'
        )
        title_label.pack(pady=(0, 25))
        
        available_layers = self.xai_processor.get_available_layers()
        if not available_layers:
            error_label = tk.Label(
                main_frame, 
                text="No convolutional layers found in the model!",
                bg='white',
                fg='red',
                font=('SF Pro Text', 11) if self.is_macos() else ('MS Sans Serif', 11)
            )
            error_label.pack()
            return
        
        layers_frame = Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        layers_frame.pack(fill=tk.X, pady=(0, 20))
        
        layers_inner = Frame(layers_frame, bg='#F5F5DC', padx=20, pady=15)
        layers_inner.pack(fill=tk.BOTH, expand=True)
        
        layers_title = tk.Label(
            layers_inner,
            text="Available Convolutional Layers",
            font=('SF Pro Text', 12, 'bold') if self.is_macos() else ('MS Sans Serif', 12, 'bold'),
            bg='#F5F5DC',
            fg='black'
        )
        layers_title.pack(anchor=tk.W, pady=(0, 12))
        
        listbox_frame = Frame(layers_inner, bg='#F5F5DC')
        listbox_frame.pack(fill=tk.X, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(
            listbox_frame, 
            bg='#E0E0E0',
            troughcolor='#F5F5DC',
            activebackground='#D0D0D0'
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        self.layers_listbox = tk.Listbox(
            listbox_frame, 
            yscrollcommand=scrollbar.set,
            height=10,
            font=('SF Mono', 10) if self.is_macos() else ('Consolas', 10),
            bg='#FFFEF7',
            fg='black',
            selectbackground='#D2B48C',
            selectforeground='black',
            relief=tk.SUNKEN,
            bd=2,
            highlightthickness=0,
            activestyle='none'
        )
        self.layers_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.layers_listbox.yview)
        
        for i, layer_name in enumerate(available_layers):
            layer_info = self.get_layer_info(layer_name)
            display_text = f"{i+1:2d}.  {layer_name:<25} {layer_info}"
            self.layers_listbox.insert(tk.END, display_text)
        
        self.layers_listbox.bind('<Double-1>', self.on_layer_double_click)
        
        instruction_label = tk.Label(
            layers_inner,
            text="Double-click a layer to choose which method to apply it to",
            font=('SF Pro Text', 10) if self.is_macos() else ('MS Sans Serif', 10),
            bg='#F5F5DC',
            fg='#8B4513',
            anchor='w'
        )
        instruction_label.pack(anchor=tk.W, pady=(8, 0))
        
        selection_frame = Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        selection_frame.pack(fill=tk.X, pady=(0, 20))
        
        inner_frame = Frame(selection_frame, bg='#F5F5DC', padx=20, pady=18)
        inner_frame.pack(fill=tk.BOTH, expand=True)
        
        selection_title = tk.Label(
            inner_frame,
            text="Method Layer Configuration",
            font=('SF Pro Text', 12, 'bold') if self.is_macos() else ('MS Sans Serif', 12, 'bold'),
            bg='#F5F5DC',
            fg='black'
        )
        selection_title.pack(anchor=tk.W, pady=(0, 15))
        
        self.layer_vars = {}
        self.layer_combos = {}
        methods = [
            ('gradcam', 'Grad-CAM++:'),
            ('guided_gradcam', 'Guided Grad-CAM++:'),
            ('consensus', 'Consensus Analysis:')
        ]
        
        for i, (method_key, method_label) in enumerate(methods):
            method_frame = Frame(inner_frame, bg='#F5F5DC')
            method_frame.pack(fill=tk.X, pady=8)
            
            label = tk.Label(
                method_frame, 
                text=method_label, 
                width=20,
                anchor='w',
                bg='#F5F5DC',
                fg='black',
                font=('SF Pro Text', 11) if self.is_macos() else ('MS Sans Serif', 11)
            )
            label.pack(side=tk.LEFT)
            
            self.layer_vars[method_key] = tk.StringVar()
            
            combo_frame = Frame(method_frame, bg='#F5F5DC')
            combo_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(15, 0))
            
            combo = ttk.Combobox(
                combo_frame,
                textvariable=self.layer_vars[method_key],
                values=available_layers,
                state="readonly",
                width=28,
                font=('SF Mono', 10) if self.is_macos() else ('Consolas', 10)
            )
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.layer_combos[method_key] = combo
            
            current_layer = self.xai_processor.get_layer_for_method(method_key)
            if current_layer and current_layer in available_layers:
                combo.set(current_layer)
            elif available_layers:
                combo.set(available_layers[-1])  # Default to last layer
            
            set_btn = tk.Button(
                combo_frame,
                text="Set",
                command=lambda m=method_key: self.set_single_method(m),
                font=('SF Pro Text', 10, 'bold') if self.is_macos() else ('MS Sans Serif', 10, 'bold'),
                bg='#D2B48C',
                fg='black',
                activebackground='#C8A882',
                activeforeground='black',
                relief=tk.RAISED,
                bd=2,
                width=6,
                cursor='hand2',
                pady=8
            )
            set_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        quick_frame = Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        quick_frame.pack(fill=tk.X, pady=(0, 25))
        
        quick_inner = Frame(quick_frame, bg='#F5F5DC', padx=20, pady=15)
        quick_inner.pack(fill=tk.BOTH, expand=True)
        
        quick_title = tk.Label(
            quick_inner,
            text="Quick Selection",
            font=('SF Pro Text', 12, 'bold') if self.is_macos() else ('MS Sans Serif', 12, 'bold'),
            bg='#F5F5DC',
            fg='black'
        )
        quick_title.pack(anchor=tk.W, pady=(0, 12))
        
        quick_buttons_frame = Frame(quick_inner, bg='#F5F5DC')
        quick_buttons_frame.pack(fill=tk.X)
        
        quick_buttons = [
            ("All → First", self.set_all_first),
            ("All → Last", self.set_all_last),
            ("All → Middle", self.set_all_middle)
        ]
        
        for text, command in quick_buttons:
            btn = tk.Button(
                quick_buttons_frame,
                text=text,
                command=command,
                font=('SF Pro Text', 10) if self.is_macos() else ('MS Sans Serif', 10),
                bg='#F0E68C',
                fg='black',
                activebackground='#EEDD82',
                activeforeground='black',
                relief=tk.RAISED,
                bd=2,
                width=14,
                cursor='hand2',
                pady=10
            )
            btn.pack(side=tk.LEFT, padx=(0, 10))
        
        buttons_frame = Frame(main_frame, bg='white')
        buttons_frame.pack(fill=tk.X, pady=(30, 15))
        
        left_buttons = Frame(buttons_frame, bg='white')
        left_buttons.pack(side=tk.LEFT, anchor='w')
        
        reset_btn = tk.Button(
            left_buttons,
            text="Reset Defaults",
            command=self.reset_defaults,
            font=('SF Pro Text', 11) if self.is_macos() else ('MS Sans Serif', 11),
            bg='#DDD',
            fg='black',
            activebackground='#CCC',
            activeforeground='black',
            relief=tk.RAISED,
            bd=2,
            width=14,
            cursor='hand2',
            pady=8
        )
        reset_btn.pack()
        
        right_buttons = Frame(buttons_frame, bg='white')
        right_buttons.pack(side=tk.RIGHT, anchor='e')
        
        apply_btn = tk.Button(
            right_buttons,
            text="Apply & Re-analyze",
            command=self.apply_and_reanalyze,
            font=('SF Pro Text', 11, 'bold') if self.is_macos() else ('MS Sans Serif', 11, 'bold'),
            bg='#87CEEB',
            fg='black',
            activebackground='#87CEFA',
            activeforeground='black',
            relief=tk.RAISED,
            bd=2,
            width=16,
            cursor='hand2',
            pady=8
        )
        apply_btn.pack(side=tk.RIGHT, padx=(8, 0))
        
        apply_only_btn = tk.Button(
            right_buttons,
            text="Apply Only",
            command=self.apply_changes,
            font=('SF Pro Text', 11) if self.is_macos() else ('MS Sans Serif', 11),
            bg='#98FB98',
            fg='black',
            activebackground='#90EE90',
            activeforeground='black',
            relief=tk.RAISED,
            bd=2,
            width=12,
            cursor='hand2',
            pady=8
        )
        apply_only_btn.pack(side=tk.RIGHT, padx=(8, 0))
        
        cancel_btn = tk.Button(
            right_buttons,
            text="Cancel",
            command=self.cancel,
            font=('SF Pro Text', 11) if self.is_macos() else ('MS Sans Serif', 11),
            bg='#DDD',
            fg='black',
            activebackground='#CCC',
            activeforeground='black',
            relief=tk.RAISED,
            bd=2,
            width=10,
            cursor='hand2',
            pady=8
        )
        cancel_btn.pack(side=tk.RIGHT)
    
    def is_macos(self):
        import platform
        return platform.system() == 'Darwin'
    
    def get_layer_info(self, layer_name):
        try:
            layer = self.xai_processor.model.get_layer(layer_name)
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
                if isinstance(output_shape, tuple) and len(output_shape) >= 3:
                    return f"(filters: {output_shape[-1]})"
            return ""
        except:
            return ""
    
    def on_layer_double_click(self, event):
        selection = self.layers_listbox.curselection()
        if selection:
            layer_text = self.layers_listbox.get(selection[0])
            parts = layer_text.split()
            if len(parts) >= 2:
                layer_name = parts[1]  
                
                self.show_method_selection_simple(layer_name)
    
    def show_method_selection_simple(self, layer_name):
        methods = [
            ('gradcam', 'Grad-CAM++'),
            ('guided_gradcam', 'Guided Grad-CAM++'),
            ('consensus', 'Consensus Analysis'),
            ('all', 'All Methods')
        ]
        
        message = f"Apply layer '{layer_name}' to which method?\n\n"
        for i, (method_key, method_label) in enumerate(methods, 1):
            message += f"{i}. {method_label}\n"
        
        choice = tk.simpledialog.askstring(
            "Select Method",
            message + "\nEnter number (1-4) or press Cancel:",
            parent=self.dialog
        )
        
        if choice and choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(methods):
                method_key = methods[choice_num - 1][0]
                self.apply_layer_to_method_simple(layer_name, method_key)
    
    def apply_layer_to_method_simple(self, layer_name, method):
        try:
            if method == 'all':
                # Apply to all methods
                for method_key in self.layer_vars.keys():
                    self.layer_vars[method_key].set(layer_name)
                message = f"Layer '{layer_name}' applied to all methods"
            else:
                if method in self.layer_vars:
                    self.layer_vars[method].set(layer_name)
                    method_display = method.replace('_', ' ').title()
                    message = f"Layer '{layer_name}' applied to {method_display}"
                else:
                    messagebox.showerror("Error", f"Unknown method: {method}")
                    return
            
            messagebox.showinfo("Layer Applied", message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply layer:\n{str(e)}")
    
    def set_single_method(self, method):
        try:
            layer_name = self.layer_vars[method].get().strip()
            if not layer_name:
                messagebox.showwarning("Warning", "Please select a layer from the dropdown")
                return
            
            self.xai_processor.set_layer_for_method(method, layer_name)
            
            method_display = method.replace('_', ' ').title()
            messagebox.showinfo("Success", f"{method_display} layer set to: {layer_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set layer:\n{str(e)}")
    
    def set_all_first(self):
        available_layers = self.xai_processor.get_available_layers()
        if available_layers:
            first_layer = available_layers[0]
            for var in self.layer_vars.values():
                var.set(first_layer)
    
    def set_all_last(self):
        available_layers = self.xai_processor.get_available_layers()
        if available_layers:
            last_layer = available_layers[-1]
            for var in self.layer_vars.values():
                var.set(last_layer)
    
    def set_all_middle(self):
        available_layers = self.xai_processor.get_available_layers()
        if available_layers:
            middle_idx = len(available_layers) // 2
            middle_layer = available_layers[middle_idx]
            for var in self.layer_vars.values():
                var.set(middle_layer)
    
    def reset_defaults(self):
        available_layers = self.xai_processor.get_available_layers()
        if available_layers:
            default_layer = available_layers[-1]  
            for var in self.layer_vars.values():
                var.set(default_layer)
    
    def apply_changes(self):
        try:
            applied_layers = {}
            for method, var in self.layer_vars.items():
                layer_name = var.get().strip()
                if layer_name:
                    self.xai_processor.set_layer_for_method(method, layer_name)
                    applied_layers[method] = layer_name
            
            self.result = {
                'applied': True,
                'reanalyze': False,
                'selections': applied_layers
            }
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply layer selections:\n{str(e)}")
    
    def apply_and_reanalyze(self):
        try:
            applied_layers = {}
            for method, var in self.layer_vars.items():
                layer_name = var.get().strip()
                if layer_name:
                    self.xai_processor.set_layer_for_method(method, layer_name)
                    applied_layers[method] = layer_name
            
            self.result = {
                'applied': True,
                'reanalyze': True,
                'selections': applied_layers
            }
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply layer selections:\n{str(e)}")
    
    def cancel(self):
        self.result = {'applied': False, 'reanalyze': False}
        self.dialog.destroy()

class MedicalXAIInterface:

    def __init__(self, root):
        self.root = root
        self.root.title("Neuroimaging Analysis Tool System - Early Alzheimer's Disease Detection")
        self.root.geometry("1400x1000")
        
        try:
            self.root.state('zoomed')
        except tk.TclError:
            self.root.attributes('-zoomed', True)
        except:
            pass

        self.model, self.xai_processor, self.current_image_data, self.prediction_data = None, None, None, None
        
        self.maximized_pane = None
        self.original_sash_positions = {}

        self.create_interface()
        
        self.root.after(200, self.load_default_model_with_status)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_default_model_with_status(self):
        """Enhanced model loading with layer configuration button activation"""
        try:
            self.update_status_with_progress("Loading AI model...", 10)
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras")
            
            if not os.path.exists(model_path):
                self.update_status_with_progress("Default model not found - please load manually", 0)
                return
            
            self.update_status_with_progress("Loading EfficientNetV2B0 model...", 30)
            self.root.update()
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            self.update_status_with_progress("Initializing XAI processor...", 60)
            self.root.update()
            
            # Initialize XAI processor
            self.xai_processor = XAIProcessor(self.model)
            
            # Enable layer configuration button
            self.layer_config_btn.config(state=tk.NORMAL)
            
            self.update_status_with_progress("EfficientNetV2B0 model loaded successfully", 100)
            
            self.root.after(1000, lambda: self.update_status_with_progress(
                "Ready - Upload MRI image to begin analysis", 0
            ))
            
            available_layers = self.xai_processor.get_available_layers()
            print(f"Successfully loaded model from: {model_path}")
            print(f"Available conv layers ({len(available_layers)}): {available_layers}")

        except Exception as e:
            self.update_status_with_progress("Model loading failed - please load manually", 0)
            print(f"Could not load default model: {e}")

    def clear_all_blurs(self):
        for window in self.windows.values():
            if hasattr(window, 'remove_blur'):
                window.remove_blur()

    def create_interface(self):
        """Enhanced interface creation with themed layer selection button"""
        self.toolbar_frame = ttk.Frame(self.root)
        self.toolbar_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(self.toolbar_frame, text="Upload Coronal MRI Slice", command=self.upload_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar_frame, text="Load Custom Model", command=self.load_model_dialog).pack(side=tk.LEFT, padx=5)
        
        # Layer configuration button with enhanced styling
        self.layer_config_btn = ttk.Button(
            self.toolbar_frame, 
            text="Configure XAI Layers", 
            command=self.show_layer_selection_dialog,
            state=tk.DISABLED  # Disabled until model is loaded
        )
        self.layer_config_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.toolbar_frame, text="Clear Analysis", command=self.clear_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(self.toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.save_report_btn = ttk.Button(self.toolbar_frame, text="Save Medical Report", command=self.save_comprehensive_report)
        self.save_report_btn.pack(side=tk.LEFT, padx=5)
        ttk.Separator(self.toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        status_frame = ttk.Frame(self.toolbar_frame)
        status_frame.pack(side=tk.RIGHT, padx=10)
        
        self.status_var = tk.StringVar(value="Initializing Analysis Tool System...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, width=30)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(
            status_frame, 
            mode='determinate', 
            length=200,
            style='TProgressbar'
        )
        self.progress.pack(side=tk.LEFT, padx=15)
        self.progress['value'] = 0

        # Continue with the rest of your interface creation...
        self.paned_window_container = ttk.Frame(self.root)
        self.paned_window_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.main_v_pane = ttk.PanedWindow(self.paned_window_container, orient=tk.VERTICAL)
        self.main_v_pane.pack(fill=tk.BOTH, expand=True)

        self.windows = {}
        self.h_panes = []
        
        window_configs = [
            ("Original MRI", 0), ("Prediction Analysis", 0), ("Confidence Map", 0), ("Class Probabilities", 0),
            ("Grad-CAM++ Raw", 1), ("Grad-CAM++ Masked", 1), ("Grad-CAM++ Overlay", 1), ("Grad-CAM++ Stats", 1),
            ("Guided Grad-CAM++ Raw", 2), ("Guided Grad-CAM++ Masked", 2), ("Guided Grad-CAM++ Overlay", 2), ("Guided Stats", 2),
            ("Consensus Raw", 3), ("Consensus Masked", 3), ("Consensus Overlay", 3), ("Medical Report", 3)
        ]

        for i in range(4):
            h_pane = ttk.PanedWindow(self.main_v_pane, orient=tk.HORIZONTAL)
            self.main_v_pane.add(h_pane, weight=1)
            self.h_panes.append(h_pane)

        for title, row in window_configs:
            window = ContentWindow(self.h_panes[row], title, app=self)
            self.h_panes[row].add(window, weight=1)
            self.windows[title] = window

    def show_layer_selection_dialog(self):
        """Show the themed layer selection dialog with reanalyze option"""
        if not self.xai_processor:
            messagebox.showwarning("Warning", "Please load an AI model first")
            return
        
        dialog = LayerSelectionDialog(self.root, self.xai_processor)
        result = dialog.show()
        
        if result and result.get('applied', False):
            selections = result.get('selections', {})
            
            # Show confirmation message
            message = "Layer configuration updated:\n"
            for method, layer in selections.items():
                method_name = method.replace('_', ' ').title()
                message += f"• {method_name}: {layer}\n"
            
            if result.get('reanalyze', False):
                # User clicked "Apply & Re-analyze"
                if hasattr(self, 'visualization_data') and hasattr(self, 'current_image_path'):
                    message += "\nRe-analyzing with new layer configuration..."
                    messagebox.showinfo("Configuration Updated", message)
                    self.analyze_image()
                else:
                    messagebox.showinfo("Configuration Updated", 
                        message + "\nPlease upload an MRI image to analyze with new settings.")
            else:
                # User clicked "Apply Only"
                messagebox.showinfo("Configuration Updated", message)

    def reset_view(self): 
        try:
            if self.maximized_pane:
                for window in self.windows.values():
                    if window.is_maximized:
                        window.toggle_maximize()
                        break
            
            for window in self.windows.values():
                if window.is_maximized:
                    window.is_maximized = False
                    window.maximize_btn.config(text="Max", bg='#C0C0C0')

                    if window.should_be_blurred() and not window.is_blurred:
                        if hasattr(window, 'figure') and window.figure:
                            window.apply_blur()
            
            self.reset_paned_layout()           
            self.update_status("Reset Layout & Windows")
            
        except Exception as e:
            print(f"Error during view reset: {e}")
            self.update_status("Error resetting view")

    def reset_paned_layout(self):
        try:
            self.root.update_idletasks()           

            container_height = self.paned_window_container.winfo_height()
            container_width = self.paned_window_container.winfo_width()
            
            num_v_panes = len(self.main_v_pane.panes())
            if num_v_panes > 1:
                row_height = container_height // num_v_panes
                for i in range(num_v_panes - 1):
                    position = (i + 1) * row_height
                    self.main_v_pane.sashpos(i, position)
            
            for h_pane in self.h_panes:
                num_h_panes = len(h_pane.panes())
                if num_h_panes > 1:
                    col_width = container_width // num_h_panes
                    for j in range(num_h_panes - 1):
                        position = (j + 1) * col_width
                        h_pane.sashpos(j, position)
                        
            self.original_sash_positions = {}
            self.maximized_pane = None
            
        except Exception as e:
            print(f"Error resetting paned layout: {e}")

    def toggle_maximize_pane(self, pane_to_toggle): 
        if self.maximized_pane == pane_to_toggle:

            self.restore_layout()
            self.maximized_pane = None
        else:

            if self.maximized_pane is not None:

                self.restore_layout()
            self.maximize_pane(pane_to_toggle)

    def maximize_pane(self, pane_to_toggle):
        self.maximized_pane = pane_to_toggle
        
        self.save_current_layout()
        
        target_row = None
        target_col = None
        
        for i, h_pane in enumerate(self.h_panes):
            panes_in_row = h_pane.panes()
            for j, pane_path in enumerate(panes_in_row):
                if str(pane_to_toggle) == pane_path:
                    target_row = i
                    target_col = j
                    break
            if target_row is not None:
                break
        
        if target_row is None or target_col is None:
            return        

        self.root.update_idletasks()  # Ensure geometry is updated
        container_height = self.paned_window_container.winfo_height()
        container_width = self.paned_window_container.winfo_width()
        
        for i in range(len(self.main_v_pane.panes()) - 1):
            if i < target_row:

                self.main_v_pane.sashpos(i, 1)
            elif i >= target_row:

                self.main_v_pane.sashpos(i, container_height - 1)
        
        target_h_pane = self.h_panes[target_row]
        for j in range(len(target_h_pane.panes()) - 1):
            if j < target_col:

                target_h_pane.sashpos(j, 1)
            elif j >= target_col:

                target_h_pane.sashpos(j, container_width - 1)

    def restore_layout(self):
        if not self.maximized_pane or not self.original_sash_positions:
            return
            
        main_positions = self.original_sash_positions.get('main', [])
        for i, pos in main_positions:
            try:
                if i < len(self.main_v_pane.panes()) - 1:
                    self.main_v_pane.sashpos(i, pos)
            except:
                pass
        
        for pane_idx, sash_positions in self.original_sash_positions.items():
            if isinstance(pane_idx, int) and pane_idx < len(self.h_panes):
                h_pane = self.h_panes[pane_idx]
                for j, pos in sash_positions:
                    try:
                        if j < len(h_pane.panes()) - 1:
                            h_pane.sashpos(j, pos)
                    except:
                        pass
        
        self.original_sash_positions = {}

    def save_current_layout(self):
        self.original_sash_positions = {}
        
        main_positions = []
        for i in range(len(self.main_v_pane.panes()) - 1):
            try:
                pos = self.main_v_pane.sashpos(i)
                main_positions.append((i, pos))
            except:
                pass
        self.original_sash_positions['main'] = main_positions
        
        for i, h_pane in enumerate(self.h_panes):
            pane_positions = []
            for j in range(len(h_pane.panes()) - 1):
                try:
                    pos = h_pane.sashpos(j)
                    pane_positions.append((j, pos))
                except:
                    pass
            self.original_sash_positions[i] = pane_positions
    
    def load_default_model(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            model_path = os.path.join(script_dir, "TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras")
            
            self.model = tf.keras.models.load_model(model_path)
            self.xai_processor = XAIProcessor(self.model)
            self.update_status("EfficientNetV2B0 Model loaded successfully")
            print(f"Successfully loaded model from: {model_path}")

        except Exception as e:
            self.update_status("Default model not found - please load manually")
            print(f"Could not load default model: {e}")

    def load_model_dialog(self):
        """Enhanced model loading dialog with layer configuration"""
        path = filedialog.askopenfilename(
            title="Select AI Model File",
            filetypes=[("Keras Models", "*.keras *.h5"), ("All Files", "*.*")])
        
        if path:
            try:
                self.update_status_with_progress("Loading custom model...", 20)
                self.root.update()
                
                self.model = tf.keras.models.load_model(path)
                
                self.update_status_with_progress("Initializing XAI processor...", 60)
                self.root.update()
                
                self.xai_processor = XAIProcessor(self.model)
                
                # Enable layer configuration button
                self.layer_config_btn.config(state=tk.NORMAL)
                
                self.update_status_with_progress("Custom model loaded successfully", 100)
                
                self.root.after(1500, lambda: self.update_status_with_progress(
                    "Upload MRI image to begin analysis", 0
                ))
                
                # Show model info without overwhelming detail
                available_layers = self.xai_processor.get_available_layers()
                layer_info = f"Model loaded successfully!\n{os.path.basename(path)}\n\nFound {len(available_layers)} convolutional layers"
                
                if available_layers:
                    # Show first few and last few layers
                    if len(available_layers) <= 5:
                        layer_info += f"\nLayers: {', '.join(available_layers)}"
                    else:
                        first_two = ', '.join(available_layers[:2])
                        last_two = ', '.join(available_layers[-2:])
                        layer_info += f"\nFirst: {first_two}\nLast: {last_two}"
                        layer_info += f"\n(+{len(available_layers)-4} more layers)"
                
                layer_info += "\n\nUse 'Configure XAI Layers' to customize analysis."
                
                messagebox.showinfo("Model Loaded", layer_info)
                
            except Exception as e:
                self.update_status_with_progress("Model loading failed", 0)
                messagebox.showerror("Error", f"Could not load model:\n{str(e)}")

    def update_status_with_progress(self, message, progress_value):
        self.status_var.set(message)
        self.progress['value'] = progress_value
        self.root.update_idletasks()


    def upload_image(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please load an AI model first")
            return
            
        path = filedialog.askopenfilename(
            title="Select Coronal MRI Slice",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            try:
                self.current_image_path = path
                filename = os.path.basename(path)
                self.update_status_with_progress(f"MRI image loaded: {filename}", 15)
                
                # Start analysis after brief delay to show status
                self.root.after(500, self.analyze_image)
                
            except Exception as e:
                self.update_status_with_progress("Image loading failed", 0)
                messagebox.showerror("Error", f"Could not load image:\n{str(e)}")

    def analyze_image(self):
        """Enhanced image analysis with better layer status reporting"""
        if not hasattr(self, 'current_image_path') or not self.model:
            messagebox.showwarning("Warning", "Please load both model and MRI image first")
            return
            
        try:
            self.update_status_with_progress("Starting AI analysis...", 5)
            
            self.update_status_with_progress("Preprocessing MRI image...", 10)
            original_img, img_array = self.get_img_array(self.current_image_path)
            
            self.update_status_with_progress("Running AI inference...", 20)
            predictions = self.model.predict(img_array, verbose=0)
            
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(1, -1)
            predictions = predictions[0]
            predicted_class_idx = np.argmax(predictions)
            predicted_label = CLASS_NAMES[min(predicted_class_idx, len(CLASS_NAMES)-1)]
            confidence = predictions[predicted_class_idx]
            
            self.prediction_data = {
                'predictions': predictions,
                'predicted_class_idx': predicted_class_idx,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'class_descriptions': CLASS_DESCRIPTIONS
            }
            
            # Get selected layers for each method
            gradcam_layer = self.xai_processor.get_layer_for_method('gradcam')
            guided_layer = self.xai_processor.get_layer_for_method('guided_gradcam')
            consensus_layer = self.xai_processor.get_layer_for_method('consensus')
            
            # Truncate layer names for status display
            def truncate_layer_name(name, max_len=12):
                return name if len(name) <= max_len else f"...{name[-(max_len-3):]}"
            
            gc_short = truncate_layer_name(gradcam_layer)
            guided_short = truncate_layer_name(guided_layer)
            
            self.update_status_with_progress(f"Grad-CAM++ ({gc_short})...", 35)
            gcpp_raw = self.xai_processor.make_gradcam_plus_plus(img_array, predicted_class_idx, gradcam_layer)
            
            self.update_status_with_progress(f"Guided Grad-CAM++ ({guided_short})...", 50)
            guided_gcpp_raw = self.xai_processor.make_guided_gradcam_plus_plus(img_array, predicted_class_idx, guided_layer)
            
            self.update_status_with_progress("Creating consensus maps...", 60)
            consensus_raw = self.xai_processor.create_consensus_map([gcpp_raw, guided_gcpp_raw])
            
            self.update_status_with_progress("Applying brain masks...", 70)
            gcpp_masked = self.xai_processor.apply_brain_mask(gcpp_raw, original_img)
            guided_gcpp_masked = self.xai_processor.apply_brain_mask(guided_gcpp_raw, original_img)
            consensus_masked = self.xai_processor.apply_brain_mask(consensus_raw, original_img)
            
            self.update_status_with_progress("Creating overlays...", 80)
            
            def normalize_heatmap(heatmap):
                h_max = np.max(heatmap)
                return heatmap / h_max if h_max > 0 else heatmap
                
            gcpp_overlay = self.xai_processor.create_overlay(normalize_heatmap(gcpp_masked), original_img, 0.6)
            guided_gcpp_overlay = self.xai_processor.create_overlay(normalize_heatmap(guided_gcpp_masked), original_img, 0.6)
            consensus_overlay = self.xai_processor.create_overlay(normalize_heatmap(consensus_masked), original_img, 0.6)
            
            self.visualization_data = {
                'original_img': original_img, 'img_array': img_array,
                'gcpp_raw': gcpp_raw, 'gcpp_masked': gcpp_masked, 'gcpp_overlay': gcpp_overlay,
                'guided_gcpp_raw': guided_gcpp_raw, 'guided_gcpp_masked': guided_gcpp_masked, 'guided_gcpp_overlay': guided_gcpp_overlay,
                'consensus_raw': consensus_raw, 'consensus_masked': consensus_masked, 'consensus_overlay': consensus_overlay,
                'selected_layers': {
                    'gradcam': gradcam_layer,
                    'guided_gradcam': guided_layer,
                    'consensus': consensus_layer
                }
            }
            
            self.update_status_with_progress("Populating windows...", 90)
            self.populate_all_windows()
            
            # Compact status message
            layer_summary = f"GC:{gc_short}, G:{guided_short}"
            self.update_status_with_progress(
                f"Complete: {predicted_label} ({confidence:.1%}) | {layer_summary}", 
                100
            )
            
            # Reset progress after delay
            self.root.after(4000, lambda: self.update_status_with_progress(
                "Ready - Configure layers or load new image", 0
            ))
            
        except Exception as e:
            self.update_status_with_progress("Analysis failed - Check layers or image", 0)
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}\n\nTry checking layer configuration.")
            import traceback
            traceback.print_exc()

    def get_img_array(self, img_path):
        # Load the image
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale')
        array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Convert to 2D for preprocessing
        if len(array.shape) == 3:
            slice_data = array[:, :, 0]  # Take first channel if grayscale
        else:
            slice_data = array
        
        # Apply preprocessing pipeline from the research
        processed_slice = MedicalImageProcessor.crop_and_resize_slice(slice_data, target_size=IMG_SIZE)
        
        # Normalize intensity to 0-255 range
        if np.max(processed_slice) > 0:
            processed_slice = cv2.normalize(processed_slice, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert back to RGB for model
        rgb_img = cv2.cvtColor(processed_slice.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Prepare for model inference
        array = tf.keras.preprocessing.image.img_to_array(rgb_img)
        original_img = array.copy()
        array_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(np.expand_dims(array, axis=0))
        
        return original_img, array_preprocessed

    def populate_all_windows(self):
        if not hasattr(self, 'visualization_data') or not hasattr(self, 'prediction_data'):
            return
        self.create_original_mri_window()
        self.create_prediction_analysis_window()
        self.create_confidence_map_window()
        self.create_class_probabilities_window()
        self.create_gradcam_windows()
        self.create_gradcam_stats_window()
        self.create_guided_gradcam_windows()
        self.create_guided_stats_window()
        self.create_consensus_windows()
        self.create_medical_report_window()

    def create_original_mri_window(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        original_img = self.visualization_data['original_img'].astype(np.uint8)
        ax.imshow(original_img, cmap='gray')
        ax.set_xlabel('Pixels', fontsize=10)
        ax.set_ylabel('Pixels', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        ax.text(10, 30, f"Patient ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.text(10, 50, f"Scan Date: {datetime.now().strftime('%Y-%m-%d')}", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.text(10, 70, f"Image Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        plt.tight_layout()
        self.windows["Original MRI"].set_content(fig)

    def create_prediction_analysis_window(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        prediction = self.prediction_data
        classes = CLASS_NAMES[:len(prediction['predictions'])]
        probabilities = prediction['predictions']
        colors = ['#e74c3c' if i == prediction['predicted_class_idx'] else '#95a5a6' for i in range(len(classes))]
        
        bars = ax1.barh(classes, probabilities, color=colors)
        ax1.set_xlabel('Confidence Score', fontsize=10)
        ax1.set_title('Diagnostic Confidence', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        for bar, prob in zip(bars, probabilities):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{prob:.3f} ({prob*100:.1f}%)', ha='left', va='center', fontsize=9)
        
        ax2.axis('off')
        description = CLASS_DESCRIPTIONS.get(prediction['predicted_label'], prediction['predicted_label'])
        summary_text = f"""
DIAGNOSTIC SUMMARY

Primary Diagnosis: {prediction['predicted_label']}
Medical Classification: {description}
Confidence Level: {prediction['confidence']:.1%}

Clinical Interpretation:
• {self.get_clinical_interpretation(prediction['predicted_label'], prediction['confidence'])}

Recommendation:
• {self.get_clinical_recommendation(prediction['predicted_label'], prediction['confidence'])}
"""
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.8))
        plt.tight_layout()
        self.windows["Prediction Analysis"].set_content(fig)

    def create_confidence_map_window(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        prediction = self.prediction_data
        confidence_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
        
        for i, prob in enumerate(prediction['predictions'][:len(CLASS_NAMES)]):
            confidence_matrix[i, i] = prob
            
        im = ax.imshow(confidence_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_title('Diagnostic Confidence Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontsize=10)
        ax.set_ylabel('True Class (Diagonal)', fontsize=10)
        
        for i in range(len(CLASS_NAMES)):
            if i < len(prediction['predictions']):
                ax.text(i, i, f'{prediction["predictions"][i]:.3f}', ha="center", va="center", color="white", fontweight='bold')
                
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Confidence Score', fontsize=10)
        plt.tight_layout()
        self.windows["Confidence Map"].set_content(fig)

    def create_class_probabilities_window(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        prediction = self.prediction_data
        classes = CLASS_NAMES[:len(prediction['predictions'])]
        probabilities = prediction['predictions']
        colors_pie = plt.cm.Set3(np.arange(len(classes)))
        
        ax1.pie(probabilities, labels=classes, autopct='%1.2f%%', colors=colors_pie, startangle=90)
        ax1.set_title('Class Distribution', fontsize=12, fontweight='bold')
        
        bars = ax2.bar(classes, probabilities, color=colors_pie, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Probability', fontsize=10)
        ax2.set_title('Probability Scores', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{prob:.4f}', ha='center', va='bottom', fontsize=9)
            
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(probabilities))
        uncertainty = entropy / max_entropy
        
        ax3.bar(['Certainty', 'Uncertainty'], [1-uncertainty, uncertainty], color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax3.set_ylabel('Score', fontsize=10)
        ax3.set_title('Model Certainty Analysis', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        ax4.axis('off')
        stats_text = f"""
STATISTICAL ANALYSIS

Max Prob: {np.max(probabilities):.4f}
Min Prob: {np.min(probabilities):.4f}
Mean Prob: {np.mean(probabilities):.4f}
Std Dev: {np.std(probabilities):.4f}

Entropy: {entropy:.4f}
Norm. Entropy: {uncertainty:.4f}
Confidence: {(1-uncertainty)*100:.1f}%

Threshold: ≥ 0.7
Score: {prediction['confidence']:.3f}
Status: {'PASS' if prediction['confidence'] >= 0.7 else 'REVIEW'}
"""
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.8))
        plt.tight_layout()
        self.windows["Class Probabilities"].set_content(fig)

    def create_gradcam_windows(self):
        reduced_size = (4, 4)
        
        fig1, ax1 = plt.subplots(1, 1, figsize=reduced_size)
        im1 = ax1.imshow(self.visualization_data['gcpp_raw'], cmap='jet', vmin=0, vmax=1)
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Attention Intensity', fontsize=10)
        plt.tight_layout()
        self.windows["Grad-CAM++ Raw"].set_content(fig1)
        
        fig2, ax2 = plt.subplots(1, 1, figsize=reduced_size)
        im2 = ax2.imshow(self.visualization_data['gcpp_masked'], cmap='jet', vmin=0, vmax=1)
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Attention Intensity', fontsize=10)
        plt.tight_layout()
        self.windows["Grad-CAM++ Masked"].set_content(fig2)
        
        fig3, ax3 = plt.subplots(1, 1, figsize=reduced_size)
        ax3.imshow(self.visualization_data['gcpp_overlay'])
        plt.tight_layout()
        self.windows["Grad-CAM++ Overlay"].set_content(fig3)

    def create_gradcam_stats_window(self):
        """Enhanced Grad-CAM stats window with layer information"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        gcpp_data = self.visualization_data['gcpp_masked']
        
        # Get the layer used for this analysis
        selected_layer = self.visualization_data.get('selected_layers', {}).get('gradcam', 'Unknown')
        
        ax1.hist(gcpp_data.flatten(), bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.set_xlabel('Activation Value', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title(f'Activation Distribution\n(Layer: {selected_layer})', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        center_line = gcpp_data[gcpp_data.shape[0]//2, :]
        ax2.plot(center_line, color='#e74c3c', linewidth=2)
        ax2.set_xlabel('Pixel Position', fontsize=10)
        ax2.set_ylabel('Activation', fontsize=10)
        ax2.set_title('Center Line Profile', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        h, w = gcpp_data.shape
        regions = {
            'Top-Left': gcpp_data[:h//2, :w//2], 'Top-Right': gcpp_data[:h//2, w//2:],
            'Bottom-Left': gcpp_data[h//2:, :w//2], 'Bottom-Right': gcpp_data[h//2:, w//2:]
        }
        region_means = [np.mean(region) for region in regions.values()]
        ax3.bar(regions.keys(), region_means, color=['#9b59b6', '#f39c12', '#2ecc71', '#e74c3c'], alpha=0.7)
        ax3.set_ylabel('Mean Activation', fontsize=10)
        ax3.set_title('Regional Activation', fontsize=11, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        ax4.axis('off')
        stats_text = f"""
    GRAD-CAM++ STATISTICS

    Layer: {selected_layer}

    Max: {np.max(gcpp_data):.6f}
    Min: {np.min(gcpp_data):.6f}
    Mean: {np.mean(gcpp_data):.6f}
    Std Dev: {np.std(gcpp_data):.6f}
    Median: {np.median(gcpp_data):.6f}

    Peak Loc: {np.unravel_index(np.argmax(gcpp_data), gcpp_data.shape)}
    Active Pixels (>0.5): {np.sum(gcpp_data > 0.5)}
    Coverage: {(np.sum(gcpp_data > 0.1)/gcpp_data.size)*100:.1f}%
    """
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.8))
        plt.tight_layout()
        self.windows["Grad-CAM++ Stats"].set_content(fig)

    def create_guided_gradcam_windows(self):
        reduced_size = (4, 4)
        
        fig1, ax1 = plt.subplots(1, 1, figsize=reduced_size)
        im1 = ax1.imshow(self.visualization_data['guided_gcpp_raw'], cmap='jet', vmin=0, vmax=1)
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Attention Intensity', fontsize=10)
        plt.tight_layout()
        self.windows["Guided Grad-CAM++ Raw"].set_content(fig1)
        
        fig2, ax2 = plt.subplots(1, 1, figsize=reduced_size)
        im2 = ax2.imshow(self.visualization_data['guided_gcpp_masked'], cmap='jet', vmin=0, vmax=1)
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Attention Intensity', fontsize=10)
        plt.tight_layout()
        self.windows["Guided Grad-CAM++ Masked"].set_content(fig2)
        
        fig3, ax3 = plt.subplots(1, 1, figsize=reduced_size)
        ax3.imshow(self.visualization_data['guided_gcpp_overlay'])
        plt.tight_layout()
        self.windows["Guided Grad-CAM++ Overlay"].set_content(fig3)

    def create_guided_stats_window(self):
        """Enhanced Guided Grad-CAM stats window with layer information"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        guided_data = self.visualization_data['guided_gcpp_masked']
        
        # Get the layer used for this analysis
        selected_layer = self.visualization_data.get('selected_layers', {}).get('guided_gradcam', 'Unknown')
        
        x, y = np.arange(guided_data.shape[1]), np.arange(guided_data.shape[0])
        X, Y = np.meshgrid(x, y)
        step = max(1, guided_data.shape[0] // 20)
        X_sub, Y_sub, Z_sub = X[::step, ::step], Y[::step, ::step], guided_data[::step, ::step]
        contour = ax1.contour(X_sub, Y_sub, Z_sub, levels=10, cmap='viridis')
        ax1.clabel(contour, inline=True, fontsize=8)
        ax1.set_title(f'Activation Contours\n(Layer: {selected_layer})', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        gy, gx = np.gradient(guided_data)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        im2 = ax2.imshow(gradient_mag, cmap='plasma')
        ax2.set_title('Gradient Magnitude', fontsize=11, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Gradient', fontsize=10)
        
        gcpp_data = self.visualization_data['gcpp_masked']
        guided_flat, gcpp_flat = guided_data.flatten(), gcpp_data.flatten()
        ax3.scatter(gcpp_flat, guided_flat, alpha=0.5, s=1, color='#e74c3c')
        ax3.set_xlabel('Grad-CAM++ Activation', fontsize=10)
        ax3.set_ylabel('Guided Grad-CAM++ Activation', fontsize=10)
        ax3.set_title('Method Correlation', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        correlation = np.corrcoef(gcpp_flat, guided_flat)[0, 1]
        ax3.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax3.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax4.axis('off')
        stats_text = f"""
    GUIDED GRAD-CAM++ ANALYSIS

    Layer: {selected_layer}

    Max: {np.max(guided_data):.6f}
    Mean: {np.mean(guided_data):.6f}
    Std Dev: {np.std(guided_data):.6f}
    Sparsity: {(np.sum(guided_data < 0.01)/guided_data.size)*100:.1f}%

    Max Grad: {np.max(gradient_mag):.6f}
    Mean Grad: {np.mean(gradient_mag):.6f}

    Correlation: {correlation:.3f}
    """
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.8))
        plt.tight_layout()
        self.windows["Guided Stats"].set_content(fig)

    def create_consensus_windows(self):

        reduced_size = (4, 4)
        
        fig1, ax1 = plt.subplots(1, 1, figsize=reduced_size)
        im1 = ax1.imshow(self.visualization_data['consensus_raw'], cmap='jet', vmin=0, vmax=1)
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Consensus Strength', fontsize=10)
        plt.tight_layout()
        self.windows["Consensus Raw"].set_content(fig1)
        
        fig2, ax2 = plt.subplots(1, 1, figsize=reduced_size)
        im2 = ax2.imshow(self.visualization_data['consensus_masked'], cmap='jet', vmin=0, vmax=1)
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Consensus Strength', fontsize=10)
        plt.tight_layout()
        self.windows["Consensus Masked"].set_content(fig2)
        
        fig3, ax3 = plt.subplots(1, 1, figsize=reduced_size)
        ax3.imshow(self.visualization_data['consensus_overlay'])
        plt.tight_layout()
        self.windows["Consensus Overlay"].set_content(fig3)

    def create_medical_report_window(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        ax.axis('off')
        prediction = self.prediction_data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_text = f"""
MEDICAL AI ANALYSIS REPORT

=================================
PATIENT INFORMATION
=================================
Scan ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}
Analysis Date: {timestamp}

=================================
DIAGNOSTIC RESULTS
=================================
PRIMARY DIAGNOSIS: {prediction['predicted_label']}
CLASSIFICATION: {CLASS_DESCRIPTIONS.get(prediction['predicted_label'], 'Unknown')}
CONFIDENCE SCORE: {prediction['confidence']:.1%}

Class Probabilities:
"""
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES[:len(prediction['predictions'])], prediction['predictions'])):
            indicator = "★" if i == prediction['predicted_class_idx'] else "  "
            report_text += f"{indicator} {class_name}: {prob:.3f} ({prob*100:.1f}%)\n"
            
        report_text += f"""
=================================
CLINICAL INTERPRETATION
=================================
{self.get_clinical_interpretation(prediction['predicted_label'], prediction['confidence'])}

Key Findings:
• Model focuses on {self.get_attention_regions()}
• Activation patterns suggest {self.get_pattern_analysis()}

=================================
RECOMMENDATIONS
=================================
{self.get_clinical_recommendation(prediction['predicted_label'], prediction['confidence'])}
"""
        ax.text(0.02, 0.98, report_text, transform=ax.transAxes, fontsize=8, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        plt.tight_layout()
        self.windows["Medical Report"].set_content(fig)

    def get_clinical_interpretation(self, predicted_label, confidence):
        if predicted_label == 'CN':
            if confidence > 0.8: 
                return "High confidence normal cognitive function."
            elif confidence > 0.6: 
                return "Moderate confidence normal cognitive function."
            else: 
                return "Low confidence normal classification."
        elif predicted_label == 'EMCI':
            if confidence > 0.8: 
                return "High confidence early mild cognitive impairment."
            elif confidence > 0.6: 
                return "Moderate confidence EMCI classification."
            else: 
                return "Low confidence EMCI classification."
        elif predicted_label == 'LMCI':
            if confidence > 0.8: 
                return "High confidence late mild cognitive impairment."
            elif confidence > 0.6: 
                return "Moderate confidence LMCI classification."
            else: 
                return "Low confidence LMCI classification."
        else: 
            return "Classification requires clinical correlation."

    def get_clinical_recommendation(self, predicted_label, confidence):
        base_recs = {
            'CN': ["Continue routine screening.", "Maintain healthy lifestyle."],
            'EMCI': ["Comprehensive neuropsychological evaluation.", "Monitor progression every 6 months."],
            'LMCI': ["Immediate comprehensive clinical evaluation.", "Implement safety measures and support systems."]
        }
        recs = base_recs.get(predicted_label, ["Consult with neurologist."])
        if confidence < 0.6:
            recs.insert(0, "Low confidence score - recommend additional diagnostic imaging.")
        return "\n• ".join(recs)

    def get_attention_regions(self):
        gcpp_data = self.visualization_data['gcpp_masked']
        h, w = gcpp_data.shape
        peak_y, peak_x = np.unravel_index(np.argmax(gcpp_data), gcpp_data.shape)
        y_region = "superior" if peak_y < h//3 else "middle" if peak_y < 2*h//3 else "inferior"
        x_region = "left" if peak_x < w//3 else "central" if peak_x < 2*w//3 else "right"
        return f"{y_region} {x_region} brain regions"

    def get_pattern_analysis(self):
        gcpp_sparsity = (np.sum(self.visualization_data['gcpp_masked'] < 0.1) / self.visualization_data['gcpp_masked'].size) * 100
        if gcpp_sparsity < 70: 
            return "diffuse activation patterns"
        elif gcpp_sparsity < 85: 
            return "moderate focal activation"
        else: 
            return "highly focal activation"

    def save_comprehensive_report(self):
        if not hasattr(self, 'visualization_data') or not hasattr(self, 'prediction_data'):
            messagebox.showinfo("Save Report", "No analysis data available to save.")
            return
            
        save_dir = filedialog.askdirectory(title="Select directory to save medical report")
        if not save_dir:
            return
            
        try:
            self.update_status_with_progress("Creating report directory...", 10)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(save_dir, f"medical_analysis_report_{timestamp}")
            os.makedirs(os.path.join(report_dir, "visualizations"), exist_ok=True)
            
            self.update_status_with_progress("Saving medical visualizations...", 30)
            self.root.update()
            
            total_windows = len(self.windows)
            for i, (window_name, window) in enumerate(self.windows.items()):
                if window.figure:
                    filename = f"{window_name.replace(' ', '_').replace('+', 'Plus')}.png"
                    filepath = os.path.join(report_dir, "visualizations", filename)
                    
                    figure_to_save = window.original_figure if window.is_blurred and window.original_figure else window.figure
                    figure_to_save.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                    
                    progress = 30 + (i / total_windows) * 60
                    self.update_status_with_progress(f"Saving {window_name}...", progress)
                    self.root.update()
            
            self.update_status_with_progress("Generating summary report...", 95)
            
            summary_path = os.path.join(report_dir, "analysis_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Medical AI Analysis Report\n")
                f.write(f"Generated: {timestamp}\n")
                f.write(f"Diagnosis: {self.prediction_data['predicted_label']}\n")
                f.write(f"Confidence: {self.prediction_data['confidence']:.1%}\n")
                f.write(f"Model: EfficientNetV2B0\n")
            
            self.update_status_with_progress("Medical report saved successfully", 100)
            
            messagebox.showinfo("Success", 
                f"Comprehensive medical report saved to:\n{os.path.basename(report_dir)}\n\n"
                f"Location: {report_dir}")
            
            self.root.after(3000, lambda: self.update_status_with_progress(
                "Ready for new analysis", 0
            ))
            
        except Exception as e:
            self.update_status_with_progress("Report save failed", 0)
            messagebox.showerror("Error", f"Failed to save medical report:\n{str(e)}")

    def clear_analysis(self):
        """Enhanced clear analysis with layer config state preservation"""
        try:
            self.update_status_with_progress("Clearing analysis...", 50)
            
            if self.maximized_pane:
                self.maximized_pane.toggle_maximize()
                
            for window in self.windows.values():
                for widget in window.content_frame.winfo_children():
                    widget.destroy()
                if window.figure:
                    plt.close(window.figure)
                    window.figure = None
                window.original_figure = None
                window.is_blurred = True
                    
            if hasattr(self, 'visualization_data'): 
                del self.visualization_data
            if hasattr(self, 'prediction_data'): 
                del self.prediction_data
            if hasattr(self, 'current_image_path'): 
                del self.current_image_path
                
            self.update_status_with_progress("Analysis cleared", 100)
            
            # Show layer config status if available
            status_msg = "Upload MRI to begin analysis"
            if self.xai_processor:
                status_msg += " (layers configured)"
            
            self.root.after(1500, lambda: self.update_status_with_progress(status_msg, 0))
            
        except Exception as e:
            self.update_status_with_progress("Error clearing analysis", 0)
            print(f"Error during clear: {e}")

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def on_close(self):
        try:
            plt.close('all')
            self.root.destroy()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.root.destroy()

def main():
    def start_main_app():
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white', 
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white', 
            'font.size': 10, 
            'axes.titlesize': 12,
            'axes.labelsize': 10, 
            'xtick.labelsize': 9, 
            'ytick.labelsize': 9
        })
        
        root = tk.Tk()
        root.withdraw() 
        
        app = MedicalXAIInterface(root)
        
        root.after(100, lambda: root.deiconify())
        root.mainloop()
    
    try:
        import sys
        import os
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from loading_window import LoadingWindow
        
        loading = LoadingWindow("Analysis Tool System", "Initializing Analysis Tool System...")
        loading.show_and_animate(2.5, start_main_app)
        
    except ImportError as e:
        print(f"Loading window not available ({e}), starting directly...")
        start_main_app()

if __name__ == '__main__':
    main()