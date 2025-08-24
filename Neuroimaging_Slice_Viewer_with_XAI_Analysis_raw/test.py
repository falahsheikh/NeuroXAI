#!/usr/bin/env python3
"""
Hardware Performance Measurement for NeuroXAI Tool
Run this script to measure actual memory usage, inference time, and system requirements
"""

import os
import sys
import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tracemalloc
import gc
from pathlib import Path
import json
from datetime import datetime

class PerformanceProfiler:
    def __init__(self):
        self.results = {
            'system_info': self.get_system_info(),
            'measurements': []
        }
        
    def get_system_info(self):
        """Get system specifications"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'platform': sys.platform,
            'python_version': sys.version,
            'tensorflow_version': tf.__version__
        }
    
    def measure_model_loading(self, model_path):
        """Measure model loading time and memory"""
        print(f"Measuring model loading performance for: {model_path}")
        
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        try:
            model = load_model(model_path)
            load_time = time.time() - start_time
            
            # Memory after loading
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            model_size_mb = self.get_model_size_mb(model_path)  # Use file path instead of model object
            
            result = {
                'operation': 'model_loading',
                'model_path': model_path,
                'load_time_seconds': round(load_time, 4),
                'memory_usage_mb': round(current_memory - initial_memory, 2),
                'peak_memory_mb': round(peak / 1024 / 1024, 2),
                'model_size_mb': model_size_mb,
                'model_parameters': model.count_params()
            }
            
            print(f"✓ Model loaded in {load_time:.4f}s")
            print(f"✓ Memory usage: {result['memory_usage_mb']:.2f} MB")
            print(f"✓ Model size: {model_size_mb:.2f} MB")
            
            return model, result
            
        except Exception as e:
            print(f"Error loading model: {e}")
            tracemalloc.stop()
            return None, {'error': str(e)}
    
    def get_model_size_mb(self, model_path):
        """Calculate model size in MB from file path"""
        try:
            size_mb = os.path.getsize(model_path) / 1024 / 1024
            return round(size_mb, 2)
        except:
            return 0
    
    def measure_inference(self, model, sample_data, num_runs=10):
        """Measure inference performance"""
        print(f"Measuring inference performance ({num_runs} runs)...")
        
        process = psutil.Process()
        
        # Warm-up runs
        for _ in range(3):
            _ = model.predict(sample_data, verbose=0)
        
        times = []
        memory_usage = []
        
        for i in range(num_runs):
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            predictions = model.predict(sample_data, verbose=0)
            end_time = time.perf_counter()
            
            inference_time = end_time - start_time
            current_memory = process.memory_info().rss / 1024 / 1024
            
            times.append(inference_time)
            memory_usage.append(current_memory - initial_memory)
        
        result = {
            'operation': 'inference',
            'num_runs': num_runs,
            'mean_inference_time_ms': round(np.mean(times) * 1000, 2),
            'std_inference_time_ms': round(np.std(times) * 1000, 2),
            'min_inference_time_ms': round(np.min(times) * 1000, 2),
            'max_inference_time_ms': round(np.max(times) * 1000, 2),
            'mean_memory_delta_mb': round(np.mean(memory_usage), 2),
            'predictions_shape': predictions.shape,
            'sample_prediction': predictions[0].tolist() if len(predictions) > 0 else None
        }
        
        print(f"✓ Mean inference time: {result['mean_inference_time_ms']:.2f} ms")
        print(f"✓ Memory delta: {result['mean_memory_delta_mb']:.2f} MB")
        
        return result
    
    def measure_xai_performance(self, model, sample_data, target_layer_name=None):
        """Measure XAI visualization generation performance"""
        print("Measuring XAI visualization performance...")
        
        try:
            # Import XAI components
            from tensorflow.keras.models import Model
            import cv2
            
            # Debug: Print model structure
            print("Model architecture inspection:")
            print(f"Model inputs: {model.input}")
            print(f"Model outputs: {model.output}")
            print(f"Model output shape: {model.output_shape}")
            
            # Find last conv layer if not specified
            if target_layer_name is None:
                conv_layers = []
                for layer in model.layers:
                    if 'conv' in layer.__class__.__name__.lower():
                        conv_layers.append(layer.name)
                        print(f"Found conv layer: {layer.name} - {layer.__class__.__name__}")
                
                if conv_layers:
                    target_layer_name = conv_layers[-1]  # Use last conv layer
                else:
                    return {'error': 'No convolutional layer found for XAI'}
            
            print(f"Using layer: {target_layer_name}")
            
            # Test the target layer exists - simplified approach
            try:
                target_layer = model.get_layer(target_layer_name)
                print(f"Target layer found: {target_layer.name} ({target_layer.__class__.__name__})")
            except Exception as e:
                print(f"Error accessing target layer: {e}")
                return {'error': f'Target layer {target_layer_name} not accessible: {str(e)}'}
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            
            # Simplified approach - create gradient model and test immediately
            try:
                print("Creating and testing gradient model...")
                
                # Create the gradient model
                grad_model = Model(
                    inputs=model.input,
                    outputs=[target_layer.output, model.output]
                )
                print("Gradient model created successfully")
                
                # Test a forward pass first
                test_outputs = grad_model(sample_data)
                print(f"Forward pass successful. Output types: {[type(out) for out in test_outputs]}")
                
                # Now do the actual gradient computation
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(sample_data)
                    
                    # Handle predictions properly
                    # The second output is a list, convert to tensor
                    if isinstance(predictions, list):
                        predictions = tf.convert_to_tensor(predictions[0])  # Take first element if it's a list
                    else:
                        predictions = tf.convert_to_tensor(predictions)
                    
                    if len(predictions.shape) == 1:
                        predictions = tf.expand_dims(predictions, 0)
                    
                    # Get the predicted class - convert to scalar
                    pred_idx = tf.argmax(predictions[0])
                    pred_idx = tf.cast(pred_idx, tf.int32)  # Ensure it's int32
                    
                    # Use tf.gather instead of indexing
                    class_output = tf.gather(predictions[0], pred_idx)
                
                # Compute gradients
                grads = tape.gradient(class_output, conv_outputs)
                
                if grads is None:
                    return {'error': 'Could not compute gradients - gradient is None'}
                
                print(f"Gradients computed successfully")
                print(f"Conv outputs shape: {conv_outputs.shape}")
                print(f"Gradients shape: {grads.shape}")
                
                # Simple Grad-CAM computation
                # Remove batch dimension
                conv_outputs = conv_outputs[0]  # [H, W, C]
                grads = grads[0]  # [H, W, C]
                
                # Compute channel weights (average of gradients for each channel)
                weights = tf.reduce_mean(grads, axis=(0, 1))  # [C]
                
                # Compute weighted combination of feature maps
                heatmap = tf.reduce_sum(weights * conv_outputs, axis=2)  # [H, W]
                
                # Apply ReLU to focus on positive contributions
                heatmap = tf.nn.relu(heatmap)
                
                # Normalize to [0, 1]
                heatmap_max = tf.reduce_max(heatmap)
                if heatmap_max > 0:
                    heatmap = heatmap / heatmap_max
                
                # Resize to match input size (224, 224)
                heatmap = tf.image.resize(heatmap[..., tf.newaxis], (224, 224))
                heatmap = tf.squeeze(heatmap)
                
                end_time = time.perf_counter()
                current_memory = process.memory_info().rss / 1024 / 1024
                
                # Prepare results
                result = {
                    'operation': 'xai_visualization',
                    'target_layer': target_layer_name,
                    'generation_time_ms': round((end_time - start_time) * 1000, 2),
                    'memory_usage_mb': round(current_memory - initial_memory, 2),
                    'heatmap_shape': heatmap.shape.as_list(),
                    'heatmap_stats': {
                        'min': float(tf.reduce_min(heatmap)),
                        'max': float(tf.reduce_max(heatmap)),
                        'mean': float(tf.reduce_mean(heatmap))
                    }
                }
                
                print(f"✓ XAI generation time: {result['generation_time_ms']:.2f} ms")
                print(f"✓ Memory usage: {result['memory_usage_mb']:.2f} MB")
                print(f"✓ Heatmap shape: {result['heatmap_shape']}")
                print(f"✓ Heatmap stats: min={result['heatmap_stats']['min']:.4f}, max={result['heatmap_stats']['max']:.4f}, mean={result['heatmap_stats']['mean']:.4f}")
                
                return result
                
            except Exception as e:
                print(f"Error in XAI computation: {e}")
                import traceback
                traceback.print_exc()
                return {'error': f'XAI computation failed: {str(e)}'}
                
        except Exception as e:
            print(f"Error in XAI measurement: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def create_sample_data(self, batch_size=1):
        """Create sample data for testing"""
        # Create synthetic MRI-like data
        sample_data = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
        sample_data = tf.keras.applications.efficientnet_v2.preprocess_input(sample_data)
        return sample_data
    
    def run_full_performance_test(self, model_path, output_file="performance_results.json"):
        """Run complete performance testing suite"""
        print("=" * 60)
        print("NeuroXAI Hardware Performance Testing")
        print("=" * 60)
        
        # Test model loading
        model, load_result = self.measure_model_loading(model_path)
        if model is None:
            print("Failed to load model. Exiting.")
            return
        
        self.results['measurements'].append(load_result)
        
        # Create sample data
        sample_data = self.create_sample_data()
        
        # Test inference
        inference_result = self.measure_inference(model, sample_data)
        self.results['measurements'].append(inference_result)
        
        # Test XAI
        xai_result = self.measure_xai_performance(model, sample_data)
        self.results['measurements'].append(xai_result)
        
        # Calculate total workflow time
        total_time_ms = (
            load_result.get('load_time_seconds', 0) * 1000 +
            inference_result.get('mean_inference_time_ms', 0) +
            xai_result.get('generation_time_ms', 0)
        )
        
        workflow_result = {
            'operation': 'complete_workflow',
            'total_time_ms': round(total_time_ms, 2),
            'inference_plus_xai_ms': round(
                inference_result.get('mean_inference_time_ms', 0) + 
                xai_result.get('generation_time_ms', 0), 2
            )
        }
        
        self.results['measurements'].append(workflow_result)
        self.results['timestamp'] = datetime.now().isoformat()
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"System: {self.results['system_info']['cpu_count']} CPU cores, "
              f"{self.results['system_info']['memory_total_gb']} GB RAM")
        print(f"Model Parameters: {load_result.get('model_parameters', 'N/A'):,}")
        print(f"Model Size: {load_result.get('model_size_mb', 'N/A')} MB")
        print(f"Inference Time: {inference_result.get('mean_inference_time_ms', 'N/A')} ms")
        print(f"XAI Generation: {xai_result.get('generation_time_ms', 'N/A')} ms")
        print(f"Total Workflow: {workflow_result['total_time_ms']} ms")
        print(f"Results saved to: {output_file}")
        
        return self.results

def main():
    """Main function to run performance tests"""
    profiler = PerformanceProfiler()
    
    # Get the correct path to the model based on the project structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw
    
    # Try multiple possible model paths
    possible_model_paths = [
        # Path from the current script location
        os.path.join(script_dir, "exnModel", "TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras"),
        # Path from project root
        os.path.join(project_root, "training", "models", "TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras"),
        # Relative path from script location
        os.path.join(script_dir, "..", "training", "models", "TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras"),
        # Direct relative path
        "training/models/TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras"
    ]
    
    model_path = None
    for path in possible_model_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            model_path = abs_path
            print(f"Found model at: {model_path}")
            break
    
    if not model_path:
        print("Model not found at any of the expected locations:")
        for path in possible_model_paths:
            abs_path = os.path.abspath(path)
            print(f"  Checked: {abs_path} - {'EXISTS' if os.path.exists(abs_path) else 'NOT FOUND'}")
        
        print("\nLooking for .keras files in current directory and training/models:")
        
        # Search in current directory
        current_dir = Path(".")
        keras_files = list(current_dir.glob("*.keras")) + list(current_dir.glob("*.h5"))
        
        # Search in training/models if it exists
        training_models_dir = Path("training/models")
        if training_models_dir.exists():
            keras_files.extend(list(training_models_dir.glob("*.keras")))
            keras_files.extend(list(training_models_dir.glob("*.h5")))
        
        # Search in exnModel if it exists
        exn_model_dir = Path("exnModel")
        if exn_model_dir.exists():
            keras_files.extend(list(exn_model_dir.glob("*.keras")))
            keras_files.extend(list(exn_model_dir.glob("*.h5")))
        
        if keras_files:
            print("Found model files:")
            for i, file in enumerate(keras_files):
                print(f"  {i+1}. {file}")
            
            try:
                choice = input(f"\nEnter number (1-{len(keras_files)}) to test, or press Enter to exit: ")
                if choice.strip():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(keras_files):
                        model_path = str(keras_files[choice_idx])
                    else:
                        print("Invalid choice.")
                        return
                else:
                    return
            except ValueError:
                print("Invalid input.")
                return
        else:
            print("No .keras or .h5 files found.")
            return
    
    # Run the tests
    results = profiler.run_full_performance_test(model_path)
    return results

if __name__ == "__main__":
    main()