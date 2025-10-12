#!/usr/bin/env python3
"""
NeuroXAI Program Requirements Test
Tests program dependencies, resource usage, and compatibility
Independent of specific hardware performance
"""

import os
import sys
import psutil
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime
import gc
import tracemalloc

class ProgramRequirementsTest:
    def __init__(self):
        self.results = {
            'program_info': {
                'name': 'NeuroXAI',
                'version': '1.0',
                'description': 'Lightweight Explainable Deep Learning for Early Alzheimer\'s Detection'
            },
            'requirements': {},
            'compatibility': {},
            'resource_usage': {}
        }
    
    def test_python_compatibility(self):
        """Test Python version requirements"""
        version = sys.version_info
        
        # Define minimum requirements
        min_major, min_minor = 3, 8
        compatible = version.major >= min_major and version.minor >= min_minor
        
        self.results['requirements']['python'] = {
            'minimum_version': f'{min_major}.{min_minor}',
            'current_version': f'{version.major}.{version.minor}.{version.micro}',
            'compatible': compatible,
            'required': True
        }
        
        return compatible
    
    def test_core_dependencies(self):
        """Test required Python packages"""
        core_packages = {
            'tensorflow': {
                'minimum_version': '2.8.0',
                'purpose': 'Deep learning model execution',
                'required': True
            },
            'numpy': {
                'minimum_version': '1.19.0',
                'purpose': 'Numerical computations',
                'required': True
            },
            'opencv-python': {
                'minimum_version': '4.5.0',
                'purpose': 'Image processing',
                'required': True
            },
            'pillow': {
                'minimum_version': '8.0.0',
                'purpose': 'Image format support',
                'required': True
            },
            'matplotlib': {
                'minimum_version': '3.3.0',
                'purpose': 'Visualization and plotting',
                'required': True
            },
            'simpleitk': {
                'minimum_version': '2.0.0',
                'purpose': 'Medical image processing',
                'required': True
            },
            'nibabel': {
                'minimum_version': '3.2.0',
                'purpose': 'NIfTI medical image format support',
                'required': True
            },
            'scipy': {
                'minimum_version': '1.7.0',
                'purpose': 'Scientific computing',
                'required': False
            },
            'scikit-learn': {
                'minimum_version': '0.24.0',
                'purpose': 'Machine learning utilities',
                'required': False
            }
        }
        
        dependencies_status = {}
        all_required_available = True
        
        for package_name, info in core_packages.items():
            try:
                if package_name == 'opencv-python':
                    import cv2
                    version = cv2.__version__
                elif package_name == 'pillow':
                    import PIL
                    version = PIL.__version__
                else:
                    pkg = __import__(package_name)
                    version = getattr(pkg, '__version__', 'unknown')
                
                dependencies_status[package_name] = {
                    'available': True,
                    'version': version,
                    'minimum_required': info['minimum_version'],
                    'purpose': info['purpose'],
                    'required': info['required']
                }
                
            except ImportError:
                dependencies_status[package_name] = {
                    'available': False,
                    'version': None,
                    'minimum_required': info['minimum_version'],
                    'purpose': info['purpose'],
                    'required': info['required']
                }
                
                if info['required']:
                    all_required_available = False
        
        self.results['requirements']['dependencies'] = dependencies_status
        return all_required_available
    
    def test_gui_support(self):
        """Test GUI framework availability"""
        gui_status = {}
        
        try:
            import tkinter
            gui_status['tkinter'] = {
                'available': True,
                'purpose': 'Main GUI framework',
                'note': 'Usually included with Python installation'
            }
        except ImportError:
            gui_status['tkinter'] = {
                'available': False,
                'purpose': 'Main GUI framework',
                'note': 'May need separate installation on some Linux distributions'
            }
        
        self.results['requirements']['gui'] = gui_status
        return gui_status['tkinter']['available']
    
    def test_model_compatibility(self, model_path=None):
        """Test model loading and architecture compatibility"""
        if not model_path:
            # Look for model files
            possible_paths = [
                "exnModel/TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras",
                "training/models/TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras"
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        model_info = {
            'architecture': 'EfficientNetV2B0',
            'input_shape': [224, 224, 3],
            'output_classes': 3,
            'class_names': ['CN', 'EMCI', 'LMCI']
        }
        
        if model_path and os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                
                model_info.update({
                    'file_size_mb': round(os.path.getsize(model_path) / (1024**2), 2),
                    'parameters': model.count_params(),
                    'layers': len(model.layers),
                    'trainable_parameters': int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
                    'loadable': True
                })
                
                # Test if we can create the architecture without weights
                try:
                    test_model = tf.keras.applications.EfficientNetV2B0(
                        weights=None,
                        include_top=False,
                        input_shape=(224, 224, 3)
                    )
                    model_info['architecture_available'] = True
                    del test_model
                except:
                    model_info['architecture_available'] = False
                
                del model
                
            except Exception as e:
                model_info.update({
                    'loadable': False,
                    'error': str(e)
                })
        else:
            model_info.update({
                'file_found': False,
                'note': 'Model file not found at expected location'
            })
        
        self.results['compatibility']['model'] = model_info
        return model_info.get('loadable', False)
    
    def measure_memory_requirements(self, model_path=None):
        """Measure program memory usage requirements"""
        tracemalloc.start()
        process = psutil.Process()
        
        baseline_memory = process.memory_info().rss / (1024**2)  # MB
        
        memory_profile = {
            'baseline_mb': round(baseline_memory, 1)
        }
        
        try:
            # Test importing main libraries
            import tensorflow as tf
            import cv2
            import matplotlib.pyplot as plt
            
            after_imports = process.memory_info().rss / (1024**2)
            memory_profile['after_imports_mb'] = round(after_imports, 1)
            memory_profile['import_overhead_mb'] = round(after_imports - baseline_memory, 1)
            
            # Test model loading if available
            if model_path and os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                after_model = process.memory_info().rss / (1024**2)
                memory_profile['after_model_load_mb'] = round(after_model, 1)
                memory_profile['model_memory_mb'] = round(after_model - after_imports, 1)
                
                # Test inference memory
                sample_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
                predictions = model.predict(sample_data, verbose=0)
                after_inference = process.memory_info().rss / (1024**2)
                memory_profile['after_inference_mb'] = round(after_inference, 1)
                memory_profile['inference_memory_mb'] = round(after_inference - after_model, 1)
                
                # Test batch processing
                batch_data = np.random.rand(10, 224, 224, 3).astype(np.float32)
                batch_predictions = model.predict(batch_data, verbose=0)
                after_batch = process.memory_info().rss / (1024**2)
                memory_profile['peak_memory_mb'] = round(after_batch, 1)
                memory_profile['batch_overhead_mb'] = round(after_batch - after_inference, 1)
                
                del model, sample_data, predictions, batch_data, batch_predictions
            
        except Exception as e:
            memory_profile['error'] = str(e)
        
        gc.collect()
        tracemalloc.stop()
        
        # Calculate recommended memory
        peak_usage = memory_profile.get('peak_memory_mb', memory_profile.get('after_imports_mb', baseline_memory))
        memory_profile['recommended_ram_gb'] = max(4, round((peak_usage * 2) / 1024, 0))  # 2x peak usage, min 4GB
        
        self.results['resource_usage']['memory'] = memory_profile
        return memory_profile
    
    def test_file_format_support(self):
        """Test supported file formats"""
        formats = {
            'medical_images': {
                'nifti': {
                    'extensions': ['.nii', '.nii.gz'],
                    'library': 'nibabel',
                    'supported': False
                }
            },
            'standard_images': {
                'png': {'extensions': ['.png'], 'library': 'PIL/OpenCV', 'supported': False},
                'jpeg': {'extensions': ['.jpg', '.jpeg'], 'library': 'PIL/OpenCV', 'supported': False},
                'tiff': {'extensions': ['.tif', '.tiff'], 'library': 'PIL/OpenCV', 'supported': False}
            },
            'models': {
                'keras': {'extensions': ['.keras', '.h5'], 'library': 'tensorflow', 'supported': False}
            }
        }
        
        # Test medical image support
        try:
            import nibabel
            formats['medical_images']['nifti']['supported'] = True
        except ImportError:
            pass
        
        # Test standard image support
        try:
            import cv2
            from PIL import Image
            formats['standard_images']['png']['supported'] = True
            formats['standard_images']['jpeg']['supported'] = True
            formats['standard_images']['tiff']['supported'] = True
        except ImportError:
            pass
        
        # Test model support
        try:
            import tensorflow as tf
            formats['models']['keras']['supported'] = True
        except ImportError:
            pass
        
        self.results['compatibility']['file_formats'] = formats
        return formats
    
    def test_platform_compatibility(self):
        """Test platform and architecture support"""
        import platform
        
        platform_info = {
            'operating_system': platform.system(),
            'architecture': platform.machine(),
            'python_implementation': platform.python_implementation(),
            'supported_platforms': {
                'Windows': '10 or later',
                'Darwin': 'macOS 10.14 or later',  # Darwin is macOS
                'Linux': 'Ubuntu 18.04 or equivalent'
            },
            'current_platform_supported': platform.system() in ['Windows', 'Darwin', 'Linux']
        }
        
        # Test CPU requirements
        cpu_info = {
            'cores_available': psutil.cpu_count(logical=False),
            'threads_available': psutil.cpu_count(logical=True),
            'minimum_recommended': 2,  # cores
            'adequate_performance': psutil.cpu_count(logical=False) >= 2
        }
        
        platform_info['cpu'] = cpu_info
        self.results['compatibility']['platform'] = platform_info
        return platform_info['current_platform_supported']
    
    def generate_requirements_summary(self):
        """Generate final requirements summary"""
        summary = {
            'minimum_system_requirements': {
                'python_version': '3.8 or later',
                'ram': '4GB minimum, 8GB recommended',
                'storage': '2GB free space (including model and dependencies)',
                'cpu': '2+ cores recommended',
                'gpu': 'Not required (CPU-only operation supported)',
                'network': 'Optional (offline operation supported)'
            },
            'required_dependencies': [
                'tensorflow>=2.8.0',
                'numpy>=1.19.0',
                'opencv-python>=4.5.0',
                'pillow>=8.0.0',
                'matplotlib>=3.3.0',
                'simpleitk>=2.0.0',
                'nibabel>=3.2.0',
                'tkinter (usually included with Python)'
            ],
            'supported_formats': {
                'input': ['NIfTI (.nii, .nii.gz)', 'PNG', 'JPEG', 'TIFF'],
                'models': ['Keras (.keras)', 'HDF5 (.h5)'],
                'output': ['PNG (visualizations)', 'JSON (reports)']
            },
            'supported_platforms': [
                'Windows 10 or later',
                'macOS 10.14 or later',
                'Linux (Ubuntu 18.04 or equivalent)'
            ]
        }
        
        self.results['summary'] = summary
        return summary
    
    def run_full_requirements_test(self, model_path=None, output_file="program_requirements.json"):
        """Run complete program requirements test"""
        print("=" * 60)
        print("NeuroXAI Program Requirements Test")
        print("=" * 60)
        
        # Run all tests
        python_ok = self.test_python_compatibility()
        deps_ok = self.test_core_dependencies()
        gui_ok = self.test_gui_support()
        model_ok = self.test_model_compatibility(model_path)
        memory_profile = self.measure_memory_requirements(model_path)
        formats = self.test_file_format_support()
        platform_ok = self.test_platform_compatibility()
        summary = self.generate_requirements_summary()
        
        # Add timestamp
        self.results['test_timestamp'] = datetime.now().isoformat()
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("REQUIREMENTS TEST SUMMARY")
        print("=" * 60)
        
        print(f"✓ Python Version: {self.results['requirements']['python']['current_version']}")
        
        required_deps = [name for name, info in self.results['requirements']['dependencies'].items() 
                        if info['required']]
        available_deps = [name for name, info in self.results['requirements']['dependencies'].items() 
                         if info['required'] and info['available']]
        print(f"✓ Dependencies: {len(available_deps)}/{len(required_deps)} required packages available")
        
        if self.results['resource_usage']['memory']:
            recommended_ram = self.results['resource_usage']['memory']['recommended_ram_gb']
            print(f"✓ Recommended RAM: {recommended_ram}GB")
        
        print(f"✓ Platform: {self.results['compatibility']['platform']['operating_system']} "
              f"({'supported' if platform_ok else 'check compatibility'})")
        
        print(f"\nProgram Status: {'READY' if all([python_ok, deps_ok, gui_ok]) else 'REQUIREMENTS NOT MET'}")
        print(f"Results saved to: {output_file}")
        
        return self.results

def main():
    """Main function to run requirements test"""
    tester = ProgramRequirementsTest()
    
    # Look for model file
    model_path = None
    possible_paths = [
        "exnModel/TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras",
        "training/models/TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("Note: Model file not found at expected locations.")
        print("Memory requirements will be estimated based on architecture only.")
    
    # Run the test
    results = tester.run_full_requirements_test(model_path)
    return results

if __name__ == "__main__":
    main()