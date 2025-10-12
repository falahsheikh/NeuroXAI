#!/usr/bin/env python3
"""
System Requirements Test for NeuroXAI
Tests actual hardware requirements and dependencies for the neuroimaging analysis tool
"""

import sys
import os
import platform
import psutil
import time
import gc
import subprocess
import pkg_resources
from pathlib import Path
import tracemalloc

def test_python_version():
    """Test Python version compatibility"""
    print("=== Python Version Test ===")
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version compatible")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False

def test_system_info():
    """Display system information"""
    print("\n=== System Information ===")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.2f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    print(f"Memory Usage: {memory.percent}%")
    
    # Disk info
    disk = psutil.disk_usage('/')
    print(f"Available Disk Space: {disk.free / (1024**3):.2f} GB")

def test_required_packages():
    """Test for required Python packages"""
    print("\n=== Required Packages Test ===")
    
    # Core packages for NeuroXAI based on the files mentioned
    required_packages = [
        'tensorflow',
        'numpy',
        'opencv-python',
        'matplotlib',
        'pillow',
        'tkinter',  # Usually built-in with Python
        'simpleitk',
        'nibabel',  # For NIfTI support
        'scipy',
        'scikit-learn'
    ]
    
    optional_packages = [
        'pandas',
        'seaborn',
        'plotly'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
                print(f"✓ {package}: Available (built-in)")
            elif package == 'opencv-python':
                import cv2
                print(f"✓ opencv-python: {cv2.__version__}")
            elif package == 'pillow':
                import PIL
                print(f"✓ pillow: {PIL.__version__}")
            else:
                pkg = __import__(package)
                if hasattr(pkg, '__version__'):
                    print(f"✓ {package}: {pkg.__version__}")
                else:
                    print(f"✓ {package}: Available")
        except ImportError:
            print(f"✗ {package}: Missing")
            missing_required.append(package)
    
    print("\n--- Optional Packages ---")
    for package in optional_packages:
        try:
            pkg = __import__(package)
            if hasattr(pkg, '__version__'):
                print(f"✓ {package}: {pkg.__version__}")
            else:
                print(f"✓ {package}: Available")
        except ImportError:
            print(f"- {package}: Not installed (optional)")
            missing_optional.append(package)
    
    return missing_required, missing_optional

def test_tensorflow_setup():
    """Test TensorFlow installation and capabilities"""
    print("\n=== TensorFlow Setup Test ===")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        
        # Test if TensorFlow can use GPU
        print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        
        # Test basic operations
        print("Testing basic TensorFlow operations...")
        start_time = time.time()
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        end_time = time.time()
        print(f"✓ Basic operations working (took {(end_time-start_time)*1000:.2f}ms)")
        
        return True
        
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"✗ TensorFlow error: {e}")
        return False

def test_model_loading():
    """Test model loading capabilities"""
    print("\n=== Model Loading Test ===")
    
    try:
        import tensorflow as tf
        
        # Test if we can create a simple EfficientNetV2B0-like model
        print("Testing EfficientNetV2B0 availability...")
        try:
            model = tf.keras.applications.EfficientNetV2B0(
                weights=None,  # Don't download weights for test
                include_top=False,
                input_shape=(224, 224, 3)
            )
            print("✓ EfficientNetV2B0 architecture available")
            
            # Calculate model size
            model_size = model.count_params() * 4 / (1024**2)  # Assuming float32
            print(f"Model size (approx): {model_size:.1f} MB")
            
            del model
            gc.collect()
            return True
            
        except Exception as e:
            print(f"✗ EfficientNetV2B0 error: {e}")
            return False
            
    except ImportError:
        print("✗ TensorFlow not available for model test")
        return False

def test_memory_usage():
    """Test memory usage during typical operations"""
    print("\n=== Memory Usage Test ===")
    
    # Start memory tracking
    tracemalloc.start()
    initial_memory = psutil.Process().memory_info().rss / (1024**2)
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Simulate loading model and processing image
        print("Simulating model operations...")
        
        # Create dummy model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        memory_after_model = psutil.Process().memory_info().rss / (1024**2)
        print(f"Memory after model creation: {memory_after_model:.1f} MB")
        
        # Simulate image processing
        dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        prediction = model.predict(dummy_image, verbose=0)
        
        memory_after_prediction = psutil.Process().memory_info().rss / (1024**2)
        print(f"Memory after prediction: {memory_after_prediction:.1f} MB")
        
        # Simulate multiple images (batch processing)
        dummy_batch = np.random.rand(10, 224, 224, 3).astype(np.float32)
        batch_predictions = model.predict(dummy_batch, verbose=0)
        
        peak_memory = psutil.Process().memory_info().rss / (1024**2)
        print(f"Peak memory usage: {peak_memory:.1f} MB")
        
        # Clean up
        del model, dummy_image, dummy_batch, prediction, batch_predictions
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / (1024**2)
        print(f"Memory after cleanup: {final_memory:.1f} MB")
        
        return peak_memory
        
    except Exception as e:
        print(f"Memory test error: {e}")
        return None

def test_image_processing():
    """Test image processing capabilities"""
    print("\n=== Image Processing Test ===")
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Test basic image operations
        print("Testing image processing operations...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Test OpenCV operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(test_image, (224, 224))
        print("✓ OpenCV operations working")
        
        # Test PIL operations
        pil_image = Image.fromarray(test_image)
        pil_resized = pil_image.resize((224, 224))
        print("✓ PIL operations working")
        
        return True
        
    except Exception as e:
        print(f"✗ Image processing error: {e}")
        return False

def test_medical_imaging():
    """Test medical imaging format support"""
    print("\n=== Medical Imaging Support Test ===")
    
    try:
        import nibabel as nib
        import SimpleITK as sitk
        
        print("✓ NiBabel available for NIfTI support")
        print("✓ SimpleITK available for medical formats")
        
        # Test creating dummy medical image
        dummy_volume = np.random.rand(64, 64, 64).astype(np.float32)
        
        # Test SimpleITK operations
        sitk_image = sitk.GetImageFromArray(dummy_volume)
        resampled = sitk.Resample(sitk_image, sitk_image)
        print("✓ Medical image processing operations working")
        
        return True
        
    except ImportError as e:
        print(f"✗ Medical imaging libraries missing: {e}")
        return False
    except Exception as e:
        print(f"✗ Medical imaging error: {e}")
        return False

def generate_requirements_report():
    """Generate final requirements report"""
    print("\n" + "="*50)
    print("SYSTEM REQUIREMENTS REPORT")
    print("="*50)
    
    # Get current system specs
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"\nTested on:")
    print(f"- OS: {platform.system()} {platform.release()}")
    print(f"- Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"- RAM: {memory.total / (1024**3):.1f} GB total")
    print(f"- CPU: {psutil.cpu_count(logical=True)} cores")
    
    print(f"\nMinimum Requirements (Recommended):")
    print(f"- Python 3.8+")
    print(f"- RAM: 4GB minimum, 8GB recommended")
    print(f"- Storage: 2GB free space (including model and dependencies)")
    print(f"- CPU: Multi-core processor (any modern CPU)")
    print(f"- GPU: Not required (CPU-only operation supported)")
    print(f"- OS: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)")
    
    print(f"\nRequired Dependencies:")
    print(f"- tensorflow>=2.8.0")
    print(f"- numpy")
    print(f"- opencv-python")
    print(f"- pillow")
    print(f"- matplotlib")
    print(f"- simpleitk")
    print(f"- nibabel")
    print(f"- tkinter (usually included with Python)")
    
    print(f"\nSupported Formats:")
    print(f"- Medical: NIfTI (.nii, .nii.gz)")
    print(f"- Images: PNG, JPEG, TIFF")
    print(f"- Models: Keras/TensorFlow (.keras, .h5)")

def main():
    """Run all system requirement tests"""
    print("NeuroXAI System Requirements Test")
    print("=" * 40)
    
    # Run tests
    python_ok = test_python_version()
    test_system_info()
    missing_req, missing_opt = test_required_packages()
    tf_ok = test_tensorflow_setup()
    model_ok = test_model_loading()
    peak_memory = test_memory_usage()
    image_ok = test_image_processing()
    medical_ok = test_medical_imaging()
    
    # Generate report
    generate_requirements_report()
    
    # Final assessment
    print(f"\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    if python_ok and not missing_req and tf_ok and model_ok:
        print("✓ System meets minimum requirements for NeuroXAI")
        if peak_memory:
            print(f"✓ Peak memory usage during test: {peak_memory:.1f} MB")
    else:
        print("✗ System does not meet all requirements")
        if missing_req:
            print(f"Missing required packages: {', '.join(missing_req)}")
    
    return {
        'python_compatible': python_ok,
        'missing_required': missing_req,
        'missing_optional': missing_opt,
        'tensorflow_working': tf_ok,
        'model_loading': model_ok,
        'peak_memory_mb': peak_memory,
        'image_processing': image_ok,
        'medical_imaging': medical_ok
    }

if __name__ == "__main__":
    results = main()