# NeuroXAI: Neuroimaging Slice Viewer with XAI Analysis

A comprehensive medical imaging application for neuroimaging slice visualization with explainable AI analysis for early Alzheimer's disease detection. This application consists of two main components: a neuroimaging slice viewer for medical image analysis and an XAI (Explainable AI) analysis tool for automated brain scan interpretation.

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Author](#author)
- [Institutional Affiliation](#institutional-affiliation)
- [License](#license)
- [Citation](#citation)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Features

### Neuroimaging Slice Viewer
- Multi-planar viewing (axial, coronal, sagittal)
- Window/level adjustment for optimal contrast
- Zoom and pan functionality with crosshair navigation
- Drawing and annotation tools with customizable colors and brush sizes
- Distance measurement capabilities with real-world units
- 3D surface model generation using marching cubes algorithm
- Session save/load functionality for workflow continuity
- Comprehensive patient information management
- Undo/redo functionality for all operations
- Multiple view layouts with maximize/minimize options

### XAI Analysis Tool
- AI-powered early Alzheimer's disease detection using EfficientNetV2B0
- Grad-CAM++ attention visualization for model interpretability
- Guided backpropagation analysis for enhanced feature attribution
- Consensus attention mapping combining multiple XAI techniques
- Brain tissue masking and automatic preprocessing
- Statistical analysis of attention patterns
- Medical report generation with clinical recommendations
- Support for multiple disease classifications (CN, EMCI, LMCI)
- Interactive visualization windows with detailed analysis

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for optimal performance)
- 5GB free disk space
- Graphics card with OpenGL support
- Display resolution: 1280x720 minimum (1920x1080 recommended)

### Supported Operating Systems
- macOS 10.14 or later (Intel and Apple Silicon)
- Windows 10 or later (64-bit)
- Linux (Ubuntu 18.04+, CentOS 7+, Fedora 30+)

## Installation

### Step 1: Prerequisites

Ensure you have Python 3.8+ installed:

```bash
python3 --version
```

If Python is not installed, download from [python.org](https://www.python.org/downloads/)

### Step 2: Clone Repository

```bash
git clone https://github.com/falahsheikh/NeuroXAI.git
cd NeuroXAI
```

### Step 3: Create Virtual Environment

```bash
python3 -m venv neuroimaging_env
```

### Step 4: Activate Virtual Environment

**macOS/Linux:**
```bash
source neuroimaging_env/bin/activate
```

**Windows Command Prompt:**
```cmd
neuroimaging_env\Scripts\activate
```

**Windows PowerShell:**
```powershell
neuroimaging_env\Scripts\Activate.ps1
```

### Step 5: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 6: Install Dependencies

```bash
pip install tkinter matplotlib numpy tensorflow opencv-python scipy scikit-image SimpleITK pillow
```

### Step 7: Verify Installation

```bash
pip list
python -c "import tensorflow, cv2, matplotlib, numpy, scipy, skimage, SimpleITK, PIL; print('All imports successful')"
```

### Platform-Specific Installation Notes

#### macOS with Apple Silicon
```bash
pip install tensorflow-macos tensorflow-metal
```

#### Linux Additional Dependencies
```bash
sudo apt update
sudo apt install python3-tk libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```

#### Windows with CUDA Support
```bash
pip install tensorflow-gpu
```

## Usage

### Starting the Applications

#### Neuroimaging Slice Viewer

```bash
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw
python manual.py
```

#### XAI Analysis Tool

```bash
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw/exnModel
python explainability_visuals.py
```

#### Alternative: Launch XAI Tool from Slice Viewer

1. Start the slice viewer: `python manual.py`
2. Click "Launch Analysis Tool" button in the toolbar
3. XAI tool will open in a separate window

### Loading and Using Test Data

#### For Slice Viewer
1. Start the slice viewer application
2. Click "Load Volume" button
3. Navigate to `test_inputs_for_tools/Neuroimaging_Slice_Viewer/`
4. Select `skull_stripped_mri.nii`
5. Use slice navigation controls to explore the volume
6. Try measurement and drawing tools
7. Save session using "Save Session" button

#### For XAI Analysis Tool
1. Start the XAI analysis tool
2. Click "Upload Coronal MRI Slice" button
3. Navigate to `test_inputs_for_tools/Analysis_Tool/CN/` (or EMCI/LMCI)
4. Select any PNG file
5. Wait for automatic processing (30-60 seconds)
6. Review generated visualization windows
7. Click "Max" on any window for detailed view
8. Save results using "Save Generated Medical Report"

### Basic Workflow

#### Slice Viewer Operations

1. **Load Volume**
   ```
   File → Load Volume → Select .nii/.nii.gz file
   ```

2. **Navigate Slices**
   - Use slice position sliders on the left panel
   - Mouse wheel scroll on any view
   - Click on views to update crosshair position

3. **Adjust Display**
   - Window/Level sliders for contrast adjustment
   - Colormap selection for different visualizations
   - Zoom controls: Ctrl+scroll wheel or zoom select tool

4. **Annotate and Measure**
   - Enable "Draw" checkbox and draw with mouse
   - Enable "Measure" checkbox and click two points for distance
   - Customize colors, brush sizes, and opacity

5. **Save Work**
   ```
   File → Save Session → Choose location and filename
   ```

6. **Generate Reports**
   ```
   File → Save Report → Select output directory
   ```

#### XAI Analysis Operations

1. **Upload and Process**
   - Upload coronal MRI slice (PNG/JPG format, 224x224 recommended)
   - Wait for automatic AI processing
   - Review prediction results and confidence scores

2. **Examine Visualizations**
   - Original MRI with metadata overlay
   - Grad-CAM++ attention maps (raw, masked, overlay)
   - Guided Grad-CAM++ enhanced visualizations
   - Consensus maps combining multiple techniques
   - Statistical analysis windows

3. **Interpret Results**
   - Check prediction confidence and class probabilities
   - Review attention regions highlighted by AI
   - Read clinical interpretation and recommendations
   - Examine statistical metrics for validation

4. **Export Results**
   - Save comprehensive medical report with all visualizations
   - Export includes original images, attention maps, and analysis

### Keyboard Shortcuts

- `Ctrl+O`: Load Volume
- `Ctrl+S`: Save Report
- `Ctrl+Shift+S`: Save Session
- `Ctrl+Shift+O`: Load Session
- `Ctrl+R`: Reset Layout
- `Ctrl+0`: Reset Zoom
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `ESC`: Cancel current operation (measurement, drawing)

### Advanced Features

#### 3D Model Generation
1. Load a volume in the slice viewer
2. Navigate to "3D" tab in control panel
3. Adjust ISO threshold value
4. Click "Generate 3D Model"
5. Rotate and examine 3D visualization

#### Session Management
1. Save current state: annotations, measurements, view settings
2. Load previous sessions to continue work
3. Export comprehensive reports with all findings

#### Custom Model Loading (XAI Tool)
1. Click "Load Custom Model" in XAI tool
2. Select your own trained Keras model file
3. Tool automatically adapts to model output classes

## Project Structure

```
NeuroXAI/
├── LICENSE                                    # CC BY-NC 4.0 License
├── README.md                                  # This documentation
├── CITATION.cff                              # Citation information
├── CONTRIBUTING.md                           # Contribution guidelines
├── Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw/
│   ├── manual.py                             # Main slice viewer application
│   ├── loading_window.py                     # Loading screen component
│   └── exnModel/
│       ├── explainability_visuals.py         # XAI analysis tool
│       └── TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras # Pre-trained model
├── test_inputs_for_tools/                    # Sample data and test files
│   ├── Analysis_Tool/                        # Test images for XAI analysis
│   │   ├── CN/                              # Cognitively Normal samples
│   │   ├── EMCI/                            # Early MCI samples  
│   │   └── LMCI/                            # Late MCI samples
│   └── Neuroimaging_Slice_Viewer/           # Test data for slice viewer
│       └── skull_stripped_mri.nii           # Sample NIfTI file
├── preprocessing_tools/                      # Data preprocessing utilities
│   ├── skull_stripping.py                   # Brain extraction tools
│   └── slice_extraction.py                  # Slice extraction utilities
├── training/                                # Model training resources
│   ├── models/                              # Trained model files
│   ├── outputs/                             # Training results and metrics
│   └── training_scripts.ipynb              # Training notebooks
└── docs/                                   # Additional documentation
```

## Testing

### Quick Functionality Test

```bash
# Navigate to project directory
cd NeuroXAI

# Test slice viewer imports
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw
python -c "import manual; print('Slice viewer imports successful')"

# Test XAI tool imports
cd exnModel
python -c "import explainability_visuals; print('XAI tool imports successful')"

# Test all required packages
python -c "
import tkinter
import matplotlib
import numpy
import tensorflow
import cv2
import scipy
import skimage
import SimpleITK
import PIL
print('All packages successfully imported')
"
```

### Test with Sample Data

#### Slice Viewer Test
```bash
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw
python manual.py
```
Then in the GUI:
1. Click "Load Volume"
2. Navigate to `../test_inputs_for_tools/Neuroimaging_Slice_Viewer/`
3. Select `skull_stripped_mri.nii`
4. Verify three orthogonal views appear
5. Test slice navigation and zoom functionality

#### XAI Analysis Test
```bash
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw/exnModel
python explainability_visuals.py
```
Then in the GUI:
1. Click "Upload Coronal MRI Slice"
2. Navigate to `../../test_inputs_for_tools/Analysis_Tool/CN/`
3. Select any PNG file
4. Wait for processing to complete
5. Verify multiple analysis windows appear
6. Test maximize/minimize functionality

### Automated Testing

```bash
# Create test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
import sys
import subprocess

def test_imports():
    try:
        import tkinter
        import matplotlib
        import numpy
        import tensorflow
        import cv2
        import scipy
        import skimage
        import SimpleITK
        import PIL
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_tensorflow():
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        print(f"✓ GPU available: {tf.config.list_physical_devices('GPU')}")
        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Neuroimaging Software Installation...")
    print("-" * 50)
    
    success = True
    success &= test_imports()
    success &= test_tensorflow()
    
    if success:
        print("\n✓ Installation test passed!")
        sys.exit(0)
    else:
        print("\n✗ Installation test failed!")
        sys.exit(1)
EOF

python test_installation.py
```

## Troubleshooting

### Common Installation Issues

#### Python Import Errors
```bash
# Verify Python version (must be 3.8+)
python3 --version

# Check if virtual environment is activated
which python
# Should show path to neuroimaging_env

# Verify package installation
pip list | grep tensorflow
pip list | grep opencv

# Reinstall problematic packages
pip uninstall tensorflow opencv-python
pip install tensorflow opencv-python
```

#### TensorFlow Issues

**For Apple Silicon Macs:**
```bash
pip uninstall tensorflow
pip install tensorflow-macos tensorflow-metal
```

**For NVIDIA GPU Support:**
```bash
pip install tensorflow-gpu
# Ensure CUDA and cuDNN are installed
```

**Memory Issues:**
```bash
# Limit TensorFlow memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### Graphics and GUI Issues

**Linux Display Issues:**
```bash
sudo apt update
sudo apt install mesa-utils libgl1-mesa-glx
sudo apt install python3-tk

# Test OpenGL
glxinfo | grep OpenGL

# For remote sessions
ssh -X username@hostname
export DISPLAY=:0
```

**macOS GUI Issues:**
```bash
# Install tkinter support
brew install python-tk

# For conda environments
conda install tk
```

**Windows Graphics Issues:**
```cmd
# Update graphics drivers
# Enable hardware acceleration in display settings
# Run as administrator if permission issues
```

#### Memory and Performance Issues

**Insufficient Memory:**
- Close other applications
- Increase virtual memory/swap space
- Use smaller image sizes for XAI analysis
- Process images individually rather than in batches

**Slow Performance:**
```bash
# Check available memory
free -h  # Linux
vm_stat # macOS

# Monitor CPU usage during processing
top  # Linux/macOS
# Task Manager on Windows
```

#### File Format Issues

**NIfTI Loading Problems:**
```bash
# Verify file integrity
python -c "
import SimpleITK as sitk
image = sitk.ReadImage('path/to/your/file.nii')
print(f'Image size: {image.GetSize()}')
print(f'Spacing: {image.GetSpacing()}')
"
```

**Image Format Issues (XAI Tool):**
- Ensure images are PNG or JPG format
- Recommended size: 224x224 pixels
- Convert if necessary:
```bash
# Using ImageMagick
convert input.dcm -resize 224x224 output.png

# Using Python
python -c "
from PIL import Image
img = Image.open('input.jpg')
img = img.resize((224, 224))
img.save('output.png')
"
```

### Error Messages and Solutions

**"ImportError: No module named 'tkinter'"**
```bash
# Ubuntu/Debian
sudo apt install python3-tk

# CentOS/RHEL
sudo yum install tkinter

# macOS
brew install python-tk
```

**"Could not load dynamic library 'libcudart.so'"**
```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit
# Or download from NVIDIA website
```

**"Failed to get convolution algorithm"**
```bash
# TensorFlow GPU memory issue
export TF_FORCE_GPU_ALLOW_GROWTH=true
# Or use CPU-only version
pip install tensorflow-cpu
```

**Application crashes on startup:**
```bash
# Run with verbose output
python manual.py --verbose

# Check system resources
df -h  # Disk space
free -h  # Memory
```

### Getting Help

1. **Check system requirements** - Ensure your system meets minimum requirements
2. **Verify installation steps** - Review installation commands for your platform  
3. **Check error messages** - Run applications from terminal to see detailed errors
4. **Test with sample data** - Use provided test files to isolate issues
5. **Update dependencies** - Ensure all packages are up to date:
   ```bash
   pip install --upgrade tensorflow opencv-python matplotlib numpy scipy
   ```

## Authors

**Falah Sheikh et al.**
- Primary Author: Falah Sheikh
- Email: sheikhfalah.sheikhha@ucalgary.ca
- GitHub: [@falahsheikh](https://github.com/falahsheikh)
- Institution: University of Calgary
- Lab: Data and Network Sciences & Applications Lab (DANSA) Lab
- Funding: Alberta Innovates Summer Research Studentship

## Institutional Affiliation

This project was developed at the **Data and Network Sciences & Applications Lab (DANSA) Lab**, University of Calgary, with funding support from **Alberta Innovates Summer Research Studentship**.

The research focuses on advancing medical imaging analysis through explainable artificial intelligence techniques, contributing to early detection and diagnosis of neurodegenerative diseases.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)** - see the [LICENSE](LICENSE) file for details.

### Permitted Uses
- Academic research and education
- Personal use and learning  
- Modification and creation of derivative works
- Distribution of original or modified versions
- Use by non-profit research organizations
- Government research applications
- Student projects and coursework

### Requirements
- Attribution must be provided to the original author and institution
- License notice must be included in redistributions
- Changes must be indicated if modifications are made
- Link to original source should be provided

### Restrictions
- Commercial use is prohibited
- Cannot be used for profit-generating activities
- Cannot be integrated into commercial products without explicit permission
- Cannot be sold or licensed to third parties

### For Commercial Use

This software is not available for commercial use under the standard license. For commercial licensing inquiries, including:
- Integration into commercial medical software
- Use in for-profit healthcare applications  
- Commercial research and development
- Consulting services utilizing this software

Please contact: sheikhfalah.sheikhha@ucalgary.ca

## Citation

If you use this software in your research, please cite:

```bibtex
@software{sheikh2025neuroimaging,
  title={Neuroimaging Slice Viewer with XAI Analysis for Early Alzheimer's Disease Detection},
  author={Sheikh, Falah and others},
  year={2025},
  institution={Data and Network Sciences & Applications Lab (DANSA) Lab, University of Calgary},
  funding={Alberta Innovates Summer Research Studentship},
  url={https://github.com/falahsheikh/NeuroXAI},
  license={CC BY-NC 4.0},
  doi={}, // Add DOI if available
  note={Software for neuroimaging analysis with explainable AI}
}

@inproceedings{sheikh2026neuroxai,
  title={NeuroXAI: lightweight explainable early Alzheimer's detection for resource-constrained clinical settings},
  author={Sheikh, Falah and Al Marouf, Ahmed and Alhajj, Reda and Rokne, Jon George},
  booktitle={Medical Imaging 2026: Imaging Informatics},
  volume={13930},
  pages={139301R},
  year={2026},
  organization={SPIE},
  address={Vancouver, BC, Canada},
  doi={10.1117/12.3088200},
  url={https://doi.org/10.1117/12.3088200}
}
```



### Alternative Citation Formats

**APA Style:**
Sheikh, F., et al. (2025). Neuroimaging Slice Viewer with XAI Analysis for Early Alzheimer's Disease Detection [Computer software]. University of Calgary. https://github.com/falahsheikh/NeuroXAI

**IEEE Style:**
F. Sheikh et al., "Neuroimaging Slice Viewer with XAI Analysis for Early Alzheimer's Disease Detection," University of Calgary, 2025. [Online]. Available: https://github.com/falahsheikh/NeuroXAI

## Contributing

Contributions are welcome from the research community. This project encourages collaboration to advance medical imaging analysis and explainable AI techniques.

### How to Contribute

1. **Fork the repository**
   ```bash
   # Click 'Fork' on GitHub or use GitHub CLI
   gh repo fork falahsheikh/NeuroXAI
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/NeuroXAI.git
   cd NeuroXAI
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/research-improvement
   ```

4. **Make your changes and commit**
   ```bash
   git add .
   git commit -m "Add research improvement: detailed description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/research-improvement
   ```

6. **Submit a pull request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Provide detailed description of changes

### Contribution Guidelines

**Code Quality:**
- Follow existing code style and conventions
- Add comments for complex algorithms
- Include docstrings for new functions
- Test your changes with provided sample data

**Documentation:**
- Update README.md for new features
- Add usage examples for new functionality
- Update installation instructions if dependencies change

**Research Contributions:**
- Include references for new algorithms
- Provide validation data or methods
- Document performance improvements
- Explain clinical relevance

### Contribution Areas

We welcome contributions in:

**Algorithm Improvements:**
- Enhanced XAI visualization techniques
- Better attention mechanism implementations
- Improved preprocessing methods
- New evaluation metrics

**Software Engineering:**
- Bug fixes and performance optimizations
- User interface enhancements
- Cross-platform compatibility improvements
- Memory and speed optimizations

**Documentation and Testing:**
- Additional usage examples and tutorials
- Comprehensive test cases
- Installation guides for new platforms
- User experience improvements

**Research Applications:**
- Support for additional medical imaging modalities
- New disease classification categories
- Validation on different datasets
- Integration with other medical AI tools

### License Agreement

By contributing, you agree that:
- Your contributions will be licensed under the same CC BY-NC 4.0 license
- Your contributions cannot be used for commercial purposes
- You retain copyright to your contributions
- You grant permission for academic and research use

## Acknowledgments

### Funding and Institutional Support
- **Alberta Innovates** for Summer Research Studentship funding
- **Data and Network Sciences & Applications Lab (DANSA) Lab, University of Calgary** for research support, infrastructure, and academic guidance
- **University of Calgary** for institutional resources and research environment

### Technical Dependencies
- **TensorFlow** team for the deep learning framework
- **OpenCV** community for computer vision tools
- **Matplotlib** developers for visualization capabilities
- **SimpleITK** team for medical image processing
- **Scikit-image** contributors for image analysis algorithms
- **NumPy** and **SciPy** communities for numerical computing foundation

### Research Community
- Medical imaging research community for datasets, validation methods, and clinical insights
- Explainable AI researchers for foundational XAI techniques and methodologies
- Open source software contributors whose tools made this project possible

### Special Recognition
- Neuroimaging researchers who provided feedback and validation
- Beta testers from the medical imaging community
- Contributors to medical imaging standards and protocols

## Contact

### Primary Contact
**Falah Sheikh et al.**
- **Email**: sheikhfalah.sheikhha@ucalgary.ca
- **GitHub**: [@falahsheikh](https://github.com/falahsheikh)
- **Institution**: Data and Network Sciences & Applications Lab (DANSA) Lab, University of Calgary

### Communication Channels

**For Technical Issues:**
- Use GitHub Issues for bug reports and feature requests
- Provide detailed error messages and system information
- Include steps to reproduce any problems

**For Research Collaboration:**
- Email: sheikhfalah.sheikhha@ucalgary.ca for academic partnerships
- Include details about your research interests and institution
- Specify collaboration goals and expected outcomes

**For Usage Questions:**
- Check this README and documentation first
- Search existing GitHub Issues for similar questions
- Create new issue with "question" label if needed
- Email: sheikhfalah.sheikhha@ucalgary.ca for complex questions

**For Commercial Licensing:**
- Email: sheikhfalah.sheikhha@ucalgary.ca with detailed inquiry
- Include organization information and intended use case
- Specify commercial application and scale of deployment

### Response Times
- Technical issues: Typically within 3-5 business days
- Research collaboration: Within 1-2 weeks
- General questions: Within 1 week
- Commercial inquiries: Within 2 weeks

---

**Note**: This software is provided for research and educational purposes. Clinical applications require appropriate validation, regulatory approval, and adherence to medical device standards. Users are responsible for ensuring compliance with applicable regulations and ethical guidelines in their jurisdiction.
