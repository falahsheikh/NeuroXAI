# Neuroimaging Slice Viewer with XAI Analysis

A comprehensive medical imaging application for neuroimaging slice visualization with explainable AI analysis for early Alzheimer's disease detection. This application consists of two main components: a neuroimaging slice viewer for medical image analysis and an XAI (Explainable AI) analysis tool for automated brain scan interpretation.

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

## Installation Instructions

### macOS Installation

#### Step 1: Install System Dependencies

First, install Homebrew package manager if not already installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install Python and required system packages:
```bash
brew install python python-tk
```

Verify Python installation:
```bash
python3 --version
```

#### Step 2: Download Project Files

Clone the repository or download the project files:
```bash
git clone https://github.com/falahsheikh/Lightweight_MRI_EAD_Detection.git
cd Lightweight_MRI_EAD_Detection
```

If downloading manually, extract the files and navigate to the project directory:
```bash
cd /path/to/Lightweight_MRI_EAD_Detection
```

#### Step 3: Create Virtual Environment

Create a new virtual environment for the project:
```bash
python3 -m venv neuroimaging_env
```

Activate the virtual environment:
```bash
source neuroimaging_env/bin/activate
```

Verify virtual environment is active (you should see the environment name in your terminal prompt):
```bash
which python
```

#### Step 4: Install Python Dependencies

Upgrade pip to the latest version:
```bash
pip install --upgrade pip
```

Install all required Python packages:
```bash
pip install tkinter matplotlib numpy tensorflow opencv-python scipy scikit-image SimpleITK pillow
```

Wait for all packages to download and install. This may take several minutes depending on your internet connection.

#### Step 5: Verify Installation

Check that all packages are installed correctly:
```bash
pip list
```

Navigate to the application directory:
```bash
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw
```

#### Step 6: Run the Application

Start the main neuroimaging slice viewer application:
```bash
python manual.py
```

The application should launch with a loading screen followed by the main interface.

### Windows Installation

#### Step 1: Install Python

Download Python 3.8 or later from the official Python website (python.org):
- Go to https://www.python.org/downloads/
- Download the latest Python 3.x version for Windows
- Run the installer with the following important settings:
  - Check "Add Python to PATH" during installation
  - Choose "Customize installation"
  - Ensure "tkinter/Tk and IDLE" is selected
  - Choose "Install for all users" if you have administrator privileges

Verify Python installation by opening Command Prompt and running:
```cmd
python --version
```

#### Step 2: Download Project Files

Download the project files to your computer. If using Git:
```cmd
git clone https://github.com/falahsheikh/Lightweight_MRI_EAD_Detection.git
cd Lightweight_MRI_EAD_Detection
```

If downloading manually, extract the files to a directory such as:
```cmd
C:\Users\YourUsername\Documents\Lightweight_MRI_EAD_Detection
```

Navigate to the project directory:
```cmd
cd C:\path\to\Lightweight_MRI_EAD_Detection
```

#### Step 3: Create Virtual Environment

Open Command Prompt as Administrator (recommended) or regular Command Prompt and navigate to the project directory:
```cmd
cd C:\path\to\Lightweight_MRI_EAD_Detection
```

Create a virtual environment:
```cmd
python -m venv neuroimaging_env
```

Activate the virtual environment:
```cmd
neuroimaging_env\Scripts\activate
```

You should see (neuroimaging_env) at the beginning of your command prompt.

#### Step 4: Install Python Dependencies

Upgrade pip to the latest version:
```cmd
pip install --upgrade pip
```

Install all required Python packages:
```cmd
pip install matplotlib numpy tensorflow opencv-python scipy scikit-image SimpleITK pillow
```

This installation process may take 10-15 minutes depending on your internet speed and computer performance.

#### Step 5: Verify Installation

Check that all packages are installed correctly:
```cmd
pip list
```

Navigate to the application directory:
```cmd
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw
```

#### Step 6: Run the Application

Start the main neuroimaging slice viewer application:
```cmd
python manual.py
```

The application should launch with a loading screen followed by the main interface.

### Linux Installation (Ubuntu/Debian)

#### Step 1: Update System and Install Dependencies

Update your package manager:
```bash
sudo apt update
sudo apt upgrade
```

Install Python and system dependencies:
```bash
sudo apt install python3 python3-pip python3-venv python3-tk
```

Install additional system libraries required for graphics and GUI:
```bash
sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```

Install development tools and libraries:
```bash
sudo apt install build-essential python3-dev libffi-dev libssl-dev
```

Verify Python installation:
```bash
python3 --version
```

#### Step 2: Download Project Files

Clone the repository or download the project files:
```bash
git clone https://github.com/falahsheikh/Lightweight_MRI_EAD_Detection.git
cd Lightweight_MRI_EAD_Detection
```

If downloading manually:
```bash
cd /path/to/Lightweight_MRI_EAD_Detection
```

#### Step 3: Create Virtual Environment

Create a virtual environment:
```bash
python3 -m venv neuroimaging_env
```

Activate the virtual environment:
```bash
source neuroimaging_env/bin/activate
```

Verify the virtual environment is active:
```bash
which python
```

#### Step 4: Install Python Dependencies

Upgrade pip to the latest version:
```bash
pip install --upgrade pip
```

Install all required Python packages:
```bash
pip install matplotlib numpy tensorflow opencv-python scipy scikit-image SimpleITK pillow
```

This installation may take 15-20 minutes depending on your system and internet connection.

#### Step 5: Verify Installation

Check that all packages are installed correctly:
```bash
pip list
```

Test tkinter installation:
```bash
python -c "import tkinter; print('tkinter working')"
```

Navigate to the application directory:
```bash
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw
```

#### Step 6: Run the Application

Start the main neuroimaging slice viewer application:
```bash
python manual.py
```

The application should launch successfully.

### Linux Installation (CentOS/RHEL)

#### Step 1: Update System and Install Dependencies

Update your system:
```bash
sudo yum update
```

Install Python and development tools:
```bash
sudo yum install python3 python3-pip python3-devel
```

Install additional system libraries:
```bash
sudo yum install tkinter gcc openssl-devel libffi-devel
```

Install EPEL repository for additional packages:
```bash
sudo yum install epel-release
```

Install graphics libraries:
```bash
sudo yum install mesa-libGL mesa-libEGL libXrandr libXss libXcursor libXcomposite alsa-lib libXi libXtst
```

#### Step 2: Download Project Files

Clone the repository or download project files:
```bash
git clone https://github.com/falahsheikh/Lightweight_MRI_EAD_Detection.git
cd Lightweight_MRI_EAD_Detection
```

If downloading manually:
```bash
cd /path/to/Lightweight_MRI_EAD_Detection
```

#### Step 3: Create Virtual Environment

Create a virtual environment:
```bash
python3 -m venv neuroimaging_env
```

Activate the virtual environment:
```bash
source neuroimaging_env/bin/activate
```

#### Step 4: Install Python Dependencies

Upgrade pip:
```bash
pip install --upgrade pip
```

Install required packages:
```bash
pip install matplotlib numpy tensorflow opencv-python scipy scikit-image SimpleITK pillow
```

#### Step 5: Verify Installation

Check installed packages:
```bash
pip list
```

Navigate to application directory:
```bash
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw
```

#### Step 6: Run the Application

Start the application:
```bash
python manual.py
```

### Linux Installation (Fedora)

#### Step 1: Update System and Install Dependencies

Update your system:
```bash
sudo dnf update
```

Install Python and development tools:
```bash
sudo dnf install python3 python3-pip python3-devel python3-tkinter
```

Install additional system libraries:
```bash
sudo dnf install gcc openssl-devel libffi-devel
```

Install graphics libraries:
```bash
sudo dnf install mesa-libGL mesa-libEGL libXrandr libXss libXcursor libXcomposite alsa-lib libXi libXtst
```

#### Step 2: Download Project Files

Clone the repository:
```bash
git clone https://github.com/falahsheikh/Lightweight_MRI_EAD_Detection.git
cd Lightweight_MRI_EAD_Detection
```

#### Step 3: Create Virtual Environment

Create virtual environment:
```bash
python3 -m venv neuroimaging_env
```

Activate virtual environment:
```bash
source neuroimaging_env/bin/activate
```

#### Step 4: Install Python Dependencies

Upgrade pip:
```bash
pip install --upgrade pip
```

Install required packages:
```bash
pip install matplotlib numpy tensorflow opencv-python scipy scikit-image SimpleITK pillow
```

#### Step 5: Verify Installation

Check installed packages:
```bash
pip list
```

Navigate to application directory:
```bash
cd Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw
```

#### Step 6: Run the Application

Start the application:
```bash
python manual.py
```

## Application Usage

### Main Application: Neuroimaging Slice Viewer

The main application (manual.py) provides comprehensive tools for medical image analysis and visualization.

#### Loading Medical Images

1. **Start the application**: The main window will appear with a toolbar and empty viewing panels
2. **Load a volume**: Click the "Load Volume" button in the toolbar
3. **Select file**: Choose a NIfTI file (.nii or .nii.gz) from the file dialog
4. **View confirmation**: The image will appear in the three orthogonal views (axial, coronal, sagittal)

#### Navigation and Viewing

1. **Slice navigation**: Use the slice position sliders on the left panel to navigate through different slices
2. **Window/Level adjustment**: Modify the Window and Level sliders to optimize image contrast and brightness
3. **Zoom functionality**: Use Ctrl+scroll wheel to zoom in/out on any view
4. **Pan images**: Use middle mouse button to pan around zoomed images
5. **Crosshair navigation**: Click on any view to update the crosshair position across all views

#### Image Analysis Tools

1. **Measurement tool**: 
   - Enable "Measure" checkbox in the toolbar
   - Click two points to create a distance measurement
   - View measurements in the Annotations tab

2. **Drawing tool**:
   - Enable "Draw" checkbox in the toolbar
   - Click and drag to draw annotations on images
   - Adjust brush size and color in the controls panel
   - Add comments to drawings for documentation

3. **Zoom selection**:
   - Enable "Zoom Select" checkbox
   - Click and drag to select an area for zooming
   - The view will zoom to the selected region

#### Session Management

1. **Save session**: Click "Save Session" to save your current work including annotations and settings
2. **Load session**: Click "Load Session" to restore a previously saved session
3. **Patient information**: Click "Patient Info" to edit patient details
4. **Export reports**: Use "Save Report" to generate comprehensive analysis reports

#### Launching XAI Analysis

1. **Access analysis tool**: Click the "Launch Analysis Tool" button in the toolbar
2. **Automatic launch**: The XAI analysis application will start in a separate window
3. **Continue work**: Both applications can run simultaneously

### XAI Analysis Tool: Explainability Visuals

The XAI analysis tool (explainability_visuals.py) provides AI-powered analysis of brain MRI images with explainable visualizations.

#### Loading Images for Analysis

1. **Application startup**: The XAI tool opens with multiple empty analysis windows
2. **Upload image**: Click "Upload Coronal MRI Slice" button in the toolbar
3. **Select file**: Choose a PNG or JPG image file (coronal brain slice)
4. **Processing**: The application will automatically process the image and generate multiple analysis views

#### Understanding Analysis Results

The tool generates multiple visualization windows:

1. **Original MRI**: Displays the input image with metadata overlay
2. **Prediction Analysis**: Shows diagnostic confidence scores and clinical interpretation
3. **Confidence Map**: Visual representation of model certainty across different classes
4. **Class Probabilities**: Detailed statistical breakdown of prediction probabilities
5. **Grad-CAM++ visualizations**: Heat maps showing areas of focus for AI decision-making
6. **Guided Grad-CAM++ analysis**: Enhanced attention maps with statistical analysis
7. **Consensus analysis**: Combined visualization from multiple AI techniques
8. **Medical Report**: Comprehensive clinical report with recommendations

#### Interacting with Analysis Windows

1. **Window maximization**: Click "Max" button on any window to view it in full size
2. **Window restoration**: Click "Min" button to restore normal view
3. **Navigation tools**: Use toolbar buttons to zoom, pan, and navigate through visualizations
4. **Save results**: Click "Save Generated Medical Report" to export all analysis results

#### Interpreting Results

1. **Confidence scores**: Higher scores indicate greater AI certainty in diagnosis
2. **Heat maps**: Red/yellow areas show regions of interest for AI decision-making
3. **Clinical recommendations**: Follow the recommendations provided in the medical report
4. **Statistical analysis**: Review detailed statistics for research and validation purposes

## Test Data and Examples

### Sample Data Location

The project includes test data in the `test_inputs_for_tools` directory:

```
test_inputs_for_tools/
├── Neuroimaging_Slice_Viewer/
│   └── skull_stripped_mri.nii
└── Analysis_Tool/
    ├── CN/          # Cognitively Normal samples
    ├── EMCI/        # Early Mild Cognitive Impairment samples
    └── LMCI/        # Late Mild Cognitive Impairment samples
```

### Testing the Slice Viewer

1. **Load test volume**:
   - Start the slice viewer application
   - Click "Load Volume"
   - Navigate to `test_inputs_for_tools/Neuroimaging_Slice_Viewer/`
   - Select `skull_stripped_mri.nii`

2. **Explore functionality**:
   - Navigate through slices using the controls
   - Try measurement and drawing tools
   - Experiment with window/level adjustments
   - Test session save/load functionality

### Testing the XAI Analysis Tool

1. **Load test images**:
   - Start the XAI analysis tool
   - Click "Upload Coronal MRI Slice"
   - Navigate to `test_inputs_for_tools/Analysis_Tool/CN/` (or EMCI/LMCI)
   - Select any PNG file

2. **Review analysis**:
   - Examine all generated visualization windows
   - Maximize different windows for detailed view
   - Review the medical report for clinical interpretation
   - Test the save functionality

## Troubleshooting

### Installation Issues

#### Python Installation Problems

**Issue**: "python command not found"
- **macOS**: Install Python via Homebrew: `brew install python`
- **Windows**: Reinstall Python and ensure "Add to PATH" is checked
- **Linux**: Install using package manager: `sudo apt install python3`

**Issue**: "pip command not found"
- **All platforms**: Python may be installed without pip
- **Solution**: Install pip manually or reinstall Python with pip included

#### Virtual Environment Issues

**Issue**: Virtual environment activation fails
- **macOS/Linux**: Check file permissions: `chmod +x neuroimaging_env/bin/activate`
- **Windows**: Try using `neuroimaging_env\Scripts\activate.bat`
- **All platforms**: Ensure you're in the correct directory

**Issue**: "Permission denied" errors
- **macOS/Linux**: Use `sudo` for system-wide installations or fix permissions
- **Windows**: Run Command Prompt as Administrator

### Package Installation Issues

#### TensorFlow Installation Problems

**Issue**: TensorFlow installation fails on macOS M1/M2
- **Solution**: Use Apple Silicon optimized version:
```bash
pip install tensorflow-macos tensorflow-metal
```

**Issue**: TensorFlow GPU support issues
- **Solution**: Install CUDA toolkit for NVIDIA GPUs or use CPU-only version

**Issue**: Memory errors during installation
- **Solution**: Increase virtual memory or install packages one by one

#### Graphics and GUI Issues

**Issue**: "ImportError: No module named 'tkinter'"
- **macOS**: Install tkinter: `brew install python-tk`
- **Ubuntu/Debian**: Install tkinter: `sudo apt install python3-tk`
- **Windows**: Reinstall Python with tkinter enabled

**Issue**: OpenGL errors or graphics issues
- **Linux**: Install Mesa drivers: `sudo apt install mesa-utils`
- **All platforms**: Update graphics drivers
- **Virtual machines**: Enable 3D acceleration in VM settings

**Issue**: "cannot connect to X server" on Linux
- **Solution**: Ensure GUI session is running or use X11 forwarding for remote connections:
```bash
ssh -X username@hostname
```

### Runtime Issues

#### Memory and Performance Problems

**Issue**: "Out of memory" errors
- **Solution**: Ensure at least 8GB RAM available
- **Solution**: Close other applications before analysis
- **Solution**: Increase virtual memory/swap space

**Issue**: Application runs slowly
- **Solution**: Use SSD storage for better I/O performance
- **Solution**: Ensure adequate RAM available
- **Solution**: Close unnecessary background applications

#### File Loading Issues

**Issue**: Cannot load NIfTI files
- **Solution**: Verify file format and integrity
- **Solution**: Try converting file format using medical imaging tools
- **Solution**: Check file permissions

**Issue**: Image files not loading in XAI tool
- **Solution**: Ensure files are PNG or JPG format
- **Solution**: Verify image dimensions are reasonable (not too large)
- **Solution**: Check file path contains no special characters

#### Application Crashes

**Issue**: Application crashes on startup
- **Solution**: Check all dependencies are installed correctly
- **Solution**: Run from terminal to see error messages
- **Solution**: Verify Python version compatibility

**Issue**: Crashes during analysis
- **Solution**: Monitor system resources (RAM, disk space)
- **Solution**: Try with smaller image files
- **Solution**: Restart application and try again

### Platform-Specific Issues

#### macOS Issues

**Issue**: "App can't be opened" security warning
- **Solution**: Go to System Preferences > Security & Privacy > Allow app

**Issue**: Homebrew installation issues
- **Solution**: Update Xcode command line tools: `xcode-select --install`

#### Windows Issues

**Issue**: Antivirus blocking installation
- **Solution**: Temporarily disable antivirus during installation
- **Solution**: Add Python and project directory to antivirus exclusions

**Issue**: Long path names causing issues
- **Solution**: Move project to shorter path (e.g., C:\neuroimaging\)

#### Linux Issues

**Issue**: Package manager permission errors
- **Solution**: Use sudo for system package installation
- **Solution**: Check user permissions and group membership

**Issue**: Display issues in remote sessions
- **Solution**: Enable X11 forwarding: `ssh -X`
- **Solution**: Use VNC for full desktop remote access

### Getting Help

If you continue to experience issues:

1. **Check system requirements**: Ensure your system meets minimum requirements
2. **Verify installation steps**: Review installation steps for your platform
3. **Check error messages**: Run applications from terminal to see detailed error messages
4. **Update dependencies**: Ensure all packages are up to date
5. **Try clean installation**: Remove virtual environment and reinstall from scratch

### Log Files and Debugging

To help diagnose issues:

1. **Run with verbose output**:
```bash
python manual.py --verbose
```

2. **Check Python import errors**:
```bash
python -c "import tensorflow, cv2, matplotlib, numpy, scipy, skimage, SimpleITK, PIL; print('All imports successful')"
```

3. **Monitor system resources**:
- **macOS**: Activity Monitor
- **Windows**: Task Manager
- **Linux**: `htop` or `top`

## File Structure and Organization

### Project Directory Structure

```
Lightweight_MRI_EAD_Detection/
├── Neuroimaging_Slice_Viewer_with_XAI_Analysis_raw/
│   ├── manual.py                           # Main neuroimaging slice viewer application
│   ├── loading_window.py                   # Loading screen component
│   └── exnModel/
│       ├── explainability_visuals.py       # XAI analysis tool
│       └── TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras  # Pre-trained model
├── test_inputs_for_tools/                  # Sample data and test files
│   ├── Analysis_Tool/                      # Test images for XAI analysis
│   │   ├── CN/                            # Cognitively Normal samples
│   │   ├── EMCI/                          # Early MCI samples
│   │   └── LMCI/                          # Late MCI samples
│   └── Neuroimaging_Slice_Viewer/         # Test data for slice viewer
│       └── skull_stripped_mri.nii         # Sample NIfTI file
├── preprocessing_tools/                    # Data preprocessing utilities
│   ├── skull_stripping.py                 # Brain extraction tools
│   └── slice_extraction.py                # Slice extraction utilities
└── training/                              # Model training resources
    ├── models/                            # Trained model files
    ├── outputs/                           # Training results and metrics
    └── training_scripts.ipynb             # Training notebooks
```

### Key Files Description

- **manual.py**: Main application entry point for the neuroimaging slice viewer
- **explainability_visuals.py**: Standalone XAI analysis tool for AI-powered brain scan analysis
- **loading_window.py**: Shared loading screen component used by both applications
- **TRAINING_WITH_INFLATED_AUGMENTED_DATA_EFV2B0.keras**: Pre-trained EfficientNetV2B0 model for early Alzheimer's detection

This comprehensive setup provides both applications with full functionality for medical image analysis and AI-powered diagnostic assistance.