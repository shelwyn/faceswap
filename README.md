# FaceSwap - Real-time Face Swapping Application

A Streamlit-based web application that performs real-time face swapping using InsightFace deep learning models. Upload or capture photos and seamlessly swap faces between images with just a few clicks.

## 🌟 Features

- **Camera Integration**: Capture face photos directly from your device camera
- **File Upload**: Support for PNG, JPG, and JPEG image formats
- **Real-time Face Swapping**: Powered by InsightFace's state-of-the-art deep learning models
- **User-friendly Interface**: Clean, modern web interface built with Streamlit
- **Download Results**: Save swapped images directly to your device
- **Responsive Design**: Works on various screen sizes

## 🖥️ Demo

[Add screenshots or GIF here]

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/shelwyn/faceswap.git
cd faceswap
```

### Set up Virtual Environment

#### Windows

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### macOS/Linux

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run streamlit_face_swap.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
faceswap/
│
├── streamlit_face_swap.py    # Main application file
├── requirements.txt          # Python dependencies
├── inswapper_128.onnx       # Face swapping model
└── README.md                # Project documentation
```

## 🧠 About the Model

### inswapper_128.onnx

This is the core face swapping model file from InsightFace. The model:

- Is based on deep learning architecture optimized for face swapping
- Provides high-quality face replacement while preserving facial expressions
- Works with the ONNX (Open Neural Network Exchange) format for cross-platform compatibility
- Has been trained on extensive facial datasets to ensure realistic results
- Maintains facial attributes like lighting, angle, and expression from the target image

The model automatically downloads on first run if not present. It's approximately 530MB in size.

## 🔧 Dependencies

- **streamlit**: Web application framework
- **opencv-python-headless**: Computer vision operations
- **insightface**: Face analysis and swapping library
- **onnxruntime**: ONNX model inference
- **Pillow**: Image processing
- **numpy**: Numerical computations

## 📋 Requirements

- Python 3.8+
- Windows/macOS/Linux
- Webcam (optional, for camera capture)
- Minimum 4GB RAM recommended
- GPU support optional (will use CPU if GPU not available)

## 🎯 Usage

1. **Start the application** using the command above
2. **Upload or capture a face photo** in the first column
3. **Upload a target image** in the second column
4. **Click "Swap Faces"** to process the images
5. **Download the result** using the download button

### Tips for Best Results

- Use clear, front-facing photos
- Ensure good lighting in both images
- Avoid extreme angles or partial face visibility
- Higher resolution images produce better results

## ⚠️ Important Notes

- First run will download the face swapping model (~530MB)
- Processing time depends on image size and system performance
- The application requires internet connection for initial model download
- Face detection may fail with obscured or angled faces

## 🐛 Troubleshooting

### Common Issues

1. **Model download fails**
   - Check your internet connection
   - Try manually downloading the model from InsightFace repository

2. **Face not detected**
   - Ensure faces are clearly visible
   - Try different images with better lighting

3. **Performance issues**
   - Reduce image resolution
   - Close other applications to free up memory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the face swapping models
- [Streamlit](https://streamlit.io/) for the web framework
- OpenCV community for computer vision tools

## 📞 Contact

- GitHub: [@shelwyn](https://github.com/shelwyn)
- Repository: [https://github.com/shelwyn/faceswap](https://github.com/shelwyn/faceswap)

---

Made with ❤️ by [shelwyn](https://github.com/shelwyn)
