import streamlit as st
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import tempfile
import os
import io

# Page configuration
st.set_page_config(page_title="Face Swap App", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    div[data-testid="column"] {
        background-color: #f9f9f9;
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        min-height: 500px;
    }
    .stImage {
        max-width: 100%;
        margin: 20px auto;
        display: block;
    }
    h3 {
        text-align: center;
        color: #333;
        margin-bottom: 20px;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        margin-top: 10px;
    }
    .stDownloadButton > button {
        width: 100%;
        background-color: #008CBA;
        color: white;
        font-weight: bold;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'face_image' not in st.session_state:
    st.session_state.face_image = None
if 'swapped_image' not in st.session_state:
    st.session_state.swapped_image = None
if 'target_image' not in st.session_state:
    st.session_state.target_image = None

def resize_image_for_display(image, max_width=280, max_height=350):
    """Resize image to fit within specified dimensions while maintaining aspect ratio"""
    width, height = image.size
    aspect_ratio = width / height
    
    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def perform_face_swap(face_image, target_image):
    """Perform face swap using InsightFace"""
    try:
        # Initialize InsightFace
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Convert PIL images to cv2 format
        face_cv2 = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
        target_cv2 = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
        
        # Get faces from both images
        face_faces = app.get(face_cv2)
        target_faces = app.get(target_cv2)
        
        if not face_faces:
            st.error("No face detected in the face image")
            return None
            
        if not target_faces:
            st.error("No face detected in the target image")
            return None
        
        # Get the face to swap (source face)
        source_face = face_faces[0]
        
        # Initialize the swapper model
        swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
        
        # Perform face swap on all faces in target image
        result = target_cv2.copy()
        for target_face in target_faces:
            result = swapper.get(result, target_face, source_face, paste_back=True)
        
        # Convert back to RGB for PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
        
    except Exception as e:
        st.error(f"Error during face swap: {str(e)}")
        return None

def main():
    # Title
    st.title("🔄 Face Swap Application")
    st.markdown("Upload or capture your face photo and select a target image to swap faces!")
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📸 Face Photo")
        
        # Option to use camera or upload
        capture_method = st.radio("Choose input method:", ("Camera Capture", "Upload Image"), key="face_method")
        
        if capture_method == "Camera Capture":
            camera_image = st.camera_input("Take a picture", key="camera_input")
            
            if camera_image is not None:
                st.session_state.face_image = Image.open(camera_image)
                resized_face = resize_image_for_display(st.session_state.face_image)
                st.image(resized_face, use_container_width=False)
        else:
            uploaded_face = st.file_uploader("Upload face image", type=['png', 'jpg', 'jpeg'], key="face_upload")
            
            if uploaded_face is not None:
                st.session_state.face_image = Image.open(uploaded_face)
                resized_face = resize_image_for_display(st.session_state.face_image)
                st.image(resized_face, use_container_width=False)
        
        if st.session_state.face_image is None:
            st.info("No face image selected")
    
    with col2:
        st.markdown("### 🖼️ Target Image")
        
        uploaded_target = st.file_uploader("Upload target image", type=['png', 'jpg', 'jpeg'], key="target_upload")
        
        if uploaded_target is not None:
            st.session_state.target_image = Image.open(uploaded_target)
            resized_target = resize_image_for_display(st.session_state.target_image)
            st.image(resized_target, use_container_width=False)
        else:
            st.info("Please upload a target image")
    
    with col3:
        st.markdown("### ✨ Result")
        
        # Face swap button
        if st.button("🔄 Swap Faces", type="primary", use_container_width=True):
            if st.session_state.face_image is not None and st.session_state.target_image is not None:
                with st.spinner("Performing face swap..."):
                    result = perform_face_swap(st.session_state.face_image, st.session_state.target_image)
                    
                    if result is not None:
                        st.session_state.swapped_image = result
                        st.success("Face swap completed!")
            else:
                st.error("Please provide both face and target images")
        
        if st.session_state.swapped_image is not None:
            resized_result = resize_image_for_display(st.session_state.swapped_image)
            st.image(resized_result, use_container_width=False)
            
            # Download button
            buf = io.BytesIO()
            st.session_state.swapped_image.save(buf, format='PNG')
            st.download_button(
                label="📥 Download Result",
                data=buf.getvalue(),
                file_name="face_swapped_result.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.info("Click 'Swap Faces' to see the result")

if __name__ == "__main__":
    main()