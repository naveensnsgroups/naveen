import streamlit as st
import os
import tempfile
import cv2
from yolo_inference import load_model, detect_image, detect_video_frame_by_frame

# Initialize the YOLO model
model = load_model()

st.title("YOLO Object Detection")
st.write("Upload an image or video to detect objects.")

# Upload an image or video
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        # Handle image upload and detection
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run detection on the image
        results_df = detect_image(model, file_path)

        # Display results for the image
        st.image(file_path, caption="Uploaded Image", use_column_width=True)
        st.write("Detection Results:")
        st.write(results_df)

    elif uploaded_file.type == "video/mp4":
        # Handle video upload and detection
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Run detection on the video frame-by-frame
        stframe = st.empty()
        output_video_path, results_df = detect_video_frame_by_frame(model, video_path, stframe)

        # Display final annotated video
        st.video(output_video_path)
        st.write("Detection Results:")
        st.write(results_df)
