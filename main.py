import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Tuple
import time
import base64
from io import BytesIO

# Page config
st.set_page_config(
    page_title="SAM2 + SmolVLM Tracker",
    page_icon="🎬",
    layout="wide"
)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TrackedObject:
    obj_id: int
    current_action: str = ""
    action_history: List[Tuple[int, str]] = field(default_factory=list)
    frame_count: int = 0

# ============================================================================
# SMOLVLM ACTION RECOGNIZER
# ============================================================================

@st.cache_resource
def load_smolvlm():
    """Load SmolVLM model (cached)"""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    return processor, model

class SmolVLMActionRecognizer:
    """Action recognition using SmolVLM"""
    
    def __init__(self):
        self.processor, self.model = load_smolvlm()
    
    def recognize_action(self, frame):
        """Recognize action from a frame crop"""
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe what this person or object is doing in 1-2 sentences."}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = self.processor(
            text=prompt,
            images=[frame],
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        generated_ids = output[0][inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        action = response.strip()
        
        for prefix in ["Assistant:", "Answer:"]:
            if action.lower().startswith(prefix.lower()):
                action = action[len(prefix):].strip()
        
        action = action.strip('"\'')
        
        if action:
            action = action[0].upper() + action[1:]
        
        return action

# ============================================================================
# SAM2 INITIALIZATION
# ============================================================================

@st.cache_resource
def load_sam2(checkpoint_path, model_cfg="sam2_hiera_l.yaml"):
    """Load SAM2 predictor (cached)"""
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path)
    return predictor

def download_sam2_checkpoint(model_size="large"):
    """Download SAM2 checkpoint if not exists"""
    import urllib.request
    
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    checkpoint_urls = {
        "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "base": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    }
    
    checkpoint_path = f"{checkpoints_dir}/sam2_hiera_{model_size}.pt"
    
    if not os.path.exists(checkpoint_path):
        with st.spinner(f"Downloading SAM2 {model_size} checkpoint..."):
            urllib.request.urlretrieve(checkpoint_urls[model_size], checkpoint_path)
    
    return checkpoint_path

# ============================================================================
# SAM2 + SMOLVLM TRACKER
# ============================================================================

class SAM2SmolVLMTracker:
    """Combined SAM2 tracking with SmolVLM action recognition"""
    
    def __init__(self, sam2_predictor):
        self.sam2_predictor = sam2_predictor
        self.action_recognizer = SmolVLMActionRecognizer()
        self.tracked_objects = {}
    
    def track_with_actions(self, video_path, initial_points, initial_labels,
                          obj_id=1, action_update_interval=30, progress_bar=None):
        """Track object and recognize actions"""
        
        # Initialize SAM2
        inference_state = self.sam2_predictor.init_state(video_path=video_path)
        
        self.sam2_predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=initial_points,
            labels=initial_labels,
        )
        
        self.tracked_objects[obj_id] = TrackedObject(obj_id=obj_id)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_segments = {}
        action_timeline = {}
        
        frame_idx = 0
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(
            inference_state
        ):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
            if obj_id in video_segments[out_frame_idx]:
                mask = video_segments[out_frame_idx][obj_id][0]
                tracked_obj = self.tracked_objects[obj_id]
                tracked_obj.frame_count += 1
                
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    x1, x2 = max(0, xs.min()-10), min(frame_rgb.shape[1], xs.max()+10)
                    y1, y2 = max(0, ys.min()-10), min(frame_rgb.shape[0], ys.max()+10)
                    object_crop = frame_rgb[y1:y2, x1:x2]
                    
                    if out_frame_idx % action_update_interval == 0:
                        action = self.action_recognizer.recognize_action(object_crop)
                        tracked_obj.current_action = action
                        tracked_obj.action_history.append((out_frame_idx, action))
                    
                    action_timeline[out_frame_idx] = tracked_obj.current_action
            
            if progress_bar:
                progress_bar.progress((out_frame_idx + 1) / total_frames)
            
            frame_idx += 1
        
        cap.release()
        return video_segments, action_timeline, self.tracked_objects[obj_id]

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_tracking_with_actions(video_path, video_segments, action_timeline,
                                   obj_id, output_path='tracked_output.mp4', progress_bar=None):
    """Create video with tracking and action labels"""
    import supervision as sv
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN, opacity=0.3)
    box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.BLACK,
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1,
        text_position=sv.Position.TOP_CENTER,
        text_padding=8,
        border_radius=4
    )
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in video_segments and obj_id in video_segments[frame_idx]:
            mask = video_segments[frame_idx][obj_id][0]
            
            ys, xs = np.where(mask)
            if len(xs) > 0:
                x1, y1 = xs.min(), ys.min()
                x2, y2 = xs.max(), ys.max()
                
                detections = sv.Detections(
                    xyxy=np.array([[x1, y1, x2, y2]]),
                    mask=np.array([mask]),
                    class_id=np.array([obj_id])
                )
                
                frame = mask_annotator.annotate(frame, detections)
                frame = box_annotator.annotate(frame, detections)
                
                if frame_idx in action_timeline:
                    action = action_timeline[frame_idx]
                    labels = [action]
                    frame = label_annotator.annotate(frame, detections, labels)
        
        out.write(frame)
        frame_idx += 1
        
        if progress_bar:
            progress_bar.progress(frame_idx / total_frames)
    
    cap.release()
    out.release()

# ============================================================================
# INTERACTIVE IMAGE COMPONENT
# ============================================================================

def get_image_base64(img_array):
    """Convert numpy array to base64 for HTML display"""
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def create_clickable_image(frame_rgb, points, width, height):
    """Create an HTML clickable image with JavaScript for point selection"""
    img_base64 = get_image_base64(frame_rgb)
    
    # Draw points on the image
    display_frame = frame_rgb.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(display_frame, (int(x), int(y)), 10, (0, 255, 0), -1)
        cv2.circle(display_frame, (int(x), int(y)), 12, (255, 255, 255), 2)
        cv2.putText(display_frame, f'P{i+1}', (int(x)+15, int(y)-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    img_with_points_base64 = get_image_base64(display_frame)
    
    html = f"""
    <div style="position: relative; display: inline-block;">
        <img id="video-frame" src="data:image/png;base64,{img_with_points_base64}" 
             style="cursor: crosshair; max-width: 100%; border: 2px solid #4CAF50; border-radius: 5px;"
             onclick="handleClick(event)">
        <div id="coords" style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; font-family: monospace;">
            Click on the image to add points
        </div>
    </div>
    
    <script>
        function handleClick(event) {{
            const img = event.target;
            const rect = img.getBoundingClientRect();
            
            // Calculate click position relative to actual image dimensions
            const scaleX = {width} / rect.width;
            const scaleY = {height} / rect.height;
            
            const x = Math.round((event.clientX - rect.left) * scaleX);
            const y = Math.round((event.clientY - rect.top) * scaleY);
            
            // Update the coords display
            document.getElementById('coords').innerHTML = 
                `Clicked at: X = ${{x}}, Y = ${{y}}<br>
                <span style="color: #4CAF50; font-weight: bold;">
                ⬇️ Enter these coordinates below to add the point
                </span>`;
            
            // Update the Streamlit number inputs
            const xInput = window.parent.document.querySelector('input[aria-label="X coordinate"]');
            const yInput = window.parent.document.querySelector('input[aria-label="Y coordinate"]');
            
            if (xInput) {{
                xInput.value = x;
                xInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
            if (yInput) {{
                yInput.value = y;
                yInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
        }}
    </script>
    """
    
    return html

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("🎬 SAM2 + SmolVLM Object Tracker")
    st.markdown("Track objects in videos and recognize their actions using SAM2 and SmolVLM")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_size = st.selectbox(
            "SAM2 Model Size",
            ["tiny", "small", "base", "large"],
            index=3
        )
        
        action_interval = st.slider(
            "Action Update Interval (frames)",
            min_value=10,
            max_value=60,
            value=30,
            step=10
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload a video
        2. Click on the image to select points
        3. Click "Add Point" to confirm
        4. Add multiple points if needed
        5. Click "Start Tracking"
        """)
    
    # Main content
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Display first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            cap.release()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📹 First Frame - Click to Select Point")
                
                # Initialize session state
                if 'points' not in st.session_state:
                    st.session_state.points = []
                if 'labels' not in st.session_state:
                    st.session_state.labels = []
                if 'clicked_x' not in st.session_state:
                    st.session_state.clicked_x = width // 2
                if 'clicked_y' not in st.session_state:
                    st.session_state.clicked_y = height // 2
                
                # Display clickable image
                html_content = create_clickable_image(frame_rgb, st.session_state.points, width, height)
                st.components.v1.html(html_content, height=min(height + 100, 700))
                
                st.caption(f"Video size: {width}x{height}")
            
            with col2:
                st.subheader("🎯 Add Points")
                
                # Point input (will be auto-filled by clicking)
                x_coord = st.number_input("X coordinate", 
                                         min_value=0, 
                                         max_value=width-1, 
                                         value=st.session_state.clicked_x,
                                         key="x_input")
                y_coord = st.number_input("Y coordinate", 
                                         min_value=0, 
                                         max_value=height-1, 
                                         value=st.session_state.clicked_y,
                                         key="y_input")
                
                st.session_state.clicked_x = x_coord
                st.session_state.clicked_y = y_coord
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("➕ Add Point", use_container_width=True):
                        new_point = [x_coord, y_coord]
                        # Check for duplicates
                        is_duplicate = False
                        for existing_point in st.session_state.points:
                            if abs(existing_point[0] - new_point[0]) < 5 and abs(existing_point[1] - new_point[1]) < 5:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            st.session_state.points.append(new_point)
                            st.session_state.labels.append(1)
                            st.success(f"Point added at ({x_coord}, {y_coord})")
                            st.rerun()
                        else:
                            st.warning("Point too close to existing point!")
                
                with col_b:
                    if st.button("🗑️ Clear All", use_container_width=True):
                        st.session_state.points = []
                        st.session_state.labels = []
                        st.rerun()
                
                # Display points
                st.markdown("---")
                st.markdown(f"**Points Added:** {len(st.session_state.points)}")
                if st.session_state.points:
                    for i, point in enumerate(st.session_state.points):
                        st.text(f"P{i+1}: ({int(point[0])}, {int(point[1])})")
                else:
                    st.info("👆 Click on the image above, then click 'Add Point'")
                
                # Start tracking button
                st.markdown("---")
                if len(st.session_state.points) > 0:
                    if st.button("🚀 Start Tracking", type="primary", use_container_width=True):
                        with st.spinner("Initializing models..."):
                            # Download checkpoint
                            checkpoint_path = download_sam2_checkpoint(model_size)
                            
                            # Load models
                            sam2_predictor = load_sam2(checkpoint_path)
                            tracker = SAM2SmolVLMTracker(sam2_predictor)
                        
                        # Convert points
                        points = np.array(st.session_state.points, dtype=np.float32)
                        labels = np.array(st.session_state.labels, dtype=np.int32)
                        
                        # Track
                        st.subheader("🔄 Tracking Progress")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Tracking objects...")
                        segments, timeline, tracked_obj = tracker.track_with_actions(
                            video_path,
                            points,
                            labels,
                            action_update_interval=action_interval,
                            progress_bar=progress_bar
                        )
                        
                        # Generate output
                        status_text.text("Creating output video...")
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        visualize_tracking_with_actions(
                            video_path,
                            segments,
                            timeline,
                            obj_id=1,
                            output_path=output_path,
                            progress_bar=progress_bar
                        )
                        
                        status_text.text("✅ Complete!")
                        
                        # Display results
                        st.subheader("📊 Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Frames", tracked_obj.frame_count)
                        with col2:
                            st.metric("Actions Detected", len(tracked_obj.action_history))
                        with col3:
                            st.metric("Update Interval", f"{action_interval} frames")
                        
                        # Action timeline
                        st.subheader("📝 Action Timeline")
                        for frame_idx, action in tracked_obj.action_history:
                            st.markdown(f"**Frame {frame_idx}:** {action}")
                        
                        # Download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Tracked Video",
                                data=f,
                                file_name="tracked_output.mp4",
                                mime="video/mp4"
                            )
                else:
                    st.warning("⚠️ Add at least one point to start tracking")

if __name__ == "__main__":
    main()
