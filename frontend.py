import streamlit as st
import os
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
import yaml
import seaborn as sns
from ultralytics import YOLO
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import plotly.figure_factory as ff
from collections import Counter
import json

# Set page configuration
st.set_page_config(page_title="Pipeline Defect Detection System", 
                   page_icon="🔍", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #2E86C1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #3498DB;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stProgress .st-bo {
        background-color: #3498DB;
    }
    .highlight {
        background-color: #ffffcc;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing history
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'conf_threshold' not in st.session_state:
    st.session_state.conf_threshold = 0.25
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "trained_yolov8n.pt"

# Load class names
class_names = ["Deformation", "Obstacle", "Rupture", "Disconnect", "Misalignment", "Deposition"]
class_colors = {
    "Deformation": "#FF5733",
    "Obstacle": "#33FF57",
    "Rupture": "#3357FF",
    "Disconnect": "#F033FF",
    "Misalignment": "#FF3366",
    "Deposition": "#33FFF7"
}

# Function to load the model
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to perform inference
def detect_defects(model, image_array, conf_threshold):
    results = model(image_array, conf=conf_threshold)[0]
    return results

# Function to draw bounding boxes on image
def draw_bboxes(image, results):
    img_with_boxes = image.copy()
    
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    
    detections = []
    
    # For debugging - check unique class indices
    unique_classes = np.unique(classes)
    st.write(f"Debug - Unique class indices detected: {unique_classes}")
    
    for box, cls, conf in zip(boxes, classes, confidences):
        # Ensure class index is valid
        if 0 <= cls < len(class_names):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[cls]
            color = class_colors[class_name]
            hex_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            rgb_color = (hex_color[2], hex_color[1], hex_color[0])  # Convert to BGR
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), rgb_color, 2)
            
            # Draw label background
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
            cv2.rectangle(img_with_boxes, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), rgb_color, -1)
            
            # Draw label text
            cv2.putText(img_with_boxes, label, (x1, y1-5), font, font_scale, (255, 255, 255), 1)
            
            # Store detection info
            detections.append({
                'class': class_name,
                'confidence': float(conf),  # Convert to Python float
                'coordinates': (int(x1), int(y1), int(x2), int(y2)),  # Convert to Python int
                'area': float((x2 - x1) * (y2 - y1))  # Convert to Python float
            })
        else:
            st.warning(f"Invalid class index detected: {cls}. Expected range: 0-{len(class_names)-1}")
    
    return img_with_boxes, detections

# Function to create a defect distribution pie chart
def create_defect_pie_chart(detections):
    if not detections:
        return None
    
    class_counts = Counter([d['class'] for d in detections])
    if not class_counts:
        return None
    
    # Debug information
    st.write(f"Debug - Defect types in pie chart: {list(class_counts.keys())}")
    
    fig = px.pie(
        names=list(class_counts.keys()),
        values=list(class_counts.values()),
        title="Defect Type Distribution",
        color=list(class_counts.keys()),
        color_discrete_map=class_colors
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend_title="Defect Types",
        height=400
    )
    
    return fig

# Function to create confidence distribution bar chart
def create_confidence_bar_chart(detections):
    if not detections:
        return None
    
    df = pd.DataFrame(detections)
    
    # Check if there are multiple defect types
    unique_classes = df['class'].unique()
    if len(unique_classes) <= 1:
        st.warning(f"Limited defect diversity detected. Classes found: {unique_classes}")
    
    avg_conf_by_class = df.groupby('class')['confidence'].mean().reset_index()
    
    fig = px.bar(
        avg_conf_by_class,
        x='class',
        y='confidence',
        color='class',
        title="Average Confidence by Defect Type",
        color_discrete_map=class_colors
    )
    fig.update_layout(
        xaxis_title="Defect Type",
        yaxis_title="Average Confidence",
        height=400
    )
    
    return fig

# Function to create a defect size distribution chart
def create_size_distribution(detections):
    if not detections:
        return None
    
    df = pd.DataFrame(detections)
    
    fig = px.box(
        df,
        x='class',
        y='area',
        color='class',
        title="Defect Size Distribution by Type",
        color_discrete_map=class_colors
    )
    fig.update_layout(
        xaxis_title="Defect Type",
        yaxis_title="Area (pixels²)",
        height=400
    )
    
    return fig

# Function to create a heatmap of detections over time
def create_detection_heatmap(history):
    if not history:
        return None
    
    # Flatten all detections
    all_detections = []
    for entry in history:
        for detection in entry['detections']:
            detection_copy = detection.copy()
            detection_copy['timestamp'] = entry['timestamp']
            all_detections.append(detection_copy)
    
    if not all_detections:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_detections)
    
    # Convert timestamps to periods for better visualization
    df['period'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Count defects by type and period
    heatmap_data = df.groupby(['period', 'class']).size().unstack(fill_value=0)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        title="Defect Detection Frequency Over Time",
        labels=dict(x="Defect Type", y="Time Period", color="Count"),
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    
    return fig

# Create sample detections function for demonstration
def create_sample_detections():
    """Create sample detections across different defect types for testing"""
    sample_detections = []
    
    # Create multiple detections of different types
    for class_name in class_names:
        # Create 1-3 detections for each class type
        num_detections = np.random.randint(1, 4)
        for _ in range(num_detections):
            sample_detections.append({
                'class': class_name,
                'confidence': float(np.random.uniform(0.25, 0.95)),
                'coordinates': (
                    int(np.random.randint(10, 100)), 
                    int(np.random.randint(10, 100)),
                    int(np.random.randint(150, 250)),
                    int(np.random.randint(150, 250))
                ),
                'area': float(np.random.randint(1000, 10000))
            })
    
    return sample_detections

# Sidebar for settings and navigation
st.sidebar.markdown("# Pipeline Defect Detection")
st.sidebar.markdown("## Settings")

# Model selection
model_option = st.sidebar.selectbox(
    "Select Model",
    ["trained_yolov8n.pt", "yolov8n_weights.pth"],
    index=0
)

if st.session_state.selected_model != model_option:
    st.session_state.selected_model = model_option

# Confidence threshold slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=st.session_state.conf_threshold,
    step=0.05,
    help="Adjust the confidence threshold for defect detection"
)
st.session_state.conf_threshold = conf_threshold

# Add demo mode option
demo_mode = st.sidebar.checkbox("Enable Demo Mode (show all defect types)", value=False)

# Navigation options
page = st.sidebar.radio(
    "Navigation",
    ["Defect Detection", "Analysis Dashboard", "History & Reports"]
)

# Try to load the selected model
try:
    model = load_model(st.session_state.selected_model)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None
    st.warning("Running in limited mode due to model loading failure.")

# Main content based on selected page
if page == "Defect Detection":
    st.markdown("<h1 class='main-header'>Pipeline Defect Detection System</h1>", unsafe_allow_html=True)
    
    # File upload section
    st.markdown("<h2 class='sub-header'>Upload Image for Defect Detection</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Read and display the original image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Process button
        if st.button("Detect Defects"):
            with st.spinner("Processing image..."):
                if demo_mode:
                    # In demo mode, create sample detections with all defect types
                    st.info("Demo Mode: Showing sample detections of all defect types")
                    detections = create_sample_detections()
                    
                    # Create a copy of the image with sample bounding boxes
                    annotated_img = image_np.copy()
                    for detection in detections:
                        x1, y1, x2, y2 = detection['coordinates']
                        class_name = detection['class']
                        conf = detection['confidence']
                        
                        color = class_colors[class_name]
                        hex_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        rgb_color = (hex_color[2], hex_color[1], hex_color[0])
                        
                        # Draw rectangle and label
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), rgb_color, 2)
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(annotated_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_color, 2)
                        
                else:
                    # Normal mode with actual model detection
                    if model is not None:
                        # Perform inference
                        results = detect_defects(model, image_np, conf_threshold)
                        
                        # Draw bounding boxes
                        annotated_img, detections = draw_bboxes(image_np, results)
                    else:
                        st.error("Model not loaded. Cannot perform detection.")
                        st.stop()
                
                # Record detection in history
                if detections:
                    history_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'image_filename': uploaded_file.name,
                        'detections': detections,
                        'model_used': st.session_state.selected_model,
                        'confidence_threshold': float(conf_threshold)
                    }
                    st.session_state.detection_history.append(history_entry)
                
                # Display annotated image
                with col2:
                    st.image(annotated_img, caption="Detected Defects", use_column_width=True)
                
                # Show detection results
                if detections:
                    st.markdown("<h3>Detection Results</h3>", unsafe_allow_html=True)
                    
                    # Create a dataframe for the detections
                    detection_df = pd.DataFrame(detections)
                    detection_df['coordinates'] = detection_df['coordinates'].apply(lambda x: f"({x[0]}, {x[1]}, {x[2]}, {x[3]})")
                    detection_df = detection_df[['class', 'confidence', 'coordinates', 'area']]
                    detection_df.columns = ['Defect Type', 'Confidence', 'Coordinates (x1,y1,x2,y2)', 'Area (pixels²)']
                    
                    # Display the detection table
                    st.dataframe(detection_df)
                    
                    # Display metrics
                    metrics_cols = st.columns(4)
                    
                    with metrics_cols[0]:
                        st.metric("Total Defects", len(detections))
                    
                    with metrics_cols[1]:
                        avg_conf = np.mean([d['confidence'] for d in detections])
                        st.metric("Avg. Confidence", f"{avg_conf:.2f}")
                    
                    with metrics_cols[2]:
                        most_common = Counter([d['class'] for d in detections]).most_common(1)
                        most_common_defect = most_common[0][0] if most_common else "None"
                        st.metric("Most Common Defect", most_common_defect)
                    
                    with metrics_cols[3]:
                        highest_conf = max([d['confidence'] for d in detections]) if detections else 0
                        st.metric("Highest Confidence", f"{highest_conf:.2f}")
                    
                    # Create and display charts
                    chart_cols = st.columns(2)
                    
                    with chart_cols[0]:
                        pie_chart = create_defect_pie_chart(detections)
                        if pie_chart:
                            st.plotly_chart(pie_chart, use_container_width=True)
                    
                    with chart_cols[1]:
                        conf_chart = create_confidence_bar_chart(detections)
                        if conf_chart:
                            st.plotly_chart(conf_chart, use_container_width=True)
                else:
                    st.info("No defects detected in the image.")

elif page == "Analysis Dashboard":
    st.markdown("<h1 class='main-header'>Pipeline Defect Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Add demo mode for dashboard
    if demo_mode and not st.session_state.detection_history:
        st.info("Demo Mode: Creating sample detection history")
        # Create sample detection history for demo
        for i in range(5):
            sample_entry = {
                'timestamp': (datetime.now() - pd.Timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S"),
                'image_filename': f"sample_image_{i+1}.jpg",
                'detections': create_sample_detections(),
                'model_used': st.session_state.selected_model,
                'confidence_threshold': float(conf_threshold)
            }
            st.session_state.detection_history.append(sample_entry)
    
    # Check if there's detection history
    if not st.session_state.detection_history:
        st.info("No detection data available. Please analyze some images first.")
    else:
        # Extract all defects from history
        all_detections = []
        for entry in st.session_state.detection_history:
            for detection in entry['detections']:
                detection_copy = detection.copy()
                detection_copy['timestamp'] = entry['timestamp']
                detection_copy['image'] = entry['image_filename']
                all_detections.append(detection_copy)
        
        # Create a DataFrame
        df = pd.DataFrame(all_detections)
        
        # Dashboard metrics
        st.markdown("<h2 class='sub-header'>Overview Metrics</h2>", unsafe_allow_html=True)
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Inspected Images", len(st.session_state.detection_history))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Defects Detected", len(all_detections))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with metric_cols[2]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            avg_defects = len(all_detections) / len(st.session_state.detection_history) if st.session_state.detection_history else 0
            st.metric("Avg. Defects per Image", f"{avg_defects:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with metric_cols[3]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            avg_conf = df['confidence'].mean() if not df.empty else 0
            st.metric("Avg. Detection Confidence", f"{avg_conf:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Defect type diversity metric
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        defect_types_detected = df['class'].nunique() if not df.empty else 0
        st.metric("Defect Types Detected", f"{defect_types_detected}/{len(class_names)}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Defect type distribution
        st.markdown("<h2 class='sub-header'>Defect Analysis</h2>", unsafe_allow_html=True)
        
        chart_cols = st.columns(2)
        
        with chart_cols[0]:
            # Defect distribution pie chart
            fig_pie = create_defect_pie_chart(all_detections)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_cols[1]:
            # Confidence by defect type
            fig_conf = create_confidence_bar_chart(all_detections)
            if fig_conf:
                st.plotly_chart(fig_conf, use_container_width=True)
        
        # Size distribution and time analysis
        chart_cols2 = st.columns(2)
        
        with chart_cols2[0]:
            # Size distribution
            fig_size = create_size_distribution(all_detections)
            if fig_size:
                st.plotly_chart(fig_size, use_container_width=True)
        
        with chart_cols2[1]:
            # Time-based heatmap
            fig_heatmap = create_detection_heatmap(st.session_state.detection_history)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Detailed defect table
        st.markdown("<h2 class='sub-header'>Detailed Defect Data</h2>", unsafe_allow_html=True)
        
        # Create a visualization of defect frequency
        if not df.empty:
            class_counts = df['class'].value_counts().reset_index()
            class_counts.columns = ['Defect Type', 'Count']
            
            fig = px.bar(
                class_counts,
                x='Defect Type',
                y='Count',
                color='Defect Type',
                title="Defect Frequency",
                color_discrete_map=class_colors
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the dataframe for all detections
            if 'coordinates' in df.columns:
                df['coordinates'] = df['coordinates'].apply(lambda x: f"({x[0]}, {x[1]}, {x[2]}, {x[3]})")
            
            # Filter and rename columns for display
            display_df = df[['class', 'confidence', 'area', 'image', 'timestamp']].copy()
            display_df.columns = ['Defect Type', 'Confidence', 'Area (pixels²)', 'Image', 'Timestamp']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download option
            csv = display_df.to_csv(index=False)
            st.download_button(
                "Download Detection Data as CSV",
                csv,
                "pipeline_defect_data.csv",
                "text/csv",
                key='download-csv'
            )

elif page == "History & Reports":
    st.markdown("<h1 class='main-header'>Detection History & Reports</h1>", unsafe_allow_html=True)
    
    # Add demo mode for history
    if demo_mode and not st.session_state.detection_history:
        st.info("Demo Mode: Creating sample detection history")
        # Create sample detection history for demo
        for i in range(5):
            sample_entry = {
                'timestamp': (datetime.now() - pd.Timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S"),
                'image_filename': f"sample_image_{i+1}.jpg",
                'detections': create_sample_detections(),
                'model_used': st.session_state.selected_model,
                'confidence_threshold': float(conf_threshold)
            }
            st.session_state.detection_history.append(sample_entry)
    
    if not st.session_state.detection_history:
        st.info("No detection history available. Please analyze some images first.")
    else:
        # Create a summary of the detection history
        history_summary = []
        for i, entry in enumerate(st.session_state.detection_history):
            defect_counts = Counter([d['class'] for d in entry['detections']])
            total_defects = len(entry['detections'])
            avg_conf = np.mean([d['confidence'] for d in entry['detections']]) if entry['detections'] else 0
            
            summary = {
                'ID': i + 1,
                'Timestamp': entry['timestamp'],
                'Image': entry['image_filename'],
                'Total Defects': total_defects,
                'Avg. Confidence': f"{avg_conf:.2f}",
                'Defect Summary': ", ".join([f"{k}: {v}" for k, v in defect_counts.items()])
            }
            history_summary.append(summary)
        
        # Display the history summary
        df_history = pd.DataFrame(history_summary)
        st.dataframe(df_history, use_container_width=True)
        
        # Option to select specific detection for detailed view
        selected_id = st.selectbox(
            "Select a detection for detailed report",
            options=df_history['ID'].tolist(),
            format_func=lambda x: f"ID {x}: {df_history[df_history['ID'] == x]['Image'].values[0]} ({df_history[df_history['ID'] == x]['Timestamp'].values[0]})"
        )
        
        if selected_id:
            # Get the selected entry
            selected_entry = st.session_state.detection_history[selected_id - 1]
            
            st.markdown("<h2 class='sub-header'>Detailed Report</h2>", unsafe_allow_html=True)
            
            # Create columns for report information
            info_cols = st.columns(3)
            
            with info_cols[0]:
                st.markdown("### Image Information")
                st.markdown(f"**Filename:** {selected_entry['image_filename']}")
                st.markdown(f"**Analysis Date:** {selected_entry['timestamp']}")
                st.markdown(f"**Model Used:** {selected_entry['model_used']}")
                st.markdown(f"**Confidence Threshold:** {selected_entry['confidence_threshold']}")
            
            with info_cols[1]:
                st.markdown("### Detection Summary")
                defect_counts = Counter([d['class'] for d in selected_entry['detections']])
                
                for defect, count in defect_counts.items():
                    st.markdown(f"**{defect}:** {count}")
                
                total_defects = len(selected_entry['detections'])
                st.markdown(f"**Total Defects:** {total_defects}")
            
            with info_cols[2]:
                st.markdown("### Statistics")
                if selected_entry['detections']:
                    confidences = [d['confidence'] for d in selected_entry['detections']]
                    areas = [d['area'] for d in selected_entry['detections']]
                    
                    st.markdown(f"**Avg. Confidence:** {np.mean(confidences):.2f}")
                    st.markdown(f"**Max. Confidence:** {np.max(confidences):.2f}")
                    st.markdown(f"**Min. Confidence:** {np.min(confidences):.2f}")
                    st.markdown(f"**Avg. Defect Size:** {np.mean(areas):.2f} pixels²")
                else:
                    st.markdown("No defects detected.")
            
            # Show detection visualizations
            if selected_entry['detections']:
                # Create visualizations for the selected detection
                chart_cols = st.columns(2)
                
                with chart_cols[0]:
                    # Defect distribution pie chart
                    fig_pie = create_defect_pie_chart(selected_entry['detections'])
                    if fig_pie:
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with chart_cols[1]:
                    # Confidence by defect type
                    fig_conf = create_confidence_bar_chart(selected_entry['detections'])
                    if fig_conf:
                        st.plotly_chart(fig_conf, use_container_width=True)
                
                # Show detailed detection table
                st.markdown("### Detection Details")
                detection_df = pd.DataFrame(selected_entry['detections'])
                
                if 'coordinates' in detection_df.columns:
                    detection_df['coordinates'] = detection_df['coordinates'].apply(lambda x: f"({x[0]}, {x[1]}, {x[2]}, {x[3]})")
                
                detection_df = detection_df[['class', 'confidence', 'coordinates', 'area']]
                detection_df.columns = ['Defect Type', 'Confidence', 'Coordinates (x1,y1,x2,y2)', 'Area (pixels²)']
                
                st.dataframe(detection_df, use_container_width=True)
                
                # Option to download report
                report_data = {
                    'Image': selected_entry['image_filename'],
                    'Analysis Date': selected_entry['timestamp'],
                    'Model Used': selected_entry['model_used'],
                    'Confidence Threshold': float(selected_entry['confidence_threshold']),
                    'Total Defects': len(selected_entry['detections']),
                    'Defect Counts': {k: int(v) for k, v in dict(defect_counts).items()},  # Convert Counter values to int
                    'Detections': []
                }
                
                # Convert all NumPy types to Python native types for JSON serialization
                for detection in selected_entry['detections']:
                    report_data['Detections'].append({
                        'class': detection['class'],
                        'confidence': float(detection['confidence']),
                        'coordinates': [int(c) for c in detection['coordinates']],
                        'area': float(detection['area'])
                    })
                
                # Convert to JSON string
                report_json = json.dumps(report_data, indent=4)
                
                st.download_button(
                    "Download Detailed Report (JSON)",
                    report_json,
                    f"defect_report_{selected_id}_{selected_entry['image_filename'].split('.')[0]}.json",
                    "application/json",
                    key='download-report'
                )
            else:
                st.info("No defects were detected in this image.")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7F8487; padding: 10px;">
        <p>Pipeline Defect Detection System | Created with Streamlit</p>
        <p>© 2025 | Version 1.0.0</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Add information about model and system capabilities
with st.expander("About This System"):
    st.markdown("""
    ### About the Pipeline Defect Detection System
    
    This application uses advanced computer vision with YOLOv8 to detect various types of defects in pipeline infrastructure:
    
    - **Deformation**: Physical changes in pipeline shape
    - **Obstacle**: Foreign objects obstructing the pipeline
    - **Rupture**: Breaks or cracks in the pipeline structure
    - **Disconnect**: Separated pipe sections
    - **Misalignment**: Improperly aligned pipe sections
    - **Deposition**: Buildup of material inside the pipeline
    
    #### How to Use
    1. Upload an image of a pipeline section
    2. Adjust the confidence threshold if needed
    3. Click "Detect Defects" to analyze the image
    4. Review results in the detection view
    5. Access the dashboard for historical analysis
    
    #### Technical Information
    The system is built on YOLOv8, a state-of-the-art object detection model, fine-tuned specifically for pipeline defect detection.
    """)

# Add help section
with st.sidebar.expander("Help & Support"):
    st.markdown("""
    ### Need Help?
    
    - **Image Requirements**: For best results, use clear, well-lit images of pipeline interiors.
    - **Confidence Threshold**: Lower values detect more defects but may increase false positives.
    - **Demo Mode**: Enable this to see examples of all defect types.
    
    For additional support, please contact technical support.
    """)

# Add performance metrics section in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### System Performance")

# Simple performance metrics display
if model is not None:
    st.sidebar.success("✅ Model loaded successfully")
else:
    st.sidebar.error("❌ Model not loaded")

# Memory usage
import psutil
memory_usage = psutil.Process().memory_info().rss / 1024**2  # in MB
st.sidebar.metric("Memory Usage", f"{memory_usage:.1f} MB")

# Add a version info and system status
st.sidebar.markdown("---")
st.sidebar.markdown(f"**System Status**: Active")
st.sidebar.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}")
st.sidebar.markdown(f"**Version**: 1.0.0")