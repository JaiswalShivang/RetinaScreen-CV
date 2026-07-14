import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- 1. Page Configuration ---
st.set_page_config(page_title="OculAI Diagnostics", page_icon="👁️", layout="wide")

# --- 2. Compact CSS Injection (Just for font and title gradient) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    .gradient-text {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Load Model Safely ---
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('best_oculai_model.keras')

try:
    model = load_ai_model()
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
except Exception as e:
    st.error(f"Error loading model: Ensure 'best_oculai_model.keras' is in this folder. Details: {e}")
    st.stop()

# --- 4. Robust Grad-CAM Engine ---
def make_gradcam_heatmap(img_array, sequential_model, last_conv_layer_name="top_conv"):
    base_net = sequential_model.layers[0]
    base_grad_model = tf.keras.models.Model(
        inputs=base_net.input,
        outputs=[base_net.get_layer(last_conv_layer_name).output, base_net.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_features = base_grad_model(img_array)
        x = base_features
        for layer in sequential_model.layers[1:]:
            x = layer(x)
        preds = x
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy(), preds[0].numpy()

def overlay_heatmap(img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# --- 6. Main Application Dashboard ---
st.markdown("<div style='text-align: center;'><span class='gradient-text'>OculAI Retinal Screener</span></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8E9BB5;'>Upload an Optical Coherence Tomography (OCT) scan to receive real-time classification and visual localization.</p>", unsafe_allow_html=True)

# Upload Section
uploader_col = st.columns([1, 2, 1])[1]
with uploader_col:
    with st.container(border=True):
        st.subheader("📷 Scan Input")
        uploaded_file = st.file_uploader("Upload patient OCT scan:", type=["jpg", "jpeg", "png"])

# Determine input source
image = None
source_desc = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    source_desc = "Uploaded Patient Scan"

# If image is ready, run pipeline
if image is not None:
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Processing Neural Network Diagnostics..."):
        heatmap, predictions = make_gradcam_heatmap(img_array, model)
        pred_idx = np.argmax(predictions)
        confidence = predictions[pred_idx] * 100
        gradcam_img = overlay_heatmap(np.array(image), heatmap)
        pred_class = class_names[pred_idx]

    # Clinical Result Notification Banner
    banner_text = f"**AI Diagnostic Screening Result: {pred_class} Detected** (Confidence: {confidence:.2f}%)"
    if pred_class in ['CNV', 'DME']:
        st.error(banner_text)
    elif pred_class == 'DRUSEN':
        st.warning(banner_text)
    else:
        st.success(banner_text)

    # Images Columns
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown(f"**📷 {source_desc}**")
            st.image(image, width='stretch')
    with col2:
        with st.container(border=True):
            st.markdown("**🔍 AI Focus Localization (Grad-CAM)**")
            st.image(gradcam_img, width='stretch')

    # Detailed statistics
    col_stats, col_recs = st.columns(2)
    with col_stats:
        with st.container(border=True):
            st.subheader("📊 Confidence Breakdown")
            for i, name in enumerate(class_names):
                val = float(predictions[i])
                st.progress(val, text=f"{name}: {val*100:.2f}%")

    with col_recs:
        with st.container(border=True):
            st.subheader("📋 Recommendation")
            if pred_class in ['CNV', 'DME']:
                st.markdown("""
                **🚨 URGENT:** Structural fluid accumulation/neovascular activity detected. 
                - **Action:** Referral to a retina specialist for clinical evaluation.
                - **Routine:** Monitor central vision regularly.
                """)
            elif pred_class == 'DRUSEN':
                st.markdown("""
                **⚠️ MONITOR:** Early signs of dry AMD (drusen deposits).
                - **Action:** Schedule routine checkups with an ophthalmologist (6-12 months).
                - **Lifestyle:** Consider sun protection and lifestyle counseling.
                """)
            else:
                st.markdown("""
                **✅ ROUTINE:** Retinal layers appear healthy with no detectable cysts or anomalies.
                - **Action:** Maintain standard annual eye examinations.
                """)
else:
    # Awaiting state
    with st.container(border=True):
        st.markdown(
            """
             <div style="text-align: center; padding: 40px 0;">
                <h3 style="margin-bottom: 10px;">Awaiting Retinal OCT Scan</h3>
                <p style="color: #8E9BB5;">Upload a patient OCT scan above to begin the diagnostic analysis.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Glossary Expander
st.markdown("---")
with st.expander("📖 Disease Glossary & Reference Guide"):
    st.markdown("""
    - **CNV (Choroidal Neovascularization)**: Abnormal blood vessels growing under the retina. Causes rapid, severe vision loss.
    - **DME (Diabetic Macular Edema)**: Macular fluid build-up due to diabetes.
    - **DRUSEN**: Early age-related macular degeneration deposits.
    - **NORMAL**: Healthy retinal structures.
    """)