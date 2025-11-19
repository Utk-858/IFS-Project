# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
import hashlib
import matplotlib.pyplot as plt
import io
import os

st.set_page_config(page_title="Dual Watermarking Demo (CPU GAN-approx)", layout="wide")

# ----------------------------
# Helpers: load model safely
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path="model.pth"):
    if not os.path.exists(path):
        return None, f"Model file not found at {path}. Please place model.pth in this folder."
    try:
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        state = torch.load(path, map_location=device)
        # if user saved full model instead of state_dict, handle gracefully
        if isinstance(state, dict) and any(k.startswith('module.') or k in model.state_dict() for k in state.keys()):
            model.load_state_dict(state)
        else:
            # attempt to load weights-only fallback
            try:
                model.load_state_dict(state)
            except Exception:
                # user's file might be a full checkpoint containing 'model_state_dict'
                if 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'])
                else:
                    return None, "Failed to interpret model.pth — save state_dict(model.state_dict())."
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading model.pth: {e}"

model, load_error = load_model("model.pth")

# ----------------------------
# Transforms and labels
# ----------------------------
transform_eval = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
labels = ["No Watermark / Tampered", "Watermark Present"]

def predict_image(np_img):
    """Returns (label_str, confidence_float). Accepts uint8 HWC RGB np array."""
    if model is None:
        return "Model Missing", 0.0
    pil = Image.fromarray(np_img.astype(np.uint8))
    x = transform_eval(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    pred = int(out.argmax(1).item())
    prob = float(torch.softmax(out, dim=1)[0][pred].item())
    return labels[pred], prob

# ----------------------------
# Watermark functions
# ----------------------------
def bits_from_string(s, length):
    h = hashlib.sha256(s.encode()).digest()
    out = []
    for byte in h:
        for i in range(8):
            out.append((byte >> i) & 1)
            if len(out) >= length: return out
    return out + [0]*(length - len(out))

def embed_dct_robust(img_np, payload_bits, block=8, alpha=8):
    # expects img_np RGB uint8
    ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = ycbcr[:,:,0]
    h, w = Y.shape
    ph = int(np.ceil(h/block)*block)
    pw = int(np.ceil(w/block)*block)
    padded = np.pad(Y, ((0, ph-h),(0, pw-w)), mode="reflect")
    bit_idx = 0
    for i in range(0, ph, block):
        for j in range(0, pw, block):
            blockY = padded[i:i+block, j:j+block]
            d = cv2.dct(blockY)
            if bit_idx < len(payload_bits):
                d[1,0] += alpha if payload_bits[bit_idx] else -alpha
                bit_idx += 1
            padded[i:i+block, j:j+block] = cv2.idct(d)
            if bit_idx >= len(payload_bits):
                break
        if bit_idx >= len(payload_bits):
            break
    ycbcr[:,:,0] = padded[:h, :w]
    out = cv2.cvtColor(ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return out

def embed_lsb_fragile(img_np, frag_bits, region=(0,0,64,64)):
    x,y,w,h = region
    out = img_np.copy()
    g = out[y:y+h, x:x+w, 1].copy()
    flat = g.flatten()
    for i, b in enumerate(frag_bits):
        if i >= flat.size: break
        flat[i] = (flat[i] & 0xFE) | (b & 1)
    out[y:y+h, x:x+w, 1] = flat.reshape((h,w))
    return out

def fragile_diff_map(orig_img, attacked_img, region=(0,0,64,64)):
    x,y,w,h = region
    g1 = orig_img[y:y+h, x:x+w, 1]
    g2 = attacked_img[y:y+h, x:x+w, 1]
    diff_bits = (g1 & 1) ^ (g2 & 1)
    return diff_bits.astype(np.float32)

# ----------------------------
# Fast CPU "GAN-approx" operations
# ----------------------------
def simulate_diffusion_noise(img_rgb, strength=0.2):
    """Add texture-level noise + bilateral smoothing to mimic diffusion remapping."""
    img = img_rgb.copy().astype(np.float32)
    # add subtle per-pixel gaussian noise
    sigma = 6.0 * strength
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img = img + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    # apply bilateral filter to smooth but keep edges (like denoising step)
    # cv2.bilateralFilter expects BGR
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    den = cv2.bilateralFilter(bgr, d=7, sigmaColor=75*strength+1, sigmaSpace=75*strength+1)
    return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)

def esrgan_like_updown(img_rgb, scale=2):
    """Fast upsample + sharpen + downsample to mimic super-resolution texture changes."""
    h,w = img_rgb.shape[:2]
    up = cv2.resize(img_rgb, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    # unsharp mask (sharpen)
    blur = cv2.GaussianBlur(up, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    down = cv2.resize(sharp, (w, h), interpolation=cv2.INTER_AREA)
    return down

def pixel_shift_texture(img_rgb, max_shift=3):
    """Random small pixel shifts / jitter in local patches."""
    img = img_rgb.copy()
    h,w = img.shape[:2]
    # split to 8x8 patches and randomly shift a few
    ph, pw = 16, 16
    for y in range(0,h,ph):
        for x in range(0,w,pw):
            if np.random.rand() < 0.08:  # shift only some patches
                dy = np.random.randint(-max_shift, max_shift+1)
                dx = np.random.randint(-max_shift, max_shift+1)
                src_y = np.clip(y, 0, h-ph)
                src_x = np.clip(x, 0, w-pw)
                patch = img[src_y:src_y+ph, src_x:src_x+pw].copy()
                ty = np.clip(src_y+dy, 0, h-ph)
                tx = np.clip(src_x+dx, 0, w-pw)
                img[ty:ty+ph, tx:tx+pw] = patch
    return img

def texture_regen(img_rgb):
    """Add mild texture re-synthesis: local contrast/stretch + slight HSV change."""
    img = img_rgb.copy().astype(np.float32)
    # local CLAHE on L channel
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    newl = clahe.apply(l)
    lab2 = cv2.merge([newl,a,b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    # small HSV perturb
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (0.95 + np.random.rand()*0.1), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out

def fake_gan_regenerate(img_rgb, strength=0.4):
    """Compose the CPU-friendly GAN-approx attack pipeline."""
    # 1) slight texture re-synthesis
    t1 = texture_regen(img_rgb)
    # 2) esrgan-like up/down
    t2 = esrgan_like_updown(t1, scale=2 if strength>0.2 else 1)
    # 3) diffusion-like noise + bilateral denoise
    t3 = simulate_diffusion_noise(t2, strength=strength)
    # 4) pixel jitter
    t4 = pixel_shift_texture(t3, max_shift=int(3*strength)+1)
    return t4

# ----------------------------
# UI layout
# ----------------------------
st.title("💧 Dual Watermarking — GAN-approx Demo (CPU Only)")
cols = st.columns([1,1,1])

with cols[0]:
    uploaded = st.file_uploader("Upload an image (256x256 recommended)", type=["jpg","png","jpeg"])
    key_robust = st.text_input("Robust key (owner)", value="robust_key_123")
    key_frag = st.text_input("Fragile key (tamper)", value="fragile_key_123")
    embed_alpha = st.slider("Robust strength (alpha)", min_value=2, max_value=20, value=8)
    st.markdown("**Notes:** This app uses a CPU-friendly GAN-approx pipeline (no Stable Diffusion).")

with cols[1]:
    if load_error:
        st.error(load_error)
    else:
        st.success("Model loaded successfully." if model is not None else "Model not loaded.")
    st.write("Model device:", device)
    run_gan = st.button("Run GAN-approx Attack")
    run_attack = st.button("Run Classical Attacks (Blur / JPEG / Noise)")
    save_all = st.button("Save All Results (.zip)")

with cols[2]:
    st.markdown("**Actions**")
    st.write("- Upload → Embed → Attack → Inspect")
    st.write("- Use GAN-approx for diffusion-like regeneration")
    st.write("- Download results using buttons below")

# ----------------------------
# Main logic
# ----------------------------
def np_to_image_bytes(arr):
    im = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

if uploaded:
    # Prepare original
    pil = Image.open(uploaded).convert("RGB").resize((256,256))
    orig_np = np.array(pil)
    st.subheader("Original")
    st.image(orig_np, use_column_width=False)

    # Embed watermarks
    robust_bits = bits_from_string(key_robust, 512)
    fragile_bits = bits_from_string(key_frag, 1024)
    wm = embed_dct_robust(orig_np, robust_bits, alpha=embed_alpha)
    wm = embed_lsb_fragile(wm, fragile_bits, region=(0,0,64,64))
    st.session_state.watermarked_img = wm


    st.subheader("Watermarked")
    st.image(wm)

    pred_wm, conf_wm = predict_image(wm)
    st.write(f"Detection (watermarked): **{pred_wm}** — Confidence: **{conf_wm:.3f}**")

    # classical attacks: produce one attacked image (blur->jpeg->noise)
    def classical_attack_pipeline(img):
        out = cv2.GaussianBlur(img, (5,5), 1.2)
        _, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:,:,::-1]
        noise = (out.astype(np.int16) + np.random.randint(-8,8,out.shape)).clip(0,255).astype(np.uint8)
        return noise

    attacked = classical_attack_pipeline(wm)

    # If user clicks classical attack button, show updated attacked and metrics
    if run_attack:
        st.subheader("Classical Attack (Blur + JPEG + Noise)")
        st.image(attacked)
        pred_att, conf_att = predict_image(attacked)
        st.write(f"Detection (attacked): **{pred_att}** — Confidence: **{conf_att:.3f}**")
        st.session_state.attacked_img = attacked


    # If user clicks GAN-approx button, generate regenerated image
    gan_img = None
    gan_strength = st.slider("GAN Attack Strength", 0.1, 0.9, 0.45, 0.05)

    if run_gan:
        with st.spinner("Running GAN-approx pipeline (CPU)..."):
            gan_img = fake_gan_regenerate(wm, strength=gan_strength)
        st.session_state.gan_img = gan_img
        st.subheader("GAN-Approx Regenerated Image")
        st.image(gan_img)
        pred_gan, conf_gan = predict_image(gan_img)
        st.write(f"Detection (GAN-approx): **{pred_gan}** — Confidence: **{conf_gan:.3f}**")

    else:
        # try to detect on existing attacked image (classical) for quick results
        pred_att, conf_att = predict_image(attacked)

    # Fragile tamper heatmaps
    st.subheader("Fragile Tamper Localization (LSB region)")
    # compare watermarked vs attacked, and watermarked vs gan if present
    map_att = fragile_diff_map(wm, attacked, region=(0,0,64,64))
    heat_att = (map_att / (map_att.max()+1e-6))
    heat_img_att = (plt.cm.jet(cv2.resize(heat_att, (256,256)))[:,:,:3]*255).astype(np.uint8)
    st.image(heat_img_att, caption="Tamper heatmap (classical attack)")

    if gan_img is not None:
        map_gan = fragile_diff_map(wm, gan_img, region=(0,0,64,64))
        heat_gan = (map_gan / (map_gan.max()+1e-6))
        heat_img_gan = (plt.cm.jet(cv2.resize(heat_gan, (256,256)))[:,:,:3]*255).astype(np.uint8)
        st.image(heat_img_gan, caption="Tamper heatmap (GAN-approx)")

    # Confidence bar chart
    st.subheader("Detection Confidence Comparison")
    # prepare metrics
    c_orig = predict_image(orig_np)[1]
    c_wm = predict_image(wm)[1]
    c_att = predict_image(attacked)[1]
    c_gan = predict_image(gan_img)[1] if gan_img is not None else None
    labels_chart = ["Original", "Watermarked", "Attacked"]
    values_chart = [c_orig, c_wm, c_att]
    if c_gan is not None:
        labels_chart.append("GAN-approx")
        values_chart.append(c_gan)

    import pandas as pd
    df = pd.DataFrame({
        "Label": labels_chart,
        "Confidence": values_chart
    })

    st.bar_chart(df, x="Label", y="Confidence")

    # side-by-side compare images
    st.subheader("Compare — Original / Watermarked / Attacked / GAN")
    cols2 = st.columns(4)
    cols2[0].image(orig_np, caption="Original")
    cols2[1].image(wm, caption=f"Watermarked\n({predict_image(wm)[0]} {predict_image(wm)[1]:.2f})")
    cols2[2].image(attacked, caption=f"Attacked\n({predict_image(attacked)[0]} {predict_image(attacked)[1]:.2f})")
    cols2[3].image(gan_img if gan_img is not None else np.zeros_like(orig_np), caption=f"GAN-approx\n({predict_image(gan_img)[0] if gan_img is not None else '—'})")

    # downloads
    st.subheader("Download Images")
    st.download_button("Download watermarked (PNG)", data=np_to_image_bytes(wm), file_name="watermarked.png", mime="image/png")
    st.download_button("Download attacked (PNG)", data=np_to_image_bytes(attacked), file_name="attacked.png", mime="image/png")
    if gan_img is not None:
        st.download_button("Download GAN-approx (PNG)", data=np_to_image_bytes(gan_img), file_name="gan_regen.png", mime="image/png")

    # optional Save All (simple zipped bytes)
    if save_all:
        import zipfile, tempfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("original.png", np_to_image_bytes(orig_np))
            z.writestr("watermarked.png", np_to_image_bytes(wm))
            z.writestr("attacked.png", np_to_image_bytes(attacked))
            if gan_img is not None:
                z.writestr("gan_approx.png", np_to_image_bytes(gan_img))
        buf.seek(0)
        st.download_button("Download results ZIP", data=buf, file_name="results.zip", mime="application/zip")

else:
    st.info("Upload an image to start. The app embeds a robust DCT watermark + fragile LSB watermark, then runs attacks and shows detection confidence & tamper heatmaps.")

# footer
st.markdown("---")
st.caption("CPU-based GAN-approx for demo only — not full Stable Diffusion. Good for local demos and PPT figures.")

# ==========================
# SAFE RESULT VISUALIZATION
# ==========================

# Prevent crashes if watermark or attack not done
# ==========================
# SAFE RESULT VISUALIZATION
# ==========================
# ==========================
# SAFE RESULT VISUALIZATION
# ==========================

# Only run if both exist
if "watermarked_img" in st.session_state and "attacked_img" in st.session_state:

    st.subheader("🔍 Fragile Watermark Tamper Localization")

    watermarked_img = st.session_state.watermarked_img
    attacked_img = st.session_state.attacked_img

    orig_np = np.array(watermarked_img)
    att_np = np.array(attacked_img)

    # Fragile map
    fragile_map = fragile_diff_map(orig_np, att_np)

    # Heat overlay
    def overlay_tamper(base_img, tamper_map):
        norm = tamper_map / (tamper_map.max() + 1e-6)
        resized = cv2.resize(norm, (base_img.shape[1], base_img.shape[0]))
        heat = plt.cm.jet(resized)[:, :, :3]
        base = base_img.astype(np.float32) / 255.0
        return np.clip(0.6 * base + 0.4 * heat, 0, 1)

    heat_overlay = overlay_tamper(att_np, fragile_map)

    # Pixel diff map
    pixel_diff = np.mean(np.abs(att_np.astype(np.float32) - orig_np.astype(np.float32)), axis=2)

    # Metrics
    mean_pixel_diff = float(pixel_diff.mean())
    fragile_flip_ratio = float(fragile_map.mean() * 100)

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🔥 Fragile Watermark Tamper Heatmap")
        st.image(heat_overlay, caption="LSB fragile watermark damage", use_column_width=True)

    with col2:
        st.write("### 🧨 Pixel Difference Heatmap")
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(pixel_diff, cmap="inferno")
        ax.axis("off")
        st.pyplot(fig)

    st.write("### 📊 Quantitative Metrics")
    st.metric("Mean Pixel Difference", f"{mean_pixel_diff:.2f}")
    st.metric("Fragile Bit Flip %", f"{fragile_flip_ratio:.2f}%")

    st.info("""
    **Interpretation**
    - **Mean Pixel Difference** → overall visual distortion  
    - **Fragile Bit Flip %** → how much LSB watermark was destroyed  
    """)
