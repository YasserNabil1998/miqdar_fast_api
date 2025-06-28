from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, UnidentifiedImageError
import numpy as np
import cv2
import io
import base64
from sklearn.cluster import KMeans
from typing import Optional
import time
import requests  # لإرسال الطلب لـ API التوليد
import torch

app = FastAPI()

def convert_units(w_px, h_px, px_per_cm):
    w_cm = w_px / px_per_cm
    h_cm = h_px / px_per_cm
    return {
        "cm": (round(w_cm, 2), round(h_cm, 2)),
        "mm": (round(w_cm * 10, 2), round(h_cm * 10, 2)),
        "in": (round(w_cm / 2.54, 2), round(h_cm / 2.54, 2))
    }

def get_edge_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def preprocess_image(pil_image):
    max_dim = 384
    if max(pil_image.size) > max_dim:
        pil_image = pil_image.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
    return np.array(pil_image), pil_image

@app.post("/process/")
async def process_image(
    file: UploadFile = File(...),
    target_height_cm: Optional[float] = Form(None),
    product_name: Optional[str] = Form(None),
    user_width: Optional[float] = Form(None),
    user_height: Optional[float] = Form(None),
    do_extract_colors: bool = Form(False),
    do_detect_dimensions: bool = Form(False),
    do_generate_image: bool = Form(False),
):
    start_all = time.time()

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)

        results = {}

        if do_extract_colors:
            start = time.time()
            image_resized = image.resize((150, 150))
            pixels = np.array(image_resized).reshape(-1, 3)
            kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)

            swatch_width = 100
            swatch_height = 100
            palette_image = Image.new("RGB", (swatch_width * 5, swatch_height), (255, 255, 255))
            draw = ImageDraw.Draw(palette_image)

            hex_colors = []
            for i, color in enumerate(colors):
                rgb = tuple(color)
                hex_code = '#%02x%02x%02x' % rgb
                hex_colors.append(hex_code)
                x0 = i * swatch_width
                draw.rectangle([x0, 0, x0 + swatch_width, swatch_height], fill=rgb)

            img_bytes = io.BytesIO()
            palette_image.save(img_bytes, format="PNG")
            encoded_img = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

            results["extract-colors"] = {
                "colors": hex_colors,
                "preview_base64": encoded_img,
                "time_sec": round(time.time() - start, 2)
            }

        if do_detect_dimensions:
            start = time.time()
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                results["detect-dimensions"] = {"error": "لم يتم العثور على كائن"}
            else:
                height, width = gray.shape
                center_x, center_y = width // 2, height // 2

                def contour_center_distance(contour):
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        return float('inf')
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return (cX - center_x) ** 2 + (cY - center_y) ** 2

                target_contour = min(contours, key=contour_center_distance)
                x, y, w, h = cv2.boundingRect(target_contour)
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

                PX_PER_CM = 38
                units = convert_units(w, h, PX_PER_CM)

                boxed_pil = Image.fromarray(image_np)
                buffer = io.BytesIO()
                boxed_pil.save(buffer, format="PNG")
                preview_base64 = base64.b64encode(buffer.getvalue()).decode()

                results["detect-dimensions"] = {
                    "dimensions_px": {"width": w, "height": h},
                    "dimensions_converted": units,
                    "preview_base64": preview_base64,
                    "time_sec": round(time.time() - start, 2)
                }

        if do_generate_image:
            start = time.time()

            if target_height_cm is None:
                results["generate-image"] = {"error": "مطلوب target_height_cm"}
            else:
                image_np_small, pil_image_resized = preprocess_image(image)
                image_bgr = cv2.cvtColor(image_np_small, cv2.COLOR_RGB2BGR)

                height_px = image_np_small.shape[0]
                px_per_cm = height_px / target_height_cm

                edge_map = get_edge_map(image_bgr)
                edge_map_pil = Image.fromarray(edge_map)

                img_bytes = io.BytesIO()
                edge_map_pil.save(img_bytes, format="PNG")
                encoded_edge_map = base64.b64encode(img_bytes.getvalue()).decode()

                prompt = f"A coffee pot with {target_height_cm} cm height."
                # timeout = 600 if not torch.cuda.is_available() else 30

                try:
                    response = requests.post(
                        "http://ec2-16-170-228-225.eu-north-1.compute.amazonaws.com:8002/generate-image/",
                        json={
                            "edge_map_base64": encoded_edge_map,
                            "prompt": prompt
                        },
                        timeout=None
                    )


                    if response.status_code == 200:
                        results["generate-image"] = {
                            "message": "تم إنشاء الصورة بنجاح",
                            "preview_base64": response.json().get("preview_base64"),
                            "product_name": product_name,
                            "user_width": user_width,
                            "user_height": user_height,
                            "time_sec": round(time.time() - start, 2)
                        }
                    else:
                        results["generate-image"] = {
                            "error": f"فشل توليد الصورة - كود: {response.status_code}",
                            "details": response.text
                        }

                except requests.exceptions.RequestException as e:
                    results["generate-image"] = {
                        "error": "فشل الاتصال بخدمة توليد الصورة",
                        "details": str(e)
                    }

        results["total_time_sec"] = round(time.time() - start_all, 2)
        return JSONResponse(content=results)

    except UnidentifiedImageError:
        return JSONResponse(content={"error": "الصورة غير صالحة"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
