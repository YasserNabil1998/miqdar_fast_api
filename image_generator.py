from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import logging

app = FastAPI()

# إعدادات الجهاز والدقة
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# تحميل الموديلات مرة واحدة فقط عند تشغيل السيرفر
controlnet_model_id = "lllyasviel/control_v11p_sd15_canny"
print("🔁 تحميل نموذج ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_id,
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None
)

print("🔁 تحميل نموذج Stable Diffusion...")
stable_diff_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    controlnet=controlnet,
    torch_dtype=dtype
).to(device)

if device == "cuda":
    stable_diff_pipe.enable_xformers_memory_efficient_attention()

print("✅ تم تحميل النماذج بنجاح")

@app.post("/generate-image/")
async def generate_image_base64(
    edge_map_base64: str = Form(...),
    prompt: str = Form(...)
):
    try:
        # تحويل base64 إلى صورة
        image_data = base64.b64decode(edge_map_base64)
        edge_map = Image.open(io.BytesIO(image_data)).convert("RGB")

        with torch.inference_mode():
            result = stable_diff_pipe(
              prompt=prompt,
              image=edge_map,
              guidance_scale=5.0,            # أقل جودة لكن أسرع
              num_inference_steps=10         # سريع جدًا مقابل 15 أو 20
          )

        generated_image = result.images[0]
        img_bytes = io.BytesIO()
        generated_image.save(img_bytes, format="PNG")
        encoded_img = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        return JSONResponse(content={"preview_base64": encoded_img})

    except Exception as e:
        logging.exception("❌ خطأ أثناء التوليد")
        return JSONResponse(
            content={"error": "فشل توليد الصورة", "details": str(e)},
            status_code=500
        )
