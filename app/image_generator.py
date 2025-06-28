from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import logging

app = FastAPI()

controlnet = None
stable_diff_pipe = None

@app.post("/generate-image/")
async def generate_image_base64(
    edge_map_base64: str = Form(...),
    prompt: str = Form(...)
):
    global controlnet, stable_diff_pipe

    if controlnet is None or stable_diff_pipe is None:
        # تعيين الجهاز بشكل ثابت على CPU لأن GPU غير متوفر
        device = "cpu"

        # استخدام float32 لأن float16 غير مدعوم بشكل جيد على CPU
        dtype = torch.float32

        controlnet_model_id = "lllyasviel/control_v11p_sd15_canny"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
            variant=None  # لا نستخدم fp16 على CPU
        )
        stable_diff_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            controlnet=controlnet,
            torch_dtype=dtype
        ).to(device)

        # على CPU لا نفعل ميزة xformers
        # if device == "cuda":
        #     stable_diff_pipe.enable_xformers_memory_efficient_attention()

    try:
        image_data = base64.b64decode(edge_map_base64)
        edge_map = Image.open(io.BytesIO(image_data)).convert("RGB")

        with torch.inference_mode():
            result = stable_diff_pipe(
                prompt=prompt,
                image=edge_map,
                guidance_scale=3.0,          # خفضت القيمة عن 4.5 لتقليل الحمل
                num_inference_steps=5       # خفضت الخطوات عن 7 لتسريع وتقليل استهلاك الذاكرة
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
