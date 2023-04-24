import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableDiffusionImg2ImgPipeline,\
    EulerAncestralDiscreteScheduler, \
    DDIMScheduler,\
    DDPMScheduler,\
    PNDMScheduler, \
    UNet2DConditionModel,\
    AutoencoderKL

from transformers import CLIPTextModel
import torch
from PIL import Image
import numpy as np

model_id = "./models/dog500_1e6_constant_0.1_xformer"

# Load models and create wrapper for stable diffusion
text_encoder = CLIPTextModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    torch_dtype=torch.float16
)

# unet.enable_xformers_memory_efficient_attention()
# vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",text_encoder=text_encoder, unet=unet,
                                                torch_dtype=torch.float16).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

man_promptlist = ["a photo of a <zxq> man as a soldier, closeup character art by donato giancola, craig mullins, digital art, trending on artstation",
              "Symmetry!!,highly detailed, digital painting, selfie portrait of <zxq> in cyber armor, dreamy and ethereal, expressive pose, black eyes, exciting expression, fantasy, intricate, elegant, many lightning, cold color, highly detailed, digital painting, artstation, concept art, cyberpunk wearing, smooth, sharp focus, led, illustration",
              "Symmetry!!,highly detailed, selfie portrait of <zxq> with white beard and hair, wearing wolf pelt, with runic geometry face tattoos",
              "Symmetry!!, selfie portrait of <zxq> is a rugged ranger, d&d, muscular, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha",
              "Symmetry!!, selfie portrait of <zxq> art by dan mumford and yusuke murata and makoto shinkai and ross tran, cosmic, heavenly, god rays, intricate detail, cinematic, 8 k, cel shaded, unreal engine, featured on artstation, pixiv"]

women_promptlist = ["Symmetry!!, Half - circuits hacker <ym> with cute - fine - face, pretty face, multicolored hair, realistic shaded perfect face, fine details by realistic shaded lighting poster by ilya kuvshinov katsuhiro otomo, magali villeneuve, artgerm, jeremy lipkin and michael garmash and rob rey",
                    "Portrait of <rb> with bangs, 1 9 6 0 s, long hair, red hairband, bangs, intricate, elegant, glowing lights, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by wlop, mars ravelo and greg rutkowski",
                    ]
# Symmetry!!,highly detailed, digital painting, selfie portrait of <sam>

prompt ="a photo of <dog2> dog" 
negative_prompt = "nsfw,lowers,bad anatomy, bad hands, text,error,missing fingers,extra digit, fewer digits,cropped,worst quality,low quality,normal quality, jpeg artifacts,signature,watermark,username,blurry"

pipe.enable_xformers_memory_efficient_attention()
result = Image.new("RGB", (512*5, 512))
for i in range(5):
    image = pipe(prompt=prompt, 
                negative_prompt=negative_prompt,
                generator=torch.Generator(device="cuda").manual_seed(i), 
                num_inference_steps=28, 
                guidance_scale=7,
                height=512,
                width=512).images[0]
    # image.save("./sample/{}.jpg".format(str(i))) 
    result.paste(image, box=(i*512, 0))
result.save("./{}.jpg".format("result"))    