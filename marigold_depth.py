import diffusers
import torch
from torchvision.utils import save_image

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
depth = pipe(image, output_type='pt').prediction
depth = depth.squeeze()
save_image(depth, "depth.png")
print(f"Depth shape: {depth.shape}")
print(f"Depth dtype: {depth.dtype}")
print(f"Depth range: {depth.min(), depth.max()}")
