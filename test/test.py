from models.modeling_geomagvit import GeoMAGVIT
import torch
from PIL import Image
from torchvision import transforms

vq = GeoMAGVIT.from_pretrained('JO-KU/Geo-MAGVIT').eval()

# Load test image
img = Image.open('./data/test.png').convert('RGB')
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
x = transform(img).unsqueeze(0)

# Test round-trip
with torch.no_grad():
    tokens = vq.get_code(x)
    recon = vq.decode_code(tokens)

print(f'Tokens shape: {tokens.shape}')
print(f'Token range: {tokens.min()}-{tokens.max()}')
print(f'Unique tokens: {len(torch.unique(tokens))}')

# Save
recon = ((recon + 1) / 2 * 255).clamp(0, 255).byte()
Image.fromarray(recon[0].permute(1,2,0).numpy()).save('test_recon.png')
print('Saved to test_recon.png - compare với original!')
