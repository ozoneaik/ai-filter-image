import torch
from PIL import Image
from torchvision import transforms
import timm
import requests
from io import BytesIO

chkpt = torch.load("best_model.pt", map_location="cpu")
class_names = chkpt["class_names"]

classes = ["GREETING", "NSFW", "OTHER"]

IMG_SIZE = 224
infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(class_names))
model.load_state_dict(chkpt["model_state"])
model.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(input_data):
    if isinstance(input_data, str):  # ถ้าเป็น path ของไฟล์
        img = Image.open(input_data).convert("RGB")
    else:  # ถ้าเป็น PIL.Image (จาก URL)
        img = input_data.convert("RGB")

    # ทำ preprocessing ตามที่เคยทำ
    x = val_tf(img).unsqueeze(0)
    with torch.no_grad():
        y = model(x)
        probs = torch.softmax(y, dim=1)[0]

    # แปลงผลลัพธ์เป็น dict
    result = {classes[i]: float(probs[i]) for i in range(len(classes))}
    pred_class = classes[probs.argmax().item()]
    return pred_class, result

def predict_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return predict(img)   # ใช้ฟังก์ชัน predict เดิม

print(predict("nomal.jpg"))
# print(predict_from_url('https://us-fbcloud.net/picpost/data/347/347162-qfxjvo-4.n.jpg'))

