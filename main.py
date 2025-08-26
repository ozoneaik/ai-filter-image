import torch
from PIL import Image
from torchvision import transforms
import timm
import requests
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

class UrlRequest(BaseModel):
    url: str

# โหลดโมเดล
chkpt = torch.load("best_model_enhanced.pt", map_location="cpu")
class_names = chkpt["class_names"]
classes = ["GREETING", "NSFW", "OTHER"]

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(class_names))
model.load_state_dict(chkpt["model_state_dict"])
model.eval()

def predict(img: Image.Image):
    x = val_tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        y = model(x)
        probs = torch.softmax(y, dim=1)[0]
    result = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return classes[probs.argmax().item()], result

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API running 🚀"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        pred_class, result = predict(img)
        return JSONResponse({"prediction": pred_class, "probabilities": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/predict_url")
async def predict_from_url(req: UrlRequest):
    try:
        response = requests.get(req.url)
        img = Image.open(BytesIO(response.content))
        pred_class, result = predict(img)
        return JSONResponse({"prediction": pred_class, "probabilities": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
