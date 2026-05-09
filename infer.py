from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

app = Flask(__name__)
device = torch.device("cpu")

print("Loading model from /mnt/model/mnist_cnn.pt ...")
model = Net().to(device)
model.load_state_dict(torch.load("/mnt/model/mnist_cnn.pt", map_location=device))
model.eval()
print("Model loaded.")

tfm = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

PAGE = """
<!doctype html>
<html><head><title>MNIST Classifier</title>
<style>
body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
h2 { color: #333; }
form { margin: 20px 0; padding: 20px; background: #f4f4f4; border-radius: 8px; }
input[type=file] { margin-bottom: 10px; }
button { padding: 8px 16px; background: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; }
button:hover { background: #3367d6; }
.result { margin-top: 20px; padding: 15px; background: #e8f0fe; border-radius: 8px; }
</style></head><body>
<h2>MNIST Digit Classifier</h2>
<p>Upload a handwritten digit image (any size, will be resized to 28x28).</p>
<form method=post enctype=multipart/form-data action=/predict>
  <input type=file name=image accept="image/*" required><br>
  <button type=submit>Predict</button>
</form>
<p>Or POST to <code>/predict</code> with multipart form field <code>image</code>.</p>
</body></html>
"""

@app.route("/")
def home():
    return render_template_string(PAGE)

@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image field"}), 400
    f = request.files["image"]
    img = Image.open(io.BytesIO(f.read()))
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(logits.argmax(1).item())
        conf = float(probs.max().item())
    return jsonify({"prediction": pred, "confidence": round(conf, 4)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
