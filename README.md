<div align="center">
  <img src="https://img.shields.io/badge/Text%20Classification-Project-blue" />
  <h1>Text Classification Project</h1>
  <p>โปรเจกต์ตัวอย่างสำหรับแยกประเภทข้อความ เช่น <b>GREETING</b>, <b>NSFW</b>, <b>OTHER</b> ด้วย <b>scikit-learn</b></p>
</div>

---

## 📂 โครงสร้างโปรเจกต์

```bash
project/
│── dataset/
│   ├── train/
│   │   ├── GREETING/
│   │   ├── NSFW/
│   │   ├── OTHER/
│   ├── val/
│   │   ├── GREETING/
│   │   ├── NSFW/
│   │   ├── OTHER/
│── requirements.txt
│── train.py
│── predict.py
│── README.md
```

---

## 🚀 การติดตั้ง

1. สร้าง Virtual Environment (แนะนำ)
   ```bash
   python -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```
2. ติดตั้ง dependencies
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 การเตรียมข้อมูล

ใส่ไฟล์ข้อความตัวอย่างในโฟลเดอร์:

```bash
dataset/train/<CLASS_NAME>/
dataset/val/<CLASS_NAME>/
```
เช่น
```bash
dataset/train/GREETING/hello.txt
dataset/train/NSFW/badword.txt
dataset/train/OTHER/random.txt
```

---

## 🏋️‍♂️ การเทรนโมเดล

```bash
python train.py
```

---

## 🔍 การทำนายข้อความใหม่

```bash
python predict.py "hello, how are you?"
```

### ผลลัพธ์ตัวอย่าง
```bash
Predicted class: GREETING
```

---

## 📦 Dependencies

| Package        | Version (แนะนำ) |
| -------------- | --------------- |
| scikit-learn   | >=1.0           |
| joblib         | >=1.0           |

---

<div align="center">
  <sub>สร้างโดย OZONEAIK สิงหา 2025</sub>
</div>