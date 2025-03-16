# Flask YOLO Object Detection API

This is a **Flask API** that uses the **YOLOv8 model** to detect fresh and rotten fruits/vegetables.  
It predicts **whether a vegetable/fruit should be removed from the shelf** and provides the **estimated shelf life** if it's fresh.

## 🚀 Features
- 📸 **Upload an image** to detect objects using YOLOv8.
- 🍎 **Classify fresh or rotten vegetables/fruits**.
- ⏳ **Estimate shelf life** for fresh items.
- ⚠️ **Warn if an item should be removed from the shelf**.

---

## 📂 Files in This Repo
| File | Description |
|------|------------|
| `app.py` | Flask API for YOLO object detection |
| `requirements.txt` | List of required dependencies |
| `test.jpg` | Sample test image |
| `best_yolov8_shelf_life.pt` | YOLOv8 ML model file |

---

### Dataset

[Link](https://universe.roboflow.com/id-card-53tam/shelf-life-prediction)

