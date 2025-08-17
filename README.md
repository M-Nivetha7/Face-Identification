# Face-Identification

A lightweight, notebook-first pipeline for detecting and identifying faces in videos using Python. Built for quick experiments, classroom demos, and hackathons—run end-to-end entirely in Jupyter/Colab.

✨ Features

Face detection: RetinaFace/MediaPipe/YOLO (configurable)

Face embeddings: FaceNet / ArcFace (InsightFace) for robust identity vectors

Recognition: cosine similarity / SVM / k-NN over embeddings

Video I/O: works with files or webcam; writes annotated MP4s

Notebook-first UX: one-click cells, progress bars, and live previews

Reproducible: environment file + deterministic seeds

🔧 Tech Stack

Python 3.9+

opencv-python, numpy, torch, torchvision

Detector: mediapipe or retinaface (via insightface)

Embeddings: insightface (ArcFace) or keras-facenet

📊 Outputs

Annotated MP4 with bounding boxes, names, and per-frame FPS

CSV: timestamped detections with identity, confidence, bbox, and embedding ID

Embeddings store: .npy vectors + label index for fast reuse

📈 Accuracy Tips

Use 3–5 high-quality, frontal enrollment images per person

Keep threshold between 0.5–0.7 (ArcFace) and tune on a small validation clip

Prefer RetinaFace + ArcFace for tougher, low-light videos

Normalize frames to consistent size (e.g., 720p) for speed & stability

Utils: tqdm, scikit-learn, pandas, matplotlib

You can switch detectors/embedders by changing a single config cell in the notebook.
