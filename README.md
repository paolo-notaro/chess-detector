# chess-detector
Yet another chess-detection project

---

## 1. Requirements

- [Blender](https://www.blender.org/download/) for dataset generation
- Python > 3.9 or **exactly 3.11** for dataset generation, bpy (Blender library) won't work otherwise.

Install requirements via

    pip install -r requirements.txt

## 2. Dataset generation

    cd dataset
    python download_pgn.py
    python gen_dataset.py
