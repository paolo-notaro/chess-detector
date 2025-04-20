# chess-detector
Yet another chess-detection project

---

## 1. Requirements

- [Blender](https://www.blender.org/download/) for dataset generation
- Python > 3.9 or **exactly 3.11** for dataset generation, bpy (Blender library) won't work otherwise.

Install requirements via

    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

## 2. Dataset generation

    cd dataset
    python download_pgn.py
    python gen_dataset.py

## 3. Training

    python diff_train.py 

# 4. Inference

With prepreocessing (i.e. start from a chessboard image):

    python .\diff_predict.py --preprocess --before dataset\images\rn1qk1nr_pb5p_1ppbp1p1_P7_2Q1p3_2N5_1PPP1PPP_R1B1KB1R.png --after dataset\images\rn1qk1nr_pb5p_2pbp1p1_Pp6_2Q1p3_2N5_1PPP1PPP_R1B1KB1R.png

Without preprocessing (i.e. start from a normalized WB warped image):

    python .\diff_predict.py --before dataset/preprocessed/r4rk1_pp1qbpp1_3p1n1p_2pNp3_4P3_1P1P2PB_PBP2P1P_R2Q1RK1.png --after dataset/preprocessed/r4rk1_pp2bpp1_3p1n1p_2pNp3_4P3_1P1P2Pq_PBP2P1P_R2Q1RK1.png