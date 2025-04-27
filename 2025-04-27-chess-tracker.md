---
title: ???
tags: [Computer Vision, PyTorch, Python, Rendering]
style: fill
color: primary
description: We built a Chess Game tracker to improve our chess skills and ended up learning more about image tracking, rendering and training vision models
---

# Introduction

**A chessboard forgets everything the moment you walk away - and that feels wrong.**

My tech-avid, chess-playing brother wanted a simple way to track his real-life chess games: in this way he could analyze games offline to spot blunders, share the occasional masterpiece, maybe even track progress over time.

*Why record face-to-face games at all?*

* **Offline analysis** - capture the full chess game so computer engines and fellow players can dissect them long after the pieces are put away  
* **Progress tracking** - a growing archive of your own moves lets you spot recurring mistakes and measure improvement over months, not just one session  
* **Hands-on ML playground** - the project itself is a chance for both of us to build a practical computer-vision pipeline: real data, real constraints, and a clear success metric (did we log the right move?)

## What is out there?

Sure, once again ready-made solutions exist for this.

Commercial e-boards, like the **DGT Smart Board**, slot Hall-effect sensors beneath every square and rely on magnetized pieces to register moves [1].  
Others, such as **Chess Classics Exclusive** or **Chessnut Pro**, embed an RFID chip in each piece and let an antenna grid read its identity and square in real time [2].
Both solutions hide the technologies under the squares or glue chips to every piece, making it quite pricey (€400+), fragile, and tied to one *"smart"* board.

We figured a minimal setup we already owned - basically a phone camera, a tripod, and a computer - should be enough to deal with this.

In short, *we said "nope, thanks" to sensor boards*, because

* Extra hardware is awkward to carry around and takes effort to maintain/set up  
* Most open places where you can play chess (parks, cafés, clubs) won't ditch their trusty wooden sets / boards  
* A decent camera lives in every pocket, so why not use it?

## What makes camera-only tracking tricky

However, the camera-only approach comes with its own challenges:

| Problem            | Why it matters                                                                 |
|----------------------|--------------------------------------------------------------------------------|
| **Microscopic change** | A single pawn moves forward; 95% of the pixels stay identical              |
| **Move overload**    | Dozens of legal moves can look almost the same in a still image               |
| **Real-world chaos** | Shadows, glare, light changes, and the occasional elbow nudge to the board call for big misalignment and mispredictions.    |
| **Timing the shot** | When to take a shot? the camera must fire after the piece lands but before the next hand enters |

## What you will find in this post

1. The different ideas we tried - and why some of them flopped  
2. The key insight that set us on the right path  
3. A step-by-step tour of the working pipeline, from board calibration to move prediction  
4. Results, quirks, and what comes next

> **Chess Basics (for anyone who zoned out during _The Queen’s Gambit_)**
>
> * **Board** – 8×8 grid = 64 squares, labelled _a–h_ (files) and _1–8_ (ranks). Two players, White and Black, face each other on opposite sides of the board. White moves first.
> * **Algebraic square names** – bottom left from White’s view is _a1_, top right is _h8_.  
> * **Universal Chess Interface (UCI) move format** – four-char string: _e2e4_ = piece from _e2_ to _e4_; promotions add the piece letter (_e7e8q_).  
> * **Pieces** – king, queen, rook, bishop, knight, pawn (each with its own move pattern).
> * **Specials** – castling (king+rook swap), en-passant capture, pawn promotion upon reaching the 8th rank.  
> * **Goal** – checkmate: king is attacked and can't escape.
>
>
> Read more about chess rules on [Wikipedia](https://en.wikipedia.org/wiki/Chess) or [Lichess](https://lichess.org/learn#/).

Pull up a chair - we are about to make plain black-and-white images speak fluent chess.

# Our Journey towards a Camera-Only Chess Tracker
*(aka "how many wrong turns can you take on 64 squares?)"*

Before we could gloriously fail with neural networks, we needed a game plan. So tried to decompose the problem into smaller pieces:
- **Global goal: track the whole game** → means keeping an accurate board state after every turn. Which means:
- **Get board state after a turn** = board state *before* + **the move** that just happened.
-  **Move detection** → give the model two images (before / after) and ask,  "Which square just emptied and which one just filled?", basically obtaining the move in UCI format (e.g. *e2e4*, *b7b8q*, etc.).
- **Move validation** → even a clever model can hallucinate illegal jumps, so we must validate its guesses with a chess engine and keep only the legal moves in the current board state.
- **State update** → apply that validated move to our running board, then repeat from step 1 until handshake or checkmate.

> Implementation note: after all the above logic is implemented, we can simply record the taken moves and intermediate board positions into a standardized format (like PGN) and use it to analyze the game later on. Thi step is not covered in this post, but it is the final step of the pipeline.

## First Attempt at Prediction: The Siamese Net That Said “GG EZ” to Our Dataset  

<!-- ✏️ placeholder: insert a simple diagram showing  
     before → Encoder  after → Encoder → concat → MLP heads (from / to / promo) -->

We started with a classic idea: shove the **before** and **after** images through the **same encoder**, glue the embeddings together, and ask an MLP to predict the move in UCI format, i.e., *from-square*, *to-square*, and *promotion*.

You can see the diagram above for a simplified version of [this]() pipeline.

Here is a quick overview of our code:

```python

# Example of a simple CNN encoder
class SmallCNNEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=5, stride=2, padding=2
            ),  # -> [B, 16, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 16, 56, 56]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> [B, 32, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 32, 28, 28]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [B, 64, 28, 28]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 64, 1, 1]
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Actual move prediction model (SiameseNetwork)
class ChessMovePredictor(nn.Module):
    def __init__(
        self, embedding_dim=256, encoder_class: Type[nn.Module] = SmallCNNEncoder
    ):
        """
        ~400k total params
        Args:
            embedding_dim (int): Dimension of the embedding.
            encoder_class (Type[nn.Module]): Class of the encoder to use.
        """
        super().__init__()
        ...
        self.encoder = encoder_class(output_dim=embedding_dim)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Output heads
        self.from_head = nn.Linear(256, 64)
        self.to_head = nn.Linear(256, 64)
        self.promotion_head = nn.Linear(256, 5)

    def forward(self, before, after):
        emb_before = self.encoder(before)  # [B, D]
        emb_after = self.encoder(after)  # [B, D]
        combined = torch.cat([emb_before, emb_after], dim=1)  # [B, 2D]

        x = self.mlp(combined)  # [B, 256]

        from_logits = self.from_head(x)  # [B, 64]
        to_logits = self.to_head(x)  # [B, 64]
        promotion_logits = self.promotion_head(x)  # [B, 5]

        return from_logits, to_logits, promotion_logits
```

For embedding extraction, we tried different architectures, including our own Convolutional Neural Network, ResNet18 and Vision Transformer (ViT), but the first one was the most promising.

## Data Generation process

In order to train the model, we needed a dataset of chess positions, with corresponding images, and annotations of the moves that were made.
We generated a dataset of circa 20k chess positions, using the [Blender](https://www.blender.org/) rendering engine. The pipeline was as follows:

### Step 1: Downloading and Analyzing Chess Games

The process begins by downloading chess games from Lichess using the `download_and_select_moves` function. The script fetches a large dataset of games and analyzes them to extract moves. Users can specify the percentage of moves to retain and set a minimum count for each move type. The selected moves are shuffled and saved to a CSV file (`entries.csv`) for further processing.

```python
moves = download_and_select_moves()
with open(MOVES, "w") as moves_file:
    moves_file.write("before_fen,move_uci,after_fen\n")
    for before_id, move, after_id in moves:
        moves_file.write(f"{before_id},{move},{after_id}\n")
```

---

### Step 2: Processing Chess Moves

For each move in the dataset, the script uses the FEN (Forsyth-Edwards Notation) strings to render the board states before and after the move. Blender generates images of these board states, which are then processed using OpenCV to rectify the perspective and normalize the images. The processed images are saved in the `preprocessed` directory.

```python
before_img = postprocessing.process_image(os.path.join(RENDER_PATH, f"{before_board_id}.png"), chessboard_corners)
postprocessing.save_image(before_img, os.path.join(PREPROCESS_PATH, f"{before_board_id}.png"))
```

---

## Step 3: Dataset Preparation for Machine Learning

The generated images and move data are used to create PyTorch datasets. For example, the `ChessMoveDatasetFromCSV` class loads preprocessed images and labels them with move information (e.g., source square, target square, promotion). This dataset can be directly used to train models for tasks like move prediction.

```python
class ChessMoveDatasetFromCSV(Dataset):
    def __getitem__(self, idx):
        before_tensor = self._load_image(before_id)
        after_tensor = self._load_image(after_id)
        label = {
            "from": torch.tensor(SQUARE_TO_IDX[from_sq], dtype=torch.long),
            "to": torch.tensor(SQUARE_TO_IDX[to_sq], dtype=torch.long),
            "promotion": torch.tensor(PROMOTION_TO_IDX.get(promo, 0), dtype=torch.long)
        }
        return before_tensor, after_tensor, label
```


This pipeline automates the generation of a chess dataset, from downloading games to creating labeled images for machine learning. By leveraging tools like Blender and OpenCV, it ensures high-quality data suitable for training models in tasks like move prediction, board state classification, and more. This modular approach allows for easy customization and scalability, making it a powerful tool for chess-related AI research.

## So, what went wrong?

Our network simply couldn't learn the task. We tried different architectures, hyperparameters, and training strategies, but the model just wouldn't generalize. 
We also figured it could be due to exceeding model capacity - so we reduce the number of parameters and tried again, but the results were still disappointing.
We ended up with a model that was overfitting like crazy, and we couldn't figure out why.

### Symptoms of Failure

| Symptom | Evidence |
|---------|----------|
| **Training loss flat-lined near 0** | The model memorised the 20 k synthetic boards in no time. |
| **Validation accuracy ≈ coin-flip** | Outside that set, it guessed like a tired chess novice. |

### Autopsy Notes  

1. **Global features drown tiny changes** – ResNet focuses on colour blobs and edges; one shifted pawn hardly registers.  
2. **20 k renders are small for a full-image model** – easy to overfit on board textures and camera angle.  
3. **One head to rule them all** – the MLP had to learn piece legality *and* visual change in one shot.

> **Key takeaway → Detect the *change*, not the *pieces*.**

*(Yes, the Blender pipeline that created those 20k boards almost stayed the same.  
We’ll revisit data synthesis in the sections below.)*

#### Pivot Time  

Realizing the network should focus only on *what was moved*, we ditched the Siamese idea and moved toward our diff-patch pipeline - the one you'll meet in the next section.


## The A-ha! Moment – Go Small, Go Local, Go Δ

*"RIP SiameseNet (2024-2024)”*

   •  Realisation: every legal move toggles at most two squares  
   •  Plan: force the network’s field of view onto each square.

## Step 1: adapt our data generation pipeline

New step:

### Step 4: Generating Difference Images (Optional)

The script generates difference images that highlight the changes between the "before" and "after" board states. These images are saved in the `diff` directory and can be used for tasks like move prediction or board state classification.

```python
diff_img = postprocessing.gen_diff(before_img, after_img)
postprocessing.save_image(diff_img, os.path.join(DIFF_PATH, f"{i}.png"))
```



# Pipeline in Detail: . Δ is the Way”
   4.1  Offline calibration (4 corner homography)  
   4.2  Capturing before / after frames  
   4.3  Pre-processing  
        • warp → grayscale 224²  
        • abs-diff → normalise  
        • slice into 64 patches → 32²  
   4.4  **PatchEncoder** (code snippet - tiny CNN)  
   4.5  **MoveScorer**  
        – add 64-slot positional encodings  
        – from_proj · to_projᵀ  →  64 × 64 logits ÷ temperature  
   4.6  Legality filter with python-chess (or Stockfish)  
   4.7  Output: UCI move, board state advances

# Training & Results
   •  20 k Blender renders → 95 % top-2 synth  
   •  Early real-world test set → 60 % top-1  
   •  600 k params ≈ Raspberry-Pi friendly

# Strengths, Weaknesses, Next Steps
   ✔  Works on any board / pieces  
   ✔  Zero extra hardware  
   ✘  Lighting + camera nudge hurt diff map  (we need an online tracker, would also reduce the need for calibration)
   ✘  Promotions not yet handled  
   Roadmap: train on bigger real dataset, augmentation, dual-exposure trick, promotion head revival and experiment with online board tracking

# Takeaways (fun-size bullets)
   •  Synthetic renders > hand labelling  
   •  Localise the learning target → small nets rock  
   •  “Sensors? Nah. We’re running this game on raw RGB.”   ← your meme line

# References

[1] DGT Electronic Chess Boards overview – chesshouse.com – <https://www.chesshouse.com/collections/dgt-electronic-chess-boards-pc-connect>  
[2] Chessnut Pro sensor discussion – chessbazaar.com – <https://www.chessbazaar.com/blog/enhancing-your-chess-experience-with-custom-chess-sets-from-chessbazaar-dgt-sensors-and-chessnut-pro-sensors/>  