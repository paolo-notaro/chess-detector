---
title: Building a Vision-only Chess Tracker
tags: [Computer Vision, PyTorch, Python, Rendering, Blender, Chess]
style: fill
color: primary
description: tried to improve our chess skills and ended up learning more about image tracking, rendering and training vision models
---

#### _We wanted to build an app to track chess games and improve our play skills, and ended up learning more about image tracking, rendering and training vision models_

# Introduction: what is this about?

**A chessboard forgets everything the moment you walk away - and that feels wrong.**

I like chess, but I'm not a grandmaster. I play for fun, and I'm not too keen on memorizing opening lines or endgame traps.

My tech-avid, chess-playing brother wanted a simple way to track his real-life chess games. 
In this way he could analyze games offline to spot blunders, share the occasional masterpiece, maybe even track progress over time.


Building such system would be useful for:

* **Offline analysis** - capture the full chess game so computer engines and fellow players can dissect them long after the pieces are put away  
* **Progress tracking** - a growing archive of your own moves lets you spot recurring mistakes and measure improvement over months, not just one session  
* **Hands-on ML playground** - the project itself is a chance for both of us to build a practical computer-vision pipeline: real data, real constraints, and a clear success metric (did we log the right move?)

And of course use ✨ **AI** ✨ to do so.
So of course, I said yes.


## Chess Tracking: What is out there?

<figure style="margin: 2em auto; text-align: center; max-width: 100%;">
  <img 
    src="/blog/images/chess-tracker/chessboard.jpg" 
    alt="A chessboard in the initia game configuration." 
    style="width: 50%; height: auto; border-radius: 8px;" 
  />
  <figcaption style="margin-top: 0.5em; font-size: 0.95em; color: var(--text-muted-color);">
    A chessboard in the initial game configuration.
  </figcaption>
</figure>

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

As always, ready-made solutions exist for this.

Commercial e-boards, like the **DGT Smart Board**, slot Hall-effect sensors beneath every square and rely on magnetized pieces to register moves [1].  
Others, such as **Chess Classics Exclusive** or **Chessnut Pro**, embed an RFID chip in each piece and let an antenna grid read its identity and square in real time [2].
Both solutions hide the technologies under the squares or glue chips to every piece, making it quite pricey (€400+), fragile, and tied to one *"smart"* board.

We figured a minimal setup we already owned - basically a phone camera, a tripod, and a computer - should be enough to deal with this.

In short, we said *"no, thanks"* to sensor boards, because

* Extra hardware is awkward to carry around and takes effort to maintain/set up  
* Most open places where you can play chess (parks, cafés, clubs) won't ditch their trusty wooden sets / boards  
* A decent camera lives in every pocket, so why not use it?


## What makes camera-only tracking tricky

However, a camera-only approach comes with its own challenges, such as microscopic changes, move overload, real-world chaos, and timing the shots.

| Problem            | Why it matters                                                                 |
|----------------------|--------------------------------------------------------------------------------|
| **Microscopic change** | A single pawn moves forward; 95% of the pixels stay identical              |
| **Move overload**    | Dozens of legal moves can look almost the same in a still image               |
| **Real-world chaos** | Shadows, glare, light changes, and the occasional elbow nudge to the board call for big misalignment and mispredictions.    |
| **Timing the shot** | When to take a shot? the camera must fire after the piece lands but before the next hand enters |

As you will see, some of these issues are easy to solve with a bit of clever engineering, while others are more challenging and still bug us today.
But let's not get ahead of ourselves.

## What you will find in this article

1. The different ideas we tried - and why some of them flopped  
2. The key insight that set us on the right path  
3. A step-by-step tour of the working pipeline, from board calibration to move prediction  
4. Results, quirks, and what comes next


Pull up a chair - we are about to make plain black-and-white images speak fluent chess.

> As always, you can {% include elements/button.html link="https://github.com/paolo-notaro/chess-detector" text="Check out the project on GitHub" style="primary" size="sm" %}


# Our Journey towards a Camera-Only Chess Tracker

*(aka "how many wrong turns can you take on 64 squares?")*

Before we could gloriously fail with neural networks, we needed a game plan. So tried to decompose the problem into smaller pieces:
- **Global goal: track the whole game** → means keeping an accurate board state after every turn. Which means:
- **Get board state after a turn** = board state *before* + **the move** that just happened.
-  **Move detection** → give the model two images (before / after) and ask,  "Which square just emptied and which one just filled?", basically obtaining the move in UCI format (e.g. *e2e4*, *b7b8q*, etc.).
- **Move validation** → even a clever model can hallucinate illegal jumps, so we must validate its guesses with a chess engine and keep only the legal moves in the current board state.
- **State update** → apply that validated move to our running board, then repeat from step 1 until handshake or checkmate.

Implementation note: after all the above logic is implemented, we can simply record the taken moves and intermediate board positions into a standardized format (like PGN) and use it to analyze the game later on. Thi step is not covered in this post, but it is the final step of the pipeline.

## First Attempt at Prediction: The Siamese Net That Said “gg ez” On Our Dataset  

<figure style="margin: 2em auto; text-align: center; max-width: 100%;">
  <img 
    src="/blog/images/chess-tracker/initial-idea.png" 
    alt="Diagram of the Siamese network architecture for chess tracking (first idea)" 
    style="width: 100%; height: auto; border-radius: 8px;" 
  />
  <figcaption style="margin-top: 0.5em; font-size: 0.95em; color: var(--text-muted-color);">
    Diagram of the Siamese network architecture (first idea).
  </figcaption>
</figure>


We started with a (possibly too) simple idea: shove the **before** and **after** images through the **same encoder network**, concatenate the embeddings together, and ask an MLP to predict the move in UCI format, i.e., *from-square*, *to-square*, and *promotion*.

You can see the diagram above for a simplified version of [this](https://github.com/paolo-notaro/chess-detector/image_pairs_models.py) pipeline.

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

# Actual move prediction model (SiameseNet)
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

For embedding extraction, we tried different architectures, including our own Convolutional Neural Network (CNN), [ResNet18](https://pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html) and Vision Transformer (ViT), but the first one was the most promising.

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

We generate a variety of images, differing in board look, piece style, and lighting conditions. This helps the model generalize better to different environments.

---

## Step 3: Dataset Preparation for Machine Learning

The generated images and move data are used to create PyTorch datasets. For example, the `ChessMoveDatasetFromCSV` class loads preprocessed images and labels them with move information (e.g., source square, target square, promotion). This dataset can be directly used to train models for the move prediction task.

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

## So, what went wrong?

[]

Our network simply couldn't learn the task. We tried different architectures, hyperparameters, and training strategies, but the model just wouldn't generalize. 
We also figured it could be due to exceeding model capacity - so we reduce the number of parameters and tried again, but the results were still disappointing.
We ended up with a model that was overfitting like crazy, and we couldn't figure out why.

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


## The Aha! Moment – Go Small, Go Local, Go Δ

*"RIP SiameseNet (2024-2024) ✝️”*

We had an aha-moment: a legal move only changes two squares*: one goes empty, one gets a visitor. So instead of letting a big CNN stare at the whole board, we applied a magnifying glass over every square and asked, "Did **you** change?" 
It's like running 64 tiny analyses rather than one blurry wide-angle shot.

But in order for this to work, cells need to be comparable across images. So we needed to make sure that the images were aligned and that the pieces were in the same position in both images.

This is where the **homography** comes in. We used OpenCV to compute a homography matrix that maps the corners of the chessboard in the first image to the corners of the chessboard in the second image. This allows us to warp the images so that they are aligned.

In short, the new pipeline looks like this:
1. **Offline calibration** – use OpenCV to compute a homography matrix that maps the corners of the chessboard in the first image to the corners of the chessboard in the second image.
2. **Capture before / after frames** – take a picture of the board before and after the move.
3. **Pre-process** – warp the images using the homography matrix, convert to grayscale, and normalize.
4. **Diff** – compute the absolute difference between the two images and normalize it.
5. **Slice into 64 patches** – crop the diff image into 64 patches, one for each square on the board.
6. **PatchEncoder** – apply a small CNN to each patch to extract features.
7. **MoveScorer** – compute a similarity score between the patches to predict the move.
8. **Legality filter** – use a chess engine (like python-chess) to check if the predicted move is legal.
9. **Output** – if the move is legal, update the board state and continue to the next turn.
10. **Save the game** – save the moves and board state to a PGN file for later analysis.
11. **Repeat** – go back to step 2 until the game is over.

*Note: this is not _entirely_ true, as some moves can involve more than two squares (e.g. castling, en-passant, and promotions). However, these cases are rare and can be handled separately.


In particular, we took the following steps:

### Step 1: adapt our data generation pipeline

We add one more step at the end of the data generation pipeline, to standardize images across poses and compute difference between before and after frames.

```python
def gen_diff(before_img, after_img, binary=False, binary_threshold=30):
    diff_img = cv.absdiff(before_img, after_img)

    if binary:
        _, diff_img_binary = cv.threshold(diff_img, binary_threshold, 255, cv.THRESH_BINARY)
        return diff_img_binary
    else:
        return diff_img

diff_img = postprocessing.gen_diff(before_img, after_img)
postprocessing.save_image(diff_img, os.path.join(DIFF_PATH, f"{i}.png"))
```

In this way we generate standardized __diff images__ that highlight the changes between the "before" and "after" board states. These images are saved in the `diff` directory and can be used for tasks like move prediction or board state classification.

[diff image example]

### Step 2: adapt our model

We ditched the Siamese network and moved to a new architecture, which we call **PatchEncoder**. This model has two key differences: 
1) focuses on the __differences between the two images__, by applying a small CNN  directly to the diff image, 
2) __operates on patches__, i.e. cell-wise crops of the diff image, which contain the pixels of a single square.

Here is the rough patching algorithm:

```python
def patch_image(img: np.ndarray, resize_size: int = None) -> torch.Tensor:
    # Split into 64 (32x32) patches
    patches = []
    PATCH_SIZE = img.shape[0] // 8  # e.g. input size 224 // 8 = 28
    for rank in range(1, 9):
        for file in "abcdefgh":
            file_index = ord(file) - ord('a')
            patch_index = SQUARE_TO_IDX[f"{file}{rank}"]
            # consider that a1 is in lower right corner
            row = 7 - file_index
            col = 8 - rank
            x = col * PATCH_SIZE
            y = row * PATCH_SIZE
            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            ...
            patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)  # (1, 32, 32)
            patches.append(patch)

    patches = torch.stack(patches)  # (64, 1, 32, 32)
    return patches
```

The new **PatchEncoder** model takes the diff image as input (cropped into 64 patches) and applies a series of convolutional layers to extract features. The output is a set of embeddings for each patch, which are then used to predict the move.
Here is the rough code for the **PatchEncoder** model:

```python
class ConvPatchEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # [B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 8, 8]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 128, 4, 4]
        )

        # Final projection to embed_dim
        self.fc = nn.Linear(128 * 4 * 4, embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = patches.shape
        patches = patches.view(B * N, C, H, W)  # [B*64, 1, 32, 32]
        features = self.cnn(patches)  # [B*64, 128, 4, 4]
        features = features.view(B * N, -1)  # [B*64, 2048]
        embeddings = self.fc(features)  # [B*64, embed_dim]
        embeddings = embeddings.view(B, N, -1)  # [B, 64, embed_dim]
        return embeddings

```

This forces the model to focus on learning whether a square was involved in the move or not, rather than trying to learn the whole board state.

### Step 3: Adapt our Move Predictor

We also changed the way we compute the final move
.
In our new **MoveScorer** model, we use an attention-like mechanism to compute the relationship between the 64 cells of the board. The model takes the 64 cell embeddings from the **PatchEncoder** and
1) adds positional encodings to each patch to retain spatial information
2) computes two linear projections of the embeddings: one for the "from" square and one for the "to" square. This enforces the model to learn the relationship between the patches and the move.
2) computes a similarity score between the "from" and "to" projections, which is then used to predict the move. The output is a 64x64 matrix of logits, which represents the similarity between each pair of patches. These logits are then divided by a learned temperature parameter to control the sharpness of the distribution.

This is the rough implementation of the **MoveScorer** model:

```python
class MoveScorer(nn.Module):
    def __init__(self, embed_dim=128, proj_size=32):
        super().__init__()
        self.from_proj = nn.Linear(embed_dim, proj_size)
        self.to_proj = nn.Linear(embed_dim, proj_size)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, 64, embed_dim]
        from_vecs = self.from_proj(embeddings)  # [B, 64, proj_size]
        to_vecs = self.to_proj(embeddings)  # [B, 64, proj_size]

        # Compute pairwise move scores using batched matrix multiplication and
        # normalize by learned temperature
        scores = (
            torch.matmul(from_vecs, to_vecs.transpose(1, 2)) * self.temperature
        )  # [B, 64, 64]
        return scores
```

Putting all together, the new model architecture looks like this:

```python
class ChessMoveModel(nn.Module):
    def __init__(
        self,
        patch_size=32,
        in_channels=1,
        embed_dim=128,
    ):
        super().__init__()
        self.encoder = ConvPatchEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.scorer = MoveScorer(embed_dim=embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(64, embed_dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, 64, C, H, W]
        embeddings = self.encoder(patches)  # [B, 64, embed_dim]
        embeddings = embeddings + self.positional_encoding.unsqueeze(
            0
        )  # [B, 64, embed_dim]
        scores = self.scorer(embeddings)  # [B, 64, 64]
        return scores
```

The final prediction scores of shape [64, 64] are then passed to a legality filter (using `python-chess`) to check if the most likely move is legal. If it is, we can update the board state and continue to the next turn.


## Training and Results



## Putting It All Together: Create a User-Friendly Chess Tracker App

Having the model work with good accuracy is great, but we also need to make it user-friendly and easy to use. In particular, we need to make sure that the app can be used in real-time, with minimal setup and configuration.

We create a simple GUI app with tkinter that allows the user to:
* calibrate the board (get the corners of the chessboard)
* record a move (take a picture of the board before and after the move)
* display the predicted move
* display the current board state
* save the game to a PGN file

The main components of the app are:
1. **Tripod** – a simple tripod to hold the smartphone camera in place. The camera should be positioned above the board, with a clear view of all squares. We typically apply it on the left side, point at about 60 degrees angle down and at few centimenters from the board.

1. **Camera** – use a smartphone with an IP camera app (like DroidCam) to stream the video to the computer. The app captures frames from the camera and processes them in real-time.

2. **Computer** – a laptop or desktop computer running the app. The app receives frames from the camera, processes them, and displays the predicted move. It also helps the user to calibrate the board, track the state of the game and save the game to a PGN file.



## Strengths, Weaknesses & Next Steps

**✔ Strengths**  
* Works on *any* board, any pieces, no fancy RFID or magnets required.  
* Zero extra hardware – just your phone cam, a tripod, and a laptop.

**✘ Weaknesses**  
* Finicky lighting or a tiny tripod bump can scramble the diff image.  
* Pawn promotions? Still on our TODO list (sorry, under-appreciated queens).

**On our Roadmap**  
1. **Bigger real-world dataset** – cafés, club nights, dodgy basement lighting.  
2. **Aggressive augmentation** – glare jitter, motion blur, piece style swaps.  
3. **Dual-exposure trick** – snap twice, merge for better low-light diff maps.  
4. **Promotion-head revival** – give pawns their rightful upgrade path.  
5. **Online board tracker** – continual pose re-calibration to shrug off nudges.

---

## Fun-Size Takeaways

* **Synthetic > manual** – Blender renders beat hours of hand-labeling of real images
* **Zoom then learn** - focus on the changed squares and even tiny CNNs can shine  
* **Simple beats slick** – two photos + clever math trump sensor-stuffed boards  


> **Sensors? Nah. We’re running this game on raw RGB.**


Big thanks to [Lorenzo Notaro](https://github.com/lorenzonotaro1) for being my companion on this project, his help with the Blender pipeline and support in the long brainstorming and debugging sessions (besides, you have a better GPU than I have).


# References

[1] DGT Electronic Chess Boards overview – chesshouse.com – <https://www.chesshouse.com/collections/dgt-electronic-chess-boards-pc-connect>  
[2] Chessnut Pro sensor discussion – chessbazaar.com – <https://www.chessbazaar.com/blog/enhancing-your-chess-experience-with-custom-chess-sets-from-chessbazaar-dgt-sensors-and-chessnut-pro-sensors/>  