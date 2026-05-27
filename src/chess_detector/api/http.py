"""Flask HTTP API exposing trained chess move models.

This module is invoked as the ``chess-detector-api`` console script.
"""

import base64
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import torch
from flask import Flask, jsonify, make_response, request

from chess_detector.data.dataset import ChessMoveFromDiffDataset
from chess_detector.inference.predict import predict_move
from chess_detector.models.diff import ChessMoveModel, ConvPatchEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModelPredictor = Callable[
    [ChessMoveModel, np.ndarray, str | None, str, int], list[tuple[str, float]]
]


@dataclass(frozen=True)
class AvailableModel:
    description: str
    create_model: Callable[[], ChessMoveModel]
    predict: ModelPredictor
    checkpoints: list[str]


SESSIONS: dict[str, tuple[str, str]] = {}

# Dict with:
# - tuple of (model_name, checkpoint_name) as key
# - model instance as value
# This is used to load the model only once and reuse it for all requests
LOADED_MODELS: dict[tuple[str, str], ChessMoveModel] = {}

# Dict information about available models
AVAILABLE_MODELS: dict[str, AvailableModel] = {
    "conv-encoder-256-v1": AvailableModel(
        description="Chess Move Diff Prediction Model with Conv Encoder",
        create_model=lambda: ChessMoveModel(embed_dim=256, encoder_class=ConvPatchEncoder),
        predict=lambda model, in_image, board_fen, turn, topk: predict_move(
            model,
            ChessMoveFromDiffDataset.patch_image(
                ChessMoveFromDiffDataset.preprocess_image(in_image, 224), 32
            ),
            device,
            board_fen=board_fen,
            turn=turn,
            topk=topk,
        ),
        checkpoints=["mercurial-stag-264_epoch11"],
    )
}

app = Flask(__name__)


@app.route("/session/begin", methods=["POST"])
def begin_session():
    """
    Begins a new session with the model and checkpoint specified in the request body.
    Sets the session_id as a cookie in the response.
    Request body format:
    {
        "model": "model_name" [Required],
        "checkpoint": "checkpoint_name" [Default: first checkpoint of the model]
    }
    """
    data = request.get_json() or {}
    model_name = data.get("model")

    if not isinstance(model_name, str) or model_name not in AVAILABLE_MODELS:
        return make_response(jsonify({"error": "Model not found"}), 404)

    model_info = AVAILABLE_MODELS[model_name]
    checkpoint_name = data.get("checkpoint", model_info.checkpoints[0])

    if not isinstance(checkpoint_name, str) or checkpoint_name not in model_info.checkpoints:
        return make_response(jsonify({"error": "Checkpoint not found"}), 404)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = (model_name, checkpoint_name)

    # If the model/checkpoint is already loaded, reuse it
    if (model_name, checkpoint_name) not in LOADED_MODELS:
        model = model_info.create_model()
        model.load_state_dict(
            torch.load(f"models/checkpoint_{checkpoint_name}.pth", map_location=device)[
                "model_state_dict"
            ]
        )
        model.to(device)
        LOADED_MODELS[(model_name, checkpoint_name)] = model

    response = make_response(jsonify({"sessionId": session_id}))
    response.set_cookie("session_id", session_id, httponly=True)
    return response


@app.route("/session/end", methods=["POST"])
def end_session():
    """
    Ends the session.
    Deletes the session_id cookie from the response.
    """
    session_id = request.cookies.get("session_id")
    if session_id is not None:
        SESSIONS.pop(session_id, None)

    response = make_response(jsonify({"message": "Session ended"}))
    response.set_cookie("session_id", "", expires=0)
    return response


@app.route("/models", methods=["GET"])
def get_models():
    """
    Returns a list of available models and their checkpoints.
    """
    models_info = list()
    for model_name, model_info in AVAILABLE_MODELS.items():
        models_info.append(
            {
                "name": model_name,
                "description": model_info.description,
                "availableCheckpoints": model_info.checkpoints,
            }
        )
    return jsonify(models_info)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts the move based on the input image and the given metadata.
    Request body format:
    {
        "image": "base64 encoded grayscale PNG image, size 224x224" [Required],
        "boardFen": <FEN string of the board before the move> [Default null],
        "turn": <'w', 'b' or 'wb'> [Default 'wb'],
        "topk": <number of top moves to return> [Default 1]
    }
    """
    if "session_id" not in request.cookies:
        return make_response(jsonify({"error": "Session not started"}), 400)

    session_id = request.cookies.get("session_id")
    if session_id not in SESSIONS:
        return make_response(jsonify({"error": "Session not started"}), 400)

    model_name, checkpoint_name = SESSIONS[session_id]
    model = LOADED_MODELS[(model_name, checkpoint_name)]

    data = request.get_json() or {}

    image_data = data.get("image")

    if not image_data:
        return make_response(jsonify({"error": "Image not provided"}), 400)

    # Decode the base64 image data
    try:
        image_data = base64.b64decode(image_data)
    except Exception:
        return make_response(jsonify({"error": "Invalid image data"}), 400)

    # Convert the image data to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    in_image = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)

    if in_image is None:
        return make_response(jsonify({"error": "Invalid image data"}), 400)

    # Check if the image is grayscale and has the correct size
    if len(in_image.shape) != 2 or in_image.shape[0] != 224 or in_image.shape[1] != 224:
        return make_response(jsonify({"error": "Image must be grayscale and size 224x224"}), 400)

    board_fen = data.get("boardFen", None)
    if board_fen is not None and not isinstance(board_fen, str):
        return make_response(jsonify({"error": "Board FEN must be a string"}), 400)

    turn = data.get("turn", "wb")

    if not isinstance(turn, str) or turn not in ["w", "b", "wb"]:
        return make_response(jsonify({"error": "Turn must be 'w', 'b' or 'wb'"}), 400)

    topk = data.get("topk", 1)

    if not isinstance(topk, int) or topk < 1:
        return make_response(jsonify({"error": "Topk must be a positive integer"}), 400)

    predictions = AVAILABLE_MODELS[model_name].predict(model, in_image, board_fen, turn, topk)

    predictions = [{"moveUci": pred[0], "confidence": pred[1]} for pred in predictions]

    response = make_response(jsonify(predictions))
    response.headers["Content-Type"] = "application/json"
    return response


def main() -> None:
    """Console-script entry point: start the Flask development server."""
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
