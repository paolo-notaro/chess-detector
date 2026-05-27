"""Lichess dataset download utilities."""

import os

import requests
import zstandard as zstd

LICHESS_URL = "https://database.lichess.org/standard/"


def download_file(url, output_path):
    """Download a file from a given URL and save it to the specified path."""

    response = requests.get(url, stream=True)

    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        print(f"Size (MB): {int(content_length) / 1_000_000:.2f}.")

    response.raise_for_status()  # Raise an error for bad responses

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"File downloaded: {output_path}")


def extract_zst(input_path, output_path):
    """Extract a .zst file to the specified output path."""

    with open(input_path, "rb") as compressed_file, open(output_path, "wb") as decompressed_file:
        dctx = zstd.ZstdDecompressor()
        dctx.copy_stream(compressed_file, decompressed_file)

    print(f"File extracted: {output_path}")


def download_from_lichess(
    name="lichess_db_standard_rated_2016-03", output_path="lichess_truncated.pgn", keep_n=10000
):
    """Download a compressed PGN file from Lichess and extract it to a specified path.
    Strip the file to keep only the first n games.
    """

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Processed PGN file already exists: {output_path}")
        return

    print(
        "Downloading and extracting positions dataset from Lichess. This whole process may take a while..."
    )

    if not os.path.exists(f"{name}.pgn.zst"):
        print("Downloading compressed file... ", end="")
        download_file(LICHESS_URL + name + ".pgn.zst", f"{name}.pgn.zst")

    if not os.path.exists(f"{name}.pgn"):
        print("Extracting compressed file...", end="")
        extract_zst(f"{name}.pgn.zst", f"{name}.pgn")

    print("Stripping and truncating file...", end="")
    with open(f"{name}.pgn") as file, open(output_path, "w") as output_file:
        lines = [line for line in file.readlines() if line.startswith("1")]
        output_file.writelines(lines[:keep_n])
    print(f"Processed PGN file saved: {output_path}")

    print("Cleaning up...", end="")
    os.remove(f"{name}.pgn.zst")
    os.remove(f"{name}.pgn")
