import requests
import zstandard as zstd
import os

ZST_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2016-03.pgn.zst"

KEEP_N = 10000  # Number of games to keep
OUTPUT_PATH = "lichess_processed.pgn" # Output path for the processed PGN file


def download_file(url, output_path):


    """Download a file from a given URL and save it to the specified path."""

    response = requests.get(url, stream=True)

    print(f'Size (MB): {int(response.headers.get("Content-Length")) / 1_000_000:.2f}.')

    response.raise_for_status()  # Raise an error for bad responses


    with open(output_path, "wb") as file:


        for chunk in response.iter_content(chunk_size=8192):


            file.write(chunk)


    print(f"File downloaded: {output_path}")





def extract_zst(input_path, output_path):


    """Extract a .zst file to the specified output path."""


    with open(input_path, "rb") as compressed_file:


        with open(output_path, "wb") as decompressed_file:


            dctx = zstd.ZstdDecompressor()


            dctx.copy_stream(compressed_file, decompressed_file)


    print(f"File extracted: {output_path}")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
        print(f"Processed PGN file already exists: {OUTPUT_PATH}")
        exit(0)
    
    print("Downloading and extracting positions dataset from Lichess. This whole process may take a while...")

    if not os.path.exists("lichess_db_standard_rated_2016-03.pgn.zst"):
        print("Downloading compressed file... ", end="")
        download_file(ZST_URL, "lichess_db_standard_rated_2016-03.pgn.zst")

    if not os.path.exists("lichess_db_standard_rated_2016-03.pgn"):
        print("Extracting compressed file...", end="")
        extract_zst("lichess_db_standard_rated_2016-03.pgn.zst", "lichess_db_standard_rated_2016-03.pgn")

    print("Stripping and truncating file...", end="")
    with open("lichess_db_standard_rated_2016-03.pgn", "r") as file, open(OUTPUT_PATH, "w") as output_file:
        lines = [line for line in file.readlines() if line.startswith('1')]
        output_file.writelines(lines[:KEEP_N])
    print(f"Processed PGN file saved: {OUTPUT_PATH}")

    print("Cleaning up...", end="")
    os.remove("lichess_db_standard_rated_2016-03.pgn.zst")
    os.remove("lichess_db_standard_rated_2016-03.pgn")
    print("Done!")
            