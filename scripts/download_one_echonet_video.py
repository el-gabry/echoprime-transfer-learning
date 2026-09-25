from __future__ import annotations

from pathlib import Path
import zipfile

import redivis


TABLE_REF = "aimi.echonet_dynamic:66s1:v1_0.echonet:fjdn"

ZIP_NAME = "EchoNet-Dynamic.zip"

MEMBER = "EchoNet-Dynamic/Videos/0X100009310A3BD7FC.avi"

OUTPUT = Path(
    "data/EchoNet-Dynamic/Videos/0X100009310A3BD7FC.avi"
)


def main() -> None:
    table = redivis.table(TABLE_REF)
    remote_zip = table.file(ZIP_NAME)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("Opening remote EchoNet archive...")

    with remote_zip.open("rb") as remote_stream:
        with zipfile.ZipFile(remote_stream) as archive:

            if MEMBER not in archive.namelist():
                raise FileNotFoundError(
                    f"{MEMBER} not found inside archive."
                )

            info = archive.getinfo(MEMBER)

            print(f"Member: {MEMBER}")
            print(f"Compressed: {info.compress_size / 1024:.1f} KB")
            print(f"Uncompressed: {info.file_size / 1024:.1f} KB")

            with archive.open(MEMBER) as src:
                with OUTPUT.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)

                        if not chunk:
                            break

                        dst.write(chunk)

    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()