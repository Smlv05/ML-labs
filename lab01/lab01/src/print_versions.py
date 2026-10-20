import sys
from importlib import metadata
from pathlib import Path

REQ = Path(__file__).resolve().parents[2] / "requirements.txt"

def main():
    print(f"Python {sys.version}")
    for line in REQ.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==")[0]
        try:
            print(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"{name}: NOT INSTALLED")

if __name__ == "__main__":
    main()