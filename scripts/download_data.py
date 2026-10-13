"""Download the UCI 'Default of Credit Card Clients' dataset (id 350) and verify it."""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
XLS = "default of credit card clients.xls"
SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"


def main(target: Path = Path("data/raw")) -> None:
    target.mkdir(parents=True, exist_ok=True)
    xls = target / XLS
    if not xls.exists():
        zpath = target / "uci_350.zip"
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extract(XLS, target)
    digest = hashlib.sha256(xls.read_bytes()).hexdigest()
    if digest != SHA256:
        sys.exit(f"checksum mismatch for {XLS}: {digest}")
    print(f"ok  {xls}")


if __name__ == "__main__":
    main()
