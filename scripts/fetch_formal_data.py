"""Fetch configured public OpenSLR archives, with official MD5 verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
import time

import yaml


def fetch(source,directory):
    path=directory/source["name"];url=source["url"]
    if not path.exists():
        temporary=path.with_suffix(".partial")
        offset=temporary.stat().st_size if temporary.exists() else 0
        headers={"User-Agent":"speechwm-M5/0.5"}
        if offset:headers["Range"]=f"bytes={offset}-"
        with urlopen(Request(url,headers=headers),timeout=45) as response:
            resume=offset and response.status==206
            if resume and not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):raise ValueError("Invalid resume response")
            total=offset if resume else 0;last=time.monotonic()
            with temporary.open("ab" if resume else "wb") as output:
                while block:=response.read(1024*1024):
                    output.write(block);total+=len(block)
                    if total>400*1024*1024:raise ValueError("Official archive exceeds 400 MiB bound")
                    if time.monotonic()-last>15:print(path.name,total,"bytes",flush=True);last=time.monotonic()
        temporary.replace(path)
    md5=hashlib.md5();sha=hashlib.sha256()
    with path.open("rb") as f:
        while block:=f.read(1024*1024):md5.update(block);sha.update(block)
    if md5.hexdigest()!=source["md5"]:raise ValueError(f"Official checksum mismatch: {path}")
    record={**source,"size_bytes":path.stat().st_size,"sha256":sha.hexdigest(),"license":"CC BY 4.0"}
    (directory/(path.name+".receipt.json")).write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,default=Path("configs/formal_data.yaml"));parser.add_argument("--only",help="Download just this configured archive")
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text());directory=Path(config["cache_directory"]);directory.mkdir(parents=True,exist_ok=True)
    sources=[s for s in config["archives"] if not args.only or s["name"]==args.only]
    if not sources:raise ValueError("Unknown archive name")
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda source:fetch(source,directory),sources))


if __name__=="__main__":main()
