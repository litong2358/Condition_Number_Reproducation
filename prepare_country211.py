"""Download Country211 and lay it out exactly as the repo's own code paths produce it.

dataset.py has two call sites that disagree about where the data lives:

  dataset.py:279  datasets.Country211(root=self.d2_path, split="train", download=True)
                  -> torchvision extracts to  <d2_path>/country211/{train,valid,test}
  dataset.py:297  create_dataset("image_folder", root=self.d2_path, split="train")
                  -> timm looks for           <d2_path>/train

<d2_path>/train does not exist, so timm's _search_split falls back to <d2_path> itself and
walks the whole tree, pooling train + valid + test into one set labelled by the leaf
directory (the country code). D_H therefore ends up as all 63,300 images rather than the
31,650 of the train split, and one epoch is 989 optimizer steps at batch size 64.

That fallback is what a fresh checkout of the repo does, so it is the behaviour reproduced
here. This script just makes it explicit and independent of whatever else is already under
data/: it downloads once, then builds <d2_path>/country211/{train,valid,test} as symlinks
so create_dataset resolves to the same 63,300 images on any machine.

Download is ~11 GB and is skipped when the data is already present.
"""
import os

from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data")
RAW = os.path.join(ROOT, "country211")                  # torchvision's extract target
OUT = os.path.join(ROOT, "country211_authorlayout")     # d2_path used by the config
SPLITS = ("train", "valid", "test")


def main():
    os.makedirs(ROOT, exist_ok=True)

    # root="data" so the archive lands at data/country211/<split>
    datasets.Country211(root=ROOT, split="train", download=True)

    # Rebuild the nested layout the repo's own paths produce: <d2_path>/country211/<split>.
    # This also stops dataset.py:279 from re-downloading, since torchvision's _check_exists
    # only tests that <d2_path>/country211 is a directory.
    nested = os.path.join(OUT, "country211")
    os.makedirs(nested, exist_ok=True)
    for split in SPLITS:
        target, link = os.path.join(RAW, split), os.path.join(nested, split)
        if os.path.isdir(target) and not os.path.exists(link):
            os.symlink(target, link)

    total = sum(len(fs) for _, _, fs in os.walk(nested, followlinks=True))
    print(f"d2_path : {OUT}")
    print(f"layout  : {OUT}/country211/{{{','.join(SPLITS)}}}")
    print(f"images  : {total}   ({total // 64} steps/epoch at batch_size=64)")
    for split in SPLITS:
        d = os.path.join(RAW, split)
        if os.path.isdir(d):
            n = sum(len(fs) for _, _, fs in os.walk(d))
            print(f"   {split:<6} {n:>6}")


if __name__ == "__main__":
    main()
