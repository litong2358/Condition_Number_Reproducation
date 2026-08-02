"""Build D_H = the full Stanford Cars set (16,185 images), as described in the paper (App. C.1).

The existing data/cars only holds 6,598 images (a prior 80/20 split of cars_train alone),
which shrinks an epoch of the shorter loader from 253 to 103 steps and therefore
under-trains the immunization by ~2.5x relative to Tab. 4's "3 epochs".

Cars labels are never used: models/model.py:reg_loss(X1, X2) consumes X2 only, and Tab. 3
reports no D_H accuracy. cars_train images are placed in their true class folders; the
8,041 cars_test images (whose labels ship separately from the devkit) go into a single
"unlabeled" folder purely so ImageFolder can enumerate them.
"""
import os
from scipy.io import loadmat

RAW = "/home/grads/tongli01/CN-fresh/data/cars_raw"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cars_full")


def safe(name):
    return str(name).replace("/", "-").replace(" ", "_")


def main():
    devkit = os.path.join(RAW, "car_devkit", "devkit")
    meta = loadmat(os.path.join(devkit, "cars_meta.mat"), squeeze_me=True)
    annos = loadmat(os.path.join(devkit, "cars_train_annos.mat"), squeeze_me=True)["annotations"]
    class_names = list(meta["class_names"])

    n = 0
    for a in annos:
        cname = safe(class_names[int(a["class"]) - 1])
        dst_dir = os.path.join(OUT, "train", cname)
        os.makedirs(dst_dir, exist_ok=True)
        src = os.path.join(RAW, "cars_train", "cars_train", str(a["fname"]))
        dst = os.path.join(dst_dir, str(a["fname"]))
        if not os.path.exists(dst) and os.path.exists(src):
            os.symlink(src, dst)
            n += 1

    unl = os.path.join(OUT, "train", "unlabeled")
    os.makedirs(unl, exist_ok=True)
    m = 0
    for f in sorted(os.listdir(os.path.join(RAW, "cars_test", "cars_test"))):
        if not f.lower().endswith(".jpg"):
            continue
        dst = os.path.join(unl, "test_" + f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(RAW, "cars_test", "cars_test", f), dst)
            m += 1

    # prepare_data() in dataset.py asserts this path exists for d2_name == "cars"
    os.makedirs(os.path.join(OUT, "cars_train"), exist_ok=True)

    total = sum(len(fs) for _, _, fs in os.walk(os.path.join(OUT, "train")))
    print(f"linked {n} train + {m} test images -> {OUT}/train  (total files: {total})")
    print("classes:", len(os.listdir(os.path.join(OUT, "train"))))
    print(f"steps/epoch at batch_size=64: {total // 64}")


if __name__ == "__main__":
    main()
