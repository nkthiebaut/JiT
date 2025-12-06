from datasets import load_dataset
import os

ds = load_dataset("zh-plus/tiny-imagenet")

# original Imagenet dataset structure (class implicit from folder)
# imagenet/
#  ├── train/
#  │    ├── n01440764/  # Class folder
#  │    │    ├── n01440764_10026.JPEG
#  │    │    ├── n01440764_10027.JPEG
#  │    │    └── ...
#  │    ├── n01443537/
#  │    │    ├── n01443537_2.JPEG
#  │    │    └── ...
#  │    └── ... (1000 folders in total)
#  ├── val/
#  │    ├── ILSVRC2012_val_00000001.JPEG
#  │    ├── ILSVRC2012_val_00000002.JPEG
#  │    └── ... (50,000 images in total)
#  └── ILSVRC2012_devkit_t12.tar.gz


for split in ["train", "valid"]:
    print(f"Number of examples in {split}: {len(ds[split])}")
    for i, example in enumerate(ds[split]):
        image = example['image']
        label = example['label']
        
        label_dir = f"/home/jovyan/data/scene_text_translation/datasets/tiny-imagenet/{split}/{label}"
        
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        image.save(f"{label_dir}/{i}.png")


