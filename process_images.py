import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

def resize_and_split_images(
    src_root="data/raw",
    dest_root="data/processed",
    image_size=(224, 224),
    split_ratio=(0.7, 0.15, 0.15)
):
    classes = os.listdir(src_root)

    for cls in classes:
        cls_path = os.path.join(src_root, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [img for img in os.listdir(cls_path) if img.lower().endswith(('jpg', 'jpeg', 'png'))]

        # Split into train, val, test
        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - split_ratio[0]), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), random_state=42)

        split_map = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }

        for split_name, split_imgs in split_map.items():
            dest_dir = os.path.join(dest_root, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)

            for img_name in split_imgs:
                src_img_path = os.path.join(cls_path, img_name)
                dest_img_path = os.path.join(dest_dir, img_name)

                try:
                    img = Image.open(src_img_path).convert("RGB")  # Ensure 3 channels
                    img = img.resize(image_size)
                    img.save(dest_img_path)
                except Exception as e:
                    print(f"Skipped {src_img_path}: {e}")

if __name__ == "__main__":
    resize_and_split_images()
