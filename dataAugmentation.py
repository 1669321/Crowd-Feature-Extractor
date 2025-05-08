import os
import random
from PIL import Image

def generating_waldos(use_background=True):
    im_num = 0
    head_path = "Waldo's Head"
    background_path = "Where's Waldo Images/Cleared"
    no_waldo_path = "Where's Waldo Images/NoWaldo"
    waldo_path = "Where's Waldo Images/Waldo"

    head_files = []
    background_files = []

    for root, _, files in os.walk(head_path):
        for file in files:
            head_files.append(os.path.join(root, file))

    for root, _, files in os.walk(background_path):
        for file in files:
            background_files.append(os.path.join(root, file))

    for head_file in head_files:
        for _ in range(2):
            for background_file in background_files:
                for _ in range(2):
                    head_image = Image.open(head_file)

                    if random.random() < 0.5:
                        head_image = head_image.rotate(random.randint(-15, 15))

                    if random.random() < 0.7:
                        scale_factor = random.uniform(0.8, 1.5)
                        width, height = head_image.size
                        head_image = head_image.resize((int(width * scale_factor), int(height * scale_factor)), Image.Resampling.LANCZOS)

                    if use_background:
                        background_image = Image.open(background_file)
                    else:
                        background_image = Image.open("black_background.jpg")

                    bg_width, bg_height = background_image.size
                    head_width, head_height = head_image.size

                    bg_x = random.randint(0, bg_width - 64)
                    bg_y = random.randint(0, bg_height - 64)
                    head_x = random.randint(0, 64 - head_width)
                    head_y = random.randint(0, 64 - head_height)

                    cropped_background = background_image.crop((bg_x, bg_y, bg_x + 64, bg_y + 64))

                    cropped_background.save(os.path.join(no_waldo_path, f"n{im_num}.jpg"))

                    cropped_background.paste(head_image, (head_x, head_y), head_image)

                    cropped_background.save(os.path.join(waldo_path, f"{im_num}.jpg"))

                    im_num += 1

generating_waldos(use_background=True)