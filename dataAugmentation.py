import os
import random
from PIL import Image

def generating_waldos(use_background=True):
    """
    Funció per generar imatges de Waldo i sense Waldo a partir de les capçaleres i fons proporcionats.
    :use_background: Si és True, utilitza fons aleatoris; si és False, utilitza un fons negre.
    """
    # Inicilitzar contador de imatges generades
    im_num = 0

    # Rutes de les carpetes d'imatges
    head_path = "Waldo's Head"
    background_path = "Where's Waldo Images/Cleared"
    no_waldo_path = "Where's Waldo Images/NoWaldo"
    waldo_path = "Where's Waldo Images/Waldo"

    
    head_files = []
    background_files = []

    # Recorre les carpetes d'imatges i afegeix els noms dels fitxers a les llistes
    for root, _, files in os.walk(head_path):
        for file in files:
            head_files.append(os.path.join(root, file))

    for root, _, files in os.walk(background_path):
        for file in files:
            background_files.append(os.path.join(root, file))

    # Recorre els caps de Waldo
    for head_file in head_files:
        for _ in range(2): # Repetir per cada cap 2 vegades
            for background_file in background_files:
                for _ in range(2): # Repetir per cada fons 2 vegades
                    head_image = Image.open(head_file)

                    # Rotar aleatoriament el cap de Waldo (50% de probabilitat)
                    if random.random() < 0.5:
                        head_image = head_image.rotate(random.randint(-15, 15))

                    # Escalar aleatòriament el cap de Waldo (70% de probabilitat)
                    if random.random() < 0.7:
                        scale_factor = random.uniform(0.8, 1.5)
                        width, height = head_image.size
                        head_image = head_image.resize((int(width * scale_factor), int(height * scale_factor)), Image.Resampling.LANCZOS)

                    if use_background:
                        background_image = Image.open(background_file)
                    else:
                        background_image = Image.open("black_background.jpg")

                    # Obtenir dimensions de la imatge de fons i del cap
                    bg_width, bg_height = background_image.size
                    head_width, head_height = head_image.size

                    # Generar posicions aleatòries per al fons i el cap
                    bg_x = random.randint(0, bg_width - 64)
                    bg_y = random.randint(0, bg_height - 64)
                    head_x = random.randint(0, 64 - head_width)
                    head_y = random.randint(0, 64 - head_height)

                    # Retallar la imatge de fons a 64x64 píxels
                    cropped_background = background_image.crop((bg_x, bg_y, bg_x + 64, bg_y + 64))

                    # Guardar la imatge de fons sense Waldo
                    cropped_background.save(os.path.join(no_waldo_path, f"n{im_num}.jpg"))

                    # Afegir el cap de Waldo a la imatge de fons
                    cropped_background.paste(head_image, (head_x, head_y), head_image)

                    # Guardar la imatge amb Waldo
                    cropped_background.save(os.path.join(waldo_path, f"{im_num}.jpg"))

                    im_num += 1

generating_waldos(use_background=True)