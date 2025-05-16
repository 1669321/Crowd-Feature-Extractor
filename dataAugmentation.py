import os
from PIL import Image
import random

# Llista d'arxius dins del directori "Wheres Waldo Images/Raw"
batches = os.listdir("Wheres Waldo Images/Raw")

def generating_waldos(use_background=True):
    """Funció per generar i guardar imatges no-existents de Waldo i No-Waldo per l'entrenament.
    :param use_background: Si és True, s'utilitza un fons real. Si es False, un fons negre.
    :return: None.
    """
    background_use_num = 2 # Número de cops que es farà servir cada fons
    head_use_num = 2 # Número de cops que es farà servir cada cap
    size = 64 # Les imatges resultants seran de 64x64 píxels. La mida del cap de Wally sol estar en un quadre de 60x60
    im_num = 0 # Contador d'imatges generades

    # Itera per cada imatge de cap de Waldo
    for head_batch in os.listdir("Wheres Waldo Images/Cleared/WaldosHead"):
        head_name = os.path.join("Wheres Waldo Images/Cleared/WaldosHead", head_batch)
        
        for _ in range(head_use_num):
            # Itera per cada imatge de fons (sense Waldo!!)
            for back_batch in os.listdir("Wheres Waldo Images/Cleared/ClearedWaldos"):
                back_name = os.path.join("Wheres Waldo Images/Cleared/ClearedWaldos", back_batch)
                
                for _ in range(background_use_num):

                    # Rotació aleatòria del cap de Waldo (50% de probabilitats)
                    if random.randint(0, 9) < 5:
                        num = random.randint(-15, 15)
                        foreground = Image.open(head_name).rotate(num)
                    else:
                        foreground = Image.open(head_name)

                    # Escala aleatòria del cap de Waldo (70% de probabilitats)
                    if random.randint(0, 9) < 7:
                        scale = random.uniform(0.8, 1.5)
                        w, h = foreground.size
                        foreground = foreground.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                    
                    # Selecció del fons real o negre
                    if use_background:
                        background = Image.open(back_name)
                    else:
                        background = Image.open("Wheres Waldo Images/Cleared/black_background.jpg")

                    # Obtenim les dimensions de la imatge de fons i la del cap
                    bck_w, bck_h = background.size
                    frg_w, frg_h = foreground.size

                    # Coordenades aleatories per retallar el fons i enganxar-hi el cap
                    bck_x = random.randint(0, bck_w - size)
                    bck_y = random.randint(0, bck_h - size)  
                    frg_x = random.randint(0, 64 - frg_w)
                    frg_y = random.randint(0, 64 - frg_h)

                    # Retallem el fons de mida 64x64
                    cropped = background.crop((bck_x, bck_y, bck_x + size, bck_y + size))

                    # Si cropped és CMYK, el convertim a RGB
                    if cropped.mode == 'CMYK':
                        cropped = cropped.convert('RGB')
                    
                    # Guardem primer la versió sense Waldo (negative image)
                    cropped.save("Wheres Waldo Images/NotWaldo/n" + str(im_num) + ".png")
                    
                    # Enganxa el cap de Waldo sobre el fons retallat
                    cropped.paste(foreground, (frg_x, frg_y), foreground)
                    
                    # Si foreground és CMYK, el convertim a RGB
                    if foreground.mode == 'CMYK':
                        foreground = foreground.convert('RGB')

                    # Guardem la imatge resultant amb Waldo i augmentem el contador
                    cropped.save("Wheres Waldo Images/Waldo/" + str(im_num)+str(use_background) + ".png")
                    im_num += 1


generating_waldos(use_background=True)


