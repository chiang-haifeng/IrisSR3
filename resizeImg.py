import os
from PIL import Image
from torchvision.transforms import Resize

data_dir = 'dataset/Iris224train'
obj_path = 'dataset/Iris128train'
for file in os.listdir(data_dir):
    img_path = os.path.join(data_dir, file).replace('\\', '/')
    image = Image.open(img_path)
    restore = Resize((128,128), interpolation=Image.BICUBIC)
    image = restore(image)
    new_path = os.path.join(obj_path, file)
    image.save(new_path)
