import glob
from PIL import Image

def make_gif(lbl):
    frames = [Image.open(image) for image in glob.glob(f"gif/{lbl}/*.png")]
    frame_one = frames[0]
    frame_one.save(f"{lbl}_my_awesome.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)
make_gif('5')
