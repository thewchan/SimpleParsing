import sys
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import *
from typing import Optional

import PIL
from PIL import ImageFilter
from PIL.Image import Image
from simple_parsing import ArgumentParser, mutable_field, subparsers
from simple_parsing.helpers import Serializable


@dataclass
class Command(Serializable):
    def __post_init__(self):
        print(f"Creating command {self}")


@dataclass
class Read(Command):
    img_path: str = "image.jpeg" 
    def __call__(self) -> Image:
        print(f"Loading image from path {self.img_path}")
        with open(self.img_path, "rb") as fp:
            img = PIL.Image.open(fp)
            img.load()
            return img


@dataclass
class Resize(Command):
    scale: float = 0.5
    def __call__(self, image: Image):
        print(f"Resizing image with scale {self.scale}")
        in_size = image.size
        out_size = tuple(int(s * self.scale) for s in in_size) 
        return image.resize(size=out_size)


@dataclass
class Sharpen(Command):
    radius: float = 0.2
    def __call__(self, image: Image):
        print(f"Sharpening image with radius {self.radius}")
        # not sure how to use the radius, but you get the idea.
        return image.filter(ImageFilter.SHARPEN)


@dataclass
class Save(Command):
    quality: float = 2
    file: Path = Path("image_edited.jpg")

    def __call__(self, image: Image):
        print(f"Saving image to path {self.file} with quality {self.quality}")
        # not sure how to use the quality, but you get the idea.
        image.save(self.file, quality=self.quality)


@dataclass
class ImageProcessingPipeline:
    read_cmd: Read = mutable_field(Read)
    resize_cmd: Optional[Resize] = None
    sharpen_cmd: Optional[Sharpen] = None
    save_cmd: Save = mutable_field(Save)

    def run(self):
        """
        Here is where you execute the 'Command' objects.

        You could use multiprocessing if you want.
        """
        img = self.read_cmd()
        if self.resize_cmd:
            img = self.resize_cmd(img)
        if self.sharpen_cmd:
            img = self.sharpen_cmd(img)
        self.save_cmd(img)
        img.show()


def main():
    """
    Example input:

    """
    parser = ArgumentParser()
    parser.add_arguments(ImageProcessingPipeline, dest="pipeline")
    args = parser.parse_args()
    
    pipeline: ImageProcessingPipeline = args.pipeline
    pipeline.run()

if __name__ == "__main__":
    main()
