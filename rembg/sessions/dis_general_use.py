import os

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class DisSession(BaseSession):
    def predict(self, img: PILImage, *args, **kwargs) -> list[PILImage]:
        """
        Predicts the mask image for the input image.

        This method takes a PILImage object as input and returns a list of
        PILImage objects as output. It performs several image processing
        operations to generate the mask image.

        Parameters:
            img (PILImage): The input image.

        Returns:
            list[PILImage]: A list of PILImage objects representing the
        generated mask image.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, (0.485, 0.456, 0.406), (1.0, 1.0, 1.0),
                           (1024, 1024)),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Downloads the pre-trained model file.

        This class method downloads the pre-trained model file from a specified
        URL using the pooch library.

        Parameters:
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            str: The path to the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the model.

        This class method returns the name of the model.

        Parameters:
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model.
        """
        return "isnet-general-use"
