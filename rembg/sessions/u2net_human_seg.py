import os

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class U2netHumanSegSession(BaseSession):
    """This class represents a session for performing human segmentation using
    the U2Net model.
    """

    def predict(self, img: PILImage, *args, **kwargs) -> list[PILImage]:
        """
        Predicts human segmentation masks for the input image.

        Parameters:
            img (PILImage): The input image.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[PILImage]: A list of predicted masks.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)
            ),
        )
        pred = ort_outs[0][:, 0, :, :]
        ma, mi = np.max(pred), np.min(pred)
        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)
        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)
        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Returns:
            str: The path to the downloaded model weights.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the U2Net model.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The name of the model.
        """
        return "u2net_human_seg"
