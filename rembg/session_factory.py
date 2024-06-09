#!/usr/bin/env python
# -*- coding: utf-8 -*-
import onnxruntime as ort

from .sessions import session_table
from .sessions.base import BaseSession


def new_session(model_name: str = "u2net", *args, **kwargs) -> BaseSession:
    """
    Create a new session object based on the specified model name.

    This function searches for the session class based on the model name in
    the 'session_table' dict. It then creates an instance of the session class
    with the provided arguments. The 'sess_opts' object is created using the
    'ort.SessionOptions()' constructor.

    Parameters:
        model_name (str): The name of the model.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        BaseSession: The created session object.
    """
    session_class: BaseSession = session_table[model_name]
    sess_opts = ort.SessionOptions()
    return session_class(model_name, sess_opts, *args, **kwargs)
