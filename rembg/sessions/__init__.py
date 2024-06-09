#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .base import BaseSession
from .dis_anime import DisSession
from .dis_general_use import DisSession as DisSessionGeneralUse
from .sam import SamSession
from .silueta import SiluetaSession
from .u2net_cloth_seg import Unet2ClothSession
from .u2net_custom import U2netCustomSession
from .u2net_human_seg import U2netHumanSegSession
from .u2net import U2netSession
from .u2netp import U2netpSession


session_table: dict[str, BaseSession] = {
    DisSession.name(): DisSession,
    DisSessionGeneralUse.name(): DisSessionGeneralUse,
    SamSession.name(): SamSession,
    SiluetaSession.name(): SiluetaSession,
    Unet2ClothSession.name(): Unet2ClothSession,
    U2netCustomSession.name(): U2netCustomSession,
    U2netHumanSegSession.name(): U2netHumanSegSession,
    U2netSession.name(): U2netSession,
    U2netpSession.name(): U2netpSession,
}
