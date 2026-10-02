"""FG-CLIP 2 model code vendored from Hugging Face `qihoo360/fg-clip2-base`.

Source revision 430fbc8a912c86fd4de601381b6245a0edab22f0, Apache License 2.0
(see file headers). Loaded locally so `trust_remote_code` stays disabled.
Only change: the torchvision `roi_align` import is lazy, used solely by region
(bounding-box) pooling, so the global image/text embeddings need no torchvision.
"""
from .configuration_fgclip2 import Fgclip2Config
from .modeling_fgclip2 import Fgclip2Model

SOURCE_REVISION = "430fbc8a912c86fd4de601381b6245a0edab22f0"
