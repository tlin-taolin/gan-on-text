# -*- coding: utf-8 -*-
"""print while store the records."""
import os
from os.path import join
import time

import code.utils.opfiles as opfile


def myprint(content, path=join(os.getcwd(), "record")):
    """print the content while store the information to the path."""
    content = time.strftime("%Y:%m:%d %H:%M:%S") + "\t" + content
    print(content)
    opfile.write_txt(content + "\n", path, type="a")
