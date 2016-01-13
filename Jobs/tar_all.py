#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tar everything, following links, in given directory (defaults to current)

Created on Wed Jan 13 12:52:50 2016

@author: nick
"""

import os
import tarfile


directory = os.getcwd()
for sub_dir, _, _ in os.walk(directory):
    # omit current directory
    if sub_dir == directory:
        continue

    # strip path
    sub_dir = os.path.split(sub_dir)[1]

    # omit hidden directories
    if sub_dir[0] == '.':
        continue

    # omit export & output directories
    if any([item in sub_dir for item in ['export', 'output']]):
        print('Skipping %s' % sub_dir)
        continue

    print('Compressing: tar -hzcvf "%s.tar.gz" "%s"' % (sub_dir, sub_dir))
    with tarfile.open(sub_dir + '.tar.gz', 'w:gz', dereference=True) as tar:
        tar.add(sub_dir)
