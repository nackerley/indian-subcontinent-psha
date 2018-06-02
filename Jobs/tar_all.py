#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Indian Subcontinent PSHA
# Copyright (C) 2016-2018 Nick Ackerley
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Tar everything, following links, in given directory (defaults to current).
"""

import os
import tarfile


directory = os.getcwd()
for sub_dir in os.listdir(directory):

    # skip files
    if os.path.isfile(sub_dir):
        continue

    # omit current directory
    if sub_dir == directory:
        continue

    # strip path
    sub_dir = os.path.split(sub_dir)[1]

    # omit hidden directories
    if sub_dir[0] == '.':
        continue

    print('Compressing: tar -hzcvf "%s.tar.gz" "%s"' % (sub_dir, sub_dir))
    with tarfile.open(sub_dir + '.tar.gz', 'w:gz', dereference=True) as tar:
        tar.add(sub_dir)
