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
Recreate symlinks needed to be able to zip entire job.
"""

import os
from glob import glob
from configparser import ConfigParser

from openquake.hazardlib import nrml


# %% definitions


def file_contains(file_name, string):

    if os.path.islink(file_name):
        file_name = os.path.realpath(file_name)

    with open(file_name, 'r') as file:
        return next((line for line in file
                     if string in line), None) is not None


def update_symlink(file_name, link_name):

    if not os.path.isfile(file_name):
        action = 'No Target'
    else:
        relative_file_name = os.path.relpath(file_name,
                                             os.path.dirname(link_name))

        if os.path.islink(link_name):
            if os.readlink(link_name) == relative_file_name:
                action = 'Exists'
            else:
                action = 'Fixing'
                os.unlink(link_name)
        elif os.path.isfile(link_name):
            action = 'Replacing'
            os.remove(link_name)
        elif os.path.exists(link_name):
            action = 'Problem'
        else:
            action = 'New'

    print('%s: %s ==> %s' % (action, link_name, file_name))

    if action not in ('No Target', 'Exists'):
        os.symlink(relative_file_name, link_name)


# %% main

logic_tree_path = 'Logic Trees'
source_model_path = 'Source Models'

for ini_file_full in sorted(glob(os.path.join('Jobs', '**', '*.ini'))):

    job_path = os.path.dirname(ini_file_full)
    print('\nJOB: ' + job_path)
    config = ConfigParser()
    config.read(ini_file_full)

    gsim_logic_xml = config['calculation']['gsim_logic_tree_file']
    gsim_logic_xml_full = os.path.join(logic_tree_path, gsim_logic_xml)
    update_symlink(gsim_logic_xml_full,
                   os.path.join(job_path, gsim_logic_xml))

    gsim_logic_pdf = gsim_logic_xml.replace('.xml', '.pdf')
    gsim_logic_pdf_full = gsim_logic_xml_full.replace('.xml', '.pdf')
    update_symlink(gsim_logic_pdf_full,
                   os.path.join(job_path, gsim_logic_pdf))

    source_logic_xml = config['calculation']['source_model_logic_tree_file']
    source_logic_xml_full = os.path.join(logic_tree_path, source_logic_xml)
    update_symlink(source_logic_xml_full,
                   os.path.join(job_path, source_logic_xml))

    source_logic_pdf = source_logic_xml.replace('.xml', '.pdf')
    source_logic_xml_pdf = source_logic_xml_full.replace('.xml', '.pdf')
    update_symlink(source_logic_xml_pdf,
                   os.path.join(job_path, source_logic_pdf))

    if not os.path.isfile(source_logic_xml_full):
        print(source_logic_xml_full, 'doesn''t exist; skipping.')
        continue

    root = nrml.read(source_logic_xml_full)

    for branch in root.logicTree.logicTreeBranchingLevel.logicTreeBranchSet:
        sources_xml = branch.uncertaintyModel.text.strip()
        sources_full = os.path.join(source_model_path, sources_xml)
        update_symlink(sources_full, os.path.join(job_path, sources_xml))
