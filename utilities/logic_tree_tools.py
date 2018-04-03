# -*- coding: UTF-8 -*-
# pylint: disable=superfluous-parens
'''
Tools relating to OpenQuake logic trees.
'''

import os
import re
import ast
from io import StringIO
from string import Template

import subprocess
from copy import deepcopy

import numpy as np
import pandas as pd

from openquake.hazardlib import nrml
from openquake.baselib.general import deprecated
from openquake.baselib.node import Node
from openquake.hazardlib.mfd.truncated_gr import TruncatedGRMFD

from toolbox import limit_precision, is_numeric


def read_tree_tsv(file_tsv):
    '''
    Enhances pandas read_csv by parsing lists of models and weights.
    '''
    tree_df = pd.read_csv(file_tsv, delimiter='\t',)
    for key in ['uncertaintyModel', 'uncertaintyWeight']:
        if key in tree_df.columns:
            for _, branch_set in tree_df.iterrows():
                branch_set[key] = ast.literal_eval(branch_set[key])
    return tree_df


MODEL_LENGTHS = {'bGRRelative': 1, 'maxMagGRRelative': 1,
                 'maxMagGRAbsolute': 1, 'abGRAbsolute': 2}


def models_with_weights(uncertainty_type, models, weights=None,
                        prefix=None, validate=True, omit=None, sub=None):
    '''
    Returns tuple of valid models and equal weights summing to unity.
    '''
    if weights is None or len(weights) != len(models):
        weights = [1.]*len(models)

    if sub is not None:
        for i, model in enumerate(models):
            if model in sub.keys():
                print('substituting %s for %s ...' % (sub[model], model))
                models[i] = sub[model]

    model_weights = list(zip(models, weights))

    def name(i, model, prefix_string):
        '''
        Adds an index and a prefix to the model name to help in debugging.
        '''
        model_string = 'model %d "%s"' % (i + 1, model)
        if prefix_string is not None:
            model_string = '%s %s' % (prefix_string, model_string)
        return model_string

    if omit is not None:
        for i, (model, _) \
                in reversed(list(zip(range(len(models)), model_weights))):
            if model in omit:
                print('Omitting %s ...' % name(i, model, prefix))
                model_weights.pop(i)

    if validate:
        for i, (model, _) \
                in reversed(list(zip(range(len(models)), model_weights))):

            if uncertainty_type == 'gmpeModel':
                try:
                    nrml.valid.gsim(model)
                except ValueError:
                    print('%s not valid GSIM. Omitting ...' %
                          name(i, model, prefix))
                    model_weights.pop(i)
            elif uncertainty_type == 'sourceModel':
                if not os.path.isfile(model):
                    print('%s not found. Omitting ...' %
                          name(i, model, prefix))
                    model_weights.pop(i)
            elif uncertainty_type in MODEL_LENGTHS.keys():
                req_size = MODEL_LENGTHS[uncertainty_type]
                if np.array(model).size != req_size:
                    print('%s has elements %d instead of %d. Omitting ...' %
                          (name(i, model, prefix), np.array(model).size,
                           req_size))
                    model_weights.pop(i)
            else:
                print('Unknown uncertainty type: %s', uncertainty_type)
                raise ValueError

    if model_weights:
        models, weights = zip(*model_weights)
        if weights:
            weights = np.array(weights).astype(float)
            weights = (weights/np.sum(weights)).round(3)
            weights[0] = (1 - np.sum(weights[1:])).round(3)
            weights = weights.tolist()
        model_weights = zip(models, weights)

    return model_weights


_NRML_STRING = '''\
<?xml version="1.0" encoding="UTF-8"?>
<nrml
xmlns="http://openquake.org/xmlns/nrml/0.5"
xmlns:gml="http://www.opengis.net/gml"
>
$CONTENT
</nrml>
'''

_LOGIC_TREE_STRING = '''\
    <logicTree
    logicTreeID="$LOGIC_TREE_ID"
    >
$BRANCHING_LEVELS
    </logicTree>'''

_BRANCH_LEVEL_STRING = '''\
        <logicTreeBranchingLevel
        branchingLevelID="$BRANCHING_LEVEL_ID"
        >
$BRANCH_SETS
        </logicTreeBranchingLevel>'''

_BRANCH_SET_STRING = '''\
            <logicTreeBranchSet
            applyToTectonicRegionType="$TECTONIC_REGION_TYPE"
            branchSetID="$BRANCH_SET_ID"
            uncertaintyType="$UNCERTAINTY_TYPE"
            >
$BRANCHES
            </logicTreeBranchSet>'''

_BRANCH_STRING = '''\
                <logicTreeBranch
                branchID="$BRANCH_ID"
                >
                    <uncertaintyModel>
                        $VALUE
                    </uncertaintyModel>
                    <uncertaintyWeight>
                        $WEIGHT
                    </uncertaintyWeight>
                </logicTreeBranch>'''


@deprecated('Use gsim_data_to_tree instead')
def write_gsim_tree_nrml(tree_df, output_file,
                         validate=True, omit=None, sub=None):
    '''
    Deprecated. Does the same job as :func:`gsim_data_to_tree`, but in a
    less robust way.
    '''
    branch_template = Template(_BRANCH_STRING)
    branch_set_template = Template(_BRANCH_SET_STRING)
    branch_level_template = Template(_BRANCH_LEVEL_STRING)

    branching_level_list = []
    for i, level in tree_df.iterrows():

        models_weights = models_with_weights(
            'gmpeModel', level['uncertaintyModel'], weights=None,
            prefix='Level %d' % (i + 1), validate=validate, omit=omit, sub=sub)

        branch_list = []
        for j, (model, weight) in enumerate(models_weights):
            branch_list += [branch_template.substitute(
                BRANCH_ID='r%dm%d' % (i + 1, j + 1),
                VALUE=model,
                WEIGHT=weight)]

        branch_set = branch_set_template.substitute(
            BRANCH_SET_ID='bs%d' % (i + 1),
            UNCERTAINTY_TYPE='gmpeModel',
            TECTONIC_REGION_TYPE=level['applyToTectonicRegionType'],
            BRANCHES='\n'.join(branch_list))
        branching_level_list += [branch_level_template.substitute(
            BRANCHING_LEVEL_ID='bl%d' % (i + 1),
            BRANCH_SETS=branch_set)]

    content = Template(_LOGIC_TREE_STRING).substitute(
        LOGIC_TREE_ID='lt1',
        BRANCHING_LEVELS='\n'.join(branching_level_list))
    document = Template(_NRML_STRING).substitute(CONTENT=content)

    with open(output_file, 'w') as file_obj:
        file_obj.write(document)


def get_template_keys(symoblic_model, all_keys):
    '''
    Parse template and check which keys are needed to evaluate it.
    '''
    template = str(symoblic_model).replace("'", '')
    labels = re.sub(r'[\[\]]', '', template).split(',')
    labels = [label.strip() for label in labels]
    required_keys = set(re.sub(r'[\[\]+,-]', '', template).split())

    for key in list(required_keys):
        if key not in all_keys:
            if key != '0':
                print('Cannot find %s in keys, setting to zero' % key)
            template = re.sub(r'\b' + key + r'\b', '0', template)
            required_keys.discard(key)

    return template, required_keys, labels


def eval_symbolic_model(template, required_keys, series):
    '''
    Given a template with keys, make substitutions from series and evaluate.
    Returns list of numbers.
    '''
    for key in required_keys:
        template = re.sub(r'\b%s\b' % key, str(series[key]), template)

    return np.array(ast.literal_eval(template)).round(6).tolist()


def expand_sources(df_in):
    '''
    For a source model logic tree expand source-specific branches
    '''
    df_out = pd.DataFrame()
    for _, row_in in df_in.iterrows():

        apply_to = row_in['applyToSources']

        if not os.path.isfile(apply_to):
            # just pass the branch level through unmolested
            row_out = row_in.copy()
            df_out = df_out.append(row_out)
            continue

        # when "apply to" is a source table file, add branch level for each row
        df_sources = pd.read_csv(apply_to, '\t')

        template, required_keys, _ = get_template_keys(
            row_in['uncertaintyModel'], df_sources.columns)

        for _, source in df_sources.iterrows():
            if source['mmax'] == 0:
                continue

            # no mmax uncertainty on zones with megathrust twins
            if (any(source['zoneid'] + 'm' == df_sources['zoneid']) and
                    (row_in['uncertaintyType'][:6] == 'maxMag')):
                continue

            model = eval_symbolic_model(template, required_keys, source)

            row_out = row_in.copy()
            row_out['applyToSources'] = source['id']
            row_out['uncertaintyModel'] = model
            df_out = df_out.append(row_out)

    all_sources = df_out['applyToSources'] == 'all'
    df_out = pd.concat((df_out[all_sources], df_out[~all_sources]))
    df_out.index = range(len(df_out))

    return df_out


def branch_mfds(mfds_in, weights_in, labels_in, branch, zone):
    '''
    Apply a branch of a logic tree to existing mfds and cumulate weights.
    '''
    template, required_keys, labels = get_template_keys(
        branch['uncertaintyModel'], zone.keys())
    models = eval_symbolic_model(template, required_keys, zone)
    weights = branch['uncertaintyWeight']

    if branch['uncertaintyType'] in MODEL_LENGTHS.keys():
        mfds_out, weights_out, labels_out = [], [], []
        for mfd_in, weight_in, label_in in zip(mfds_in, weights_in, labels_in):
            for model, weight, label in zip(models, weights, labels):
                mfd = deepcopy(mfd_in)

                if mfd is not None:
                    if branch['uncertaintyType'] == 'bGRRelative':
                        mfd.modify_increment_b(model)
                    elif branch['uncertaintyType'] == 'maxMagGRRelative':
                        if mfd.max_mag + model < mfd.min_mag + mfd.bin_width:
                            mfd = None
                        else:
                            mfd.modify_increment_max_mag(model)
                    elif branch['uncertaintyType'] == 'maxMagGRAbsolute':
                        if model < mfd.min_mag + mfd.bin_width:
                            mfd = None
                        else:
                            mfd.modify_set_max_mag(model)
                    elif branch['uncertaintyType'] == 'abGRAbsolute':
                        mfd.modify_set_ab(model[0], model[1])

                weight_out = weight_in*weight
                if label_in:
                    label_out = ', '.join((label_in, label))
                else:
                    label_out = label

                mfds_out.append(mfd)
                weights_out.append(weight_out)
                labels_out.append(label_out)
    else:
        mfds_out, weights_out, labels_out = mfds_in, weights_in, labels_in

    weights_out = np.array(weights_out)

    return (mfds_out, weights_out, labels_out)


def get_rates(mfds):
    '''
    Convert list of MFDs to matrix of occurrence rates.
    '''
    num_rates = [int(round((mfd.get_min_max_mag()[1] -
                            mfd.get_min_max_mag()[0])/mfd.bin_width)) + 1
                 if mfd is not None else 0
                 for mfd in mfds]

    rates = np.zeros((max(num_rates), len(mfds)))
    for i, mfd in enumerate(mfds):
        if mfd is not None:
            branch_rates = list(zip(*(mfd.get_annual_occurrence_rates())))[1]
            rates[:len(branch_rates), i] = branch_rates

    return rates


def collapse_sources(source_df, source_tree_symbolic_df, bin_width=0.1):
    '''
    Given a tree dataframe of branches and a source model dataframe of zones,
    for every branch affecting MFDs, collapse all possible MFDs into one
    using those weights, add the combined MFD to the zone in the source model
    and finally remove the corresponding branch from the tree.
    '''
    collapsible = np.array([
        branch['uncertaintyType'] in MODEL_LENGTHS.keys()
        for _, branch in source_tree_symbolic_df.iterrows()])

    source_df = source_df.loc[source_df['mmax'] != 0].copy()

    zone_rates, all_rates, all_weights = [], [], []
    for _, zone in source_df.iterrows():

        mfds = [TruncatedGRMFD(min_mag=zone['mmin'], max_mag=zone['mmax'],
                               a_val=zone['a'], b_val=zone['b'],
                               bin_width=bin_width)]
        weights = [1.]
        labels = ['']

        # apply logic tree branches in successsion
        for _, branch in source_tree_symbolic_df.loc[collapsible].iterrows():
            try:
                mfds, weights, labels = branch_mfds(
                    mfds, weights, labels, branch, zone)
            except ValueError:
                print(zone)
                mfds, weights, labels = branch_mfds(
                    mfds, weights, labels, branch, zone)
        rates = get_rates(mfds)

        # compute the weighted sum of the rates for this zones
        collapsed_rates = (rates*weights.reshape(1, -1)).sum(axis=1)
        zone_rates.append(limit_precision(collapsed_rates, 5))

        # save intermediate results from each zone for diagnostic purposes
        all_rates.append(rates)
        all_weights.append(weights)

    source_df['occurRates'] = zone_rates
    source_df['magBin'] = bin_width
    source_df['all_rates'] = all_rates

    return (source_df,
            source_tree_symbolic_df.loc[np.logical_not(collapsible)],
            all_weights, labels)


def df_to_tree(tree_df, validate=True, omit=None, sub=None):
    '''
    Converts logic tree :class:`pandas.DataFrame` to tree of
    :class:`openquake.baselib.node.Node` objects which then be written to a
    file using :func:`openquake.hazardlib.nrml.write`.
    '''
    tree = Node('logicTree', {'logicTreeID': 'lt1'}, None)

    for i, level in tree_df.iterrows():

        branching_level_attr = {'branchingLevelID': 'bl%d' % (i + 1)}
        branching_level = Node(
            'logicTreeBranchingLevel', branching_level_attr, None)

        branch_set_attr = {
            'branchSetID': 'bs%d' % (i + 1),
            'uncertaintyType': level['uncertaintyType']}
        for key in level.keys():
            if 'applyTo' in key and level[key] != 'all':
                branch_set_attr.update({key: level[key]})

        if 'uncertaintyWeight' in level.keys():
            weights = level['uncertaintyWeight']
        else:
            weights = None

        models_weights = models_with_weights(
            level['uncertaintyType'], level['uncertaintyModel'], weights,
            branch_set_attr['branchSetID'],
            validate=validate, omit=omit, sub=sub)

        if not models_weights:
            continue

        add_branch_set(branching_level, branch_set_attr, models_weights)

        tree.append(branching_level)

    return tree


def add_branch_set(branching_level, branch_set_attr, models_weights):
    '''
    Add a branch set to a branching level.
    '''
    branch_set = Node(
        'logicTreeBranchSet', branch_set_attr, None)
    branch_index_string = re.sub('[^0-9]', '', branch_set_attr['branchSetID'])

    if branch_index_string:
        branch_index = int(branch_index_string)
    else:
        branch_index = 999

    for j, (model, weight) in enumerate(models_weights):
        branch_attr = {'branchID': 'b%dm%d' % (branch_index, j + 1)}
        branch = Node('logicTreeBranch', branch_attr, None)
        branch.append(Node('uncertaintyModel', {}, model))
        branch.append(Node('uncertaintyWeight', {}, weight))

        branch_set.append(branch)

    branching_level.append(branch_set)


def get_dict_key_match(attrib, string):
    '''
    Finds a the first partial match among the keys in a dictionary
    '''
    keys = [key for key in attrib.keys() if string.lower() in key.lower()]
    if keys == []:
        return ''

    return keys[0]


def strip_fqtag(tag):
    '''
    Get the short representation of a fully qualified tag

    :param str tag: a (fully qualified or not) XML tag
    '''
    string = str(tag)
    # split on '}', to remove the namespace part
    pieces = string.rsplit('}', 1)
    if len(pieces) == 2:
        string = pieces[1]
    return string


class StreamingTexWriter(object):
    '''
    A stream-based TEX writer based on
    openquake.nrmllib.writers.StreamingXMLWriter
    '''
    def __init__(self, stream, include_ids=False, indent=4,
                 max_branches=10, encoding='utf-8'):
        '''
        :param stream: the stream or a file where to write the TEX
        :param int indent: the indentation to use in the TEX file
        '''
        self.stream = stream
        self.include_ids = include_ids
        self.indent = indent
        self.max_branches = int(max_branches)
        self.encoding = encoding
        self.indentlevel = 0
        self.variables = None

    def _write(self, text):
        '''
        Write text while respecting current indentation level
        '''
        if not isinstance(text, str):
            text = text.encode(self.encoding)
        spaces = ' ' * (self.indent * self.indentlevel)
        self.stream.write(spaces + text + '\n')

    def start_branch(self, name, attrs=None):
        '''
        Open a TEX branch
        '''
        name = strip_fqtag(name)
        for (attr, value) in sorted(attrs.items()):
            name = name + (r'\\ \texttt{%s}=\texttt{%s}' % (attr, value))
        self._write(r'"%s"[box] ->[link] {' % name)
        self.indentlevel += 1

    def end_branch(self):
        '''
        Close a TEX branch
        '''
        self.indentlevel -= 1
        self._write(r'},')

    def add_branch(self, node):
        '''Add branch node'''

        tag = strip_fqtag(node.tag)
        attrib = deepcopy(node.attrib)
        node_id = attrib.pop(get_dict_key_match(attrib, 'id'), None)

        if tag == 'logicTreeBranchSet':
            attrib.pop('uncertaintyType', None)
            keys = node.attrib.keys()
            if 'applyToTectonicRegionType' in keys:
                name = attrib.pop('applyToTectonicRegionType')
                name = name.replace(' ', r'\\ ')
            elif 'applyToSourceType' in keys:
                name = attrib.pop('applyToSourceType')
            elif 'applyToSources' in keys:
                apply_to = attrib.pop('applyToSources')
                if os.path.isfile(apply_to):
                    name = r'each\\ source'
                else:
                    name = r'sources:\\ ' + apply_to
            elif 'applyToBranches' in keys:
                name = r'branches:\\ ' + attrib.pop('applyToBranches')
            else:
                name = ''
        elif tag == 'logicTree':
            name = r'%s\\ %s' % (
                node[0][0]['uncertaintyType'], tag)
        elif 'omitted' in tag:
            name = tag
        else:
            name = '\texttt{%s}' % tag

        if self.include_ids:
            name = '%s: %s' % (node_id, name)

        self.start_branch(name, attrib)
        if node.text and node.text.strip():
            print('Ignoring node "%s" text "%s"' % (name, node.text.strip()))
        if len(node) > self.max_branches:
            print('Too many (%d) nodes in %s, '
                  'abbreviating TEX to first & last %d'
                  % (len(node), tag, self.max_branches/2))
            n_omitted = len(node) - self.max_branches
            ellipsis_text = str(n_omitted) + r' branches\\ omitted'
            ellipsis_node = Node(ellipsis_text, {}, None)
            nodes = node[:int(self.max_branches/2)] + [ellipsis_node] + \
                node[-int(self.max_branches/2):]
        else:
            nodes = node
        for subnode in nodes:
            self.serialize(subnode)
        self.end_branch()

    def add_leaf(self, node):
        '''
        Add terminal node (attributes are discarded)
        '''
        name = ''
        for subnode in node:
            tag = strip_fqtag(subnode.tag)
            if tag == 'uncertaintyModel':
                model = subnode.text.strip()
                if self.variables is not None:
                    numbers = [str(float(x)) for x in model.split()]
                    pairs = zip(self.variables, numbers)
                    model = ', '.join(['%s = %s' % item for item in pairs])
                elif any([string in model for string, _ in self._TEX_SUBS]):
                    for string, replacement in self._TEX_SUBS:
                        model = model.replace(string, replacement)
                else:
                    model = '\\texttt{%s}' % model.replace('_', r'\_')

            elif tag == 'uncertaintyWeight':
                weight = '%.3g' % float(subnode.text)
            else:
                print('Tag %s not recognized, ignoring ...' % tag)
        name = model + ': ' + weight

        id_key = get_dict_key_match(node.attrib, 'id')
        if self.include_ids and id_key != '':
            name = '%s: %s' % (node[id_key], name)

        self._write('"%s",' % name)

    def set_model_variables(self, node):
        '''Determine model variable names for this branch'''

        uncertainty = node['uncertaintyType']
        if uncertainty == 'abGRAbsolute':
            variables = ['$a$', '$b$']
        elif uncertainty == 'bGRRelative':
            variables = [r'$\Delta b $']
        elif uncertainty == 'maxMagGRAbsolute':
            variables = ['$m_{max}$']
        elif uncertainty == 'maxMagGRRelative':
            variables = [r'$\Delta m_{max}$']
        else:
            variables = None

        self.variables = variables

    def serialize(self, node):
        '''Serialize a node object'''

        tag = strip_fqtag(node.tag)
        if tag == 'nrml' or (tag == 'logicTreeBranchingLevel' and
                             len(node) == 1):
            for subnode in node:
                self.serialize(subnode)
        elif tag == 'logicTreeBranchSet':
            model = node[0].uncertaintyModel.text
            if isinstance(model, list):
                model = model[0]
            if is_numeric(model.strip().split()[0]):
                self.set_model_variables(node)
            self.add_branch(node)
        elif tag == 'logicTreeBranch':
            self.add_leaf(node)
        else:
            self.add_branch(node)

    def __enter__(self):
        '''
        Write the TEX preamble
        '''
        self._write(self._TEX_START)
        self.indentlevel += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''
        Write the TEX postscript
        '''
        self.indentlevel -= 1
        self._write(self._TEX_END)

    _TEX_START = r'''\documentclass[border=2mm]{standalone}

\usepackage{tikz}
\usetikzlibrary{graphdrawing, graphs}
\usegdlibrary{trees}
\usepackage{fontspec} % support for setmonofont
\setmonofont{Ubuntu Mono}[Scale=MatchUppercase]

\begin{document}
\begin{tikzpicture}[>=latex, align=center, anchor=west,
link/.style={out=east, in=west},
box/.style={rectangle, rounded corners, draw}]
\graph [tree layout, components go down center aligned, grow'=right,
fresh nodes, layer sep=2cm, sibling distance=0mm, sibling sep=0.5mm] {
    '''

    _TEX_END = r'''};
\end{tikzpicture}
\end{document}
    '''

    _TEX_SUBS = [
        ['stda', r'$\sigma_a$'],
        ['stdb', r'$\sigma_b$'],
        ['stdmmax', r'$\sigma_{m_{max}}$'],
        ['avalue', '$a$'],
        ['bvalue', '$b$'],
        ['mmax', '$m_{max}$'],
        ['$ = $', ' = '],
        ['$ + $', ' + '],
        ['$ - $', ' - '],
        ['$*$', '*'],
        ['$/$', '/'],
    ]


def tree_to_tex(root, include_ids=False, indent=4):
    '''
    Convert a tree into an TEX string by using the StreamingTexWriter.
    This is useful for testing purposes.

    :param node: a node object (typically an ElementTree object)
    :param include_ids: include or omit node ids from diagram
    :param indent: the indentation to use in the XML (default 4 spaces)
    '''
    out = StringIO()
    with StreamingTexWriter(out, include_ids, indent) as writer:
        writer.serialize(root)
    return out.getvalue()


def nrml_to_pdf(file_nrml, include_ids=False, verbose=False):
    '''
    Convert NRML logic tree into a PDF diagram. Output file name is same
    as input except with .pdf extension. An intermediate .tex file is
    generated. Lualatex must be installed and present on the system path.

    :param file_nrml: file name of NRML logic tree in XML format
    :param include_ids: include or omit node ids from diagram
    '''

    if verbose:
        print('Reading %s' % file_nrml)
    root = nrml.read(file_nrml)

    file_tex = file_nrml.replace('.xml', '') + '.tex'
    if verbose:
        print('Writing %s' % file_tex)
    with open(file_tex, 'w+') as f:
        with StreamingTexWriter(f, include_ids) as writer:
            writer.serialize(root)

    out_dir = 'build'
    file_pdf = file_tex.replace('.tex', '.pdf')
    build_pdf = os.path.join(out_dir, file_pdf)
    if verbose:
        print('Converting %s to %s' % (file_tex, build_pdf))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    subprocess.call(['lualatex', '-output-directory=' + out_dir,
                     '-interaction=nonstopmode', file_tex])

    if verbose:
        print('Moving %s to %s' % (build_pdf, file_pdf))
    if os.path.exists(file_pdf):
        os.remove(file_pdf)
    os.rename(build_pdf, file_pdf)
