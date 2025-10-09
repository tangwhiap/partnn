# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3.8
#     language: python
#     name: p38
# ---

# ## I/O Utilities for Pre-processing the Config Dictionaries

import numpy as np
from collections import OrderedDict
from itertools import product, chain
from collections import ChainMap
from textwrap import dedent
import platform
import graphlib
import subprocess
import h5py
import fnmatch
import time
from os.path import exists, getmtime
import pandas as pd
import json
import yaml
import math
import re
import io
import os
import glob
import ast
from ruamel import yaml as ruyaml
from collections import defaultdict
from pandas.api.types import union_categoricals
from partnn.io_cfg import configs_dir, results_dir, nullstr


# + 
(pyver_major, pyver_minor, pyver_patch) = map(int, platform.python_version_tuple())
is_dictordrd = (pyver_major > 3) or ((pyver_major == 3) and (pyver_minor >= 6))
odict = dict if is_dictordrd else OrderedDict
h5drvr_dflt = 'sec2'

# + code_folding=[3, 112, 209, 239, 826]
#########################################################
############ Pre-processing JSON Config Files ###########
#########################################################
class Looper:
    """
    A Looper object that recurively creates looper children for
    itself if it gets a string child. The `opt2vals` input will be
    mutated and emptied out after the `pop_vals` calls, so make sure
    you take a backup copy for yourself before giving it to this
    constructor.

    Parameters
    ----------
    childern: (list of str or Looper): a list of either option names
        (as string) or child Looper objects.

        Example: childern = ['lr', OvatLooper(...), 'tau',
                             'problem', ZipLooper(...),
                             ...]

    opt2vals: (dict) a config dictionary with keys mapped to
        a list of looped values. All options (even the ones
        that are fixed and don't need looping over) must be
        mapped to a list (albeit a single item list).

        Example: opt2vals = {'lr': [0.001],
                             'problem': ['poisson'],
                             'tau': [0.999, 0.9999, 0.999999],
                             ...}
    """

    def __init__(self, children, opt2vals):
        self.opt2vals = opt2vals
        self.children = []
        self.patterns = []
        for child in children:
            if isinstance(child, Looper):
                self.children.append(child)
                self.patterns.append(None)
            elif isinstance(child, str):
                pat = child
                self.patterns.append(pat)
                if pat == "rest":
                    child_ = AtomicLooper(None, opt2vals)
                    self.children.append(child_)
                else:
                    opts = fnmatch.filter(opt2vals.keys(), pat)
                    msg_ = f'Did not find "{child}" for looping: '
                    msg_ += f"{list(opt2vals.keys())}"
                    assert len(opts) >= 1, msg_
                    child_ = AtomicLooper(opts, opt2vals)
                    self.children.append(child_)
        self.lstrip_prfxs = []

    def pop_vals(self, turn):
        """
        Recursively calls pop_vals on its children until it reaches all
        leaf (i.e., AtomicLooper) objects.
        """
        for child in self.children:
            child.pop_vals(turn)

    def lstrip(self, *lstrip_prfxs):
        """
        Memorizes a bunch of lstrip prefixes so that it applies the
        stripping at the end of the get_cfgs() calls (i.e., during the
        inner pack call). Returns self.
        """
        self.lstrip_prfxs += list(lstrip_prfxs)
        return self

    def pack(self, cfgs):
        """
        This function does two things:
          1. Wraps the input in another list. This is needed for
             singleton loopers such as (cart, ovat, zip), which merge
             all the children dictionaries togather.

          2. Applies the lstrip function with the stored prefixes on each
             of the dictionary keys.

        Parameters
        ----------
        cfgs: (list of dict) a list of dictionaries
            Example:
                cfgs = [{'a': 1,  'b': 4,  'c': 7},
                        {'a': 2,  'b': 5,  'c': 8},
                        {'a': 3,  'b': 6,  'c': 9},
                        {'d': 10, 'e': 13},
                        {'d': 11, 'e': 14},
                        {'d': 12, 'e': 15}]

        Output
        ------
        out: (list of list of dict) a list-wrapped version of the input.
            Example:
                cfgs = [[{'a': 1,  'b': 4,  'c': 7},
                         {'a': 2,  'b': 5,  'c': 8},
                         {'a': 3,  'b': 6,  'c': 9},
                         {'d': 10, 'e': 13},
                         {'d': 11, 'e': 14},
                         {'d': 12, 'e': 15}]]
        """
        if len(self.lstrip_prfxs) == 0:
            out = [cfgs]
        else:
            out = [
                [
                    odict(
                        (lstrip(k, self.lstrip_prfxs), v)
                        for k, v in cfg.items()
                    )
                    for cfg in cfgs
                ]
            ]
        return out

    def get_cfgs(self):
        raise NotImplementedError("This should have been overriden!")


class AtomicLooper:
    """
    A class resembling the Looper object with a slight difference
    that its instances will be placed at the leaves of the looping
    tree. This does not make any recursive calls to its children
    (since it has no children).

    Parameters
    ----------
    optnames: (list of str): a list of option names.

        Example: optnames = ['lr', 'tau', 'problem']

    opt2vals: (dict) a config dictionary with keys mapped to
        a list of looped values. All options (even the ones
        that are fixed and don't need looping over) must be
        mapped to a list (albeit a single item list).

        Example: opt2vals = {'lr': [0.001],
                             'problem': ['poisson'],
                             'tau': [0.99, 0.999, 0.9999],
                             ...}
    """

    def __init__(self, optnames, opt2vals):
        self.opt2vals = opt2vals
        self.optnames = optnames
        self.values = None
        self.is_rest = optnames is None
        self.cfg = None

    def pop_vals(self, turn):
        """
        Transfers the options and their values from the globally shared
        `self.opt2vals` dictionary to its own `self.cfg` dictionary.
        """
        if turn == "nonrest":
            if not self.is_rest:
                cfg = odict()
                for opt in self.optnames:
                    msg_ = f'Option "{opt}" was either (a) not defined '
                    msg_ += "in the original config, or (b) two loopers"
                    msg_ += " attempted to use it."
                    assert opt in self.opt2vals, msg_
                    cfg[opt] = self.opt2vals.pop(opt)
                self.cfg = cfg
        elif turn == "rest":
            if self.is_rest:
                rmng_opts = list(self.opt2vals.keys())
                cfg = odict()
                for opt in rmng_opts:
                    cfg[opt] = self.opt2vals.pop(opt)
                msg_ = 'A "rest" option turned out to be empty in the '
                msg_ += 'looping tree. Either \n\t(1) "rest" was used '
                msg_ += "more than once in the looping tree, or \n\t(2) "
                msg_ += "all options were already specified in the looping "
                msg_ += 'tree and nothing was left for "rest" to claim.'
                assert len(cfg) > 0, msg_
                self.cfg = cfg
        else:
            raise ValueError(f"turn {turn} undefined.")

    def get_cfgs(self, pass_list=False):
        """
        The main function applying the looping process and
        generating dictionaries.

        Example 1:
            pass_list = False

            self.cfg = {'a': [1, 2],
                        'b': [3, 4],
                        'c': [5, 6]}

            out = [[{'a': 1}, {'a': 2}],
                   [{'b': 3}, {'b': 4}],
                   [{'c': 5}, {'c': 6}]]

        Example 2:
            pass_list = True

            self.cfg = {'a': [1, 2],
                        'b': [3, 4],
                        'c': [5, 6]}

            out = [[{'a/list': [1, 2]},
                   [{'b/list': [3, 4]},
                   [{'c/list': [5, 6]}]]
        """
        if not pass_list:
            out = [
                [{opt: val} for val in vals] for opt, vals in self.cfg.items()
            ]
        else:
            out = [[{f"{opt}/list": vals}] for opt, vals in self.cfg.items()]
        return out


class CartLooper(Looper):
    def get_cfgs(self):
        """
        A few examples will clarify the data structure:

        Example:
            child_cfglists = [[[{'a': 1}, {'a': 2}],
                               [{'b': 3}, {'b': 4}],
                               [{'c': 5}, {'c': 6}]],
                              [[{'d': 7}, {'d': 8}]],
                              [[{'e': 9}, {'e': 0}]]]

            child_cfgs = [[{'a': 1}, {'a': 2}],
                          [{'b': 3}, {'b': 4}],
                          [{'c': 5}, {'c': 6}],
                          [{'d': 7}, {'d': 8}],
                          [{'e': 9}, {'e': 0}]]

            cfgs = [{'a':1, 'b':3, 'c':5, 'd':7, 'e':9},
                    {'a':1, 'b':3, 'c':5, 'd':7, 'e':0},
                    {'a':1, 'b':3, 'c':5, 'd':8, 'e':9},
                    ...]

            output = [cfgs]
        """
        child_cfglists = [child.get_cfgs() for child in self.children]
        child_cfgs = chain.from_iterable(child_cfglists)
        cfgs = list(chain_dicts(dtup) for dtup in product(*child_cfgs))
        return self.pack(cfgs)


class ZipLooper(Looper):
    def get_cfgs(self):
        """
        The main function applying the looping process and
        generating the output config dictionaries. Recursive
        calls of this method in the children function will
        build up the user config dictionaries.

        A few examples will clarify the data structure:

        Example:
            child_cfglists = [[[{'a': 1}, {'a': 2}],
                            [{'b': 3}, {'b': 4}],
                            [{'c': 5}, {'c': 6}]],
                            [[{'d': 7}, {'d': 8}]],
                            [[{'e': 9}, {'e': 0}]]]

            child_cfgs = [[{'a': 1}, {'a': 2}],
                          [{'b': 3}, {'b': 4}],
                          [{'c': 5}, {'c': 6}],
                          [{'d': 7}, {'d': 8}],
                          [{'e': 9}, {'e': 0}]]

            cfgs = [{'a':1, 'b':3, 'c':5, 'd':7, 'e':9},
                    {'a':2, 'b':4, 'c':6, 'd':8, 'e':0}]

            output = [cfgs]
        """
        child_cfglists = [child.get_cfgs() for child in self.children]
        child_cfgs = chain.from_iterable(child_cfglists)
        cfgs = list(chain_dicts(dtup) for dtup in zip(*child_cfgs))
        return self.pack(cfgs)


class OvatLooper(Looper):
    def get_cfgs(self):
        """
        The main function applying the looping process and
        generating the output config dictionaries. Recursive
        calls of this method in the children function will
        build up the user config dictionaries.

        A few examples will clarify the data structure:

        Example:
            child_cfglists = [[[{'a': 1}, {'a': 2}],
                            [{'b': 3}, {'b': 4}],
                            [{'c': 5}, {'c': 6}]],
                            [[{'d': 7}, {'d': 8}]],
                            [[{'e': 9}, {'e': 0}]]]

            child_cfgs = [[{'a': 1}, {'a': 2}],
                        [{'b': 3}, {'b': 4}],
                        [{'c': 5}, {'c': 6}],
                        [{'d': 7}, {'d': 8}],
                        [{'e': 9}, {'e': 0}]]

            cfgs = [{'a':1, 'b':3, 'c':5, 'd':7, 'e':9},
                    {'a':2, 'b':3, 'c':5, 'd':7, 'e':9},
                    {'a':1, 'b':4, 'c':5, 'd':7, 'e':9},
                    {'a':1, 'b':3, 'c':6, 'd':7, 'e':9},
                    {'a':1, 'b':3, 'c':5, 'd':8, 'e':9},
                    {'a':1, 'b':3, 'c':5, 'd':7, 'e':0}]

            output = [cfgs]
        """
        child_cfglists = [child.get_cfgs() for child in self.children]
        child_cfgs = chain.from_iterable(child_cfglists)
        cfgs = list(chain_dicts(dtup) for dtup in ovat(*child_cfgs))
        return self.pack(cfgs)


class AsListLooper(Looper):
    def get_cfgs(self):
        """
        The main function applying the looping process and
        generating the output config dictionaries. Recursive
        calls of this method in the children function will
        build up the user config dictionaries.

        A few examples will clarify the data structure:

        Example:
            child_cfglists = [[[{'a': 1}],
                            [{'b': 3}],
                            [{'c': 5}]],
                            [[{'d': 7}]],
                            [[{'e': 9}]]]

            child_cfgs = [[{'a': 1}],
                        [{'b': 3}],
                        [{'c': 5}],
                        [{'d': 7}],
                        [{'e': 9}]]

            cfgd_lst = [{'a': 1},
                        {'b': 3},
                        {'c': 5},
                        {'d': 7},
                        {'e': 9}]

            cfgs = [{'a':1, 'b':3, 'c':5, 'd':7, 'e':9}]

            output = [cfgs]
        """
        msg_ = "aslist only accepts options and cant work "
        msg_ += "with loopers such as cart/zip/..."
        assert all(
            isinstance(child, AtomicLooper) for child in self.children
        ), msg_

        child_cfglists = [
            child.get_cfgs(pass_list=True) for child in self.children
        ]
        child_cfgs = list(chain.from_iterable(child_cfglists))
        assert all(len(lst) == 1 for lst in child_cfgs)
        cfgd_lst = [lst[0] for lst in child_cfgs]
        cfgs = [chain_dicts(cfgd_lst)]
        return self.pack(cfgs)


class CatLooper(Looper):
    def get_cfgs(self):
        """
        The main function applying the looping process and
        generating the output config dictionaries. Recursive
        calls of this method in the children function will
        build up the user config dictionaries.

        A few examples will clarify the data structure:

        Example 1:
            child_cfglists_ = [[[{'a': 1,  'b': 4,  'c': 7},
                                {'a': 2,  'b': 5,  'c': 8},
                                {'a': 3,  'b': 6,  'c': 9}],
                            [[{'d': 10, 'e': 13},
                                {'d': 11, 'e': 14},
                                {'d': 12, 'e': 15}]]]

            child_cfglists = child_cfglists_

            child_cfgs = [[{'a': 1,  'b': 4,  'c': 7},
                        {'a': 2,  'b': 5,  'c': 8},
                        {'a': 3,  'b': 6,  'c': 9}],
                        [{'d': 10, 'e': 13},
                        {'d': 11, 'e': 14},
                        {'d': 12, 'e': 15}]]

            cfgs = [{'a': 1,  'b': 4,  'c': 7},
                    {'a': 2,  'b': 5,  'c': 8},
                    {'a': 3,  'b': 6,  'c': 9},
                    {'d': 10, 'e': 13},
                    {'d': 11, 'e': 14},
                    {'d': 12, 'e': 15}]


        Example 2 (curating singletons):
            child_cfglists_ = [[[{'a': 1}],
                                [{'b': 4}],
                                [{'c': 7}]]
                            [[{'d': 10, 'e': 13},
                                {'d': 11, 'e': 14},
                                {'d': 12, 'e': 15}]]]

            child_cfglists = [[[{'a': 1, 'b': 4, 'c': 7}]]
                            [[{'d': 10, 'e': 13},
                                {'d': 11, 'e': 14},
                                {'d': 12, 'e': 15}]]]

            child_cfgs = [[{'a': 1,  'b': 4,  'c': 7}],
                        [{'d': 10, 'e': 13},
                        {'d': 11, 'e': 14},
                        {'d': 12, 'e': 15}]]

            cfgs = [{'a': 1,  'b': 4,  'c': 7},
                    {'d': 10, 'e': 13},
                    {'d': 11, 'e': 14},
                    {'d': 12, 'e': 15}]

            output = [cfgs]
        """
        child_cfglists_ = [child.get_cfgs() for child in self.children]

        # If the `child_cfglists` are uncrated, yet each one of them is
        # a singleton, it doesn't matter if you combine them using cart
        # or zip or ovat. For this, we just combine them ourselves.
        # See Example 2 for a demonstration.
        child_cfglists = []
        for i, cfglst in enumerate(child_cfglists_):
            all_singltons = all(len(lst) == 1 for lst in cfglst)
            if all_singltons:
                merged_lst = [chain_dicts([lst[0] for lst in cfglst])]
                child_cfglists.append([merged_lst])
            else:
                child_cfglists.append(cfglst)

        # Making sure no uncurated options are left
        msg_ = "You seem to have passed a bunch of uncurated options "
        msg_ += "to the cat looper. Make sure you merge each group of "
        msg_ += "cat options using one of the merger loopers (e.g., "
        msg_ += "cart, zip, or ovat) before passing it to cat.\n"
        msg_ += f"    child_cfglists = {child_cfglists}"
        assert all(len(cfglst) == 1 for cfglst in child_cfglists), msg_

        child_cfgs = chain.from_iterable(child_cfglists)
        cfgs = list(chain.from_iterable(child_cfgs))
        return self.pack(cfgs)


def ovat(*lists):
    """
    Gets a bunch of lists and loops over them in a
    "one variable at a time" manner.

    Parameters
    ----------
    lists (list of list): a list of lists.

        Example:
            lists = [[{'a': 1}, {'a': 2}],
                     [{'b': 3}, {'b': 4}],
                     [{'c': 5}, {'c': 6}],
                     [{'d': 7}, {'d': 8}],
                     [{'e': 9}, {'e': 0}]]

    Output
    ------
    out: (list of list) the ovat loopeings of the input.

        Example:
            out = [[{'a': 1}, {'b': 3}, {'c': 5}, {'d': 7}, {'e': 9}],
                   [{'a': 2}, {'b': 3}, {'c': 5}, {'d': 7}, {'e': 9}],
                   [{'a': 1}, {'b': 4}, {'c': 5}, {'d': 7}, {'e': 9}],
                   [{'a': 1}, {'b': 3}, {'c': 6}, {'d': 7}, {'e': 9}],
                   [{'a': 1}, {'b': 3}, {'c': 5}, {'d': 8}, {'e': 9}],
                   [{'a': 1}, {'b': 3}, {'c': 5}, {'d': 7}, {'e': 0}]]
    """
    assert all(isinstance(ll, list) for ll in lists)
    cntrdval = tuple(vv[0] for vv in lists)
    othvals = [
        (*cntrdval[:vv_idx], v, *cntrdval[(vv_idx + 1):])
        for vv_idx, vv in enumerate(lists)
        for v in vv[1:]
    ]
    out = [cntrdval] + othvals
    return out


def chain_dicts(dicts_iterable):
    """
    Chains (i.e., comibnes) a bunch of dictionaries into a single
    ordered dictionary.

    Parameters
    ----------
    dicts_iterable: (iterable of dict) a list of dictionaries that need
        to be combined together.

        Example:
            dicts_iterable = [{'a': 1}, {'b': 2}, {'c': 3}]

    Output
    ------
    out: (odict) a single combined dictionary.

        Example:
            out = {'a': 1, 'b': 2, 'c': 3}
    """
    return odict(ChainMap(*dicts_iterable[::-1]))


def lstrip(s, prfx_lst):
    """
    Left-strips a bunch of prefixes from a string. If there
    is more than a single prefix, the order of prefixes for
    stripping must be chosen carefully (see the example).

    Parameters
    ----------
    s: (str) the input string to be lstripped.

        Example: s = 'g0/g1/g3/g2/opt'

    prfx_lst: (list of str) a list of prefixes for
        left-stripping in an ordered fashion.

        Example: prfx_lst = ['g0/', 'g1/', 'g2/', 'g3/']

    Output
    ------
    out: (str) the left stripped string

        Example: out = 'g3/g2/opt'
    """
    out = s
    for prfx in prfx_lst:
        out = out.lstrip(prfx)
    return out


def make_looper(looping_str, opt2vals, cmpl_type='eval'):
    """
    This function takes a string python command (which is defined in a
    json config file), and then executes it using the eval function in
    python so that it produces an Looper object.

    Parameters
    ----------
    looping_str: (str) the python command describing the looping process.

        Example 1: "cart('all')"

        Example 2: "cart('dim', 'n_srf', 'n_epochs', 'lr')"

        Example 3: "cart('dim', zip('n_epochs', ovat('n_srf', 'lr')))"
    
    cmpl_type: (str) Whether to use `eval` or `exec` to compile the looper.

    Outputs
    ----------
    looper: (Looper) An object created recursively by the eval function.
    """
    def cart(*args):
        return CartLooper(args, opt2vals)

    def ovat(*args):
        return OvatLooper(args, opt2vals)

    def zip(*args):
        return ZipLooper(args, opt2vals)

    def cat(*args):
        return CatLooper(args, opt2vals)

    def aslist(*args):
        return AsListLooper(args, opt2vals)

    _ = (cart, ovat, zip, cat, aslist)

    if cmpl_type == 'eval':
        looper = eval(looping_str)
    elif cmpl_type == 'exec':
        exec_outputs = dict()
        exec(looping_str, locals(), exec_outputs)
        assert 'looper' in exec_outputs, dedent(f'''
            The `looper` variable was not defined after 
            executing the looping command lines:
                looping_str: {looping_str}
                variables: {exec_outputs.keys()}''')
        looper = exec_outputs['looper']
    else:
        raise ValueError(f'undefined cmpl_type={cmpl_type}')

    return looper


def cfg_iters2list(cfg_dict):
    """
    Converting "range" or "linspace" specifications into a list of values.

    Parameters
    ----------
    cfg_dict: (dict) the config dictionary coming from a json file.

        Example: {"desc/lines": ["hello", "world!"],
                  "rng_seeds/range": [0, 5000, 1000],
                  "dim/list": [2, 3, 4],
                  "n_vol/linspace": [400, 100, 4],
                  "n_epochs/loglinspace": [200, 25, 4],
                  "lr/logrange": [0.001, 0.000001, 0.1],
                  "tau": 0.999}

    Outputs
    ----------
    proced_cfg_dict: (dict) the same dictionary with all keys having a
        "/list", "/range", "/logrange", "/linspace", or "/loglinspace"
        postfix removed and treated. The values will now be a list
        even if they originally were not (see "tau" for example).

        Example: {"desc": ["hello\nworld!"],
                  "rng_seeds": [0, 1000, 2000, 3000, 4000],
                  "dim": [2, 3, 4],
                  "n_vol": [400, 300, 200, 100],
                  "n_epochs": [200, 100, 50, 25],
                  "lr": [0.001, 0.0001, 0.00001],
                  "tau": [0.999]}
    """
    proced_cfg_dict = odict()
    for key, val in cfg_dict.items():
        if key.endswith("/range") or key.endswith("/logrange"):
            assert len(val) == 3
            need_logtrans = key.endswith("/logrange")
            proced_key = "/".join(key.split("/")[:-1])
            st, end, step = val
            if need_logtrans:
                st, end = math.log(st), math.log(end)
            proced_val_arr = np.arange(st, end, step)
            if need_logtrans:
                proced_val_arr = proced_val_arr.exp()
            caster = int if all(isinstance(v, int) for v in val) else float
        elif key.endswith("/linspace") or key.endswith("/loglinspace"):
            assert len(val) == 3
            need_logtrans = key.endswith("/loglinspace")
            proced_key = "/".join(key.split("/")[:-1])
            st, end, num = val
            if need_logtrans:
                st, end = math.log(st), math.log(end)
            proced_val_arr = np.linspace(st, end, num)
            if need_logtrans:
                proced_val_arr = np.exp(proced_val_arr)
            caster = int if all(isinstance(v, int) for v in val) else float
        elif key.endswith("/lines"):
            msg_ = f'The value for "{key}" option is not a list of strings: '
            msg_ += 'Options ending in "/lines" must specify '
            msg_ += "a list of strings."
            assert isinstance(val, list), msg_
            assert all(isinstance(vln, str) for vln in val), msg_
            proced_key = key[:-6]
            proced_val_arr = ["\n".join(val)]
            caster = None
        elif key.endswith("/list"):
            assert isinstance(val, list), f'"{key}" should specify a list!'
            proced_key = key[:-5]
            proced_val_arr = val
            caster = None
        else:
            proced_key, proced_val_arr, caster = key, [val], None

        if caster is not None:
            proced_val = [caster(v) for v in proced_val_arr]
            assert all(
                np.all(v1 == v2) for v1, v2 in zip(proced_val, proced_val_arr)
            )
        else:
            proced_val = proced_val_arr

        msg_ = f'"{proced_key}" was specified in multiple ways! '
        msg_ += 'Remember that for an option "X", only one of '
        msg_ += '"X" or "X/list" or "X/range" or "X/logrange" '
        msg_ += 'or "X/linspace" or "X/loglinspace" or "X/lines"'
        msg_ += " should be specified."
        assert proced_key not in proced_cfg_dict, msg_
        proced_cfg_dict[proced_key] = proced_val
    return proced_cfg_dict


def preproc_cfgdict(cfg_dict, opt_order="config", trnsfrmtn='hie'):
    """
    This function pre-process a config dict read from a json
    file, and returns a list of config dicts after applying the
    desired looping strategy.

    Parameters
    ----------
    cfg_dict: (dict) config dict read from a json file.

        Example: {
          "desc": "Example scratch json config",
          "date": "December 20, 2022",
          "problem": "poisson",
          "rng_seed/range": [0, 5000, 1000],
          "dim/list": [2, 3, 4],
          "n_srf/list": [400, 200, 100],
          "n_srfpts_mdl": 0,
          "n_srfpts_trg": 2,
          "trg/w": 1.0,
          "n_epoch/list": [200000, 100000],
          "do_detspacing": false,
          "do_bootstrap": false,
          "do_dblsampling": true,
          "lr/list": [0.001, 0.0001],
          "tau": 0.999,
          "looping": "cart(aslist('rng_seed'), zip('dim', 'lr'), 'rest')"
        }

    opt_order: (str) determines the order of the options in the
        key should be either 'config' or 'looping'.

    Outputs
    ----------
    all_cfgdicts: (list of dicts) list of processed config dicts.

        Example: [{'desc': 'Example scratch json config',
                   'date': 'December 20, 2022',
                   'problem': 'poisson',
                   'rng_seed/list': [0, 1000, 2000, 3000, 4000],
                   'dim': 2,
                   'n_srf': 400,
                   'n_srfpts_mdl': 0,
                   'n_srfpts_trg': 2,
                   'trg/w': 1.0,
                   'n_epoch': 200000,
                   'do_detspacing': False,
                   'do_bootstrap': False,
                   'do_dblsampling': True,
                   'lr': 0.001,
                   'tau': 0.999},
                    ...]
    """
    # Applying any necessary `hie2deep` or `deep2hie` transformations.
    opt2vals_prsd = tnsfm_cfgdict(cfg_dict, trnsfrmtn=trnsfrmtn)

    # Removing "/list", "/range", "/linspace", etc. and
    # replacing them with a proper list of hp values.
    opt2vals_orig = cfg_iters2list(opt2vals_prsd)

    # `opt_order='config'` will try to preserve the order of options
    # in the output dictionary as the json config file. Otherwise, the
    # order will follow the DFS order of options in the looping tree.
    # `opt_order='looping'` will make the options follow the DFS order
    # of the looping tree.
    assert opt_order in ("config", "looping")

    if (("looping" in opt2vals_orig) or  ("looping/eval" in opt2vals_orig) or 
        ("looping/exec" in opt2vals_orig)):

        opt2vals_ = opt2vals_orig.copy()
        n_opts = len(opt2vals_orig)

        if "looping" in opt2vals_orig:
            looping_str = opt2vals_.pop("looping")[0]
            looping_cmplr = "eval"
        elif "looping/eval" in opt2vals_orig:
            looping_str = opt2vals_.pop("looping/eval")[0]
            looping_cmplr = "eval"
        elif "looping/exec" in opt2vals_orig:
            looping_str = opt2vals_.pop("looping/exec")[0]
            looping_cmplr = "exec"
        else:
            raise ValueError('undefined case')
        
        assert all(opt not in opt2vals_ for opt in ("looping",
            "looping/eval", "looping/exec")), dedent('''
                Found multiple looping* instances in the 
                config dictionary. You should provide at 
                most one of "looping", "looping/eval", or 
                "looping/exec" keys.''')

        # The looper orderings are the same as to the looping tree.
        # The trick to preserve the config order in the options, is
        # to store the index of options along with the values, and
        # then sorting the keys at the end using those indexes.
        if opt_order == "config":
            # Example:
            #     opt2vals_ = {'a': [10, 20, 30],
            #                  'b': [40, 50],
            #                  'c': [60]}
            #
            #     opt2vals  = {'a': [(0, 10), (0, 20), (0, 30)],
            #                  'b': [(1, 40), (1, 50)],
            #                  'c': [(2, 60)]}
            opt2vals = odict()
            for optidx, (opt, vals) in enumerate(opt2vals_.items()):
                opt2vals[opt] = [(optidx, val) for val in vals]
        elif opt_order == "looping":
            opt2vals = opt2vals_.copy()
        else:
            raise ValueError(f"Unknown opt_order={opt_order}")

        # Obtaining the looper object using python's `eval()` method
        # on the looping string defined in the json config file.
        looper = make_looper(looping_str, opt2vals, looping_cmplr)

        # First, we need to ask all non-'rest' looping leaf objects
        # (i.e., AtomicLooper) to claim their values, and pop their
        # options out of the `opt2vals` dictionary
        looper.pop_vals(turn="nonrest")

        # Next, we ask the 'rest' looping leaf objects to claim any
        # options that are left for themselves. There should only be
        # a single 'rest' looper. If there are more than one, the
        # second one will be empty of options.
        looper.pop_vals(turn="rest")

        # Now we should make sure that 'opt2vals' is emptied out and
        # all options were somehow used in the looping process.
        msg_ = f"""\
        The following options were left unspecified in the
        looping tree. Make sure you include all options in
        the looping tree. As a hint, you can use the "rest"
        key to insert the remaining options:
            {list(opt2vals.keys())}
        """
        assert len(opt2vals) == 0, dedent(msg_)

        # The main function call: Getting the list of all looped
        # config dictionaries.
        rt_cfgs = looper.get_cfgs()

        # Making sure nothing ambiguous is left to deal with.
        msg_rt = "The looping tree seems ambiguous and some of the "
        msg_rt += "options still seem uncurated. As a hint for solving "
        msg_rt += "this, you should put one of singleton loopers (e.g., "
        msg_rt += "cart, zip, or ovat) at the root of the looping tree!"
        assert len(rt_cfgs) == 1, msg_rt
        all_cfgs = rt_cfgs[0]

        # Taking care of the option sorting after the looper generated
        # looping-sorted dictionaries :)
        if opt_order == "config":
            all_srtdcfgs = []
            for cfg in all_cfgs:
                srtdcfg = odict()

                # Since the 'aslist' looper keeps the '*/list' option values
                # intact, the option indecis remain distributed among the
                # values. We should refactorize the index values before going
                # any further.
                def cure_aslist(opt, idx_vals):
                    if opt.endswith("/list"):
                        # Example:
                        #   opt = 'a/list'
                        #   idx_vals = [(0, 10), (0, 20), (0, 30)]
                        #   out = (0, [10, 20, 30])
                        opt_idxs = [x[0] for x in idx_vals]
                        assert len(set(opt_idxs)) <= 1
                        vals = [x[1] for x in idx_vals]
                        if len(opt_idxs) > 0:
                            opt_idx = opt_idxs[0]
                        else:
                            opt_idx = n_opts
                        out = (opt_idx, vals)
                    else:
                        out = idx_vals

                    # Some sanity checks
                    assert isinstance(out, tuple)
                    assert len(out) == 2
                    assert isinstance(out[0], int)

                    return out

                cfg = odict(
                    [
                        (opt, cure_aslist(opt, idx_vals))
                        for opt, idx_vals in cfg.items()
                    ]
                )

                # Applying the sorting process
                def srt_key(opt):
                    return cfg[opt][0]

                for opt in sorted(cfg, key=srt_key):
                    srtdcfg[opt] = cfg[opt][1]
                all_srtdcfgs.append(srtdcfg)
            all_cfgs = all_srtdcfgs
    else:
        # No modifications are necessary if the `'looping'` key is
        # absent from the config dictionary.
        all_cfgs = [cfg_dict]

    return all_cfgs

#########################################################
######### Parsing Confing References Utilities ##########
#########################################################

def ruyaml_safedump(data):
    """
    There is an order preservation problem when dumping dictionaries into yaml.

    * PyYaml does some weird type of sorting before dumping.

        * To preserve the original order, you have to specify `sort_keys=False`

    * Ruaml.YAML does the same sorting thing as well. 

        * However, it doesn't even have a `sort_keys` argument.

        * To dump with original order, you have to exactly do the following:

            ```
            import sys
            myyaml = ruamel.yaml.YAML()
            myyaml.dump(dictionary, sys.stout)
            ```

        Specifying `typ=safe` to `ruamel.yaml.YAML` will cause you to lose the original order!

    * If you want the most-inner lists/dicts to appear in-line: 

        * you have to specify `default_flow_style=None` to PyYaml.

        * In Ruamel.Yaml, you must set `myyaml.default_flow_style = None` (see the example below).

    ----------
    Here's a PyYaml same-order dumping example:

    ```
    import yaml
    yaml.safe_dump(dictionary, sort_keys=False, default_flow_style=None)
    ```

    Here's a Ruamel.Yaml same-order dumping example:

    ```
    import io, ruamel
    stream = io.StringIO()
    myyaml = ruamel.yaml.YAML()
    myyaml.default_flow_style = None
    myyaml.dump(dictionary, stream)
    print(stream.getvalue())
    ```

    Here's a smaller version if you want to directly stream to `sys.stdout`:

    ```
    import sys
    myyaml = ruamel.yaml.YAML()
    myyaml.default_flow_style = None
    myyaml.dump(dictionary, sys.stout)
    ```

    Parameters
    ----------
    data: (dict) the dictionary to be dumped

    Returns
    ------
    data_str: (str) the order-preserved data dumped as a string.
    """
    stream = io.StringIO()
    myyaml = ruyaml.YAML()
    myyaml.default_flow_style = None
    myyaml.dump(data, stream)
    data_str = stream.getvalue()
    return data_str


def load_yaml(path):
    """
    Loads a json or yaml file from a given path. It can handle 
    an incomplete path by pre-pending the `configs_dir`.

    Parameters
    ----------
        path: (str) the path to the yaml/json file.
    
    Outputs
    ----------
        json_cfgdict: (dict) the loaded config dictionary.
    """
    assert isinstance(path, str)
    
    prfxd_path = configs_dir + os.sep + path
    if exists(path):
        existing_path = path
    elif exists(prfxd_path):
        existing_path = prfxd_path
    else:
        raise ValueError(f'The path "{path}" does not exist')
    
    if existing_path.endswith('.json'):
        with open(existing_path, 'r') as fp:
            json_cfgdict = json.load(fp, object_pairs_hook=odict)
    elif existing_path.endswith('.yml'):
        with open(existing_path, "r") as fp:
            json_cfgdict = dict(ruyaml.safe_load(fp))
    else:
        raise RuntimeError(f'unknown config extension: {existing_path}')
    
    return json_cfgdict


def tnsfm_cfgdict(cfg_dict: dict, trnsfrmtn: dict | None = None):
    """
    Transforms a given config dictionary. The output is not 
    in-place-related to the input.

    Parameters
    ----------
        cfg_dict: (dict) the config dictionary.

        trnsfrmtn: (str | None) the transformation. Must be one 
            of `['hie', 'deep', 'none', None]`.
    
    Output
    ----------
        tnsfmed_cfgdict: (dict) The transformed config dictionary.
    """
    trnsfrmtn = trnsfrmtn if trnsfrmtn is not None else 'none'
    assert trnsfrmtn in ('hie', 'deep', 'none')

    if trnsfrmtn == 'hie':
        tnsfmed_cfgdict = deep2hie(cfg_dict)
    elif trnsfrmtn == 'deep':
        tnsfmed_cfgdict = hie2deep(cfg_dict)
    elif trnsfrmtn == 'none':
        tnsfmed_cfgdict = odict(cfg_dict)
    else:
        raise ValueError(f'undefined trnsfrmtn={trnsfrmtn}')
    
    return tnsfmed_cfgdict


def move_stub2end(cfg_dict: dict, sep: str = '/'):
    """
    Takes a hierarchical config dictionary and re-arranges the 
    key parts by relocating the stub parts of the keys (i.e., 
    `('ref', 'def', 'pop')`) to the end of them.

    Parameters
    ----------
        cfg_dict: (dict) the config dictionary.

        Example:
            ```
            cfg_dict = {'a/def/b/c': "1+1"}
            ```

        sep (str): The string seperator of hierarchy within 
            the keys. Usually, it is set to `'/'`.
    
    Output
    ----------
        out_cfgdict: (dict) The same as the input, but with 
        the stubs moved to the end of the key.

        Example:
            ```
            cfg_dict = {'a/b/c/def': "1+1"}
            ```
    """
    out_cfgdict = odict()
    all_stubs = ('def', 'ref', 'pop')
    for key, val in cfg_dict.items():
        assert not isinstance(val, dict), dedent(f'''
            I was supposed to get a hierarchical config dictionary 
            as input, yet one of the keys specifies a dictionary:
                key = "{key}"
                type(val) = "{type(val)}"
                val.keys() = "{val.keys()}"
        ''')
        
        key_splt = key.split(sep)
        n_stbs = sum(key_splt.count(stub) for stub in all_stubs)
        if n_stbs == 0:
            out_cfgdict[key] = val
        elif n_stbs == 1:
            assert 'ref' not in key_splt[:-1], dedent(f'''
                A "ref" stub can only appear a single time 
                at the end of the key:
                    key = {key}
            ''')
            
            for stub in ('def', 'pop'):
                n_stb = key_splt.count(stub)
                if n_stb == 0:
                    pass
                elif n_stb == 1:
                    i_stb = key_splt.index(stub)
                    key_splt.append(key_splt.pop(i_stb))
                else:
                    raise ValueError(dedent(f'''
                        Found {n_stb} "{stub}" stubs in the same key. Each 
                        hierarchical key must only carry a single stub 
                        at most:
                            key = "{key}"
                    '''))
            
            if key_splt[-1] == 'def':
                def_substubs = ('crd', 'evl')
                n_subdefstbs = sum(key_splt.count(stub) for stub in def_substubs)

                assert n_subdefstbs < 2, ValueError(dedent(f'''
                    Found multiple ({n_stbs}) def sub-stubs in the same key. 
                    Each hierarchical key must only carry a single stub at most:
                        key = "{key}"
                '''))

                for stub in def_substubs:
                    n_stb = key_splt.count(stub)
                    if n_stb == 0:
                        pass
                    elif n_stb == 1:
                        i_stb = key_splt.index(stub)
                        key_splt.insert(-1, key_splt.pop(i_stb))
                    else:
                        raise ValueError(dedent(f'''
                            Found two "{stub}" stubs in the same key. Each 
                            hierarchical key must only carry a single stub 
                            at most:
                                key = "{key}"
                        '''))
            
            newkey = sep.join(key_splt)
            out_cfgdict[newkey] = val
        else:
            raise ValueError(dedent(f'''
                Found multiple ({n_stbs}) stubs in the same key. 
                Each hierarchical key must only carry a single 
                stub at most:
                    key = "{key}"
            '''))
    
    for kkk in out_cfgdict:
        assert '/def/' not in kkk, f'kkk = {kkk}'
    
    return out_cfgdict


def frmlz_config(config: dict, cfg_path: str, pathsep: str, 
    dep_dict: dict, inplace: bool = True):
    """
    Gets a config dictionary, and does two things:

        1. for any key that ends with `'/ref'`:
            * its values will be prepended with the `cfg_path`.
            * The key and the values will be added to the `/ref` 
              entry of `dep_dict`.
        
        2. for any key that ends with `'/def'` or `/pop`:
            * The key and the values will be added to the `/def` 
              or `/pop` entry of `dep_dict`.
        
        3. Return `rfrd_paths` as a list of extra files that need 
           to be loaded to complete the references.

    For example: 
        ```
        config = {
            'a/b/c/ref': ['d/e/f', 'g/h/i']
            'x/def': 'something'
            'z/ref': '01_runs/02_conf.yml -> d/e/f'
            'y': 23}

        cfg_path = '01_runs/01_conf.yml'

        pathsep = ' -> '

        dep_dict = {'/ref': dict(), 
                    '/def': dict(), 
                    '/pop': dict(),
                    'none': list()}
        ```

        will turn `config` into 
        
        ```
        config = {
            'a/b/c/ref': [
                '01_runs/01_conf.yml -> d/e/f', 
                '01_runs/01_conf.yml -> g/h/i']
            'x/def': 'something'
            'z/ref': '01_runs/02_extra.yml -> d/e/f'
            'y': 23}

        rfrd_paths = ['01_runs/02_extra.yml']

        dep_dict = {
            '/ref': {
                '01_runs/01_conf.yml -> a/b/c': 
                    ['01_runs/01_conf.yml -> d/e/f', 
                     '01_runs/01_conf.yml -> g/h/i']}, 
            '/def': {
                '01_runs/01_conf.yml -> x': []},
            '/pop': dict(),
            'none': list()}
        ```
    
    Parameters
    ----------
        config (dict): The input config dictionary. The `*/ref`, 
            `*/def` and `*/pop` keys will have their values modified 
            in-place. 
        
        cfg_path (str): The path identifier for the input config 
            file. This will be used to prepedend the key names.

        pathsep (str): The seperator for telling which files 
            the keys are from. Usually, it's ' -> '.
        
        dep_dict (dict): A dictionary with 4 entries:
            
            * `'/ref'` must specify a dictionary.
            
            * `'/def'` must specify a dictionary.
            
            * `'/pop'` must specify a dictionary.
            
            * `'none'` must specify a list.
        
        inplace (bool): Whether `dep_dict` entries will be modified in-place 
            or not. Currently, this dictionary will mandatorily be modified 
            in-place.

    Output
    ----------
        config (dict): The completed version of the input.
        
        rfrd_paths (list): The list of external config files referred 
            to in the input config dictionary (see the intro example). 
        
        dep_dict (dict): The completed version of the input (see the 
            intro example).
    """
    assert inplace
    rfrd_paths = []

    ref_deps = dep_dict['/ref']
    assert isinstance(ref_deps, dict)
    def_deps = dep_dict['/def']
    assert isinstance(def_deps, dict)
    pop_deps = dep_dict['/pop']
    assert isinstance(pop_deps, dict)
    none_deps = dep_dict['none']
    assert isinstance(none_deps, list)

    cfg_pathdir = os.path.dirname(cfg_path)

    config_out = odict()
    for key_stub, value in config.items():
        assert pathsep not in key_stub, dedent(f'''
            Cannot accept a key with pathsep="{pathsep}" in it:
                key = {key_stub}
            ''')
        path_key_stub = pathsep.join((cfg_path, key_stub))
        if key_stub.endswith('/ref'):
            path_key = path_key_stub[:-4]
            vallst = [value] if isinstance(value, str) else value
            assert isinstance(vallst, (tuple, list))
            path_vallst = []
            for val in vallst:
                assert isinstance(val, str), dedent(f'''
                    The "{key_stub}" option specified a non-string 
                    value even though it is a reference type.
                        offending value: {val}
                        full option: {vallst}''')
                
                assert not any(val.endswith(stub) 
                    for stub in ('/ref', '/def', '/pop')), dedent(f'''
                        An referred key cannot have a `/ref`, `/def`, or 
                        `/pop` at the end. Please strip these stubs.
                           offending path -> key     ==    {path_key}
                           offending reference       ==    {val}
                    ''')
                        
                val_split = val.split(pathsep)
                if len(val_split) == 1:
                    val_split.insert(0, cfg_path)
                assert len(val_split) == 2, f'val="{val}" pathsep="{pathsep}"'
                
                ############# Handling relative referral paths #############
                # This next part tries to handle relative referral path locations.
                # Here is an ideal example:
                #   1. Let's say we are processing a config file at `a/b/c/cfg1.yml`.
                #   2. Inside this config file, we have a `key: ./d/e/cfg2.yml -> var` line.
                #   3. Ideally, we would like to translate this line into 
                #        `key: a/b/c/d/e/cfg2.yml -> var`.
                rfrd_path1 = val_split[0]
                # Checking whether this referred path is relative at all.
                need_prpnd = rfrd_path1.startswith('./') or rfrd_path1.startswith('../')
                # In case this referred path was relative, we augment it by the prepending 
                # the current config file's parent directory.
                rfrd_path2 = f'{cfg_pathdir}/{rfrd_path1}' if need_prpnd else rfrd_path1
                # The `os.path.normpath` will get rid of path redundancies:
                #   * For example, it replaces `a/b/.///../c` with `a/c`.
                #   * This will helpy unify references to the same file 
                #       * For example, 'a/b/..//c' and 'a/c' refer to the same file, and 
                #         don't need to be loaded twice.
                rfrd_path3 = os.path.normpath(rfrd_path2)
                # Now, we need to update `val_split`.
                val_split = [rfrd_path3, val_split[1]]
                
                rfrd_paths.append(val_split[0])
                path_val = pathsep.join(val_split)

                path_vallst.append(path_val)
                # adding the dependency 
                # `path_key_stub` is the child, and `path_val` is parent.
                # `path_val` must be processed first.
                if path_key not in ref_deps:
                    ref_deps[path_key] = list()
                if path_val not in ref_deps[path_key]:
                    ref_deps[path_key].append(path_val)
            
            value_out = path_vallst[0] if isinstance(vallst, str) else path_vallst
            config_out[path_key_stub] = value_out
        elif key_stub.endswith('/def'):
            path_key = path_key_stub[:-4]
            def_deps[path_key] = list()
            config_out[path_key_stub] = value
        elif key_stub.endswith('/pop'):
            path_key = path_key_stub[:-4]
            pop_deps[path_key] = list()
            config_out[path_key_stub] = value
        else:
            none_deps.append(path_key_stub)
            config_out[path_key_stub] = value

    return config_out, rfrd_paths, dep_dict


def impose_rules(dep_dict, sep='/', do_report=False):
    """
    This function imposes four new sets of dependency rules in order:

        1. If `alice` is a `*/pop` key, then it must be processed 
           after all of its `*/def`- and `*/ref`-"relatives".

        2. If `alice` is a `*/def` key, then it must be processed 
           after all of its `*/ref`-"relatives".
    
        3. If `alice` refers to `bob`, then `alice` must be 
           processed after all the `*/ref`- and `*/def`- and 
           `*/pop`-relatives of `bob` are processed.

           P.S. Here's an example of what we mean by `alice` 
           referring to `bob`:

           ```
           alice = 'a/b/c/ref'
           bob = 'g/h/ref'
           dep_dict['/ref'] = {'a/b/c/ref': 'g/h', ...} 
           ```
        
        4. If `alice` is a `*/ref` key, then it must be processed 
           before all of its `*/ref`-"child" keys.
    
    The `dep_dict` will change in-place.

    * Definition of "relative":

        The `bob` key is a `*/ref`-"relative" of `alice` if

            * `bob` has to end with `/ref`.
            
            * `alice` ends with one of the `/ref`, `/def` or 
            `/pop` stubs.
            
            * The non-stub part of one of them is a subset of 
            the other one's non-stub part.

        For example, the following `bob1` and `bob2` are 
        `*/ref`-"relative" of `alice`:
            
            ```
            alice = 'a/b/c/pop'
            bob1 = 'a/b/c/d/e/ref'
            bob2 = 'a/ref'
            ```
        
        However, these are not any kind of "relative":

            ```
            alice = 'a/b/c/ref'
            bob = 'a/f/c/d/e/ref'
            ```

    * Definition of "child":

        The `bob` key is a "child" of `alice` if

            * `alice` has to end with `/ref`.
            
            * `bob` ends with one of the `/ref`, `/def` or 
              `/pop` stubs.
            
            * The non-stub part of `alic` must be a subset of 
              `bob`'s non-stub part.
        
        For example, the following `bob` is a "child" of `alice`:
            
            ```
            alice = 'a/b/c/ref'
            bob = 'a/b/c/d/e/pop'
            ```
        
        However, the next `bob` is not a "child" of `alice`:
            
            ```
            alice = 'a/b/c/ref'
            bob = 'a/ref'
            ```
        
    Parameters
    ----------        
        dep_dict (dict): A dictionary with 4 entries:
            
            * `'/ref'` must specify a dictionary.
            
            * `'/def'` must specify a dictionary.
            
            * `'/pop'` must specify a dictionary.
            
            * `'none'` must specify a list.
        
        sep (str): The string seperator of hierarchy within the keys. 
            Usually, it is set to `'/'`.
    """
    ref_deps = dep_dict['/ref']
    assert isinstance(ref_deps, dict)
    def_deps = dep_dict['/def']
    assert isinstance(def_deps, dict)
    pop_deps = dep_dict['/pop']
    assert isinstance(pop_deps, dict)
    none_deps = dep_dict['none']
    assert isinstance(none_deps, list)

    # Doing the splitting once and for all
    ref_keysplts = [(key, key.split(sep)) for key in ref_deps]
    def_keysplts = [(key, key.split(sep)) for key in def_deps]
    pop_keysplts = [(key, key.split(sep)) for key in pop_deps]

    # Preparing the reporting material
    if do_report:
        rule_type1 = 'If `alice` is a `*/pop` key, then it must be '
        rule_type1 += 'processed after all of its `*/def`- and '
        rule_type1 += '`*/ref`-"relatives".'
        report_type1 = []

        rule_type2 = 'If `alice` is a `*/def` key, then it must be '
        rule_type2 += 'processed after all of its `*/ref`-"relatives"'
        report_type2 = []

        rule_type3 = 'If `alice` refers to `bob`, then `alice` must be '
        rule_type3 += 'processed after all the `*/ref`- and `*/def`- and '
        rule_type3 += '`*/pop`-relatives of `bob` are processed.'
        report_type3 = []

        rule_type4 = 'If `alice` is a `*/ref` key, then it must be '
        rule_type4 += 'processed before all of its `*/ref`-"child" keys.'
        report_type4 = []
    else:
        report_type1, rule_type1 = None, None
        report_type2, rule_type2 = None, None
        report_type3, rule_type3 = None, None
        report_type4, rule_type4 = None, None

    ############################ Imposing Rule (1) ############################
    #         If `alice` is a `*/pop` key, then it must be processed          #
    #            after all of its `*/def`- and `*/ref`-"relatives".           #
    ###########################################################################
    trg_keysplts = (ref_keysplts + def_keysplts)
    for key_pop, keysplt_pop in pop_keysplts:
        assert len(pop_deps[key_pop]) == 0
        for stub_trg, trg_keysplts in (('/ref', ref_keysplts), ('/def', def_keysplts)):
            for (key_trg, keysplt_trg) in trg_keysplts:
                if ((keysplt_pop == keysplt_trg[:len(keysplt_pop)]) or 
                    (keysplt_trg == keysplt_pop[:len(keysplt_trg)])):
                    keystub_trg = key_trg + stub_trg
                    pop_deps[key_pop].append(keystub_trg)
                    if do_report:
                        report_type1.append((key_pop + '/pop', keystub_trg))
    
    ############################ Imposing Rule (2) ############################
    #          If `alice` is a `*/def` key, then it must be processed         #
    #                 after all of its `*/ref`-"relatives".                   #
    ###########################################################################
    for key_def, keysplt_def in def_keysplts:
        assert len(def_deps[key_def]) == 0
        for (key_ref, keysplt_ref) in ref_keysplts:
            if ((keysplt_def == keysplt_ref[:len(keysplt_def)]) or 
                (keysplt_ref == keysplt_def[:len(keysplt_ref)])):
                keystub_ref = key_ref + '/ref'
                def_deps[key_def].append(keystub_ref)
                if do_report:
                    report_type2.append((key_def + '/def', keystub_ref))
    
    ############################ Imposing Rule (3) ############################
    #            If `alice` refers to `bob`, then `alice` must be             #
    #            processed after all the `*/ref`- and `*/def`- and            #
    #               `*/pop`-relatives of `bob` are processed.                 #
    ###########################################################################
    for key_ref, keysplt_ref in ref_keysplts:
        key_refdeps = []
        for val in ref_deps[key_ref]:
            valsplt = val.split(sep)
            for stub_trg, trg_keysplts in (('/ref', ref_keysplts), 
                                           ('/def', def_keysplts), 
                                           ('/pop', pop_keysplts)):
                for (key_trg, keysplt_trg) in trg_keysplts:
                    if ((valsplt == keysplt_trg[:len(valsplt)]) or 
                        (keysplt_trg == valsplt[:len(keysplt_trg)])):
                        keystub_trg = key_trg + stub_trg
                        key_refdeps.append(keystub_trg)
                        if do_report:
                            report_type3.append((key_ref + '/ref', keystub_trg))
        ref_deps[key_ref] = key_refdeps
    
    ############################ Imposing Rule (4) ############################
    #          If `alice` is a `*/ref` key, then it must be processed         #
    #                 after all of its `*/ref` "child" keys.                  #
    ###########################################################################
    # Making a call to the following recursive function
    impose_childrules({'/ref': ref_deps}, sep='/', report_lst=report_type4)

    if do_report:
        report_dict = {
            'Type 1': (rule_type1, report_type1),
            'Type 2': (rule_type2, report_type2),
            'Type 3': (rule_type3, report_type3),
            'Type 4': (rule_type4, report_type4)}
    else:
        report_dict = None

    return dep_dict, report_dict


def impose_childrules(dep_dict: dict, hieset: (set | None) = None, 
    prefix: (str | None) = None, sep: str = "/", report_lst: list | None = None, 
    tablvls: int = 0, verbose: int = 0):
    """
    This function imposes the following rule of dependency:

        If `alice` is a `*/ref` key, then it must be processed 
        after all of the "child"-keys.

    That being said, this function is written in a general manner, and can 
    apply these rules to all kinds of provided `/ref`, `/def`, and `/chef`(!) 
    stubs as long as they're provided in the input `dep_dict`.

    * Definition of "child":

        The `bob` key is a "child" of `alice` if

            * `alice` has to end with `/ref`.
            
            * `bob` ends with one of the `/ref`, `/def` or 
              `/pop` stubs.
            
            * The non-stub part of `alic` must be a subset of 
              `bob`'s non-stub part.
        
        For example, the following `bob` is a "child" of `alice`:
            
            ```
            alice = 'a/b/c/ref'
            bob = 'a/b/c/d/e/pop'
            ```
        
        However, the next `bob` is not a "child" of `alice`:
            
            ```
            alice = 'a/b/c/ref'
            bob = 'a/b/ref'
            ```
    
    Parameters
    ----------
        dep_dict (dict): A dictionary with a variable number of 
            stub (str) to dependency (dict) entries:
            
            Example: 
                ```
                dep_dict = {
                    '/ref': {
                        'a/b/c': ['a/g', 'f/h']
                        'a/d/m/n': ['a/g', 'f/h']
                    },
                    '/def': {
                        'a/g': ['p/e']
                    }
                }
                ```

        hieset: (set | None) an internally created set for recursion 
            purposes. do not provide one if you're making an initial 
            call to this function.

        prefix: (set | None ) an internally created string for recursion 
            purposes. do not provide one if you're making an initial 
            call to this function.

        sep: (str) the hierarchy seperator string, which is usually `'/'`.

        tablvls: (int) The number of tab levels when being a verbose 
            printer.
        
        verbose: (int) The verbosity level; zero is silent.
    """
    if hieset is None:
        hieset = set(
            (key, stub, i_stub) 
            for i_stub, (stub, graph_stub) in enumerate(dep_dict.items())
            for key in graph_stub
        )
    
    if verbose > 1:
        print((tablvls * '   ') + f'new call: hieset = {list(hieset)}')

    prefixsep = '' if (prefix is None) else (prefix + sep)
    lvl1keys = set(key.split(sep)[0] for (key, stub_key, i_stbkey) in hieset)
    for key in lvl1keys:
        if verbose > 2:
            print((tablvls * '   ') + f' loop:  {key}')
        subset = set()
        prfx = key + sep
        for (kk, stub_kk, i_stbkk) in hieset:
            if kk.startswith(prfx):
                newmmbr = (kk[len(prfx):], stub_kk, i_stbkk)
                subset.add(newmmbr)
        
        all_stbs = [(stb, i_stb) for (kkk, stb, i_stb) in hieset if (kkk == key)]
        for stub2_key, i2_stbkey in all_stbs:
            parent = prefixsep + key
            parent_depdict = dep_dict[stub2_key]
            for (kk, stub_kk, i_stbkk) in subset:
                child = prefixsep + key + sep + kk
                child_depdict = dep_dict[stub_kk]
                
                if verbose > 0:
                    print((tablvls * '   ') + ' -> edge added: parent = ' + 
                          f'{parent}, child = {child}, prefixsep=' +
                          f'{prefixsep}, key={key}, stub2_key={stub2_key}')
                
                assert parent in parent_depdict, dedent(f'''
                    prefixsep={prefixsep}, key={key}, 
                    stub2_key={stub2_key}, parent={parent}
                ''')

                assert child in child_depdict, dedent(f'''
                    prefixsep={prefixsep}, key={key}, kk={kk}, 
                    stub_kk={stub_kk}, child={child}
                ''')

                parent_stub2 = parent + stub2_key
                child_depdict[child].append(parent_stub2)
                if report_lst is not None:
                    report_lst.append((child + stub_kk, parent_stub2))
        
        if len(subset) > 0:
            impose_childrules(dep_dict, subset, prefixsep + key, 
                sep, report_lst, tablvls + 1, verbose)


def parse_refs(cfg_dict, trnsfrmtn='hie', pathsep=' -> ',
    cfg_path='original', do_report=False):
    """
    Pasrse the keys with a "/ref", "/def", or "/pop" postfix.
    
      * The "/ref" postfix adds the reference dictionary keys in a 
    hierarchical fashion (see the example).

      * The "/def" postfix evaluates the string formula given in the .

      * The "/pop" postfix removes the existing keys (see the example).
    
    The "*/ref" keys cannot evaluate a formula, but they can trace a 
    cascading reference multiple levels.

    The "*/def" keys can evaluate a formula, but they cannot reference 
    values in other files. Each "something/def" key will be popped and be 
    replaced with an evaluated "something" key.

    If you need to both reference a value from another file and apply a 
    function to it, you will have to first "/ref" it temporarily, then 
    "/def" whatever you want, and then "/pop" the temporary "/ref"-ed key.

    Parameters
    ----------
    cfg_dict: (dict) the config dictionary coming from a json file.

        Example: {"hello/ref": "lvl/opts01",
                  "world/ref": ["opts02", "opts03"],

                  "of/def": "2 * config['opts04'][-1]",

                  "tomorrow/ref": "opts03",
                  "tomorrow/pop": ["c", "f"],

                  "lvl: {"opts01": {"a": 1, "b": 2}},
                  "opts02": {"c": 3, "d": 4},
                  "opts03": {"c": 5, "e": 6, "f": 7},
                  "opts04": ["x", "y", 7]
                }

    trnsfrmtn: (str) The transformation that would be applied to the 
        input config. The output will also follow the same form. Note 
        that this should be set to 'hie', and is more set there to 
        raise the user's awareness about this transformation.

    pathsep: (str) the seperator for designating the originating file, 
        and has a default value of' -> '.

    do_report: (bool) whether to print what type of predescence rules 
        and relations were applied for parsing.

    Outputs
    ----------
    proced_cfg_dict: (dict) the same dictionary with the references 
        being parsed.

        Example: {"hello/a": 1,
                  "hello/b": 2,
                  "world/c": 5,
                  "world/d": 4,
                  "world/e": 6,
                  "world/f": 7,
                  "of": 14,
                  "tomorrow/e": 6,
                  
                  "lvl/opts01/a": 1,
                  "lvl/opts01/b": 2,
                  "opts02/c": 3,
                  "opts02/d": 4,
                  "opts03/c": 5,
                  "opts03/e": 6, 
                  "opts03/f": 7, 
                  "opts04": ["x", "y", 7]}
    """
    assert trnsfrmtn == 'hie', 'code is not tested for non-hie transformations'

    # First, we have to find all the files containing references and load 
    # them. Also, we should resolve all the `/ref`-ending keys into 
    # their full `file:rfrd_name` form.
    dep_dict = {'/ref': dict(), '/def': dict(), '/pop': dict(), 'none': list()}

    # Applying a hierarchical or deep trnsfrmtn on the input
    cfg_dict = tnsfm_cfgdict(cfg_dict, trnsfrmtn)
    # Moving the stubs to the end of keys
    cfg_dict = move_stub2end(cfg_dict)
    # Formalizing the config
    cfg_dict, rfrd_paths, dep_dict = frmlz_config(cfg_dict, cfg_path=cfg_path, 
        pathsep=pathsep, dep_dict=dep_dict, inplace=True)

    # Compiling all the references from the mentioned files
    all_configs = cfg_dict
    # The set of files that were 
    loaded_files = set((cfg_path,))
    while len(rfrd_paths) > 0:
        rfrd_path = rfrd_paths.pop(0)
        if rfrd_path in loaded_files:
            continue
        
        assert rfrd_path != cfg_path
        rfrd_prntcfg = load_yaml(rfrd_path)

        # Applying a hierarchical or deep trnsfrmtn on the config
        rfrd_prntcfg = tnsfm_cfgdict(rfrd_prntcfg, trnsfrmtn)
        # Moving the stubs to the end of keys
        rfrd_prntcfg = move_stub2end(rfrd_prntcfg)
        # Formalizing the config
        rfrd_prntcfg, rfrdpaths_new, dep_dict = frmlz_config(rfrd_prntcfg, 
            cfg_path=rfrd_path, pathsep=pathsep, dep_dict=dep_dict, 
            inplace=True)
        
        # adding all the newly discovered reference files to the search list
        rfrd_paths += rfrdpaths_new
        # Marking this path as loaded
        loaded_files.add(rfrd_path)
        # Compiling the new config dict into the 
        all_configs.update(rfrd_prntcfg)

    # Making sure no key references its children and the referred keys are santized
    for key, rfrd_keys in dep_dict['/ref'].items():
        for rfrd_key in rfrd_keys:            
            assert not rfrd_key.startswith(key), dedent(f'''
                An option cannot refer to its descendent options. This 
                violates the fourth rule of parsing.
                    Offending reference: "{key}" refers to "{rfrd_key}".
            ''')

    # Imposing all the reference rules
    dep_dict, report_dict = impose_rules(dep_dict, sep='/', do_report=do_report)

    # Printing the report if necessary
    if do_report:
        msg = ''
        df_rprtdict = defaultdict(list)
        for rule_name, (rule, rule_pairs) in report_dict.items():
            rule_indntd = rule.replace("\n", "\n" + " " * 2)
            msg += dedent(f'''
                Reporting the "{rule_name}" Rule:
                    * Rule Description: 
                        "{rule_indntd}"
                    * The keys affected by this rule are:
            ''')
            for child, prnt in rule_pairs:
                child_loc, child_var = child.split(pathsep)
                prnt_loc, prnt_var = prnt.split(pathsep)
                df_rprtdict['rule'].append(rule_name)
                df_rprtdict['before/path'].append(child_loc)
                df_rprtdict['after/path'].append(prnt_loc)
                df_rprtdict['before/key'].append(child_var)
                df_rprtdict['after/key'].append(prnt_var)
                msg += ' ' * 8 + '*  '
                msg += f'"{child}" should be processed after "{prnt}"' 
                msg += '\n'
            msg += '-' * 40
        print(msg, flush=True)
        return pd.DataFrame(df_rprtdict)

    # Creating the dependency graph
    ref_deps = dep_dict['/ref']
    def_deps = dep_dict['/def']
    pop_deps = dep_dict['/pop']
    none_deps = dep_dict['none']

    # Note: 
    #   1. The `key` variable in the following loops never 
    #       has a '/ref', '/def/', or '/pop' stub. This is 
    #       why we need to append their stub back.
    #   2. All the values in the `dep_list` are keys that 
    #      end with either a '/ref', '/def/', or '/pop' stub. 
    #      That's why we don't add any stubs to them.
    graph = {key + stub: dep_list
        for stub, stub_deps in [('/ref', ref_deps), 
            ('/def', def_deps), ('/pop', pop_deps)]
        for key, dep_list in stub_deps.items()}
    
    # Making sure the imposed rules so far only reference real keys
    all_keys = set(graph) | set(none_deps)
    for key, vallst in graph.items():
        assert key in all_keys
        for val in vallst:
            assert val in all_keys, f'key = "{key}", val = "{val}"'

    # Finding the topological order of parsing
    dag = graphlib.TopologicalSorter(graph)
    keys_ts = list(dag.static_order())
    keys_tsset = set(keys_ts)
    assert all(key not in keys_tsset for key in none_deps)
    static_order = none_deps + keys_ts

    # We will start with `all_configs`, and process the references 
    # and definitions, and then slash the unnecessary material.
    prsd_cfgdict = all_configs
    
    # To keep the insertion order of the new keys at the same order 
    # of `key`, we have to displace the items after `key` to another 
    # `prsd_cfgafterdict` dictionary, and replace them back after `key`
    # was processed.
    ranks = {key: i_key for i_key, key in enumerate(prsd_cfgdict)}

    for key in static_order:
        value = prsd_cfgdict[key]
        if key.endswith('/ref'):
            ######################## Processing the `ref` commands #######################
            keysprfx = key[:-4]
            keyrank = ranks[key]

            vallst = [value] if isinstance(value, str) else value
            assert isinstance(vallst, list)

            for val in vallst:
                tmplt, n_pops = get_subdict(all_configs, prefix=val, postfix=None, 
                    sep='/', pop=False, ret_npop=True)
                assert n_pops > 0, dedent(f'''
                    We could not find any refernce for "{val}":
                        valid references: {all_configs.keys()}
                ''')
                updt_dict = deep2hie({keysprfx: tmplt}, sep='/', maxdepth=2)
                prsd_cfgdict.update(updt_dict)

                for ii, kk in enumerate(updt_dict):
                    ranks[kk] = keyrank + (0.1 * ii / len(updt_dict))
            prsd_cfgdict.pop(key)
        elif key.endswith('/def'):
            ######################## Processing the `def` commands #######################
            pathcfg = key.split(pathsep)[0]
            drct_acscfg = get_subdict(prsd_cfgdict, prefix=pathcfg, sep=pathsep)
            drct_acscfg.update(prsd_cfgdict)
            defntn_ref = {'config': drct_acscfg, 'cfg': drct_acscfg, 'c': drct_acscfg}
            
            if key.endswith('/crd/def'):
                need_eval, do_catch, newkey = False, False, key[:-8]
            elif key == 'crd/def':
                need_eval, do_catch, newkey = False, False, key[:-7]
            elif key.endswith('/evl/def'):
                need_eval, do_catch, newkey = True, True, key[:-8]
            elif key == 'evl/def':
                need_eval, do_catch, newkey = True, True, key[:-7]
            else:
                need_eval, do_catch, newkey = True, True, key[:-4]
            
            val_frmla = value
            if not need_eval:
                val = val_frmla
            else:
                try:
                    val = eval_formula(val_frmla, defntn_ref, catch=do_catch)
                except Exception as exc:
                    exc.add_note(dedent(f'''
                        This error happend while trying to evaluate the 
                        "{key}" option when parsing the "*/def" options.
                        
                        * If you have cascading definitions, make sure you 
                            define them in order.
                        
                        * If you wish to evaluate a non-string value for a 
                            "*/def" key, make sure you provide the key as a 
                            "*/def/evl" or "*/evl/def" option.
                        
                        The definitions reference:
                            defntn_ref = {defntn_ref.keys()}
                    '''))
                    raise exc


            # Removing the `*/def` key         
            prsd_cfgdict.pop(key)

            # adding the new values
            updt_dict = deep2hie({newkey: val}, sep='/')
            prsd_cfgdict.update(updt_dict)

            # adding the ranks
            newkeysep = f'{newkey}/'
            if newkey in ranks:
                keyrank = ranks[newkey]
            elif any(kk.startswith(newkeysep) for kk in ranks):
                key_lastsimilar = [kk for kk in ranks if kk.startswith(newkeysep)][-1]
                keyrank = ranks[key_lastsimilar]
            else:
                keyrank = ranks[key]

            for ii, kk in enumerate(updt_dict):
                if kk not in ranks:
                    ranks[kk] = keyrank + (0.1 * ii / len(updt_dict))

        elif key.endswith('/pop'):
            ######################## Processing the `pop` commands #######################
            drop_prfx, drop_pstfxs_ = key[:-4], value
            drop_pstfxs = [drop_pstfxs_] if drop_pstfxs_ is None else drop_pstfxs_
            drop_pstfxlst = [drop_pstfxs] if isinstance(drop_pstfxs, str) else drop_pstfxs
            assert isinstance(drop_pstfxlst, list)
            
            for drop_pstfx in drop_pstfxlst:
                drop_root = f'{drop_prfx}/{drop_pstfx}'
                drpd, n_pops = get_subdict(prsd_cfgdict, prefix=drop_root,
                    sep='/', pop=True, ret_npop=True)

                assert n_pops > 0, dedent(f'''
                    We could not find and pop anything with a "{drop_prfx}" 
                    prefix and "{drop_pstfx}" postfix. This pop request was 
                    made by {key}.
                        valid references: {all_configs.keys()}
                ''')

            # Removing the `*/pop` key            
            prsd_cfgdict.pop(key)
        else:
            assert not any(key.endswith(stub) 
                for stub in ('/ref', '/def', '/pop'))

    # Replacing the displaced items back into the `prsd_cfgdict` dictionary
    prsd_cfglist = sorted(prsd_cfgdict.items(), key=lambda keyval: ranks[keyval[0]])
    prsd_cfgdict = dict(prsd_cfglist)

    # Subsetting and removing the "original -> " prefix 
    prsd_cfgdict = get_subdict(prsd_cfgdict, prefix=cfg_path, postfix=None,
        sep=pathsep, pop=True, ret_npop=False)

    return prsd_cfgdict

#########################################################
################### Generic Utilties ####################
#########################################################


def get_git_commit():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode()
        )
    except Exception:
        commit_hash = "no git repo found"
    return commit_hash


#########################################################
################### Data Writer Class ###################
#########################################################


class DataWriter:
    def __init__(self, flush_period=10, compression_level=9, driver='sec2'):
        self.flush_period = flush_period
        self.file_path = None
        self.key2cdi = odict()
        self.extndedkeys = set()
        self.gnrlidx = 0
        self.compression_level = compression_level
        self.driver = driver

    def set_path(self, file_path):
        if self.file_path is not None:
            if file_path != self.file_path:
                print(
                    f"DW: need to flush after {self.gnrlidx} rows "
                    "due to a set_path call",
                    flush=True,
                )
                self.flush()
                self.gnrlidx = 0
        self.file_path = file_path

    @property
    def data_len(self):
        dlen = 0
        if len(self.key2cdi) > 0:
            dlen = max(
                len(index)
                for key, (colt_t, data, index) in self.key2cdi.items()
            )
        return dlen

    @property
    def file_ext(self):
        assert self.file_path is not None
        if self.file_path.endswith(".csv"):
            return "csv"
        elif self.file_path.endswith(".h5"):
            return "h5"
        else:
            raise ValueError(f"Unknown extension for {self.file_path}")

    def add(self, data_tups, file_path):
        if (file_path is None) or (len(data_tups) == 0):
            return

        self.set_path(file_path)

        # Making sure all options in all hdf keys have the same
        # length (i.e., number of rows)
        opt2len = dict()
        for key, rows_dict, col_type in data_tups:
            if rows_dict is None:
                assert key in self.key2cdi, dedent(f'''
                    cannot specify a null rows_dict if no prior 
                    data has been pushed:
                        "{key}" has null data specified.
                ''')
                self.extndedkeys.add(key)
                col_t, data, index = self.key2cdi[key]
                for opt, vals in data.items():
                    if col_t == 'np.arr':
                        assert len(vals) == 1, dedent(f'''
                            Somehow a non-single array was pushed into the 
                            data cache for "{key}/{opt}". This is bad:
                                key = {key}
                                opt = {opt}
                                col_t = {col_t}
                                len(vals) = {len(vals)}
                        ''')
                        assert isinstance(vals[0], np.ndarray), dedent(f'''
                            Not sure how a non-numpy array was pushed into 
                            the data cache for "{key}/{opt}":
                                key = {key}
                                opt = {opt}
                                col_t = {col_t}
                                offending value type = {type(vals[0])}
                        ''')
                        vals_len = len(vals[0])
                    elif col_t in ('pd.yml', 'pd.qnt', 'pd.cat'):
                        vals_len = len(vals)
                    else:
                        raise ValueError(f'undefined col_t = {col_t}')
                    opt2len[f"{key}/{opt}"] = vals_len
            else:
                assert key not in self.extndedkeys, dedent(f'''
                    The "{key}" specified null data once 
                    before. Once you specify null data, you 
                    cannot add any new data to the key.
                ''')
                for opt, vals in rows_dict.items():
                    opt2len[f"{key}/{opt}"] = len(vals)

        assert len(set(opt2len.values())) == 1, dedent(f'''
            The number of rows must be the same across
            all keys and options:
                {opt2len}
        ''')
        n_rows = list(opt2len.values())[0]
        newidxs = list(range(self.gnrlidx, self.gnrlidx + n_rows))

        # Checking if shapes have changed and therefore flush if needed.
        need_flush = False
        for key, rows_dict, col_t_ in data_tups:
            if rows_dict is None:
                continue
            if key in self.key2cdi:
                col_t, data, index = self.key2cdi[key]
                if (col_t == "np.arr") and (self.file_ext == "h5"):
                    for i, (opt, val) in enumerate(rows_dict.items()):
                        assert isinstance(val, np.ndarray)
                        if opt in data:
                            a, b = val, data[opt][0]
                            issubd = np.issubdtype(a.dtype, b.dtype)
                            if (a.shape[1:] != b.shape[1:]) or not (issubd):
                                need_flush = True

        if need_flush:
            print(
                f"DW: need to flush after {self.gnrlidx} rows "
                "due to shape mismatch.",
                flush=True,
            )
            self.flush(forget=True)

        # Inserting the new data
        for key, rows_dict, col_t_ in data_tups:
            if key not in self.key2cdi:
                col_t, data, index = col_t_, odict(), []
                self.key2cdi[key] = (col_t, data, index)
                
                assert rows_dict is not None, dedent(f'''
                    The "{key}" had no data before, and 
                    has null data now. This is unacceptable.
                ''')
            else:
                col_t, data, index = self.key2cdi[key]
                assert col_t == col_t_, f"{key} was {col_t} not {col_t_}"

                if rows_dict is not None:
                    assert set(data.keys()) == set(rows_dict.keys()), dedent(f'''
                        {key} input opts and my columns are different:
                            input opts: {set(rows_dict.keys())}
                            my columns: {set(data.keys())}
                    ''')

            assert col_t in ("np.arr", "pd.cat", "pd.qnt", "pd.yml")
            assert (self.file_ext != "csv") or (col_t in ("pd.cat", "pd.qnt"))
            
            index += newidxs
            if rows_dict is not None:
                for i, (opt, val) in enumerate(rows_dict.items()):
                    if opt not in data:
                        data[opt] = []
                    if col_t == "np.arr":
                        val_lst = [val]
                    else:
                        val_lst = list(val)
                    data[opt] += val_lst

        self.gnrlidx += n_rows
        if (self.data_len >= self.flush_period) and (self.flush_period > 0):
            print(
                f"DW: need to flush after {self.gnrlidx} "
                "rows due to io/flush frequency.",
                flush=True,
            )
            self.flush(forget=True)

    def flush(self, forget=True):
        if self.data_len == 0:
            return None
        assert self.file_path is not None

        flsh_sttime = time.time()
        if self.file_ext == "csv":
            for key, (col_t, data, index) in self.key2cdi.items():
                assert col_t in ("pd.cat", "pd.qnt"), "np.arr not impl. yet"
                pddtype = "category" if col_t == "pd.cat" else None
                data_df = pd.DataFrame(data, dtype=pddtype)
                data_df["ioidx"] = index
                columns = list(data_df.columns)
                csvfpath = self.file_path.replace(
                    ".csv", f'_{key.replace("/", "_")}.csv'
                )
                # Appending the latest row to file_path
                if not os.path.exists(csvfpath):
                    data_df.to_csv(
                        csvfpath,
                        mode="w",
                        header=True,
                        index=False,
                        columns=columns,
                    )
                else:
                    # First, check if we have the same columns
                    old_cols = pd.read_csv(csvfpath, nrows=1).columns.tolist()
                    old_cols_set = set(old_cols)
                    my_cols_set = set(columns)
                    msg_assert = "file columns and my columns are different:\n"
                    msg_assert = msg_assert + f"  file cols: {old_cols_set}\n"
                    msg_assert = msg_assert + f"  my columns: {my_cols_set}\n"
                    assert old_cols_set == my_cols_set, msg_assert
                    data_df.to_csv(
                        csvfpath,
                        mode="a",
                        header=False,
                        index=False,
                        columns=old_cols,
                    )
        elif self.file_ext == "h5":
            assert all(
                col_t in ("pd.cat", "pd.qnt", "np.arr", "pd.yml")
                for key, (col_t, data, index) in self.key2cdi.items()
            )
            ##################################################################
            ##################### Pandas HDF Operations ######################
            ##################################################################
            # Getting the next part index to write
            nextpart = 0
            if exists(self.file_path):
                h5hdf = h5py.File(self.file_path, mode="r", driver=self.driver)
                roots = set(x for x in h5hdf.keys() if x.startswith("P"))
                roots = set(x[1:] for x in roots)
                roots = set(x for x in roots if x.isdigit())
                roots = set(int(x) for x in roots)
                nextpart = max(roots) + 1 if len(roots) > 0 else 0
                assert f"{nextpart:08d}" not in h5hdf.keys()
                h5hdf.close()

            # Writing the pandas dataframes
            pdhdf = pd.HDFStore(self.file_path, mode="a")
            for key, (colt_t, data, index) in self.key2cdi.items():
                if colt_t in ("pd.cat", "pd.qnt"):
                    # Writing the main table
                    pddtype = "category" if colt_t == "pd.cat" else None
                    pdfmt = "table" if colt_t == "pd.cat" else None
                    data_df = pd.DataFrame(data, dtype=pddtype)
                    bools_d = {
                        col: "bool"
                        for col, v_lst in data.items()
                        if all(isinstance(v, bool) for v in v_lst)
                    }
                    if len(bools_d) > 0:
                        data_df = data_df.astype(bools_d)
                    data_df.to_hdf(
                        pdhdf,
                        key=f"/P{nextpart:08d}/{key}",
                        format=pdfmt,
                        index=False,
                        append=False,
                        complib="zlib",
                        complevel=self.compression_level,
                    )
            pdhdf.close()

            # Writing the numpy array datasets
            h5hdf = h5py.File(self.file_path, mode="a", driver=self.driver)
            for key, (colt_t, data, index) in self.key2cdi.items():
                np_idx = np.array(index)
                h5hdf.create_dataset(
                    f"/P{nextpart:08d}/{key}/ioidx",
                    shape=np_idx.shape,
                    dtype=np_idx.dtype,
                    data=np_idx,
                    compression="gzip",
                    compression_opts=9,
                )
                if colt_t in ("np.arr",):
                    for opt, np_arr_list in data.items():
                        np_arr = np.concatenate(np_arr_list, axis=0)
                        h5hdf.create_dataset(
                            f"/P{nextpart:08d}/{key}/{opt}",
                            shape=np_arr.shape,
                            dtype=np_arr.dtype,
                            data=np_arr,
                            compression="gzip",
                            compression_opts=self.compression_level,
                        )
                elif colt_t in ("pd.yml",):
                    try:
                        data_str = ruyaml_safedump(data)
                    except Exception as exc:
                        exc.add_note(dedent(f'''
                            This error happened while processing the following:
                                key = {key}
                                colt_t = {colt_t}'''))
                        self.tblocals = locals()
                        self.tbglobals = globals()
                        raise exc
                    assert isinstance(data_str, str)
                    h5hdf.create_dataset(
                        f"/P{nextpart:08d}/{key}/yml",
                        shape=1,
                        dtype=h5py.string_dtype(),
                        data=[data_str],
                        compression="gzip",
                        compression_opts=self.compression_level,
                    )
                    
            h5hdf.close()
        else:
            raise ValueError(
                f"file extension not implemented for {self.file_path}."
            )

        self.key2cdi.clear()
        self.extndedkeys.clear()
        print(f"DW: flush took {time.time()-flsh_sttime:.3f} seconds.")

    def close(self):
        if self.file_path is not None:
            msg_ = f"DW: need to flush after {self.gnrlidx}"
            msg_ += " rows due to close call."
            print(msg_, flush=True)
        return self.flush(forget=True)


#########################################################
######## Hierarchical-Deep Dictionary Conversion ########
#########################################################

def hie2deep(hie_dict, dictcls=None, mindepth=1, maxdepth=None, sep="/", kind=None):
    """
    Gets a hierarchical dictionary and returns a deep
    dictionary conversion of it. Raises an error if a
    bare key is present along with its children (e.g.,
    {'a': 1.0, 'a/b': 2.0} will raise an error).

    Parameters
    ----------
    hie_dict: (dict) a hierarchical dictionary
        Example: {'a/b/c': 1.,
                  'a/b/d': 2.}

    dictcls: (class or None) the dict constructor

    maxdepth: (int) the maximum depth until which the
        conversion recursion can go. If None, no max depth
        constraint will be applied.

    sep: (str) the directory seperator character, such as
         '/' in unix addressing.

    kind: (str | None) Whether the keys are 'str' or 'tuple'. 
        The default of None will infer the kind.

    Output
    ----------
    deep_dict: (dict) a deep dictionary
        Example: {'a': {'b': {'c': 1., 'd': 2.}}}
    """
    if dictcls is None:
        dictcls = odict if isinstance(hie_dict, odict) else dict
    
    if (kind is None):
        if all(isinstance(key, str) for key in hie_dict):
            kind = 'str'
        elif all(isinstance(key, tuple) for key in hie_dict):
            kind = 'tuple'
        else:
            raise ValueError('Keys should either all be string or tuple!')
    
    if kind == 'tuple':
        assert tuple() not in hie_dict, dedent('''The hierarchical dictionary 
            should not have an empty tuple in it.''')

    chmaxdepth = None if maxdepth is None else (maxdepth - 1)
    cangodeeper = (chmaxdepth is None) or (chmaxdepth > 0)

    deep_dict = dictcls()
    piped_keys = set()
    if (kind == 'str'):
        for key, val in hie_dict.items():
            if key.count(sep) >= mindepth:
                key_seq = key.split(sep, mindepth)
                keylvl1 = sep.join(key_seq[:mindepth])
                keyrst = key_seq[-1]

                assert keylvl1 not in piped_keys, dedent(f'''
                    cannot have "{keylvl1}" when its children 
                    are present: 
                        keys: {list(hie_dict.keys())}''')

                if keylvl1 not in deep_dict:
                    subdict = dictcls()
                    deep_dict[keylvl1] = subdict
                else:
                    subdict = deep_dict[keylvl1]
                subdict[keyrst] = val
            else:
                assert key not in deep_dict, dedent(f'''
                    cannot have "{key}" when its children 
                    are present: 
                        keys: {list(hie_dict.keys())}''')
                deep_dict[key] = val
                piped_keys.add(key)
    elif (kind == 'tuple'):
        for key_seq, val in hie_dict.items():
            if len(key_seq) > mindepth:
                keylvl1seq, keyrst = key_seq[:mindepth], key_seq[mindepth:]
                keylvl1 = keylvl1seq[0] if (mindepth == 1) else keylvl1seq

                assert keylvl1 not in piped_keys, dedent(f'''
                    cannot have "{keylvl1}" when its children are present: 
                        keys: {list(hie_dict.keys())}''')

                if keylvl1 not in deep_dict:
                    subdict = dictcls()
                    deep_dict[keylvl1] = subdict
                else:
                    subdict = deep_dict[keylvl1]
                subdict[keyrst] = val
            else:
                keylvl1 = key_seq[0] if (mindepth == 1) else key_seq

                assert keylvl1 not in deep_dict, dedent(f'''
                    cannot have "{keylvl1}" when its children are present: 
                        keys: {list(hie_dict.keys())}''')

                deep_dict[keylvl1] = val
                piped_keys.add(keylvl1)
    else:
        raise ValueError(f'undefined kind={kind}')

    if cangodeeper:
        for key, subdict in deep_dict.items():
            if (key not in piped_keys):
                deep_dict[key] = hie2deep(subdict, dictcls, 1, chmaxdepth, sep, kind)
    
    return deep_dict


def deep2hie(deep_dict, dictcls=None, maxdepth=None, sep="/", 
    digseqs=False, isleaf=None, kind='str'):
    """
    Gets a deep dictionary and returns a hierarchical
    dictionary conversion of it.

    Parameters
    ----------
    deep_dict: (dict) a deep dictionary
        Example: {'a': {'b': {'c': 1., 'd': 2.}}}


    dictcls: (class or None) the dict constructor

    maxdepth: (int) the maximum depth until which the
        conversion recursion can go. If None, no max depth
        constraint will be applied.

    sep: (str) the directory seperator character, such as
         '/' in unix addressing.

    digseqs: (bool) whether to continue digging when you hit a list and 
        treat as an indexible dictionary.
    
    isleaf: (callable) an indicator function which outputs `True` if a 
        value is a leaf and `False` otherwise. A good method for stopping 
        this function from going deeper.

    kind: (str) Whether the keys should be compiled as 'str' or 'tuple'. 

    Output
    ----------
    hie_dict: (dict) a hierarchical dictionary
        Example: {'a/b/c': 1.,
                  'a/b/d': 2.}
    """
    if dictcls is None:
        dictcls = odict if isinstance(deep_dict, odict) else dict
    if isleaf is not None:
        assert callable(isleaf)
    if digseqs:
        intfmtr = "{}"

    assert kind in ('str', 'tuple')
    
    chmaxdepth = None if maxdepth is None else (maxdepth - 1)
    cangodeeper_glbl = (chmaxdepth is None) or (chmaxdepth > 0)

    hie_dict = dictcls()
    for key, val in deep_dict.items():
        if cangodeeper_glbl and (isleaf is not None):
            cangodeeper = (not isleaf(val))
        elif cangodeeper_glbl and (isleaf is None):
            cangodeeper = True
        else:
            cangodeeper = False

        if cangodeeper and isinstance(val, dict):
            subdict = deep2hie(val, dictcls, chmaxdepth, sep, digseqs, isleaf, kind)
            for k, v in subdict.items():
                hie_dict[f"{key}{sep}{k}" if (kind == 'str') else (key, *k)] = v
        elif digseqs and cangodeeper and isinstance(val, (tuple, list)):
            valdict = dictcls([(intfmtr.format(i), v) for i, v in enumerate(val)])
            subdict = deep2hie(valdict, dictcls, 
                chmaxdepth, sep, digseqs, isleaf, kind)
            for k, v in subdict.items():
                hie_dict[f"{key}{sep}{k}" if (kind == 'str') else (key, *k)] = v
        else:
            hie_dict[key if (kind == 'str') else (key,)] = val

    return hie_dict


def match_prepostfix(key: str, prefix: str | None, postfix: str | None):
    """
    Utility function for checking whether a string key both 
        * starts with the prefix, and 
        * ends the postfix as well.
    
    If either of the `prefix` or `postfix` are None, the related 
    starting/ending condition will be automatically satisfied.

    Parameters
    ----------
    key: (str) The string to be checked.

    prefix: (str | None) The prefix string. Provide `None` for 
        disabling the prefix matching condition.
    
    postfix: (str | None) The postfix string. Provide `None` for 
        disabling the postfix matching condition.
    
    Returns
    ----------
    output: (bool) whether the key satisfies the prefix and postfix 
        conditions.
    """
    return (((prefix is None) or key.startswith(prefix)) and
            ((postfix is None) or key.endswith(postfix)))


#########################################################
######## Dictionary Key Selection and Processing ########
#########################################################

def get_subdict(cfg_dict: dict, prefix: str, postfix: str | None = None, 
    sep: str = '/', dictcls=None, pop: bool=False, ret_npop: bool=False):
    """
    Gets a dictionary and pops out the keys with a particular
    prefix from it. The popped keys will be returned in a
    different dictionary in the output.

    Parameters
    ----------
    cfg_dict: (dict) a dictionary
        Example: {'a/b/c': 1.,
                  'a/e/c': 3.,
                  'a/b/d': 2.}

    dictcls: (class or None) the dict constructor

    prefix: (str, optional) the prefix of the keys to be popped.
        Example: 'a/b/'

    pop (bool, optional): Whether to pop out the options out of the
            input `cfg_dict` dictionary. Defaults to False.

    Returns
    ----------
    pop_dict: (dict) the dictionary with the popped
        keys and the prefix string removed.
        Example: {'c': 1., 'd': 2.}
    """
    if dictcls is None:
        dictcls = odict if isinstance(cfg_dict, odict) else dict
    
    if (prefix is None) and (postfix is None):
        return dictcls()

    baseprfx = None if (prefix is None) else prefix + sep
    basepstfx = None if (postfix is None) else postfix + sep
    basekey = sep.join([fx for fx in (prefix, postfix) if fx is not None])

    n_pop = 0
    if basekey in cfg_dict:
        assert not any(match_prepostfix(key, baseprfx, basepstfx)
            for key in cfg_dict), dedent(f'''
            Both the "{basekey}" key and other keys 
            {("  * starting with " + baseprfx) if (baseprfx is not None) else ""}
            {("  * ending with " + basepstfx) if (basepstfx is not None) else ""}
            exist in the input dictionary. This is not a proper hierarchical 
            dict:
                existing key: {basekey}
                offending keys: {[key for key in cfg_dict if 
                    match_prepostfix(key, baseprfx, basepstfx)]}
            ''')

        n_pop += 1
        if pop:
            output = cfg_dict.pop(basekey) 
        else:
            output = cfg_dict[basekey]
    else:
        keys2pop = [key for key in cfg_dict if 
            match_prepostfix(key, baseprfx, basepstfx)]
        l_prfx = len(baseprfx) if baseprfx is not None else 0
        l_pstfx = len(basepstfx) if basepstfx is not None else 0
        
        output = dictcls()
        for key in keys2pop:
            newkey = key[l_prfx:] if l_pstfx == 0 else key[l_prfx:-l_pstfx]
            n_pop += 1
            if pop:
                output[newkey] = cfg_dict.pop(key) 
            else:
                output[newkey] = cfg_dict[key]

    return (output, n_pop) if ret_npop else output


def rnm_fmtdstr(pat_inp, pat_out, query):
    """
    Takes an input pattern, checks if the query matches it.
    
        1. If the query matches the input pattern, it will renames 
            it according to the output pattern and return it.

        2.  If the query does not match the input pattern, it will 
            return None. 

    The patterns must be re-friendly.
    
    Parameters
    ----------
    pat_inp: (str) The input pattern.
        Example: 
            `pat_inp = 'perf/aerosol/test/{ymtrc}/mean/{stat}'`
    
    pat_out: (str) The output pattern.
        Example: 
            `pat_out = '{ymtrc}::{stat}'`

    query: (str) The query string.
        Example: 
            `query = 'perf/aerosol/test/foo/bar/mean/zoo'`

    Output
    ------
    query_rnmd: (str | None) The output formatted query if the 
        query was a match to the input.

        Example: 
            `output = 'foo/bar::zoo'`
    """    
    pattern = re.escape(pat_inp)
    pattern = re.sub(r'\\\{(\w+)\\\}', r'(?P<\1>.*)', pattern)
    matchres = re.match(pattern, query)
    
    query_rnmd = None if (matchres is None) else pat_out.format(**matchres.groupdict())
    
    return query_rnmd


def get_keyrnmngs(inp_itrtr, rnm_patdict, strict_fwd=True, strict_inv=True):
    """
    Takes an input iterator/generator, and produces an old to new name mapping.

    Only the keys that match an input pattern will be included in the output.

    Parameters
    ----------
    inp_itrtr: (list[str] | tuple[str] | set[str] | dict[str:any]) The 
        input iterator, whose members are strings.

        Example:

            ```
            inp_itrtr = [
                'perf/aerosol/test/hello1/mean/world4',
                'perf/aerosol/test/hello2/mean/world3',
                'nn/pp/layer5',
                'nothing']
            ```
        
    rnm_patdict: (dict) A dictionary of old to new renamings conventions.

        Example:
            
            ```
            rnm_patdict = {
                'perf/aerosol/test/{ymtrc}/mean/{stat}': '{ymtrc}::{stat}',
                'nn/pp/{varbl}': '{varbl}', 
                'model': 'model'}
            ```
    
    strict_fwd: (bool) Whether to be strict about the uniqueness of the new 
        name for each old name.

    strict_inv: (bool) Whether to be strict about the uniqueness of the old 
        name for each new name.
    
    Returns
    -------
    fwd_rnmngs: (dict) a dictionary of the matching names from 
        the input patterns to their new names.

        Example:
            
            ```
            fwd_rnmngs = {
                'perf/aerosol/test/hello1/mean/world4': 'hello1::world4',
                'perf/aerosol/test/hello2/mean/world3': 'hello2::world3',
                'nn/pp/layer5': 'layer5'}
            ```

    """
    fwd_rnmngs, inv_rnmngs = dict(), dict()
    for key in inp_itrtr:
        assert isinstance(key, str), dedent(f'''
            One of the input keys was not a string:
                offending key: {key}''')
        for pat_inp, pat_out in rnm_patdict.items():
            key_rnmd = rnm_fmtdstr(pat_inp, pat_out, key)
            if (key_rnmd is None):
                pass
            elif strict_fwd and (key in fwd_rnmngs):
                assert fwd_rnmngs[key] == key_rnmd, dedent(f'''
                    Conflicting forward renaming for "{key}":
                        Name 1: "{fwd_rnmngs[key]}"
                        Name 2: "{key_rnmd}"''')
            elif strict_inv and (key_rnmd in inv_rnmngs):
                assert inv_rnmngs[key_rnmd] == key, dedent(f'''
                    Conflicting inverse renaming for "{key_rnmd}":
                        Name 1: "{inv_rnmngs[key_rnmd]}"
                        Name 2: "{key}"''')
            else:
                assert not(strict_fwd) or (key not in fwd_rnmngs)
                fwd_rnmngs[key] = key_rnmd
                assert not(strict_inv) or (key_rnmd not in inv_rnmngs)
                inv_rnmngs[key_rnmd] = key

    return fwd_rnmngs


def get_subdictrnmd(inp_dict, rnm_patdict, strict_fwd=True, strict_inv=True):
    """
    Takes an input dictionary, and a renaming pattern dictionary.
    
    Then it returns a subset dictionary with the keys renamed
    according to the pattern dictionary. Only the keys matching 
    the specified input patterns will be present in the output 
    dictionary.

    Parameters
    ----------
    inp_itrtr: (list[str] | tuple[str] | set[str] | dict[str:any]) The 
        input iterator, whose members are strings.

        Example:

            ```
            inp_dict = {
                'perf/aerosol/test/hello1/mean/world4': 'hi',
                'perf/aerosol/test/hello2/mean/world3': None,
                'nn/pp/layer5': np.randn(5, 4),
                'nothing': 'bye'}
            ```
        

    rnm_patdict: (dict) A dictionary of old to new renamings conventions.

        Example:
            
            ```
            rnm_patdict = {
                'perf/aerosol/test/{ymtrc}/mean/{stat}': '{ymtrc}::{stat}',
                'nn/pp/{varbl}': '{varbl}', 
                'model': 'model'}
            ```
    
    strict_fwd: (bool) Whether to be strict about the uniqueness of the new 
        name for each old name.

    strict_inv: (bool) Whether to be strict about the uniqueness of the old 
        name for each new name.
    
    Returns
    -------
    sub_dictrnmd: (dict) a dictionary of the matching names from 
        the input patterns to their new names.

        Example:
            
            ```
            sub_dictrnmd = {
                'hello1::world4': 'hi',
                'hello2::world3': None,
                'nn/pp/layer5': np.randn(5, 4)}
            ```

    """
    fwd_rnmngs = get_keyrnmngs(inp_dict, rnm_patdict, 
        strict_fwd=strict_fwd, strict_inv=strict_inv)
    sub_dictrnmd = {key_rnmd: inp_dict[key] 
        for key, key_rnmd in fwd_rnmngs.items()}
    return sub_dictrnmd


def test_getsubdictrnmd():
    inp_dict1 = {
        'perf/aerosol/test/hello1/mean/world4': 'hi',
        'perf/aerosol/test/hello2/mean/world3': None,
        'nn/pp/layer5': 12345,
        'nothing': 'bye'}

    rnm_patdict1 = {
        'perf/*/test/{ymtrc}/mean/{stat}': '{ymtrc}::{stat}',
        'nn/pp/{varbl}': '{varbl}', 
        'model': 'model'}

    out_dict1 = get_subdictrnmd(inp_dict1, rnm_patdict1)

    assert out_dict1 == {
        'hello1::world4': 'hi', 
        'hello2::world3': None, 
        'layer5': 12345}, out_dict1


def sort_keys(inp_itrtr, sort_ptrns):
    """
    Sorts the elements of the input iterator (either dictionary, list or tuple), 
    and returns the same type of iterator with the keys being sorted.

    Any key that matches none of the sorting patterns will be sent to the end 
    of the iterator.

    Parameters
    ----------
    inp_itrtr: (dict | list | tuple) The input iterator.
        
        Example:
            ```
            inp_itrtr = {
                'xtsne/11:train/orig:pnts': np.randn(3, 4),
                'xtsne/11:train/rcnst:pnts': np.randn(3, 4),
                'xtsne/11:test/orig:pnts': np.randn(3, 4),
                'xtsne/11:test/rcnst:pnts': np.randn(3, 4),
                'xtsne/11:normal/genr:pnts': np.randn(3, 4)}
            ```
    
    sort_ptrns: (list[str]) The sorting patterns.
        
        Example:
            ```
            sort_ptrns = ['*train*', '*normal*']
            ```

    Returns
    -------
    out_itrtr: (dict | list | tuple) The output sorted iterator.

        Example:
            ```
            out_itrtr = {
                'xtsne/11:train/orig:pnts': np.randn(3, 4),
                'xtsne/11:train/rcnst:pnts': np.randn(3, 4),
                'xtsne/11:normal/genr:pnts': np.randn(3, 4),
                'xtsne/11:test/orig:pnts': np.randn(3, 4),
                'xtsne/11:test/rcnst:pnts': np.randn(3, 4)}
            ```
    """
    assert isinstance(inp_itrtr, (list, dict, tuple))
    assert all(isinstance(pat, str) for pat in sort_ptrns)
    rank_keys = []
    for key in inp_itrtr:
        rank_key = len(sort_ptrns)
        if isinstance(key, str):
            for i_pat, pat in enumerate(sort_ptrns):
                if fnmatch.fnmatch(key, pat):
                    rank_key = i_pat
                    break
        rank_keys.append(rank_key)
    
    keys_srtd = sorted(enumerate(inp_itrtr), key=lambda ix: rank_keys[ix[0]])
    if isinstance(inp_itrtr, dict):
        out_itrtr = {key: inp_itrtr[key] for i_key, key in keys_srtd}
    elif isinstance(inp_itrtr, (list, tuple)):
        out_itrtr = type(inp_itrtr)(key for i_key, key in keys_srtd)
    else:
        raise ValueError(f'undefined case for {out_itrtr}')
    
    return out_itrtr


def _filter_incdictsrd(rowdicts, incdicts, has_wildcards=False, invert=False):
    """
    Filters the dataframe according to a list of "inclusion dictionaries" 
    or `incdict`s. 
    
    This function is implemented in an efficient manner, where 
    if a row must logically be included, it won't be subjected to further 
    unnecessary searches. This will minimize the query calls by reducing 
    the number of queried rows in consecutive calls.

    For instance, having 

        ```
            incdicts = [
                {'col1': [1, 2], 'col2': 3}, 
                {'col3': 5, 'col2': [3, 4], 'col4': ['b', 'c']},
                {'col5': 'a'}]
        ```

        * with `invert = False`, results in the following output:

            ```
                i1 = df['col1'].isin([1, 2])
                i2 = df['col2'] == 3
                i3 = df['col3'] == 5
                i4 = df['col2'].isin([3, 4])
                i5 = df['col4'].isin(['b', 'c'])
                i6 = df['col5'] == 'a'

                ii = (i1 & i2) | (i3 & i4 & i5) | (i6)
                df_fltrd = df[ii]
            ```
        
        * with `invert = True`, results in the following output:

            ```
                ii = (~i1 & ~i2) | (~i3 & ~i4 & ~i5) | (~i6)
                df_fltrd = df[ii]
            ```

    Application: This function is not an end-user function, 
        and is mainly designed to be called in the 
        `_filter_incdicts` function.

    Parameters
    ----------
    rowdicts (list[dict]): The data-frame as a list of row 
        dicts.

    incdicts: (list[dict]): A list of inclusion dictionaries.

    has_wildcards (bool): Whether the str matchings must 
        be carried out be fnmatch.
    
    invert (bool): Whether the `incdicts` should be treated as 
        "exclusion dictionaries" rather than "inclusion dictionaries".
    
    Returns
    -------
    rowdicts_fltrd (list[dict]): The filtered row dicts.
    """
    assert isinstance(rowdicts, (list, tuple)), dedent(f'''
        The rowdicts argument is not a list/tuple:
            rowdicts: {rowdicts}''')
    assert isinstance(incdicts, (list, tuple)), dedent(f'''
        The incdicts argument is not a list/tuple:
            incdicts: {incdicts}''')
    
    rowdicts_fltrd = []
    for rowdict in rowdicts:
        assert isinstance(rowdict, dict), dedent(f'''
            One of `rowdicts` is not a dict:
                offending member: {rowdict}''')
        any_incsok = False
        for incdict in incdicts:
            assert isinstance(incdict, dict), dedent(f'''
                One of `incdicts` is not a dict:
                    offending member: {incdict}''')
            
            all_incok = True
            for col, pattern in incdict.items():
                assert col in rowdict, dedent(f'''
                    The "{col}" key exists in an inc/exc dictionary, but 
                    one of the row dicts does not have this key:
                        col name: {col}
                        pattern: {pattern}
                        row missing the col: {rowdict}''')
                
                val = rowdict[col]
                if isinstance(pattern, str) and (not has_wildcards):
                    all_incok = all_incok and (val == pattern)
                elif isinstance(pattern, str) and (has_wildcards):
                    all_incok = all_incok and isinstance(val, str) and fnmatch.fnmatch(val, pattern)
                elif isinstance(pattern, (list, tuple, set)) and (not has_wildcards):
                    all_incok = all_incok and (val in pattern)
                elif isinstance(pattern, (list, tuple, set)) and (has_wildcards):
                    all_incok = all_incok and any(
                        (isinstance(val, str) and fnmatch.fnmatch(val, pat)) 
                        if isinstance(pat, str) else (val == pat) 
                        for pat in pattern)
                else:
                    assert not isinstance(pattern, str)
                    all_incok = all_incok and (val == pattern)

                if not all_incok:
                    break
            
            any_incsok = any_incsok or all_incok
            if any_incsok:
                break
        
        if ((not invert) and any_incsok) or (invert and (not any_incsok)):
            rowdicts_fltrd.append(rowdict)
    
    return rowdicts_fltrd


def filter_rowdicts(rows, incs=None, excs=None, has_wildcards=False):
    """
    This function is a similar one to `filter_df` and is supposed to provide 
    the same functionality, except that it only works with row dicts instead 
    of data-frames, and avoids the usage of the pandas library.

    In an essence, the following must always stand:
        filter_rowdicts(*args, **kwargs) == filter_df(*args, **kwargs)

    Filters the dataframe according to a list of "inclusion" and "exclusion" 
    dictionaries (i.e., `incs` and `excs`, respectively). 

    This function is implemented in an efficient manner, where 
    if a row must logically be included, it won't be subjected to further 
    unnecessary searches. This will minimize the query calls by reducing 
    the number of queried rows in consecutive calls.

    For instance, having 

        ```
            incs = [
                {'col1': [1, 2], 'col2': 3}, 
                {'col3': 5, 'col2': [3, 4], 'col4': ['b', 'c']},
                {'col5': 'a'}]

            excs = [
                {'col1': [8, 9], 'col2': 6}, 
                {'col3': 8},
                {'col5': 'a', 'col2': [1, 9], 'col4': ['d', 'e']}]
        ```

    results in the following output:

        ```
            # The inclusion indicators
            i1 = df['col1'].isin([1, 2])
            i2 = df['col2'] == 3
            i3 = df['col3'] == 5
            i4 = df['col2'].isin([3, 4])
            i5 = df['col4'].isin(['b', 'c'])
            i6 = df['col5'] == 'a'

            # The exclusion indicators
            j1 = df['col1'].isin([8, 9])
            j2 = df['col2'] == 6
            j3 = df['col3'] == 8
            j4 = df['col5'] == 'a'
            j5 = df['col2'].isin([1, 9])
            j6 = df['col4'].isin(['d', 'e'])

            ii = (i1 & i2) | (i3 & i4 & i5) | (i6)
            jj = ~((j1 & j2) | (j3) | (j4 & j5 & j6))
            df_fltrd = df[ii &  jj]
        ```

    Note that `jj` is equvialently the following as well 
    in the above example:
        ```
        jj = ~(j1 & j2) & ~(j3) & ~(j4 & j5 & j6)
        jj = (~j1 | ~j2) & (~j3) & (~j4 | ~j5 | ~j6)
        ```

    There are a few caveats to consider:

        1. Having `incs == None` will cause the function to 
           ignore it and include all the rows as far as the 
           inclusion criteria is concerned.

           Similarly, `excs == None` will cause the function 
           to avoid imposing any exclusions.

        2. `incs == None` is not the same as `incs == []`. The 
            former will include all the rows, whereas the latter 
            will include no rows since no row could be deemed fit 
            to be included.

    Parameters
    ----------
    df: (pd.DataFrame | dict | list) The data-frame to be searched. 
        If not an instance of pd.DataFrame, it must be a dict or 
        list that can be conveted into a dataframe.

    incs: (None | list[dict]): A list of inclusion dictionaries.

    has_wildcards: (bool) Whether the string matchings must 
        be carried out by fnmatch.
    
    Returns
    -------
    The filtered data-frame with the same type as the input. 
    """
    assert isinstance(rows, (list, tuple))
    assert all(isinstance(row, dict) for row in rows)

    # Initializing the output
    rows_fltrd = rows 

    # Processing the inclusions
    if incs is not None:
        rows_fltrd = _filter_incdictsrd(rows_fltrd, incs, 
            has_wildcards=has_wildcards, invert=False)
    
    # Processing the exclusions
    if excs is not None:
        rows_fltrd = _filter_incdictsrd(rows_fltrd, excs, 
            has_wildcards=has_wildcards, invert=True)

    return rows_fltrd


#########################################################
########### Sanity Checking Utility Functions ###########
#########################################################


msg_bcast = "{} should be np broadcastable to {}={}. "
msg_bcast += "However, it has an inferred shape of {}."


def get_arr(name, trgshp_str, trns_opts):
    """
    Gets a list of values, and checks if it is broadcastable to a
    target shape. If the shape does not match, it will raise a proper
    assertion error with a meaninful message. The output is a numpy
    array that is guaranteed to be broadcastable to the target shape.

    Parameters
    ----------
    name: (str) name of the option / hyper-parameter.

    trgshp_str: (str) the target shape elements representation. Must be a
        valid python expression where the needed elements .

    trns_opts: (dict) a dictionary containing the variables needed
        for the string to list translation of val.

    Key Variables
    -------------
    `val = trns_opts[name]`: (list or str) list of values read
        from the config file. If a string is provided, python's
        `eval` function will be used to translate it into a list.

    `trg_shape = eval_formula(trgshp_str, trns_opts)`: (tuple)
        the target shape.

    Output
    ----------
    val_np: (np.array) the numpy array of val.
    """
    msg_ = f'"{name}" must be in trns_opts but it isnt: {trns_opts}'
    assert name in trns_opts, msg_
    val = trns_opts[name]

    if isinstance(val, str):
        val_list = eval_formula(val, trns_opts)
    else:
        val_list = val
    val_np = np.array(val_list)
    src_shape = val_np.shape
    trg_shape = eval_formula(trgshp_str, trns_opts)
    msg_ = msg_bcast.format(name, trgshp_str, trg_shape, src_shape)

    assert len(val_np.shape) == len(trg_shape), msg_

    is_bcastble = all(
        (x == y or x == 1 or y == 1) for x, y in zip(src_shape, trg_shape)
    )
    assert is_bcastble, msg_

    return val_np


def multiline_eval(expr, context=None):
    """
    Evaluate several lines of input, returning the result of the last line.

    Credit: 
    https://stackoverflow.com/questions/12698028/why-is-pythons-eval-rejecting-this-multiline-string-and-how-can-i-fix-it
    """
    context = context if context is not None else dict()
    tree = ast.parse(expr)
    eval_exprs = []
    exec_exprs = []
    for module in tree.body:
        if isinstance(module, ast.Expr):
            eval_exprs.append(module.value)
        else:
            exec_exprs.append(module)
    exec_expr = ast.Module(exec_exprs, type_ignores=[])
    exec(compile(exec_expr, 'file', 'exec'), context)
    results = []
    for eval_expr in eval_exprs:
        results.append(eval(compile(ast.Expression((eval_expr)), 'file', 'eval'), context))
    return '\n'.join([str(r) for r in results])


def eval_formula(formula, variables, catch=False):
    """
    Gets a string formula and uses the `eval` function of python to
    translate it into a python variable. The necessary variables for
    translation are provided through the `variables` argument.

    Regarding the globals argument in eval, consider the 
    following example: 
      ```
      variables = {'y': 4}
      locals().update(variables)

      # Completely ok and returns 5!
      eval('y + 1')
      ```
    
    In this example, we can run the following with no error:
      `eval('y + 1')`
    This is because `y` is in the `locals()` and running eval 
    without a globals/locals argument means that they will be 
    inherited from the context calling eval (i.e., the 
    `eval_formula` function context).

    However, the followings evals will error out:
      ```
      variables = {'y': 4}
      locals().update(variables)

      # The next line will give a NameError that `y` is not defined.
      eval('[y for x in range(5)]')

      # The next line will give a NameError that `y` is not defined.
      eval('(lambda: y)()')`
      ```

    These two `eval` expressions define inner contexts; both 
        1. the list comprehension, and 
        2. the lambda function 
    are independent inner contexts.
    
    The error happens because `y` is a "nonlocal" variable 
    inside those inner contexts:
      
      1. `y` is not a global:
      
        * We never inserted it in `globals()`.
      
        * We do not want to contaminate our globals with these temporary variables.
      
      2. `y` is not a local inside the inner context:
      
        * `y` is a local variable inside the `eval_formula` function context. 
      
        * However, it is not a local variable inside the inner contexts.
      
      3. `y` is coming from an "enclosing scope"
      
        * That is, somewhere between locals and globals.

    Here are some references to read more about why this happens:
      
      1. https://bugs.python.org/issue36300
      
      2. https://stackoverflow.com/questions/68959680/python-how-to-let-eval-see-local-variables
      
      3. https://realpython.com/python-scope-legb-rule/
      
      4. https://docs.python.org/3/library/functions.html#eval

    To solve the issue and make those lines runnable, we have two options:

      1. We could insert a `nonlocal y` line inside those inner contexts. 
        
        * However, that is difficult and requires asking the users to do that.
      
      2. Instead, we will make all `variables` accessible by inserting 
         them inside the globals argument to eval. 
         
        * This way, the `variables` keys will be treated as if they 
           were global variables, but only for the eval execution context.
        
        * This is essentially why I am adding the `{**globals(), **variables}`
          argument to eval.
        
        * Note that because we are doing this, there is no need to update the 
          `locals()` dict with `variables`; The `variables` will be accessible 
          through the `globals` we provided to eval.

    TODO: 
        * Test `mutliline_eval` for expressions with multiple lines 
          and insert it instead of eval.
    
    Parameters
    ----------
    formula (str): a string that can be passed to `eval`.
        Example: "[np.sqrt(dim), 'a', None]"

    variables (dict): a dictionary of variables used in the formula.
        Example: {"dim": 4}

    catch (bool): whether to catch the name errors and making sure 
        that the eval input is a string.

    Output
    ------
    pyobj (object): the translated formula into a python object
        Example: [2.0, 'a', None]

    """
    
    
    # Unnecessary line since we are updating the eval globals with `variables`.
    # locals().update(variables)
    if catch:
        try:
            pyobj = eval(str(formula), {**globals(), **variables})
        except (NameError, SyntaxError) as exc:
            pyobj = formula
    else:
        pyobj = eval(formula, {**globals(), **variables})
    return pyobj


def chck_dstrargs(opt, cfgdict, dstr2args, opt2req, parnt_optdstr=None):
    """
    Checks if the distribution arguments are provided correctly. Works
    with hirarchical models through recursive applications. Proper error
    messages are displayed if one of the checks fails.

    Parameters
    ----------
    opt: (str) the option name.

    cfgdict: (dict) the config dictionary.

    dstr2args: (dict) a mapping between distribution and their
        required arguments.

    opt2req: (dict) required arguments for an option itself, not
        necessarily required by the option's distribution.
    """
    opt_dstr = cfgdict.get(f"{opt}/dstr", "fixed")

    msg_ = f"Unknown {opt}_dstr: it should be one of {list(dstr2args.keys())}"
    assert opt_dstr in dstr2args, msg_

    opt2req = dict() if opt2req is None else opt2req
    optreqs = opt2req.get(opt, tuple())
    must_spec = list(dstr2args[opt_dstr]) + list(optreqs)
    avid_spec = list(
        chain.from_iterable(v for k, v in dstr2args.items() if k != opt_dstr)
    )
    avid_spec = [k for k in avid_spec if k not in must_spec]

    if opt_dstr == "fixed":
        # To avoid infinite recursive calls, we should end this here.
        msg_ = f'"{opt}" must be specified.'
        if parnt_optdstr is not None:
            parnt_opt, parnt_dstr = parnt_optdstr
            msg_ += f'"{parnt_opt}" was specified as "{parnt_dstr}", and'
        msg_ += f' "{opt}" was specified as "{opt_dstr}".'
        if len(optreqs) > 0:
            msg_ += f'Also, "{opt}" requires "{optreqs}" to be specified.'
        opt_val = cfgdict.get(opt, None)
        assert opt_val is not None, msg_
    else:
        for arg in must_spec:
            opt_arg = f"{opt}{arg}"
            chck_dstrargs(opt_arg, cfgdict, dstr2args,
                          opt2req, (opt, opt_dstr))

    for arg in avid_spec:
        opt_arg = f"{opt}{arg}"
        opt_arg_val = cfgdict.get(opt_arg, None)
        msg_ = f'"{opt_arg}" should not be specified, since "{opt}" '
        msg_ += f'appears to follow the "{opt_dstr}" distribution.'
        assert opt_arg_val is None, msg_


# -

#####################################################
########## Dataframe Manipulation Utilties ##########
#####################################################

def drop_unqcols(df, exclist=None):
    """
    Returns a copy of the dataframe where all columns
    contaning a single unique value are dropped.

    Parameters
    -----------
    df: (pd.DataFrame) the dataframe with redundant
        columns.
    exclist: (List or None) the columns excluded from  
        being dropped even if they had unique values.
    """
    exclist = exclist if exclist is not None else []

    dropd_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1 and (col not in exclist):
            dropd_cols.append(col)
    df = df.drop(columns=dropd_cols)
    return df


def get_dfidxs(df, cdict):
    """
    Generates a boolean index array for the rows of the `df`
    dataframe. Each row will be ture iff it has identical values
    to the ones in `cdict`.

    Parameters
    ----------
    df: (pd.DataFrame) the main dataframe.
        Example:
            d = {'col1': [ 0,   1,   2,   3 ],
                 'col2': ['a', 'b', 'a', 'd']}
            df = pd.DataFrame(d)

    cdict: (dict) a dictionary of conditioning values
        for the indexing.
        Example: cdict = {'col2': 'a'}

    Output:
    -------
    idx: (np.array) a boolean numpy array indicating whether
        each row of `df` has the same values as `cdict`.

        Example: idx = np.array([0, 2])
    """
    idx = np.full((df.shape[0],), True)
    for col, val in cdict.items():
        idx = np.logical_and(idx, (df[col] == val).values)
    return idx


def _check_colpat(column, pattern, has_wildcards=False):
    """
    Returns a boolean index of rows that satisfy a given 
    `pattern` for a single column.

    * If the given `pattern` is a single item (i.e., an 
        int or float or a non-wildcard string) this 
        function acts just like an equality check:

        `is_okay = (colum == pattern)`
    
    * When `pattern` is a list of single items, this 
        function acts just like an membership check:

        `is_okay = (colum.isin(pattern))`

    * When `pattern` is a wildcard, it will be translated 
        to a list of matching values.

    Application: This function is not an end-user function, 
        and is mainly designed to be called in the 
        `_filter_incdicts` function.

    Parameters
    ----------
    column (pd.Series): The column to be searched.

    pattern (str, int, float, tuple, list): The compliant 
        pattern or patterns to be searched.

    has_wildcards (bool): Whether the str matchings must 
        be carried out be fnmatch.
    
    Returns
    -------
    is_okay (pd.Series) A boolean index of all rows, and whether 
        they satisfy the input pattern/patterns.
    """
    if isinstance(pattern, str) and (not has_wildcards):
        is_okay = (column == pattern)
    elif isinstance(pattern, (int, float)):
        is_okay = (column == pattern)
    elif isinstance(pattern, str) and (has_wildcards):
        unq_colvals = column.unique()
        good_vals = set(val for val in unq_colvals 
            if isinstance(val, str) and fnmatch.fnmatch(val, pattern))
        is_okay = column.isin(good_vals)
    elif isinstance(pattern, (list, tuple)) and (not has_wildcards):
        is_okay = (column.isin(set(pattern)))
    elif isinstance(pattern, (list, tuple)) and (has_wildcards):
        unq_colvals = None 
        good_vals = set()
        for pat in pattern:
            if not isinstance(pat, str):
                good_vals.add(pat)
            else:
                if unq_colvals is None:
                    unq_colvals = column.unique()
                for val in unq_colvals:
                    if isinstance(val, str) and fnmatch.fnmatch(val, pat):
                        good_vals.add(val)
        is_okay = column.isin(good_vals)
    else:
        raise ValueError(f'undefined case: {pattern}')
    
    return is_okay


def _filter_incdicts(df, incdicts, has_wildcards=False, invert=False):
    """
    Filters the dataframe according to a list of "inclusion dictionaries" 
    or `incdict`s. 
    
    This function is implemented in an efficient manner, where 
    if a row must logically be included, it won't be subjected to further 
    unnecessary searches. This will minimize the query calls by reducing 
    the number of queried rows in consecutive calls.

    For instance, having 

        ```
            incdicts = [
                {'col1': [1, 2], 'col2': 3}, 
                {'col3': 5, 'col2': [3, 4], 'col4': ['b', 'c']},
                {'col5': 'a'}]
        ```

        * with `invert = False`, results in the following output:

            ```
                i1 = df['col1'].isin([1, 2])
                i2 = df['col2'] == 3
                i3 = df['col3'] == 5
                i4 = df['col2'].isin([3, 4])
                i5 = df['col4'].isin(['b', 'c'])
                i6 = df['col5'] == 'a'

                ii = (i1 & i2) | (i3 & i4 & i5) | (i6)
                df_fltrd = df[ii]
            ```
        
        * with `invert = True`, results in the following output:

            ```
                ii = (~i1 & ~i2) | (~i3 & ~i4 & ~i5) | (~i6)
                df_fltrd = df[ii]
            ```

    Application: This function is not an end-user function, 
        and is mainly designed to be called in the 
        `_filter_incdicts` function.

    Parameters
    ----------
    df: (pd.DataFrame) The data-frame to be searched.

    incdicts: (list[dict]): A list of inclusion dictionaries.

    has_wildcards: (bool) Whether the string matchings must 
        be carried out by fnmatch.

    invert: (bool) Whether the `incdicts` should be treated as 
        "exclusion dictionaries" rather than "inclusion dictionaries".
    
    Returns
    -------
    (pd.DataFrame, pd.Series) A pair:

        1. The first item is the filtered dictionary.

        2. The second item is the boolean indices for indexing the 
            original `df` input.
    """
    
    assert isinstance(incdicts, (list, tuple))
    assert all(isinstance(incdict, dict) for incdict in incdicts)
    assert isinstance(invert, bool)
    # Note: The shallow copy is only necessary to avoid modifying 
    # the original `df` when adding a new `__isok__` column.
    df_shlwcp = df.copy(deep=False)
    df_shlwcp['__isok__'] = invert

    # Question: What is `df_srch`?
    #   `df_srch` is a data-frame which gets shrunken after an entire `incdict` is 
    #   processed. The complying rows according to that `incdict` should then be 
    #   removed from `df_srch`, since they're already verified and checking them 
    #   against other incdicts is a waste of time.

    renew_dfsrch = False
    if not renew_dfsrch:
        # Shrinking the `dfsrch` after processing each `incdict`:
        #   In this method, the search database `df_srch` will start with `df_shlwcp`, 
        #   and with each `incdict`, we will remove the rows that already have their 
        #   outcome determined.
        df_srch = df_shlwcp.copy(deep=False)

    for incdict in incdicts:
        if renew_dfsrch:
            # Renewing the searching data-base:
            #   Having `renew_dfsrch == True` makes the implementation easy. 
            #   However, this may not be efficient since we have to go back 
            #   and rule out all the rows in the original `df_shlwcp` database.
            isok = df_shlwcp['__isok__']
            df_srch = df_shlwcp[isok if invert else ~isok]
        
        # This is the main part
        df_passed = df_srch
        for col, pattern in incdict.items():
            assert col in df_passed.columns, dedent(f'''
                Column {column} does not exist in the data-frame.
                    Available columns: {df.columns.tolist()}''')
            indctr_idx = _check_colpat(df_passed[col], pattern, has_wildcards)
            df_passed = df_passed[indctr_idx]
        df_shlwcp.loc[df_passed.index, '__isok__'] = not(invert)

        if not renew_dfsrch:
            # At this point, we will narrow down the search database `df_srch`, 
            # by only keeping the rows that are still undetermined and need 
            # further search.
            # Note that the rows in the `df_passed` dataframe should be disposed 
            # from the search, because they are satisfying one of the inc dicts 
            # and are already guaranteed to be in the `df_fltrd` output. Therefore, 
            # they no longer need to be included in the search database.
            # The following lines seem the most reasonable approach to remove 
            # the `df_passed` rows from the `df_srch` data-frame.
            df_srch['__isok2__'] = True
            df_srch.loc[df_passed.index, '__isok2__'] = False
            df_srch = df_srch[df_srch['__isok2__']]

    boolidxs = df_shlwcp['__isok__']
    df_fltrd = df[boolidxs]

    return df_fltrd, boolidxs


def filter_df(df, incs=None, excs=None, has_wildcards=False, ret_boolidxs=False):
    """
    Filters the dataframe according to a list of "inclusion" and "exclusion" 
    dictionaries (i.e., `incs` and `excs`, respectively). 

    This function is implemented in an efficient manner, where 
    if a row must logically be included, it won't be subjected to further 
    unnecessary searches. This will minimize the query calls by reducing 
    the number of queried rows in consecutive calls.

    For instance, having 

        ```
            incs = [
                {'col1': [1, 2], 'col2': 3}, 
                {'col3': 5, 'col2': [3, 4], 'col4': ['b', 'c']},
                {'col5': 'a'}]

            excs = [
                {'col1': [8, 9], 'col2': 6}, 
                {'col3': 8},
                {'col5': 'a', 'col2': [1, 9], 'col4': ['d', 'e']}]
        ```

    results in the following output:

        ```
            # The inclusion indicators
            i1 = df['col1'].isin([1, 2])
            i2 = df['col2'] == 3
            i3 = df['col3'] == 5
            i4 = df['col2'].isin([3, 4])
            i5 = df['col4'].isin(['b', 'c'])
            i6 = df['col5'] == 'a'

            # The exclusion indicators
            j1 = df['col1'].isin([8, 9])
            j2 = df['col2'] == 6
            j3 = df['col3'] == 8
            j4 = df['col5'] == 'a'
            j5 = df['col2'].isin([1, 9])
            j6 = df['col4'].isin(['d', 'e'])

            ii = (i1 & i2) | (i3 & i4 & i5) | (i6)
            jj = ~((j1 & j2) | (j3) | (j4 & j5 & j6))
            df_fltrd = df[ii &  jj]
        ```

    Note that `jj` is equvialently the following as well 
    in the above example:
        ```
        jj = ~(j1 & j2) & ~(j3) & ~(j4 & j5 & j6)
        jj = (~j1 | ~j2) & (~j3) & (~j4 | ~j5 | ~j6)
        ```

    There are a few caveats to consider:

        1. Having `incs == None` will cause the function to 
           ignore it and include all the rows as far as the 
           inclusion criteria is concerned.

           Similarly, `excs == None` will cause the function 
           to avoid imposing any exclusions.

        2. `incs == None` is not the same as `incs == []`. The 
            former will include all the rows, whereas the latter 
            will include no rows since no row could be deemed fit 
            to be included.

    Parameters
    ----------
    df: (pd.DataFrame | dict | list) The data-frame to be searched. 
        If not an instance of pd.DataFrame, it must be a dict or 
        list that can be conveted into a dataframe.

    incs: (None | list[dict]): A list of inclusion dictionaries.

    has_wildcards: (bool) Whether the string matchings must 
        be carried out by fnmatch.

    ret_boolidxs: (bool) Whether to return a boolean set 
        of indices for the inclusion of the selected rows 
        in the input `df`.
    
    Returns
    -------
    The filtered data-frame with the same type as the input. 
    If `ret_boolidxs` was true, a numpy array of booleans with the 
    same length as the input data-frame will also be returned.
    """
    if isinstance(df, (dict, list)):
        pddf = pd.DataFrame(df)
    else:
        pddf = df
    
    assert isinstance(pddf, pd.DataFrame), dedent(f'''
        The input must be a data-frame like object:
            df type: {type(df)}''')

    # Initializing the output
    df_fltrd = pddf 

    # Processing the inclusions
    if incs is not None:
        df_fltrd, boolidxsinc = _filter_incdicts(df_fltrd, incs, 
            has_wildcards=has_wildcards, invert=False)
    else:
        boolidxsinc = None
    
    # Processing the exclusions
    if excs is not None:
        df_fltrd, boolidxsexc = _filter_incdicts(df_fltrd, excs, 
            has_wildcards=has_wildcards, invert=True)
    else:
        boolidxsexc = None
    
    if ret_boolidxs:
        n_rows = pddf.shape[0]
        if (boolidxsinc is None) and (boolidxsexc is None):
            boolidxs = np.full((n_rows,), True, dtype=bool)
        elif (boolidxsinc is None) and (boolidxsexc is not None):
            boolidxs = boolidxsexc.values
        elif (boolidxsinc is not None) and (boolidxsexc is None):
            boolidxs = boolidxsinc.values
        elif (boolidxsinc is not None) and (boolidxsexc is not None):
            boolidxs = boolidxsinc.values
            boolidxs[boolidxs] = boolidxsexc.values
        else:
            raise ValueError('undefined case')

        assert isinstance(boolidxs, np.ndarray)
        assert boolidxs.shape == (n_rows,)
    else:
        boolidxs = None
    
    if isinstance(df, dict):
        df_fltrdout = df_fltrd.to_dict(orient='list')
    elif isinstance(df, list):
        df_fltrdout = df_fltrd.to_dict(orient='records')
    else:
        df_fltrdout = df_fltrd

    return (df_fltrdout, boolidxs) if ret_boolidxs else df_fltrdout


def get_ovatgrps(hpdf):
    """
    Takes a hyper-parameters data-frame and divides the
    `fpidx` values into groups. This function assumes an
    ovat-style expriment (i.e., one variable at a time).
    The central hyper-parameters must be located at the
    top of the data-frame.

    Parameters
    ----------
    hpdf (pd.DataFrame) a data-frame only containing the
        hyper-parameters of the experiments. Having duplicate
        rows is fine, but do not
    """
    uhpdf = hpdf.drop_duplicates().copy()
    uhpdf = uhpdf.drop("fpidx", axis=1)
    uhpdf = drop_unqcols(uhpdf, exclist="fpidxgrp")
    uhpdf = uhpdf.reset_index(drop=True)

    mainrow = uhpdf.iloc[0]
    diffgrps = defaultdict(list)
    for i, row in uhpdf.iterrows():
        aa = row.eq(mainrow)
        diffs = {
            bb: row[bb]
            for bb, cc in dict(aa).items()
            if (not (cc) and (row[bb] != nullstr))
        }
        fpidx = row["fpidxgrp"]
        if len(diffs) == 0:
            continue
        diffs["fpidxgrp"] = fpidx
        diffgrps[tuple(sorted(tuple(diffs.keys())))].append(diffs)

    fpgrps = []
    for dgi, (dgkeys, dglist) in enumerate(diffgrps.items()):
        # dglist is a list of diff dictionaries
        dglist.append({key: mainrow[key] for key in dgkeys})
        dglist = sorted(dglist, key=lambda diffs: diffs[dgkeys[0]])
        dgdf = pd.DataFrame(dglist).drop_duplicates()
        if dgdf.shape[1] > 1:
            sortcols = list(dgdf.columns)
            sortcols.remove("fpidxgrp")
            dgdf = dgdf[["fpidxgrp"] + sortcols]
            
            # making sure the `sort_values` call does not fail due to 
            # multiple dtypes existing in the same column. This is a 
            # bandaid for now.
            for col in sortcols:
                coldtypes = {type(x) for x in dgdf[col]}
                if coldtypes == {str, float}:
                    dgdf[col] = dgdf[col].astype(str)
                    
            dgdf = dgdf.sort_values(by=sortcols)
        fpgrps.append(dgdf["fpidxgrp"].tolist())

    # Example:
    #   fpgrps = [['01_poisson/02_mse2d.0.0',
    #              '01_poisson/02_mse2d.3.0',
    #              '01_poisson/02_mse2d.5.0',
    #              '01_poisson/02_mse2d.0.1'],
    #             ['01_poisson/02_mse2d.2.2',
    #              '01_poisson/02_mse2d.2.1',
    #              '01_poisson/02_mse2d.0.0'],
    #             ['01_poisson/02_mse2d.0.3',
    #              '01_poisson/02_mse2d.0.0']]
    return fpgrps


def pd_concat(dfs, *args, **kwargs):
    """
    Concatenate while preserving categorical columns.

    Parameters
    ----------
    dfs: (list of pd.DataFrame) a list of data-frames
        with categorical columns to be concatenated.

    *args: arguments to be piped to `pd.concat`.

    **kwargs: keyword arguments to be piped to `pd.concat`.

    Output
    ------
    catdf: (pd.DataFrame) a concatenated pandas dataframe.
    """
    # Iterate on categorical columns common to all dfs
    dfs = [df.copy(deep=False) for df in dfs]
    for col in set.intersection(
        *[set(df.select_dtypes(include="category").columns) for df in dfs]
    ):
        # Generate the union category across dfs for this column
        uc = union_categoricals([df[col] for df in dfs])
        # Change to union category for all dataframes
        for df in dfs:
            df[col] = pd.Categorical(df[col].values, categories=uc.categories)
    catdf = pd.concat(dfs, *args, **kwargs)
    return catdf


def to_pklhdf(df, path, key, mode="a", driver=None, compression="gzip", compression_opts=0):
    """
    Writes the dataframe into a byte-stream on the RAM, and
    then transfers it into a numpy array inside the hdf file.

    This function was written due to exhausting number of errors
    associated with saving categorical pandas dataframes into hdf
    files directly.

    Parameters
    ----------
    df: (pd.DataFrame) a pandas data-frame either with categorical
        or non-categorical data.

    path: (str) the path of the hdf file on the disk

    key: (str) the hdf key to write the df into

    mode: (str) piped to `h5py.File`

    driver: (str) piped to `h5py.File`

    compression: (str) piped to `hdf.create_dataset`

    compression_opts: (int) piped to `hdf.create_dataset`
    """
    stream = io.BytesIO()
    df.to_pickle(stream)
    np_arr = np.frombuffer(stream.getbuffer(), dtype=np.uint8)
    driver = driver if driver is not None else h5drvr_dflt
    h5hdf = h5py.File(path, mode=mode, driver=driver) if isinstance(path, str) else path
        
    try:
        h5hdf.create_dataset(
            key,
            shape=np_arr.shape,
            dtype=np_arr.dtype,
            data=np_arr,
            compression=compression,
            compression_opts=compression_opts,
        )
    except Exception as exc:
        if isinstance(path, str):
            h5hdf.close()
        raise exc


def read_pklhdf(path, key, driver=None):
    """
    Reads the pickled numpy array from the hdf file into RAM,
    transforms it into a bytesarray, and then reads it with
    pd.read_pkl.

    This function was written due to exhausting number of errors
    associated with saving categorical pandas dataframes into hdf
    files directly.

    Parameters
    ----------
    path: (str or h5py.File or h5py.Group or h5py.Dataset) the path
        of the hdf file on the disk. It can be an already opened file
        pointer using the h5py library as well.

    key: (str) the hdf key to write the df into

    driver: (None | str) piped to `h5py.File`

    Output
    ------
    df: (pd.DataFrame) a pandas data-frame either with categorical
        or non-categorical data.
    """
    open_fp = isinstance(path, str)
    if open_fp:
        driver = driver if driver is not None else h5drvr_dflt
        h5hdf = h5py.File(path, mode="r", driver=driver)
    else:
        h5hdf = path
    rstream = io.BytesIO(h5hdf[key][:].tobytes())
    if open_fp:
        h5hdf.close()
    df = pd.read_pickle(rstream)
    return df


def to_ymlhdf(data, path, key, mode="a", driver=None, compression="gzip", compression_opts=0, overwrite=False):
    """
    Writes the dictionary into a string, and then transfers it 
    into a str array inside the hdf file.

    Parameters
    ----------
    data: (dict | list) a dictionary or list.

    path: (str) the path of the hdf file on the disk

    key: (str) the hdf key to write the df into

    mode: (str) piped to `h5py.File`

    driver: (str) piped to `h5py.File`

    compression: (str) piped to `hdf.create_dataset`

    compression_opts: (int) piped to `hdf.create_dataset`
    """

    data_str = ruyaml_safedump(data)
    assert isinstance(data_str, str)
    driver = driver if driver is not None else h5drvr_dflt
    h5hdf = h5py.File(path, mode=mode, driver=driver) if isinstance(path, str) else path
    try:
        if (key in h5hdf) and overwrite:
            h5hdf[key][0] = data_str
        else:
            h5hdf.create_dataset(
                key,
                shape=1,
                dtype=h5py.string_dtype(),
                data=[data_str],
                compression=compression,
                compression_opts=compression_opts,
            )
    except Exception as exc:
        if isinstance(path, str):
            h5hdf.close()
        raise exc


def read_ymlhdf(path, key, driver=None, catch=False):
    """
    Reads the yaml string numpy array from the hdf file,
    and then reads it with `ruyaml.safe_load`.

    This function was written due to exhausting number of errors
    associated with saving categorical pandas dataframes into hdf
    files directly.

    Parameters
    ----------
    path: (str or h5py.File or h5py.Group or h5py.Dataset) the path
        of the hdf file on the disk. It can be an already opened file
        pointer using the h5py library as well.

    key: (str) the hdf key to write the df into

    driver: (str) piped to `h5py.File`
    
    catch: (bool) whether to return `None` if the key does not 
        exist rather than raising an exception.

    Output
    ------
    data: (dict | list | None) a dictionary or a list
    """
    open_fp = isinstance(path, str)
    if open_fp:
        driver = driver if driver is not None else h5drvr_dflt
        h5hdf = h5py.File(path, mode="r", driver=driver)
    else:
        h5hdf = path

    if (key not in h5hdf) and catch:
        data = None
    else:
        assert key in h5hdf, dedent(f'''
            "{key}" does not exist in "{path}"''')
        h5ds = h5hdf[key]
        assert isinstance(h5ds, h5py.Dataset), dedent(f'''
            "{key}" is not a leaf dataset in "{path}"''')
        assert len(h5ds) == 1
        ymlstr = h5ds[0].decode("utf-8")
        assert isinstance(ymlstr, str)
        data = ruyaml.safe_load(ymlstr)
    
    if open_fp:
        h5hdf.close()
    
    return data


def augmnt_yamlhdf(data, path, key, do_raise=True, **kwargs):
    """
    This function:
        
        1. Reads the `key` path inside the disk hdf.
        
            * This should be a dictionary.
        
        2. If do_raise, asserts that all items in the disk 
            data are identical to those in the input. Otherwise, 
            overwrites the value in the disk with the input value.
        
        3. Augments the disk data with any missing keys and values.
        
        4. Writes this augmented data back into disk.
    
    Parameters
    ----------
    data: (dict) the input data dictionary.

    path: (str | h5py.File) the hdf path/file-pointer on the disk.

    key: (str) the hdf key to read/write the data from/into.

    do_raise: (bool) whether to assert if all the input and disk 
        data items must have identical values.

    kwargs: (dict) the extra arguments passed to `to_ymlhdf`.
    """
    assert isinstance(data, dict)
    data_inpt = dict(data)

    data_disk = read_ymlhdf(path, key, catch=True)
    data_disk = dict() if (data_disk is None) else data_disk
    for kkk, disk_val in data_disk.items():
        new_val = data_inpt.get(kkk, disk_val)
        assert not (do_raise) or (new_val == disk_val), dedent(f'''
            The disk data and the new data do not match:
                disk data['{kkk}'] = {disk_val}
                new data['{kkk}'] = {new_val}''')
        data_inpt[kkk] = new_val

    assert 'overwrite' not in kwargs
    to_ymlhdf(data_inpt, path, key, overwrite=True, **kwargs)


def read_pdymlhdf(*args, **kwargs):
    """
    Reads the yaml string numpy array from the hdf file,
    and then reads it with `ruyaml.safe_load`.

    This function was written due to exhausting number of errors
    associated with saving categorical pandas dataframes into hdf
    files directly.

    Parameters
    ----------
    path: (str or h5py.File or h5py.Group or h5py.Dataset) the path
        of the hdf file on the disk. It can be an already opened file
        pointer using the h5py library as well.

    key: (str) the hdf key to write the df into

    driver: (str) piped to `h5py.File`

    Output
    ------
    df: (pd.DataFrame) a pandas data-frame either with categorical
        or non-categorical data.
    """
    data = read_ymlhdf(*args, **kwargs)
    df = pd.DataFrame(data)
    return df


def decomp_df(df, columns, validate=True):
    """
    This function takes a dataframe and a grouping of columns, and checks 
    if the data-frame can be cartesian-decomposed between those columns.
    If this is possible, it will return the decomposition, otherwise it 
    will raise an error.

    Parameters
    ----------
    df: (pd.DataFrame) The input dataframe to be factorized.

    columns: (list(list(str))) The decomposition of columns into groups.

        Example:

            columns = [
                ('fpidx',), 
                ('eid', 'vid', 'v_space', 'v_node', 'v_split', 'v_varname', 'v_repr')
            ]
    
    validate: (bool) Whether to check if the cartesian product of the 
        output gives the input data-frame. The ordering of rows will not 
        be checked/validated.

    Output
    ------
    fac_dfs: (list(pd.DataFrame)) The list of factorized data-frames.
    """
    dfcolslst = list(chain.from_iterable(columns))
    assert set(dfcolslst) == set(df.columns), dedent(f'''
        The decomposition of df seems strange and you 
        may have under/over-specified some columns:
            missing columns: {set(df.columns) - set(dfcolslst)}
            over-spec columns: {set(dfcolslst) - set(df.columns)}''')

    fac_dfs = [df[list(cols_seq)].drop_duplicates().reset_index(drop=True) for cols_seq in columns]

    # Next, we make sure that if we decompose the data-frame into those columns, and 
    # then take the cartesian product, we would be reconstructing the same data-frame.
    if validate:
        dfsrtd = df[dfcolslst].sort_values(dfcolslst).reset_index(drop=True)

        dfcross1 = fac_dfs[0]
        for facdf in fac_dfs[1:]:
            dfcross1 = pd.merge(dfcross1, facdf, how='cross')
        
        dfcross2 = dfcross1.sort_values(dfcolslst).reset_index(drop=True)

        assert (dfcross2.index == dfsrtd.index).all(), dedent(f'''
            The indices are not the same after factorization:
                missing: {set(dfcross2.index) - set(dfsrtd.index)}
                extra:   {set(dfsrtd.index) - set(dfcross2.index)}''')
        
        assert (set(dfcross2.columns) == set(dfsrtd.columns)), dedent(f'''
            The columns are not the same after factorization:
                missing: {set(dfcross2.columns) - set(dfsrtd.columns)}
                extra:   {set(dfsrtd.columns) - set(dfcross2.columns)}''')

        assert (dfcross2 == dfsrtd).all().all(), dedent(f'''
            The provided data-frame is not a complete cartesian product 
            of the requested columns. Some fpidx values may have missing 
            data or the stored data may not be homogeneuous:
                Column decomposition: {columns}''')
    
    return fac_dfs


def downcast_df(df):
    """
    This function tries to safely downcast the numerical columns.

    Args:

        df: (pd.DataFrame) the input dataframe

    Returns:

        df_dncastd: (pd.DataFrame) the input dataframe with safely 
        down-casted numerical columns.
    """
    castings = dict()

    ##################### Float Down-castings ######################
    fltdtypes = ['float16', 'float32', 'float64', 'float128']
    df_float = df.select_dtypes(include=fltdtypes)
    for col, dtype in df_float.dtypes.items():
        assert dtype in fltdtypes

        col_vals1 = df[col].values
        col_vals2 = col_vals1[np.isfinite(col_vals1) & 
            ~np.isnan(col_vals1) & (col_vals1 != 0.0)]
        if len(col_vals2) == 0:
            continue
        col_vals3 = np.abs(col_vals2)
        min_val, max_val = col_vals3.min(), col_vals3.max()

        # Finding the most down-casted safe data-type
        safe_dtype = None
        for dtype_ in fltdtypes:
            if ((min_val >= np.finfo(dtype_).tiny) and 
                (max_val <= np.finfo(dtype_).max)):
                
                safe_dtype = dtype_
                break
        
        assert safe_dtype is not None

        i1, i2 = fltdtypes.index(dtype), fltdtypes.index(safe_dtype)

        # Somehow pandas can represent `5e-324` in `float64`!
        if i2 > i1: 
            continue
        
        # assert i1 >= i2, dedent(f'''
        #     Somehow the suggested down-casting for "{dtype}" 
        #     turned out to be "{safe_dtype}":
        #         column = {col}
        #         min value = {min_val}
        #         max value = {max_val} ''')

        if dtype != safe_dtype:
            castings[col] = safe_dtype

    #################### Integer Down-castings #####################
    intdtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 
        'int64', 'uint64']
    df_intgr = df.select_dtypes(include=intdtypes)
    for col, dtype in df_intgr.dtypes.items():
        col_vals1 = df[col].values
        col_vals2 = col_vals1[np.isfinite(col_vals1) & ~np.isnan(col_vals1)]
        if len(col_vals2) == 0:
            continue
        min_val, max_val = col_vals2.min(), col_vals2.max()

        # Finding the most down-casted safe data-type
        safe_dtype = None
        for dtype_ in intdtypes:
            if ((min_val >= np.iinfo(dtype_).min) and 
                (max_val <= np.iinfo(dtype_).max)):
                
                safe_dtype = dtype_
                break

        assert safe_dtype is not None
        i1, i2 = intdtypes.index(dtype), intdtypes.index(safe_dtype)
        assert i1 >= i2, dedent(f'''
            Somehow the suggested down-casting for "{dtype}" 
            turned out to be "{safe_dtype}":
                column = {col}
                min value = {min_val}
                max value = {max_val} ''')
        
        if dtype != safe_dtype:
            castings[col] = safe_dtype

    df_dncastd = df.astype(castings)
    return df_dncastd


class CartDF:
    def __init__(self, data):
        self.data = dict()
        self.col2comp = dict()
        for compid, compdata in data.items():
            self.extend(compid, compdata)
    
    @property
    def columns(self):
        return list(self.col2comp.keys())

    def get_comp(self, column):
        assert column in self.col2comp, dedent(f'''
            Column {column} does not exist in this cartesian dataframe:
                Existing columns: {list(self.col2comp.keys())}
                Existing components: {list(self.data.keys())}''')
        return self.col2comp[column]
    
    def extend(self, compid, compdata):
        assert compid not in self.data
        if isinstance(compdata, dict):
            compdf = pd.DataFrame(compdata)
        elif isinstance(compdata, pd.DataFrame):
            compdf = compdata
        else:
            raise ValueError(f'undefined type for compid={compid}: {compdata}')
        
        for col in compdf.columns.tolist():
            assert col not in self.col2comp
            self.col2comp[col] = compid
        
        self.data[compid] = compdf

    def copy(self, deep=False):
        data_cp = {key: df.copy(deep=deep) for key, df in self.data.items()}
        return CartDF(data=data_cp)
    
    def select(self, incdict, excdict, copy=False, reset_index=False, has_wildcards=False):
        """
        Selects the data from a cartesian data-frame and returns a new cartesian df.

        This function does not support getting multiple `incdict`s like the general 
        `filter_df` function. This is because multiple `incdicts` can induce multiple 
        cartesian dfs and this function returns a single cartesian df.

        Parameters
        ----------
        incdict: (dict) A single inclusion dictionary. 

        excdict: (dict) A single exclusion dictionary.

        copy: (bool | str) Whether to make a deep or shallow copy. 

        reset_index: (bool) Whether to reset and drop the index in each component's 
            data-frame.

        Output
        ------
        (CartDF) The selected cartesian data-frame.
        """
        outdata = dict()
        
        for compid, compdf in self.data.items():
            # Narrowing down the inclusion to columns in this component
            if incdict is None:
                incdicts_grp = None
            else:
                incdicts_grp = {col: vals for col, vals in incdict.items() if col in compdf.columns}
                incdicts_grp = [incdicts_grp] if len(incdicts_grp) > 0 else None
            
            # Narrowing down the exclusions to columns in this component
            if excdict is None:
                excdicts_grp = None
            else:
                excdicts_grp = {col: vals for col, vals in excdict.items() if col in compdf.columns}
                excdicts_grp = [excdicts_grp] if len(excdicts_grp) > 0 else None

            if (incdicts_grp is None) and (excdicts_grp is None):
                # There were no inc/exclusion criteria for this component. Therefore, the 
                # component remains untouched.
                compdf_slctd = compdf
            else:
                compdf_slctd = filter_df(compdf, incs=incdicts_grp, excs=excdicts_grp, 
                    has_wildcards=has_wildcards, ret_boolidxs=False)

            if reset_index: 
                compdf_slctd = compdf_slctd.reset_index(drop=True)
            
            if copy in (True, 'deep'):
                compdf_slctd = compdf_slctd.copy(deep=True)
            else:
                assert copy in (False, 'shallow')

            outdata[compid] = compdf_slctd
        
        cartdf_out = CartDF(outdata)
        
        return cartdf_out

    def groupby(self, by, sort=False, observed=True, **kwargs):
        """
        Applies the groupby operation to this cartesian data-frame.

        Parameters
        ----------
            by: (str | list | tuple) The grouping column names.
        """
        # Finding the component sepecific grouping columns
        do_peel = isinstance(by, str)
        columns = [by] if isinstance(by, str) else by
        assert isinstance(columns, (list, tuple)), dedent(f'''
            The groupby argument must either be a str, list of str, or tuple of str.
            Instead I got the following:
                by={by}''')
        if len(columns) == 0:
            yield (tuple(), self)
            return 
        

        comp2gbcols = defaultdict(list)
        for col in columns:
            assert col in self.col2comp, dedent(f'''
                The specified grouping column {col} does not exist in this 
                cartesian dataframe:
                    Offending column: {col}
                    Existing columns: {self.columns}''')
            comp_col = self.col2comp[col]
            comp2gbcols[comp_col].append(col)

        compids = list(self.data.keys())
        cols_unsrtd = []
        gb_factors = []
        for compid in compids:
            compdf = self.data[compid]
            gbcols = comp2gbcols[compid]
            if len(gbcols) == 0:
                gb_factor = [(tuple(), compdf)]
            else:
                gb_factor = list(compdf.groupby(gbcols, sort=sort, observed=observed, **kwargs))
            gb_factors.append(gb_factor)
            cols_unsrtd += gbcols

        assert sorted(cols_unsrtd) == sorted(columns)
        col2idx = {col: i_col for i_col, col in enumerate(cols_unsrtd)}
        cols_srtidx = tuple(col2idx[col] for col in columns)

        for gb_comps in product(*gb_factors):
            gb_valsunsrtd = []
            gb_cartdata = dict()
            for compid, (gb_compvals, gb_compdf) in zip(compids, gb_comps):
                gb_valsunsrtd += list(gb_compvals)
                gb_cartdata[compid] = gb_compdf
            
            grp_vals1 = tuple(gb_valsunsrtd[i_col] for i_col in cols_srtidx)
            grp_cdf = CartDF(gb_cartdata)
            assert not(do_peel) or (len(grp_vals1) == 1)
            grp_vals2 = grp_vals1[0] if do_peel else grp_vals1
            yield (grp_vals2, grp_cdf)
    
    def dense(self):
        crossdf = None
        for compid, compdf in self.data.items():
            if crossdf is None:
                crossdf = compdf
            else:
                crossdf = pd.merge(crossdf, compdf, how='cross')
        return crossdf

    def __repr__(self):
        reprdict = {compid: {'rows': compdf.shape[0], 'columns': compdf.columns.tolist()}
            for compid, compdf in self.data.items()}
        reprstr = yaml.safe_dump(reprdict, default_flow_style=None, sort_keys=False, indent=4)
        return reprstr

    def test_groupby(self, by):
        cdfgb_test = self.groupby(by)
        grp_dfs = [grp_cdf.dense() for grp_vals, grp_cdf in cdfgb_test]

        df1 = pd.concat(grp_dfs, axis=0, ignore_index=True)
        df2 = self.dense()

        assert df1.columns.tolist() == df2.columns.tolist(), dedent(f'''
            The columns or their ordering is not preserved after a groupby call:
                Original Dense Columns: {df1.columns.tolist()}
                Grouped  Dense Columns: {df2.columns.tolist()}''')
        
        assert df1.shape[0] == df2.shape[0], dedent(f'''
            The number of rows is incorrect:
                Expected number of rows: {df1.shape[0]}
                Actual number of rows: {df2.shape[0]}''')

        srtcols = df1.columns.tolist() 
        df3 = df1.sort_values(srtcols).reset_index(drop=True)
        df4 = df2.sort_values(srtcols).reset_index(drop=True)

        if not (df3 == df4).all().all():
            diffdf = df3.merge(df4.drop_duplicates(), on=srtcols, 
                how='left', indicator=True)
            diffdf = diffdf[~(diffdf['_merge'] == 'both')]
            raise ValueError(dedent(f'''
                The pandas groupby is different from our cartesian groupby:
                    Delta DF = {diffdf}'''))

    def unqs(self, column):
        compid = self.get_comp(column)
        vals_allnp = self.data[compid][column].values
        vals_allunq = np.unique(vals_allnp) if isinstance(vals_allnp, np.ndarray) else vals_allnp.unique()
        unqvals = vals_allunq.tolist()
        return unqvals
    
    @property
    def shape(self):
        nrows = math.prod([compdf.shape[0] for compid, compdf in self.data.items()])
        ncols = sum([compdf.shape[1] for compid, compdf in self.data.items()])
        return (nrows, ncols) 

#####################################################
########### HDF File Manipulation Utilties ##########
#####################################################


def get_ioidxkeys(h5fp):
    """
    Gets all keys that in the h5 file/group that end with the
    '/ioidx' string.

    Parameters
    ----------
    h5fp: (h5py.File or h5py.Group) a file pointer opened with
        the h5py library, or an h5py group handle.

    Outputs
    -------
    all_ioidxs: (list of str) a list of keys, where `f{key}/ioidx`
        exists in the `h5fp` group.
    """
    all_ioidxs = []

    def list_ioidxs(name, obj):
        if name.endswith("/ioidx"):
            all_ioidxs.append(name[:-6])

    h5fp.visititems(list_ioidxs)
    return all_ioidxs


def get_ioidx(h5fp, key):
    """
    Gets the `ioidx` array for a particular key in the hdf file.
    It traverses all its parents, until it finds the first grandparent
    that has an `ioidx` child.

    Parameters
    ----------
    h5fp: (h5py.File or h5py.Group) a file pointer opened with
        the h5py library, or an h5py group handle.

    key: (str) a key for which the ioidx must be found.
        Example: key = 'var/eval/ug/sol/mdl'

    Outputs
    -------
    ioidx: (np.array) a numpy array which is the `ioidx` of key.
    """
    hirar = key.split("/")
    n_hirar = len(hirar)
    possioilocs = [
        "/".join(hirar[: n_hirar - i] + ["ioidx"]) for i in range(n_hirar)
    ]
    ioidxpath = max(possioilocs[::-1], key=lambda loc: loc in h5fp)
    ioidx = h5fp[ioidxpath][:]
    return ioidx


def get_h5leafs(h5fp):
    """
    Gets all leafs in the h5 file/group. Leafs are array datasets
    that do not have any further sub-directories.

    Parameters
    ----------
    h5fp: (h5py.File or h5py.Group) a file pointer opened with
        the h5py library, or an h5py group handle.

    Outputs
    -------
    leafs: (list of str) a list of dataset keys.
    """
    if isinstance(h5fp, h5py.Dataset):
        leafs = [""]
    else:
        leafs = []

        def addl(name, obj):
            if isinstance(obj, h5py.Dataset):
                leafs.append(name)

        h5fp.visititems(addl)
    return leafs


def get_h5dtypes(h5grp):
    """
    Queies all the keys inside the hdf file and returns a
    dictionary specifying them and their respective dtypes.

    The output will be a dictionary mapping each key to its
    dtype.  The dtype values are one of 'np.arr', 'pd.cat',
    or 'pd.qnt'.

    The output will only include keys corresponding to h5py
    datasets or pandas dataframes. In other words, h5py groups
    (i.e., directories) will not be included in the output.

    Parameters
    ----------
    h5grp: (h5py.File or h5py.Group) a file pointer opened with
        the h5py library, or an h5py group handle.

    Outputs
    -------
    key2dtype: (dict) a key-to-dtype mapping of the
        existing keys to their data types.
        Example:

        key2dtype = {
            'etc': 'pd.cat',
            'hp': 'pd.cat',
            'stat': 'pd.qnt',
            'mon': 'pd.qnt',
            'var/eval/ug/sol/mdl': 'np.arr',
            'var/eval/ug/sol/gt': 'np.arr',
            'mdl/layer_first.0': 'np.arr',
            ...
        }

    """
    aaa = get_h5leafs(h5grp)
    bbb = [x for x in aaa if not x.endswith("/ioidx")]
    pdpklkeys = set(x.split("/pkl")[0] for x in bbb if "/pkl" in x)
    pdymlkeys = set(x.split("/yml")[0] for x in bbb if "/yml" in x)
    rymlkeys = set(x.split("/ryml")[0] for x in bbb if "/ryml" in x)
    pdcatkeys = set(x.split("/meta")[0] for x in bbb if "/meta" in x)
    pdqntkeys = set(
        x.split("/axis")[0]
        for x in bbb
        if ("/axis" in x) and (x not in pdcatkeys)
    )
    dfkeys = pdcatkeys.union(pdqntkeys).union(pdpklkeys).union(pdymlkeys).union(rymlkeys)
    npkeys = [x for x in bbb if not any(x.startswith(y) for y in dfkeys)]
    key2dtype = {
        **{x: "pd.cat" for x in pdcatkeys},
        **{x: "pd.pkl" for x in pdpklkeys},
        **{x: "pd.yml" for x in pdymlkeys},
        **{x: "pd.qnt" for x in pdqntkeys},
        **{x: "ryml" for x in rymlkeys},
        **{x: "np.arr" for x in npkeys},
    }
    return key2dtype


def get_h5keys(h5grp):
    return list(get_h5dtypes(h5grp).keys())


def save_h5datav2(data, path, driver=None):
    """
    Saves a dictionary with pands/array values into an hdf file.

    Note: This is a much faster version of the `save_h5data` function, 
    since it only opens the file for writing once. However, due to lack 
    of testing and to ensure backward compatibility, it hasn't replaced 
    the old version yet.

    Parameters
    ----------
    data: (dict) a dictionary mapping hdf keys to their values.
        The values could either be pandas dataframes or numpy
        arrays.

    path: (str) the path of the hdf file.
    """

    dtypes = dict()
    needs_pdtohdf = False
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            dtype = "np.arr"
        elif isinstance(val, pd.DataFrame):
            is_cat = any(
                val[col].dtype.name == "category" for col in val.columns
            )
            dtype = "pd.cat" if is_cat else "pd.qnt"
        elif isinstance(val, (dict, list, tuple, str, int, float)):
            dtype = "ryml"
        else:
            dtype = None
            raise ValueError(f"{type(val)} is not acceptable")
        
        dtypes[key] = dtype

    needs_pdtohdf = any(dtype == 'pd.qnt' for key, dtype in dtypes.items())
    if needs_pdtohdf:
        # Writing the pandas dataframes
        pdhdf = pd.HDFStore(path, mode="a")
        for key, val in data.items():
            dtype = dtypes[key]
            if dtype in ("pd.qnt",):
                val.to_hdf(
                    pdhdf,
                    key=key,
                    format=None,
                    index=False,
                    append=False,
                    complib="zlib",
                    complevel=0)
        pdhdf.close()

    needs_createds = any(dtype in ('pd.cat', 'ryml', 'np.arr') for key, dtype in dtypes.items())
    if needs_createds:
        # Writing the numpy array datasets
        driver = driver if driver is not None else h5drvr_dflt
        with h5py.File(path, mode="a", driver=driver) as h5hdf:
            for key, val in data.items():
                dtype = dtypes[key]
                if dtype == "pd.cat":
                    to_pklhdf(val, h5hdf, f'{key}/pkl', mode="a")
                elif dtype == "ryml":
                    to_ymlhdf(val, h5hdf, f'{key}/ryml', mode="a")
                elif dtype == "pd.qnt":
                    pass
                elif dtype == "np.arr":
                    h5hdf.create_dataset(
                        key,
                        shape=val.shape,
                        dtype=val.dtype,
                        data=val,
                        compression="gzip",
                        compression_opts=0,
                    )
                else:
                    raise ValueError(f"{dtype} not defined")


def save_h5data(data, path, driver=None):
    """
    Saves a dictionary with pands/array values into an hdf file.

    Parameters
    ----------
    data: (dict) a dictionary mapping hdf keys to their values.
        The values could either be pandas dataframes or numpy
        arrays.

    path: (str) the path of the hdf file.
    """

    for key, val in data.items():
        if isinstance(val, np.ndarray):
            dtype = "np.arr"
        elif isinstance(val, pd.DataFrame):
            is_cat = any(
                val[col].dtype.name == "category" for col in val.columns
            )
            dtype = "pd.cat" if is_cat else "pd.qnt"
        elif isinstance(val, (dict, list, tuple, str, int, float)):
            dtype = "ryml"
        else:
            dtype = None
            raise ValueError(f"{type(val)} is not acceptable")

        if dtype == "pd.cat":
            to_pklhdf(val, path, f'{key}/pkl', mode="a")
        elif dtype == "ryml":
            to_ymlhdf(val, path, f'{key}/ryml', mode="a")
        elif dtype == "pd.qnt":
            val.to_hdf(
                path,
                key=key,
                mode="a",
                format=None,
                index=False,
                append=False,
                complib="zlib",
                complevel=0,
            )
        elif dtype == "np.arr":
            driver = driver if driver is not None else h5drvr_dflt
            with h5py.File(path, mode="a", driver=driver) as h5hdf:
                h5hdf.create_dataset(
                    key,
                    shape=val.shape,
                    dtype=val.dtype,
                    data=val,
                    compression="gzip",
                    compression_opts=0,
                )
        else:
            raise ValueError(f"{dtype} not defined")


def load_h5data(path, driver=None, bcast_ioidx=False):
    """
    Loads the hdf data into a dictionary

    Parameters
    ----------
    path: (str) the path of the hdf file to be read

    Output
    ------
    data: (dict) a dictionary mapping hdf keys to their values.
        The values could either be pandas dataframes or numpy
        arrays.

    bcast_ioidx: (bool) whether to broadcast the data to have 
        the same length as its own `iodix` array.
    """
    driver = driver if driver is not None else h5drvr_dflt
    h5fp = h5py.File(path, mode="r", driver=driver)
    grpdtypes = get_h5dtypes(h5fp)
    data = dict()
    for key, dtype in grpdtypes.items():
        if dtype in ("pd.cat", "pd.qnt"):
            data_key = pd.read_hdf(path, key=key)
        elif dtype in ("pd.pkl",):
            data_key = read_pklhdf(h5fp, f'{key}/pkl')
        elif dtype in ("pd.yml",):
            data_key = read_pdymlhdf(h5fp, f'{key}/yml')
        elif dtype in ("ryml",):
            data_key = read_ymlhdf(h5fp, f'{key}/ryml')
        elif dtype in ("np.arr",):
            data_key = h5fp[key][:]
        else:
            raise ValueError(f"dtype={dtype} not defined")

        if bcast_ioidx:
            keyioidx = f'{key}/ioidx'
            assert keyioidx in h5fp, dedent(f'''
                "{keyioidx}" does not exist in "{path}"''')
            data[key] = bcast_data2ioidx(data=data_key, 
                ioidx=h5fp[keyioidx], key=key, dtype=dtype, 
                fpidx=path)
        else:
            data[key] = data_key

    h5fp.close()
    return data


def bcast_data2ioidx(data, ioidx, key=None, dtype=None, fpidx=None):
    """
    Broadcasts the data to have the same number of rows as `ioidx`.
    The other arguments (i.e., `key`, `dtype`, and `fpdix`) are 
    only there for making the assertion message informative.

    Parameters
    ----------
        data: (pd.DataFrame | np.ndarray) the input data

        ioidx: (np.ndarray) The io index for the data.
    
    Output
    ----------
        data_bcast: (pd.DataFrame | np.ndarray) the broadcasted data to 
            have the same number of rows as the input `ioidx`.
    """
    n_ioidx = len(ioidx)
    n_datarows = data.shape[0]
    assert n_datarows > 0, dedent(f'''
        The data for the "{key}" key with the "{dtype}" data type 
        in the "{fpidx}" fpidx has zero rows.''')
    assert n_ioidx % n_datarows == 0, dedent(
        resio.msg_bcast.format(key=key, dtype=dtype, 
            fpidx=fpidx, n_ioidx=n_ioidx, nrows=n_datarows, 
            datashape=data.shape))
    rep_idxs = (np.arange(n_ioidx) % n_datarows).astype(int)
    if isinstance(data, pd.DataFrame):
        data_bcast = data.iloc[rep_idxs, :].reset_index(drop=True)
    elif isinstance(data, np.ndarray):
        data_bcast = data[rep_idxs, ...]
    else:
        raise ValueError(f'undefined type(data)="{type(data)}"')

    return data_bcast


def print_duinfo(duinfo):
    tot_nbyes = 0
    for key, nbytes in duinfo.items():
        a = key + " " * (77 - len(key))
        b = f"{nbytes/1e6:.3f}"
        b = " " * (10 - len(b)) + b
        print(f"{a}{b} MB")
        tot_nbyes += nbytes
    print("-" * 90)
    c = f"{tot_nbyes/1e9:.3f}"
    c = " " * (10 - len(c)) + c
    print("Total" + " " * 72 + f"{c} GB", flush=True)


def get_h5du(h5fp, verbose=False, detailed=True, driver=None):
    """
    Computes the disk usage of all the child datasets
    under a h5 group.

    Parameters
    ----------
    h5fp: (str or h5py.File or h5py.Group) a file pointer opened
        with the h5py library, or an h5py group handle. If a string
        is provided, the function will treat it as a path and open
        it in read mode.

    verbose: (bool) whether to print the disk usage information in
        a human readable table.

    detailed: (bool) whether to open up the panads/pytables inner keys
        or hide them in a single dataframe sum.

    Outputs
    -------
    key2nbytes: (dict) a dictionary mapping each child dataset to
        its disk usage in bytes.
    """

    needs_open = isinstance(h5fp, str)
    if needs_open:
        if verbose:
            print(f'Disk usage for "{h5fp}"')
        driver = driver if driver is not None else h5drvr_dflt
        h5grp = h5py.File(h5fp, mode="r", driver=driver)
    else:
        h5grp = h5fp

    leafs = get_h5leafs(h5grp)
    h5dss = [h5grp if k == "" else h5grp[k] for k in leafs]
    h5dss = [(k, h5grp if k == "" else h5grp[k]) for k in leafs]
    key2nbytes = {k: ds.attrs.get("nelements", ds.nbytes) for k, ds in h5dss}

    if not detailed:
        key2nbytes = {
            key: sum(v for k, v in key2nbytes.items() if k.startswith(key))
            for key in get_h5keys(h5grp)
        }

    if verbose:
        print_duinfo(key2nbytes)

    if needs_open:
        h5grp.close()

    return key2nbytes


def get_fpidxlst(fpidx: any):
    """
    This is a simple helper function that converts different forms of 
    the fpidx argument into a list of strings.

    Parameters
    ----------
    fpidx: (any) a list, tuple, string, pandas data-frame or series.

    Returns
    -------
    fpidx_list: (list[str]) a list of str fpidx values.
    """
    if isinstance(fpidx, (list, tuple)):
        fpidx_list = fpidx
    elif isinstance(fpidx, pd.DataFrame):
        fpidx_list = fpidx["fpidx"].tolist()
    elif isinstance(fpidx, pd.Series):
        fpidx_list = fpidx.tolist()
    else:
        assert isinstance(fpidx, str)
        fpidx_list = [fpidx]
    
    return fpidx_list


def rslv_fpidx(fpidx, h5_opener=None, resdir=None):
    """
    Resolves an `fpidx` string into a list of resolved
    `fpidx` strings.

    A resolved `fpidx` must be formatted as
        f'{cfg_tree}.{fidx}.{pidx}',
    where
        1. `cfg_tree` is the config tree,
        Example:  cfg_tree = '01_poisson/02_mse2d'

        2. `fidx` is the file index, and should be
        an integer. For example, `fidx=5` points to
        the '01_poisson/02_mse2d_05.h5' hdf file.

        3. `pidx` is an int refering to the partition
        number inside the hdf file.
        For example, 'pidx=3' refers to the
        `/P00000003/` group inside the hdf file.

    Unresolved `fpidx` strings are ones which either
        1. do not specify the file and partition index, or
        2. specify one or both of them with a '*'.

    If `fpidx` is already resolved, it will be included in
    the output as-is.

    Parameters
    ----------
    fpidx: (str) an unresolved `fpidx` string
        Example:
        fpidx = '01_poisson/02_mse2d'

        fpidx = '01_poisson/02_mse2d.*.*'

        fpidx = '01_poisson/02_mse2d.0.*'

        fpidx = '01_poisson/02_mse2d.*.0'

    h5_opener: (callable) a function that takes the `cfg_tree`
        and `fidx` arguments and returns an h5py.File object.
        The `fidx` argument is an integer, and `cfg_tree` is
        a string.

    resdir: (None | str) The results directory.

    fpidx_list: (list of str) a list of resolved `fpidx`s.
        Example:
        fpidx = ['01_poisson/02_mse2d.0.0',
                    '01_poisson/02_mse2d.1.0',
                    ...,
                    '01_poisson/02_mse2d.5.8']

        fpidx = ['01_poisson/02_mse2d.0.0',
                    '01_poisson/02_mse2d.1.0',
                    ...,
                    '01_poisson/02_mse2d.5.8']

        fpidx = ['01_poisson/02_mse2d.0.0',
                    '01_poisson/02_mse2d.0.1',
                    ...,
                    '01_poisson/02_mse2d.0.8']

        fpidx = ['01_poisson/02_mse2d.0.0',
                    '01_poisson/02_mse2d.1.0',
                    ...,
                    '01_poisson/02_mse2d.5.0']

    """
    fpidx_list = get_fpidxlst(fpidx)

    all_fpidxs = []
    for fpidx_ in fpidx_list:
        if fpidx_.count(".") == 0:
            cfg_tree, fidx_, pidx_ = fpidx_, "*", "*"
        elif fpidx_.count(".") == 2:
            cfg_tree, fidx_, pidx_ = fpidx_.split(".")
        else:
            raise ValueError(f"fpidx={fpidx_} is not understood")

        if fidx_ == "*":
            resdir2 = results_dir if resdir is None else resdir
            allh5paths = glob.glob(f"{resdir2}/{cfg_tree}_*.h5")
            fidxs = [int(x.split("_")[-1].split(".")[0]) for x in allh5paths]
            fidxs, allh5paths = zip(*sorted(zip(fidxs, allh5paths)))
        else:
            assert fidx_.isdigit()
            fidxs = [int(fidx_)]

        if h5_opener is None:
            all_fpidxs += [f"{cfg_tree}.{fi}.{pidx_}" for fi in fidxs]
        elif pidx_.isdigit():
            all_fpidxs += [f"{cfg_tree}.{fi}.{int(pidx_)}" for fi in fidxs]
        elif pidx_ == "*":
            for fi in fidxs:
                h5fp = h5_opener(cfg_tree, fi)
                pidxs = [
                    int(k.split("P")[1].split("/")[0]) for k in h5fp.keys()
                ]
                all_fpidxs += [f"{cfg_tree}.{fi}.{pi}" for pi in pidxs]
        else:
            raise ValueError(f"Not sure what to do with pidx={pidx_}")

    return all_fpidxs


def get_mtimes(fpidxs, resdir=None):
    fpi2mtime = dict()
    resdir = resdir if (resdir is not None) else results_dir
    for fpidx in rslv_fpidx(fpidxs):
        cfgtree, fidx, _ = fpidx.split('.')
        fidx = int(fidx)
        hdfpath = f'{resdir}/{cfgtree}_{fidx:02d}.h5'
        fpi2mtime[fpidx] = getmtime(hdfpath)
    return fpi2mtime

#####################################################
######### The Result Directory Accessories  #########
#####################################################


class resio:
    fp_cache = odict()
    n_fp = 50

    msg_bcast = '''
        The "{key}" key with the "{dtype}" type within 
        the "{fpidx}" fpidx has an ioidx length of {n_ioidx}. 
        However, the loaded data for this key has "{nrows}"
        rows. I can only broadcast when the ioidx length 
        is divisible by the number of rows in the data. 
        This data seems troubled when it was written to 
        the disk:
            fpidx = {fpidx}
            key = {key}
            dtype = {dtype}
            data shape = {datashape}
            n_ioidx = {n_ioidx}
    '''
    """
    A helper class for retrieving the results directly
    from the hdf files in the results directory.

    This class does not provide any summarization.

    The methods of this class provide a lot of flexibility, but
    they may be expensive to evaluate. This class is written
    for painless occasional use for either plotting heatmaps,
    inspecting hdf files, etc.

    Global Attributes
    -----------------
    fp_cache: (OrderedDict) A cache of opened hdf file
        pointers in read mode along with their relavent data
        (e.g., the ioidx and epoch and rng_seed of the stat
        dataframes).

    n_fp: (int) the maximum number of opened hdf file pointer
        to hold in the cache at one time
    """

    def __init__(self, fpidx=None, full=True, resdir=None, driver=None):
        """
        Class constructor

        Parameters
        ----------
        fpidx: (str, list of str) unresolved `fpidx` values. This will
            be used in the public methods (i.e., `__call__`, `dtypes`,
            `du`, `keys`, `values`, and `items`).

        full: (bool) determines whether the fpidx values should be
            included in the key strings. This will be the treated as
            the default `full` argument for the `keys`, `values`, and
            `items` methods.
        """
        self.fpidx = fpidx
        self.full = full
        self.resdir = resdir if resdir is not None else results_dir
        self.driver = driver if driver is not None else h5drvr_dflt
        self.dtdictcache = dict()
        self.rslvd_cache = dict()

    def get_fpidx_key(self, fpidx, key, dtype, rng_seed=None, epoch=None, ret_info=False):
        """
        Gets the key value inside a **single** `fpidx`.

        The only difference between this method and the `__call__` method
        is that it only accepts a single `fpidx` value as a string, whereas
        `__call__` is a wrapper for getting multiple `fpidx` values and
        can accept multiple `fpidx` values provided in a dataframe.

        If the input `epoch` values did not exist in the file, the rows
        with the same rng_seed and closest epoch values will be provided
        instead.

        If the input `rng_seed` values did not exist in the file, a
        python `ValueError` will be raised.

        If `rng_seed` and `epoch` are not None, they must have the same
        length, as they will be zipped together.

        Parameters
        ----------
        fpidx: (str) a single resolved fpidx value
            Example: fpidx = '01_poisson/02_mse2d.0.0'

        key: (str) the hdf key to pull from the h5 file.
            Example: key = '/var/eval/ug/sol/mdl'

        dtype: (str) the data type of the key. It should be one of
            the 'np.arr', 'pd.cat', or 'pd.qnt' values.

        rng_seed: (None or list or np.ndarray) the set of 'rng_seed' values
            to include in the output. If None is provided, all 'rng_seed'
            values will be included in the output.

        epoch: (None, or list or np.ndarray) the set of 'epoch' values to
            include in the output. If None is provided, all 'epoch' values
            will be included in the output.

        ret_info: (bool) whether to return the selection information. If True,
            a data-frame with the 'fpidx', 'epoch', 'rng_seed', and 'ioidx'
            columns will be attached to the output.
        """
        msg_bcast = self.msg_bcast

        val, h5fp, h5grp, tdf = self.load_key_sngl(key=key, dtype=dtype, 
            fpidx=fpidx, load_pdata=True, ret_flpntrs=True)
        if dtype in ("np.arr",):
            h5ds = val
        elif dtype in ("pd.yml",):
            resdict, resdf = val, pd.DataFrame(val)
        elif dtype in ("pd.cat", "pd.qnt"):
            resdf = val
        else:
            raise ValueError(f"dtype={dtype} not defined")

        dsioidx = get_ioidx(h5grp, key)
        n_ioidx = dsioidx.size
        tdf = tdf[tdf["ioidx"].isin(dsioidx)]
        if epoch is not None:
            epoch = np.array(epoch).reshape(-1)
            if rng_seed is None:
                unqrs = sorted(tdf["rng_seed"].unique().tolist())
                rng_seed = np.array([rs for ep in epoch for rs in unqrs])
                epoch = np.array([ep for ep in epoch for rs in unqrs])
            else:
                rng_seed = np.array(rng_seed).reshape(-1)
            assert rng_seed.shape == epoch.shape
            utdf = pd.DataFrame(dict(rng_seed=rng_seed, epoch=epoch))
            cat_list = []
            info_list = []
            sorter = np.argsort(dsioidx)
            for rseed, rsutdf in utdf.groupby("rng_seed"):
                adf = tdf[tdf["rng_seed"] == rseed]
                msg_ = f'rng_seed={rseed} does not exist in "{fpidx}"'
                assert adf.shape[0] > 0, msg_
                aa = rsutdf["epoch"].values.reshape(1, -1)
                bb = adf["epoch"].values.reshape(-1, 1)
                ii = np.abs(aa - bb).argmin(axis=0)
                rsinfo = adf.iloc[ii]
                ll = rsinfo["ioidx"].values
                mm = sorter[np.searchsorted(dsioidx, ll, sorter=sorter)]

                if dtype in ("np.arr",):
                    jj, jjxx = np.unique(mm, return_inverse=True)
                    jjas = jj.argsort()
                    jjbs = jjas.argsort()

                    nrows_h5ds = h5ds.shape[0]
                    if nrows_h5ds == n_ioidx:
                        vals_idxd = h5ds[jj[jjas]][jjbs][jjxx]
                    else:
                        assert n_ioidx % nrows_h5ds == 0, dedent(
                            msg_bcast.format(key=key, dtype=dtype, 
                            fpidx=fpidx, n_ioidx=n_ioidx, nrows=nrows_h5ds, 
                            datashape=h5ds.shape))
                        jj1 = jj[jjas] % nrows_h5ds
                        jj2 = jjbs % nrows_h5ds
                        jj3 = jjxx % nrows_h5ds
                        vals_idxd = h5ds[jj1][jj2][jj3]
                elif dtype in ("pd.cat", "pd.qnt", "pd.yml"):
                    nrows_resdf = resdf.shape[0]
                    if nrows_resdf == n_ioidx:
                        vals_idxd = resdf.iloc[mm, :]
                    else:
                        assert n_ioidx % nrows_resdf == 0, dedent(
                            msg_bcast.format(key=key, dtype=dtype, 
                            fpidx=fpidx, n_ioidx=n_ioidx, nrows=nrows_resdf, 
                            datashape=resdf.shape))
                        vals_idxd = resdf.iloc[mm % nrows_resdf, :]
                else:
                    raise ValueError(f"dtype={dtype} not defined")
                cat_list.append([rsutdf.index.values, vals_idxd])
                info_list.append(rsinfo)
            iii = np.concatenate([x[0] for x in cat_list], axis=0)
            iii_as = iii.argsort()

            info = pd.concat(info_list, axis=0, ignore_index=True)
            info = info.iloc[iii_as].reset_index(drop=True)

            if dtype in ("np.arr",):
                vvv = np.concatenate([x[1] for x in cat_list], axis=0)
                outvals = vvv[iii_as]
            elif dtype in ("pd.cat", "pd.qnt", "pd.yml"):
                vvv = pd.concat(
                    [x[1] for x in cat_list], axis=0, ignore_index=True
                )
                outvals = vvv.iloc[iii_as, :]
            else:
                raise ValueError(f"dtype={dtype} not defined")
        else:
            assert rng_seed is None
            info = tdf.reset_index(drop=True)
            if dtype in ("np.arr",):
                nrows_h5ds = h5ds.shape[0]
                if nrows_h5ds == n_ioidx:
                    outvals = h5ds[:]
                else:
                    assert n_ioidx % nrows_h5ds == 0, dedent(
                        msg_bcast.format(key=key, dtype=dtype, 
                        fpidx=fpidx, n_ioidx=n_ioidx, nrows=nrows_h5ds, 
                        datashape=h5ds.shape))
                    outvals = h5ds[np.arange(n_ioidx) % nrows_h5ds]
            elif dtype in ("pd.cat", "pd.qnt", "pd.yml"):
                nrows_resdf = resdf.shape[0]
                if nrows_resdf == n_ioidx:
                    outvals = resdf
                else:
                    assert n_ioidx % nrows_resdf == 0, dedent(
                        msg_bcast.format(key=key, dtype=dtype, 
                        fpidx=fpidx, n_ioidx=n_ioidx, nrows=nrows_resdf, 
                        datashape=resdf.shape))
                    outvals = resdf.iloc[np.arange(n_ioidx) % nrows_resdf, :]
                    outvals = outvals.reset_index(drop=True)
            else:
                raise ValueError(f"dtype={dtype} not defined")

        self.close(self.n_fp)

        out = outvals
        if ret_info:
            out = (out, info)
        return out

    def load_flpntrs(self, fpidx, load_pdata=True):
        """
        Creates the hdf file pointer for `fpidx` in read mode, and
        caches it into the `self.fp_cache` variable.

        If `load_pdata` was set as True, the entire 'epoch', 'rng_seed',
        and the 'ioidx' for the 'stat' dataframe will also be loaded
        into the objects's memory.

        Parameters
        ----------
        fpidx: (str) a single fidx-resolved fpidx to load into memory.
            Example: fpidx = '01_poisson/02_mse2d.0.0'

        load_pdata: (bool) whether to load the 'epoch', 'rng_seed', and
            'ioidx' values into memory from the `stat` dataframe of
            the hdf file and partition.
        """
        cfg_tree, fidx, pidx = fpidx.split(".")
        fidx = int(fidx)
        hdfpath = f"{self.resdir}/{cfg_tree}_{fidx:02d}.h5"
        if hdfpath not in self.fp_cache:
            h5fp = h5py.File(hdfpath, mode="r", driver=self.driver)
            pdata = dict()
            self.fp_cache[hdfpath] = (h5fp, pdata)
        else:
            h5fp, pdata = self.fp_cache[hdfpath]

        if pidx == "*":
            h5grp, tdf = None, None
        else:
            pidx = int(pidx)
            if (pidx not in pdata) and load_pdata:
                tdf = pd.read_hdf(hdfpath, key=f"P{pidx:08d}/stat")
                tdf = tdf[["epoch", "rng_seed"]].copy(deep=True)
                stioidx = h5fp[f"P{pidx:08d}/stat/ioidx"][:]
                tdf["fpidx"] = fpidx
                tdf["ioidx"] = stioidx
                pdata[pidx] = tdf
            else:
                tdf = pdata.get(pidx, None)
            h5grp = h5fp[f"P{pidx:08d}"]
        return h5fp, h5grp, tdf

    def close(self, n_fp=0):
        """
        A utility function for closing extra cached hdf file pointers.

        To avoid os complaints, we keep at most `n_fp` hdf files
        open at once.

        All of the open files were opened in the read mode.
        """
        while len(self.fp_cache) > n_fp:
            for ohdfpath, (oh5fp, opdata) in self.fp_cache.items():
                break
            oh5fp, opdata = self.fp_cache.pop(ohdfpath)
            oh5fp.close()
            del opdata

    def rslv_fpidx(self, fpidx):
        """
        a wrapper around the `io_utils.rslv_fpidx` function
        for convinience.
        """
        fpidx_tup = tuple(get_fpidxlst(fpidx))

        # Looking up our cache
        if fpidx_tup in self.rslvd_cache:
            return self.rslvd_cache[fpidx_tup]

        def h5_opener(cfg_tree, fidx):
            return self.load_flpntrs(f"{cfg_tree}.{fidx}.*", load_pdata=False)[0]

        fpidx_rslvd = rslv_fpidx(fpidx, h5_opener, resdir=self.resdir)

        # Updating the cache
        self.rslvd_cache[fpidx_tup] = fpidx_rslvd
        
        return fpidx_rslvd

    def __call__(self, key, dtype=None, fpidx=None, rng_seed=None, epoch=None, ret_info=False):
        """
        A generalization wrapper around 'self.get_fpidx_key'; this method
        can get a dataframe for the `fpidx` input argument, and takes care
         of the grouping, key collection, and output compilation of each
         `fpidx` group.

        See the `get_fpidx_key` for more information.

        The `epoch` and `rng_seed` values can be specified in two ways:
          1. they can be included in the `fpidx` data-frame, or
          2. they can be specified with the `rng_seed` and `epoch` arguments.

        The `epoch` and `rng_seed` values should only be specified in only
        one of the above ways, otherwise the function will complain with a
        `ValueError`.

        If the `epoch` and `rng_seed` values were not specified in any way,
        all 'epoch' and 'rng_seed' values will be returned.

        The `fpidx` argument can be specified in one of three ways:
          1. a single string, or
          2. a list of strings, or
          3. a data-frame with an `fpidx` column, and possibly the
            `epoch` and `rng_seed` values.

        If `fpidx` is provided as a string or a list of strings, they can be
        unresolved (i.e., be incomplete or contain "*" in them).

        Parameters
        ----------
        fpidx: (str or pd.DataFrame) either a single `fpidx` value as string,
            or a df specifying the `fpidx` and possibly the `epoch` and
            `rng_seed` values. The

        key: (str) the hdf key to pull from the h5 file.
            Example: key = '/var/eval/ug/sol/mdl'

        dtype: (str) the data type of the key. It should be one of
            the 'np.arr', 'pd.cat', or 'pd.qnt' values.

        rng_seed: (None or list or np.ndarray) the set of 'rng_seed' values
            to include in the output. If None is provided, all 'rng_seed'
            values will be included in the output.

        epoch: (None, or list or np.ndarray) the set of 'epoch' values to
            include in the output. If None is provided, all 'epoch' values
            will be included in the output.

        ret_info: (bool) whether to return the selection information. If True,
            a data-frame with the 'fpidx', 'epoch', 'rng_seed', and 'ioidx'
            columns will be attached to the output.

        """
        fpidx = fpidx if fpidx is not None else self.fpidx

        if dtype is None:
            dtype = self.dtrmn_dtype(key=key, fpidx=fpidx)

        if isinstance(fpidx, (str, list)):
            all_fpidxs = self.rslv_fpidx(fpidx)

            all_outs = []
            all_infos = []
            for fpidx_ in all_fpidxs:
                fpout, info = self.get_fpidx_key(
                    fpidx_, key, dtype, rng_seed, epoch, ret_info=True
                )
                all_outs.append(fpout)
                all_infos.append(info)

            info = pd.concat(all_infos, axis=0, ignore_index=True)
            if dtype == "np.arr":
                all_nparr = all(isinstance(x, np.ndarray) for x in all_outs)
                if all_nparr:
                    shapes_set = set(x.shape[1:] for x in all_outs)
                    all_sameshape = len(shapes_set) == 1
                else:
                    all_sameshape = False

                if all_nparr and all_sameshape:
                    sdata = np.concatenate(all_outs, axis=0)
                else:
                    sdata = list(chain.from_iterable(all_outs))
            elif dtype in ("pd.cat", "pd.qnt", "pd.yml"):
                sdata = pd.concat(all_outs, axis=0, ignore_index=True)
            else:
                raise ValueError(f"dtype={dtype} not defined")
        else:
            assert isinstance(fpidx, pd.DataFrame)
            df = fpidx
            assert "fpidx" in df.columns
            df = df.reset_index(drop=True).copy(deep=True)
            if rng_seed is not None:
                msg_ = "cannot specify rng_seed when it is in df"
                assert "rng_seed" not in df.columns, msg_
                df["rng_seed"] = rng_seed
            if epoch is not None:
                msg_ = "cannot specify epoch when it is in df"
                assert "epoch" not in df.columns, msg_
                df["epoch"] = epoch
            idx_list = []
            data_list = []
            info_list = []
            for fpidx, fpidf in df.groupby("fpidx"):
                rng_seed_fpi, epoch_fpi = rng_seed, epoch
                if "rng_seed" in fpidf.columns:
                    msg_ = "cannot specify rng_seed when it is in df"
                    assert rng_seed is None, msg_
                    rng_seed_fpi = fpidf["rng_seed"].values
                if "epoch" in fpidf.columns:
                    msg_ = "cannot specify epoch when it is in df"
                    assert epoch is None, msg_
                    epoch_fpi = fpidf["epoch"].values
                vals, info_ = self.get_fpidx_key(
                    fpidx, key, dtype, rng_seed_fpi, epoch_fpi, ret_info=True
                )
                data_list.append(vals)
                info_list.append(info_)

                idxs_ = fpidf.index.values
                n_ireps = vals.shape[0] / len(idxs_)
                assert int(n_ireps) == n_ireps
                idx_list.append(idxs_.repeat(int(n_ireps)))

            idxs = np.concatenate(idx_list, axis=0)
            idxs_as = idxs.argsort(kind="mergesort")
            info = pd.concat(info_list, axis=0, ignore_index=True)
            info = info.iloc[idxs_as].reset_index(drop=True)
            if dtype == "np.arr":
                same_shape = (
                    len(set(np_arr.shape[1:] for np_arr in data_list)) == 1
                )
                if same_shape:
                    data = np.concatenate(data_list, axis=0)
                    sdata = data[idxs_as]
                else:
                    aa = np.cumsum([np_arr.shape[0] for np_arr in data_list])

                    def get_ii(i):
                        i1 = (aa > i).argmax()
                        c = data_list[i1].shape[0]
                        i2 = c - (i - aa[i1])
                        assert i2 >= 0
                        assert i2 < c
                        return i1, i2

                    data = list(chain.from_iterable(data_list))
                    ii_list = [get_ii(i) for i in idxs_as]
                    sdata = [data_list[i1][i2] for i1, i2 in ii_list]
            elif dtype in ("pd.cat", "pd.qnt", "pd.yml"):
                data = pd.concat(data_list, axis=0, ignore_index=True)
                sdata = data.iloc[idxs_as]
            else:
                raise ValueError(f"dtype={dtype} not defined")

        out = sdata
        if ret_info:
            out = (out, info)
        return out

    def dtypes(self, full=None, fpidx=None):
        """
        Queies all the keys inside the hdf file and partition
        specified by `fpidx`, and returns a dictionary specifying
        them and their respective dtypes.

        The `fpidx` argument can be unresolved for this method.

        The output will be a dictionary mapping each key to its
        dtype.  The dtype values are one of 'np.arr', 'pd.cat',
        or 'pd.qnt'.

        When `full=True`, the output dictionary keys
        will have f'{fpidx}:{key}' format.

        When `full=False`, the output dictionary keys
        will have a simple f'{key}' format.

        The output will only include keys corresponding to h5py
        datasets or pandas dataframes. In other words, h5py groups
        (i.e., directories) will not be included in the output.

        **Note:** If you suspect that different hdf files/partitions
        have different keys or dtypes, it's best to specify `full=True`
        so that you get a full report of which partition has what
        keys and dtypes.

        Parameters
        ----------
        fpidx: (str, list of str) unresolved `fpidx` values.

        full: (bool) an indicator specifying whether the keys in
          the output must have a leading `fpidx` specifier.

        Outputs
        -------
        info: (OrderedDict) a key-to-dtype mapping of the
          existing keys to their data types.
          Example:

            info = {
              'etc': 'pd.cat',
              'hp': 'pd.cat',
              'stat': 'pd.qnt',
              'mon': 'pd.qnt',
              'var/eval/ug/sol/mdl': 'np.arr',
              'var/eval/ug/sol/gt': 'np.arr',
              'mdl/layer_first.0': 'np.arr',
              ...
            }

            info = {
              '01_poisson/02_mse2d.0.0:etc': 'pd.cat',
              '01_poisson/02_mse2d.0.0:hp': 'pd.cat',
              '01_poisson/02_mse2d.0.0:stat': 'pd.qnt',
              '01_poisson/02_mse2d.0.0:mon': 'pd.qnt',
              '01_poisson/02_mse2d.0.0:mdl/layer_first.0': 'np.arr',
              ...
            }
        """
        fpidx = fpidx if fpidx is not None else self.fpidx
        all_fpidxs = self.rslv_fpidx(fpidx)
        all_fpidxs = list(odict.fromkeys(all_fpidxs))
        full = full if full is not None else self.full

        all_outs = odict()
        for fpidx_ in all_fpidxs:
            h5fp, h5grp, tdf = self.load_flpntrs(fpidx_, load_pdata=False)
            key2dtype = get_h5dtypes(h5grp)
            all_outs[f"{fpidx_}:"] = key2dtype

        info = deep2hie(all_outs)
        info = odict([(k.replace(":/", ":", 1), v) for k, v in info.items()])
        if not (full):
            info = odict([(k.split(":")[1], v) for k, v in info.items()])
        return info

    def du(self, verbose=False):
        """
        Returns the disk-usage information.

        The `fpidx` argument can be unresolved for this method.

        The output will be a dictionary mapping each key to its
        number of bytes.

        Parameters
        ----------
        fpidx: (str, list of str) unresolved `fpidx` values.

        verbose: (bool) whether to print the disk-usage information
            in a text table.

        Outputs
        -------
        duinfo: (dict) a dictionary mapping each key to its number
            of bytes.
        """
        fpidx = self.fpidx
        dtypes = self.dtypes(fpidx=fpidx, full=True)
        duinfo = dict()

        for fpkey, dtype in dtypes.items():
            fpidx, key = fpkey.split(":", maxsplit=1)
            h5fp, h5grp_, tdf = self.load_flpntrs(fpidx, load_pdata=False)
            h5grp = h5grp_[key]
            duinfo[fpkey] = sum(get_h5du(h5grp).values())

        if verbose:
            print_duinfo(duinfo)
        return duinfo

    def keys(self, full=None):
        return list(self.dtypes(full=full).keys())

    def values(self, full=None, **kwargs):
        return list(self.items(full=full, **kwargs).values())

    def items(self, full=None, **kwargs):
        full = full if full is not None else self.full
        dtdict = self.dtypes(full=True)

        keyfps = defaultdict(list)
        for fpikey, dtype in dtdict.items():
            fpidx_, key_ = fpikey.split(":")
            key = fpikey if full else key_
            keyfps[key].append(fpidx_)
        badkeyfps = {k: v for k, v in keyfps.items() if len(v) > 1}
        msg_ = "Multiple files/parts have the same keys. "
        msg_ += 'Specify "full=True" to get all of them.\n'
        for key, fps in badkeyfps.items():
            msg_ += f"{key}: {fps}\n"
        assert len(badkeyfps) == 0, msg_

        out = odict()
        for fpikey, dtype in dtdict.items():
            fpidx_, key_ = fpikey.split(":")
            key = fpikey if full else key_
            out[key] = self.__call__(key_, dtype, fpidx_, **kwargs)
        return out

    def load_key(self, key, dtype=None, fpidx=None):
        """
        Loads the **raw form** of a key from a resolvable `fpidx`.

        The raw form means that there are no epoch/rng_seed filtering happening 
        at this stage. Those things are handled in the `get_fpidx_key` method.

        This function is public and intended for general use.

        Parameters
        ----------
        fpidx: (str) a single resolved fpidx value
            Example: fpidx = '01_poisson/02_mse2d.0.0'

        key: (str) the hdf key to pull from the h5 file.
            Example: key = '/var/eval/ug/sol/mdl'

        dtype: (str) the data type of the key. It should be one of
            the 'np.arr', 'pd.cat', or 'pd.qnt' values.
        
        load_pdata: (bool) whether to load the 'epoch', 'rng_seed', and
            'ioidx' values into memory from the `stat` dataframe of
            the hdf file and partition.

            Note: This argument was only added for internal usage in 
                the `get_fpidx_key` method. General users should disable 
                it and never use this argument directly.

        ret_flpntrs: (bool) whether to return the loaded file pointers.

            Note: This argument was only added for internal usage in 
                the `get_fpidx_key` method. General users should disable 
                it and never use this argument directly.
        """
        fpidx2 = self.fpidx if fpidx is None else fpidx
        resdir = self.resdir

        # In this case, the provided fpidx is non-singular.
        fpidxs_lst = self.rslv_fpidx(fpidx2)

        if len(fpidxs_lst) == 0:
            raise ValueError(dedent(f'''
                fpidx={fpidx} could not be resolved.'''))

        out = dict()
        for fpidx_sngl in fpidxs_lst:
            out[fpidx_sngl] = self.load_key_sngl(key, dtype=dtype, fpidx=fpidx_sngl)
        
        if (len(fpidxs_lst) == 1) and isinstance(fpidx2, str) and (fpidxs_lst[0] == fpidx2):
            out = out[fpidx2]

        return out

    def load_key_sngl(self, key, dtype=None, fpidx=None, load_pdata=False, ret_flpntrs=False):
        """
        Loads the **raw form** of a single key from a **single** `fpidx`.

        The raw form means that there are no epoch/rng_seed filtering happening 
        at this stage. Those things are handled in the `get_fpidx_key` method.

        This function is private and only intended for internal use.

        Parameters
        ----------
        fpidx: (str) a single resolved fpidx value
            Example: fpidx = '01_poisson/02_mse2d.0.0'

        key: (str) the hdf key to pull from the h5 file.
            Example: key = '/var/eval/ug/sol/mdl'

        dtype: (str) the data type of the key. It should be one of
            the 'np.arr', 'pd.cat', or 'pd.qnt' values.
        
        load_pdata: (bool) whether to load the 'epoch', 'rng_seed', and
            'ioidx' values into memory from the `stat` dataframe of
            the hdf file and partition.

            Note: This argument was only added for internal usage in 
                the `get_fpidx_key` method. General users should disable 
                it and never use this argument directly.

        ret_flpntrs: (bool) whether to return the loaded file pointers.

            Note: This argument was only added for internal usage in 
                the `get_fpidx_key` method. General users should disable 
                it and never use this argument directly.
        """
        fpidx = self.fpidx if fpidx is None else fpidx
        resdir = self.resdir
        
        h5fp, h5grp, tdf = self.load_flpntrs(fpidx, load_pdata=load_pdata)
        assert h5grp is not None, dedent(f'''
            The provided fpidx does not define a single file-part index.
            This method can only handle unique fpidx values.
                fpidx = {fpidx}''')
        
        if dtype is None:
            dtype = self.dtrmn_dtype(key, fpidx=fpidx)

        if dtype in ("np.arr",):
            assert key in h5grp, dedent(f'''
                "{key}" does not exist in "{fpidx}"''')
            h5ds = h5grp[key]
            assert isinstance(h5ds, h5py.Dataset), dedent(f'''
                "{key}" is not a leaf dataset in "{fpidx}"''')
            val = h5ds
        elif dtype in ("pd.yml",):
            keyyml = f'{key}/yml'
            assert keyyml in h5grp, dedent(f'''
                "{keyyml}" does not exist in "{fpidx}"''')
            h5ds = h5grp[keyyml]
            assert isinstance(h5ds, h5py.Dataset), dedent(f'''
                "{keyyml}" is not a leaf dataset in "{fpidx}"''')
            assert len(h5ds) == 1
            ymlstr = h5ds[0].decode("utf-8")
            assert isinstance(ymlstr, str)
            resdict = ruyaml.safe_load(ymlstr)
            val = resdict
        elif dtype in ("pd.cat", "pd.qnt"):
            cfg_tree, fidx, pidx = fpidx.split(".")
            fidx, pidx = int(fidx), int(pidx)
            
            hdfpath = f"{resdir}/{cfg_tree}_{fidx:02d}.h5"
            resdf = pd.read_hdf(hdfpath, key=f"P{pidx:08d}/{key}")
            val = resdf
        else:
            raise ValueError(f"dtype={dtype} not defined")
        
        if ret_flpntrs:
            out = (val, h5fp, h5grp, tdf)
        else:
            out = val
            self.close(self.n_fp)
        
        return out

    def dtrmn_dtype(self, key, fpidx=None):
        """
        Determines the dtype by running `self.dtypes` full once on `fpidx` 
        and caching the dtype dict.

        Parameters
        ----------
            key: (str) The stored key.

            fpidx: (str | list[str] | tuple[str] | set[str] | None) 
                The provided fpidx. If None, it will default to self.fpidx.
                This function can only handle str or sequence of strs as fpidx.

        Returns
        -------
            dtype: (str) the type of the stored key.
        """
        fpidx = self.fpidx if fpidx is None else fpidx
        
        if isinstance(fpidx, str):
            fpidxhashble = fpidx
        elif isinstance(fpidx, (list, tuple, set)):
            assert all(isinstance(fpi, str) for fpi in fpidx), dedent(f'''
                Automated determination of dtype is only supported when 
                the provided fpidx is either an str or a sequence of strs:
                    fpidx={fpidx}''')
            fpidxhashble = tuple(sorted(fpidx))
        else:
            raise ValueError(f'undefined fpidx={fpidx}')
        
        if fpidxhashble not in self.dtdictcache:
            self.dtdictcache[fpidxhashble] = self.dtypes(full=True, fpidx=fpidx)
        dtdict = self.dtdictcache[fpidxhashble]
        
        dts = {
            fpikey: dtype_
            for fpikey, dtype_ in dtdict.items()
            if ':'.join(fpikey.split(':')[1:]) == key
        }
        assert len(set(dts.values())) == 1, dedent(f'''
            not sure what the dtype for {key} is:\n{dts}''')
        dtype = list(dts.values())[0]

        return dtype

