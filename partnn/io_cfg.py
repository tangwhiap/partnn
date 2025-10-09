import os
from textwrap import dedent

projname = 'code_partnn'
cwd_list = os.getcwd().split(os.sep)
msg = f"""
    The project name "{projname}" must appear exactly
    once in the current working directory:
        >> os.getcwd() == {os.getcwd()}"""
assert cwd_list.count(projname) == 1, dedent(msg)
PROJPATH = os.sep.join(cwd_list[:cwd_list.index(projname) + 1])

cache_dir = f'{PROJPATH}/.cache'
configs_dir = f'{PROJPATH}/configs'
results_dir = f'{PROJPATH}/results'
storage_dir = f'{PROJPATH}/storage'
data_dir = f'{PROJPATH}/data'
datalfs_dir = f'{PROJPATH}/../data_partnn'
summary_dir = f'{PROJPATH}/summary'
nullstr = '-'

keyspecs = [('hp',         'last',    'pd.cat'),
            ('stat',       'mean',        None),
            ('etc',        'last',    'pd.cat'),
            ('mon',        'mean',        None),
            ('mdl',        'last',        None),
            ('trg',        'last',        None),
            ('var/eval/*', 'last',        None)]