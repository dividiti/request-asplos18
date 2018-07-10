#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os

def ck_preprocess(i):
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)

  # Setup parameters for program
  new_env = {}
  files_to_push_by_path = {}
  run_input_files = []

  if i['target_os_dict'].get('remote','') == 'yes':
    LIB_DIR = dep_env('library', 'CK_ENV_LIB_ARMCL')
    LIB_NAME = dep_env('library', 'CK_ENV_LIB_ARMCL_DYNAMIC_CORE_NAME')

    files_to_push_by_path['CK_ENV_ARMCL_CORE_LIB_PATH'] = os.path.join(LIB_DIR, 'lib', LIB_NAME)
    run_input_files.append('$<<CK_ENV_LIB_STDCPP_DYNAMIC>>$')

    new_env['RUN_OPT_GRAPH_FILE'] = '.'
  else:
    new_env['RUN_OPT_GRAPH_FILE'] = dep_env('weights', 'CK_ENV_MOBILENET')

  new_env['RUN_OPT_RESOLUTION'] = dep_env('weights', 'CK_ENV_MOBILENET_RESOLUTION')
  new_env['RUN_OPT_MULTIPLIER'] = dep_env('weights', 'CK_ENV_MOBILENET_MULTIPLIER')

  print('--------------------------------\n')
  return {
    'return': 0,
    'new_env': new_env,
    'run_input_files': run_input_files,
    'files_to_push_by_path': files_to_push_by_path,
  }
