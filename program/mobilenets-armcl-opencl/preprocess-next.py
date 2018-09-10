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

  WEIGHTS_DIR = dep_env('weights', 'CK_ENV_MOBILENET')
  CONV_METHOD_FILE = i['env'].get('CK_CONVOLUTION_METHOD_FILE', 'conv_methods.txt')

  if i['target_os_dict'].get('remote','') == 'yes':
    if i['env'].get('CK_PUSH_LIBS_TO_REMOTE', 'yes').lower() == 'yes':
      lib_dir = dep_env('library', 'CK_ENV_LIB_ARMCL')
      lib_name = dep_env('library', 'CK_ENV_LIB_ARMCL_DYNAMIC_CORE_NAME')
      files_to_push_by_path['CK_ENV_ARMCL_CORE_LIB_PATH'] = os.path.join(lib_dir, 'lib', lib_name)
      run_input_files.append('$<<CK_ENV_LIB_STDCPP_DYNAMIC>>$')

    if i['env'].get('CK_PUSH_WEIGHTS_TO_REMOTE', 'yes').lower() == 'yes':
      file_index = 0
      for file_name in os.listdir(WEIGHTS_DIR):
        if file_name.endswith('.npy'):
          var_name = 'CK_ENV_WEIGHTS_' + str(file_index)
          files_to_push_by_path[var_name] = os.path.join(WEIGHTS_DIR, file_name)
          file_index += 1

    if os.path.isfile(CONV_METHOD_FILE):
      run_input_files.append(os.path.join(os.getcwd(), CONV_METHOD_FILE))

    new_env['RUN_OPT_GRAPH_FILE'] = '.'
  else:
    new_env['RUN_OPT_GRAPH_FILE'] = WEIGHTS_DIR

  new_env['RUN_OPT_RESOLUTION'] = dep_env('weights', 'CK_ENV_MOBILENET_RESOLUTION')
  new_env['RUN_OPT_MULTIPLIER'] = dep_env('weights', 'CK_ENV_MOBILENET_MULTIPLIER')
  new_env['RUN_OPT_CONVOLUTION_METHOD_FILE'] = CONV_METHOD_FILE
  new_env['RUN_OPT_DATA_LAYOUT'] = i['env'].get('CK_DATA_LAYOUT', 'NCHW')

  print('--------------------------------\n')
  return {
    'return': 0,
    'new_env': new_env,
    'run_input_files': run_input_files,
    'files_to_push_by_path': files_to_push_by_path,
  }
