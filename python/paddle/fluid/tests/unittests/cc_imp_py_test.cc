//  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <iostream>

#include "Python.h"

TEST(CC, IMPORT_PY) {
  // Initialize python env
  Py_Initialize();
  ASSERT_FALSE(!Py_IsInitialized());

  // 1. C++ Run Python simple string
  ASSERT_FALSE(!PyRun_SimpleString("import paddle"));
  ASSERT_FALSE(!PyRun_SimpleString("print(paddle.to_tensor(1))"));

  // 2. C++ Run Python funciton
  PyObject* pModule = PyImport_ImportModule("test_install_check");
  PyObject* pFunc = PyObject_GetAttrString(pModule, "TestFunc");
  PyObject* pArg = PyEval_CallObject(pFunc, NULL);
  int result;
  PyArg_Parse(pArg, "i", &result);
  EXPECT_EQ(result, 1);

  // 3. C++ Run Python file
  FILE* fp = _Py_fopen("test_install_check.py", "r+");
  ASSERT_FALSE(!PyRun_SimpleFile(fp, "test_install_check.py"));

  // Uninitialize python env
  Py_Finalize();
  ASSERT_TRUE(!Py_IsInitialized());
}
