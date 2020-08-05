# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
decorator to deprecate a function or class
"""

import warnings
import functools
import paddle


def deprecated(update_to="", since="", reason=""):
    """Decorate a function to signify its deprecation.

       This function wraps a method that will soon be removed and does two things:
           - The docstring of the API will be modified to include a notice
             about deprecation."
           - Raises a :class:`~exceptions.DeprecatedWarning` when old API is called.
       Args:
           since(str): The version at which the decorated method is considered deprecated.
           update_to(str): The new API users should use.
           reason(str): The reason why the API is deprecated.
       Returns:
           decorator: decorated function or class.
    """

    def _compare_version(v1, v2):
        """compare two versions."""

        _v1 = [int(i) for i in v1.split(".")]
        _v2 = [int(i) for i in v2.split(".")]
        _v1 += [0] * (4 - len(v1))
        _v2 += [0] * (4 - len(v1))
        for i in range(4):
            if _v1[i] > _v2[i]:
                return 1
            elif _v1[i] < _v2[i]:
                return -1
            return 0

    def decorator(func):
        """construct warning message, and return a decorated function or class."""
        assert isinstance(update_to, str), 'type of "update_to" must be str.'
        assert isinstance(since, str), 'type of "since" must be str.'
        assert isinstance(reason, str), 'type of "reason" must be str.'

        _since = since.strip()
        _update_to = update_to.strip()
        _reason = reason.strip()

        msg = 'API "{}.{}" is deprecated'.format(func.__module__, func.__name__)
        if len(_since) > 0:
            msg += " since {}".format(_since)
        msg += ", and may be removed in future versions."
        if len(_update_to) > 0:
            assert _update_to.startswith(
                "paddle."
            ), 'Argument update_to must start with "paddle.", your value is "{}"'.format(
                update_to)
            msg += ' Use "{}" instead.'.format(_update_to)
        if len(_reason) > 0:
            msg += "\n reason: {}".format(_reason)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """deprecated warning should be fired in 3 circumstances:
               1. current version is develop version, i.e. "0.0.0", because we assume develop version is always the latest version.
               2. since version is empty, in this case, API is deprecated in all versions.
               3. current version is newer than since version.
            """
            if paddle.__version__ == "0.0.0" or _since == "" or _compare_version(
                    paddle.__version__, _since) >= 0:
                warnings.simplefilter('always',
                                      DeprecationWarning)  # turn off filter
                warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
                warnings.simplefilter('default',
                                      DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return wrapper

    return decorator
