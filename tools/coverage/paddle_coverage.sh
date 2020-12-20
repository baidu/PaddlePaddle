#!/usr/bin/env bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

set -xe

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"

# install lcov
curl --connect-timeout 600 --retry 10 --retry-delay 10 -o /lcov-1.14.tar.gz -x "" -s https://paddle-ci.gz.bcebos.com/coverage/lcov-1.14.tar.gz || exit 101
tar -xf /lcov-1.14.tar.gz -C /
cd /lcov-1.14
make install

# run paddle coverage

cd /paddle/build

python3.7 ${PADDLE_ROOT}/tools/coverage/gcda_clean.py ${GIT_PR_ID}

lcov --capture -d ./ -o coverage.info --rc lcov_branch_coverage=0

# full html report

function gen_full_html_report() {
    lcov --extract coverage.info \
        '/paddle/paddle/fluid/framework/*' \
        '/paddle/paddle/fluid/imperative/*' \
        '/paddle/paddle/fluid/inference/*' \
        '/paddle/paddle/fluid/memory/*' \
        '/paddle/paddle/fluid/operators/*' \
        '/paddle/paddle/fluid/recordio/*' \
        '/paddle/paddle/fluid/string/*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info

    lcov --remove coverage-full.info \
        '/paddle/paddle/fluid/framework/*_test*' \
        '/paddle/paddle/fluid/*/*test*' \
        '/paddle/paddle/fluid/*/*/*test*' \
        '/paddle/paddle/fluid/inference/tests/*' \
        '/paddle/paddle/fluid/inference/api/demo_ci/*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info
}

gen_full_html_report || true

# diff html report

function gen_diff_html_report() {
    if [ "${GIT_PR_ID}" != "" ]; then

        COVERAGE_DIFF_PATTERN="`python3.7 ${PADDLE_ROOT}/tools/coverage/pull_request.py files ${GIT_PR_ID}`"

        python3.7 ${PADDLE_ROOT}/tools/coverage/pull_request.py diff ${GIT_PR_ID} > git-diff.out
    fi

    lcov --extract coverage-full.info \
        ${COVERAGE_DIFF_PATTERN} \
        -o coverage-diff.info \
        --rc lcov_branch_coverage=0

    python3.7 ${PADDLE_ROOT}/tools/coverage/coverage_diff.py coverage-diff.info git-diff.out > coverage-diff.tmp

    mv -f coverage-diff.tmp coverage-diff.info

    genhtml -o coverage-diff -t 'Diff Coverage' --no-function-coverage --no-branch-coverage coverage-diff.info
}

gen_diff_html_report || true

# python coverage

export COVERAGE_FILE=/paddle/build/python-coverage.data

set +x
coverage combine `ls python-coverage.data.*`
set -x

coverage xml -i -o python-coverage.xml

python3.7 ${PADDLE_ROOT}/tools/coverage/python_coverage.py > python-coverage.info

# python full html report
#
function gen_python_full_html_report() {
    lcov --extract python-coverage.info \
        '/paddle/python/*' \
        -o python-coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f python-coverage-full.tmp python-coverage-full.info

    lcov --remove python-coverage-full.info \
        '/*/tests/*' \
        -o python-coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f python-coverage-full.tmp python-coverage-full.info
}

gen_python_full_html_report || true

# python diff html report

function gen_python_diff_html_report() {
    if [ "${GIT_PR_ID}" != "" ]; then
        COVERAGE_DIFF_PATTERN="`python3.7 ${PADDLE_ROOT}/tools/coverage/pull_request.py files ${GIT_PR_ID}`"

        python3.7 ${PADDLE_ROOT}/tools/coverage/pull_request.py diff ${GIT_PR_ID} > python-git-diff.out
    fi

    lcov --extract python-coverage-full.info \
        ${COVERAGE_DIFF_PATTERN} \
        -o python-coverage-diff.info \
        --rc lcov_branch_coverage=0

    python3.7 ${PADDLE_ROOT}/tools/coverage/coverage_diff.py python-coverage-diff.info python-git-diff.out > python-coverage-diff.tmp

    mv -f python-coverage-diff.tmp python-coverage-diff.info

    genhtml -o python-coverage-diff \
        -t 'Python Diff Coverage' \
        --no-function-coverage \
        --no-branch-coverage \
        --ignore-errors source \
        python-coverage-diff.info
}

gen_python_diff_html_report || true

# assert coverage lines

echo "Assert Diff Coverage"

python3.7 ${PADDLE_ROOT}/tools/coverage/coverage_lines.py coverage-diff.info 0.9 || COVERAGE_LINES_ASSERT=1

echo "Assert Python Diff Coverage"

python3.7 ${PADDLE_ROOT}/tools/coverage/coverage_lines.py python-coverage-diff.info 0.9 || PYTHON_COVERAGE_LINES_ASSERT=1

if [ "$COVERAGE_LINES_ASSERT" = "1" ] || [ "$PYTHON_COVERAGE_LINES_ASSERT" = "1" ]; then
    echo "exit 9" > /tmp/paddle_coverage.result
    exit 9
fi
