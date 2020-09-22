# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
fleetrun is a module that spawns multiple distributed
process on each training node for gpu training and cpu training.
Usage:
    In both of single node training or multiple node training, this module
launch a process on each of the given gpu card or cpu machine.
    GPU training:
    1. for single node training with all visible gpu cards:
       fleetrun your_training_py (arg1 arg2 and all others)
    2. for single node training with [0,4) cards
       fleetrun --gpus="0,1,2,3" your_training_py (arg1 arg2 and all others)
    3. for multiple node training such as two node:192.168.0.16, 192.168.0.17
        on 192.168.0.16:
            fleetrun --ips="192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --ips="192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
    CPU training:
    1. for single node training with multi servers and workers:
        fleetrun --server_num=2 --worker_num=2 your_training_py (arg1 arg2 and all others)
    2. for multiple node training such as two node:192.168.0.16, 192.168.0.17 \
        with 2 servers and 4 workers.
        on 192.168.0.16:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16,192.168.0.17,192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6171" \
                --workers="192.168.0.16,192.168.0.17,192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
    3. use gloo backend for multiple node training such as two node:192.168.0.16, 192.168.0.17 \
        with 2 servers and 4 workers. (workers should set port)
        on 192.168.0.16:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16:6171,192.168.0.17:6171,192.168.0.16:6172,192.168.0.17:6172" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16:6171,192.168.0.17:6171,192.168.0.16:6172,192.168.0.17:6172" \
                your_training_py (arg1 arg2 and all others)
"""

from __future__ import print_function

import shutil
import sys
import tempfile
from sys import version
import subprocess
import os
import time
import six
import copy
from argparse import ArgumentParser, REMAINDER
import paddle
import paddle.fluid as fluid

from paddle.distributed.fleet.launch_utils import *
import paddle.distributed.fleet.cloud_utils as cloud_utils


def _print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def _parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description='''start paddle training using multi-process mode.
see: http://www.paddlepaddle.org/documentation/docs/zh/1.6/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-
''')
    base_group = parser.add_argument_group("Base Parameters")

    base_group.add_argument(
        "-d",
        "--distributed_mode",
        type=str,
        choices=["collective", "ps", "ps_heter", "ps_gpu", ""],
        default="",
        help="Distributed running mode: collective/ps/ps_gpu/ps_heter")

    base_group.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="The path for each process's log.If it's not set, the log will printed to default pipe."
    )

    base_group.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script")

    # Optional arguments for the launch helper
    # for collective
    collective_group = parser.add_argument_group("Collective Parameters")
    collective_group.add_argument(
        "--ips",
        type=str,
        default="127.0.0.1",
        help="Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..")
    collective_group.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="It's for gpu training and the training process will run on the gpus,"
        "each process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training."
    )

    ps_group = parser.add_argument_group("Parameter-Server Parameters")
    # for parameter server
    ps_group.add_argument(
        "--servers", type=str, default="", help="User defined servers ip:port")
    ps_group.add_argument(
        "--workers", type=str, default="", help="User defined workers ip:port")
    ps_group.add_argument(
        "--heter_workers",
        type=str,
        default="",
        help="User defined heter workers ip:port")

    ps_group.add_argument("--worker_num", type=int, help="number of workers")
    ps_group.add_argument("--server_num", type=int, help="number of servers")
    ps_group.add_argument(
        "--heter_worker_num", type=int, help="number of heter_workers")

    ps_group.add_argument(
        "--heter_worker_device",
        type=str,
        default="gpu",
        choices=["gpu", "xpu"],
        help="heter worker device")

    return parser.parse_args()


def get_cluster_from_args(args, gpus):
    node_ips = [x.strip() for x in args.ips.split(',')]
    if len(node_ips) == 1:
        node_ip = node_ips[0]
    else:
        _, node_ip = get_host_name_ip()

    # node_ip = args.node_ip
    assert node_ip in node_ips, "Can't find your local ip {%s} in node_ips: {%s}" \
        % (node_ip, node_ips)
    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args: node_ips:{} node_ip:{} node_rank:{}".format(
        node_ips, node_ip, node_rank))

    free_ports = None
    if not cloud_utils.use_paddlecloud() and len(
            node_ips) <= 1 and os.environ.get('FLAGS_START_PORT') is None:
        free_ports = find_free_ports(len(gpus))
        if free_ports is not None:
            free_ports = list(free_ports)
    else:
        start_port = 6070
        if os.environ.get('FLAGS_START_PORT') is not None:
            start_port = int(os.environ.get('FLAGS_START_PORT'))

        free_ports = [x for x in range(start_port, start_port + len(gpus))]

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, gpus)


def get_gpus(gpus):
    if gpus is None:
        gpus_num = fluid.core.get_cuda_device_count()
        res_gpus = [str(x) for x in range(0, gpus_num)]
    else:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or cuda_visible_devices == "":
            res_gpus = [x.strip() for x in gpus.split(',')]
        else:
            # change gpus into relative values
            # e.g. CUDA_VISIBLE_DEVICES=4,5,6,7; args.gpus=4,5,6,7;
            # therefore gpus=0,1,2,3
            cuda_visible_devices_list = cuda_visible_devices.split(',')
            for x in gpus.split(','):
                assert x in cuda_visible_devices_list, "Can't find "\
                    "your gpus %s in CUDA_VISIBLE_DEVICES[%s]."\
                    % (x, cuda_visible_devices)
            res_gpus = [
                cuda_visible_devices_list.index(x.strip())
                for x in gpus.split(',')
            ]
            logger.info("Change selected_gpus into reletive values. --ips:{} "
                        "will change into relative_ips:{} according to your "
                        "CUDA_VISIBLE_DEVICES:{}".format(
                            gpus, res_gpus, cuda_visible_devices_list))

    return res_gpus


def launch_collective(args):
    # parse arguments, used for cloud-single-machine and local
    gpus = get_gpus(args.gpus)
    trainers_num = cloud_utils.get_trainers_num()
    logger.debug("parsed from args trainerss_num:{} gpus:{}".format(
        trainers_num, gpus))

    cluster = None
    pod = None

    start_port = 6170
    if os.environ.get('FLAGS_START_PORT') is not None:
        start_port = os.environ.get('FLAGS_START_PORT')
    if cloud_utils.use_paddlecloud() and trainers_num != 1:
        cluster, pod = cloud_utils.get_cloud_cluster(args.ips, gpus, start_port)
        logger.debug("get cluster from cloud:{}".format(cluster))
    else:
        # trainers_num = 1 or not use paddlecloud ips="a,b"
        cluster, pod = get_cluster_from_args(args, gpus)
        logger.debug("get cluster from args:{}".format(cluster))

    global_envs = copy.copy(os.environ.copy())
    gloo_rendezvous_dir = tempfile.mkdtemp()
    # add gloo env
    global_envs["PADDLE_WITH_GLOO"] = "1"
    global_envs["PADDLE_GLOO_RENDEZVOUS"] = "2"
    global_envs["PADDLE_GLOO_FS_PATH"] = gloo_rendezvous_dir

    procs = start_local_trainers(
        cluster,
        pod,
        training_script=args.training_script,
        training_script_args=args.training_script_args,
        log_dir=args.log_dir,
        envs=global_envs)

    while True:
        alive = watch_local_trainers(procs, cluster.trainers_nranks())

        if not alive:
            logger.info("Local processes completed.")
            logger.debug("POD info:{}".format(pod))
            break

        time.sleep(3)

    if os.path.exists(gloo_rendezvous_dir):
        shutil.rmtree(gloo_rendezvous_dir)


def launch_ps(args):
    cloud_flag = cloud_utils.use_paddlecloud()

    # for ps-cpu on paddlecloud
    direct_start_mode = ["ps", ""]
    if cloud_flag and (args.distributed_mode in direct_start_mode):
        direct_start(args)
        return
    elif cloud_flag and args.distributed_mode == "ps_heter":
        cloud_ps_heter_env_set(args)
        args.trainers = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        args.workers = os.getenv("PADDLE_PSERVERS_IP_PORT_LIST")
        args.heter_workers = os.getenv("PADDLE_HETER_TRAINER_IP_PORT_LIST")

    ps_launcher = ParameterServerLauncher(args)
    ps_launcher.start_ps(args)
    return


def launch():
    args = _parse_args()
    logger = get_logger()
    _print_arguments(args)
    ps_args = [
        '--worker_num', '--server_num', '--heter_worker_num', '--servers',
        '--workers', '--heter_worrkers', 'heter_worker_device'
    ]
    collective_args = ['--ips', '--gpus']
    has_ps_args = [
        ps_arg for ps_arg in ps_args if ps_arg in " ".join(sys.argv[1:-1])
    ]
    has_collective_args = [
        co_arg for co_arg in collective_args
        if co_arg in " ".join(sys.argv[1:-1])
    ]
    if fluid.core.is_compiled_with_cuda():
        cuda_device_num = fluid.core.get_cuda_device_count()
    else:
        cuda_device_num = 0

    ps_mode = ['ps', 'ps_gpu', 'ps_heter']
    if len(has_ps_args) > 0 or args.distributed_mode in ps_mode:
        logger.info(
            "Run parameter-sever mode. pserver arguments:{}, cuda count:{}".
            format(has_ps_args, cuda_device_num))
        launch_ps(args)
    elif len(has_collective_args) > 0:
        logger.info("Run collective gpu mode. gpu arguments:{}, cuda count:{}".
                    format(has_collective_args, cuda_device_num))
        launch_collective(args)
    else:
        logger.warning(
            "Not found distinct arguments. Default use gpu collective mode")
        launch_collective(args)


if __name__ == "__main__":
    launch()
