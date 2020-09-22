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

import functools
import logging
import socket
import time
import os
import signal
import copy
import sys
import subprocess
import tempfile
import shutil
from contextlib import closing
import socket

import paddle
import paddle.fluid as fluid
logger = logging.getLogger("root")
logger.propagate = False


class Cluster(object):
    def __init__(self, hdfs):
        self.job_server = None
        self.pods = []
        self.hdfs = None
        self.job_stage_flag = None

    def __str__(self):
        return "job_server:{} pods:{} job_stage_flag:{} hdfs:{}".format(
            self.job_server, [str(pod) for pod in self.pods],
            self.job_stage_flag, self.hdfs)

    def __eq__(self, cluster):
        if len(self.pods) != len(cluster.pods):
            return False

        for a, b in zip(self.pods, cluster.pods):
            if a != b:
                return False

        if self.job_stage_flag != cluster.job_stage_flag:
            return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(cluster):
        self.pods = copy.copy(cluster.pods)

    def trainers_nranks(self):
        return len(self.trainers_endpoints())

    def pods_nranks(self):
        return len(self.pods)

    def trainers_endpoints(self):
        r = []
        for pod in self.pods:
            for t in pod.trainers:
                r.append(t.endpoint)
        return r

    def pods_endpoints(self):
        r = []
        for pod in self.pods:
            ep = "{}:{}".format(pod.addr, pod.port)
            assert pod.port != None and pod.addr != None, "{} not a valid endpoint".format(
                ep)
            r.append(ep)

        return r

    def get_pod_by_id(self, pod_id):
        for pod in self.pods:
            if str(pod_id) == str(pod.id):
                return pod

        return None


class JobServer(object):
    def __init__(self):
        self.endpoint = None

    def __str__(self):
        return "{}".format(self.endpoint)

    def __eq__(self, j):
        return self.endpint == j.endpoint

    def __ne__(self, j):
        return not self == j


class Trainer(object):
    def __init__(self):
        self.gpus = []
        self.endpoint = None
        self.rank = None

    def __str__(self):
        return "gpu:{} endpoint:{} rank:{}".format(self.gpus, self.endpoint,
                                                   self.rank)

    def __eq__(self, t):
        if len(self.gpus) != len(t.gpus):
            return False

        if self.endpoint != t.endpoint or \
                self.rank != t.rank:
            return False

        for a, b in zip(self.gpus, t.gpus):
            if a != b:
                return False

        return True

    def __ne__(self, t):
        return not self == t

    def rank(self):
        return self.rank


class Pod(object):
    def __init__(self):
        self.rank = None
        self.id = None
        self.addr = None
        self.port = None
        self.trainers = []
        self.servers = []
        self.workers = []
        self.heter_workers = []
        self.gpus = []

    def __str__(self):
        return "rank:{} id:{} addr:{} port:{} visible_gpu:{} trainers:{} servers:{} \
            workers:{} heter_workers:{}".format(
            self.rank, self.id, self.addr, self.port, self.gpus, [
                str(t) for t in self.trainers
            ], [str(s) for s in self.servers], [str(w) for w in self.workers],
            [str(h) for h in self.heter_workers])

    def __eq__(self, pod):
        if self.rank != pod.rank or \
                self.id != pod.id or \
                self.addr != pod.addr or \
                self.port != pod.port:
            logger.debug("pod {} != pod".format(self, pod))
            return False

        if len(self.trainers) != len(pod.trainers):
            logger.debug("trainers {} != {}".format(self.trainers,
                                                    pod.trainers))
            return False

        for i in range(len(self.trainers)):
            if self.trainers[i] != pod.trainers[i]:
                logger.debug("trainer {} != {}".format(self.trainers[i],
                                                       pod.trainers[i]))
                return False

        if len(self.servers) != len(pod.servers):
            logger.debug("servers {} != {}".format(self.servers, pod.servers))
            return False

        for i in range(len(self.servers)):
            if self.servers[i] != pod.servers[i]:
                logger.debug("servers {} != {}".format(self.servers[i],
                                                       pod.servers[i]))
                return False

        if len(self.workers) != len(pod.workers):
            logger.debug("workers {} != {}".format(self.workers, pod.workers))
            return False

        for i in range(len(self.workers)):
            if self.workers[i] != pod.workers[i]:
                logger.debug("workers {} != {}".format(self.workers[i],
                                                       pod.workers[i]))
                return False

        return True

    def __ne__(self, pod):
        return not self == pod

    def parse_response(self, res_pods):
        pass

    def rank(self):
        return self.rank

    def get_visible_gpus(self):
        r = ""
        for g in self.gpus:
            r += "{},".format(g)

        assert r != "", "this pod {} can't see any gpus".format(self)

        r = r[:-1]
        return r


def get_logger(log_level=20, name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    log_handler = logging.StreamHandler()
    log_format = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    return logger


def get_cluster(node_ips, node_ip, trainer_endpoints, selected_gpus):
    assert type(trainer_endpoints) is list, "trainer_endpoints must be list"
    cluster = Cluster(hdfs=None)
    trainer_rank = 0
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.addr = ip
        cur_node_endpoints = trainer_endpoints[node_rank]
        # when use paddlecloud, endpoints may > selected_gpus(user_defined)
        assert len(cur_node_endpoints) >= len(
            selected_gpus
        ), "current trainer_endpoints size should be greater equal than selected_gpus size."
        for i in range(len(selected_gpus)):
            trainer = Trainer()
            trainer.gpus.append(selected_gpus[i])
            trainer.endpoint = "%s" % (cur_node_endpoints[i])
            trainer.rank = trainer_rank
            trainer_rank += 1

            pod.trainers.append(trainer)
        cluster.pods.append(pod)

    pod_rank = node_ips.index(node_ip)
    return cluster, cluster.pods[pod_rank]


def terminate_local_procs(procs):
    for p in procs:
        if p.proc.poll() is None:
            p.proc.terminate()
            if p.log_fn:
                p.log_fn.close()
            logger.debug("terminate process id:{}".format(p.proc.pid))

    # wait all process terminiated
    time.sleep(3)
    for step in range(0, 50):
        alive = False
        for p in procs:
            if p.proc.poll() is None:  # not termniate
                os.kill(p.proc.pid, signal.SIGKILL)
                alive = True

        if not alive:
            logger.info("terminate all the procs")
            return

        time.sleep(3)

    logger.fatal("can't kill all process and exit")
    exit(1)


def get_host_name_ip():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_name, host_ip
    except:
        return None


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.
    Usage:
    .. code-block:: python
        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def find_free_ports(num):
    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port_set = set()
    step = 0
    while True:
        port = __free_port()
        if port not in port_set:
            port_set.add(port)

        if len(port_set) >= num:
            return port_set

        step += 1
        if step > 100:
            print(
                "can't find avilable port and use the specified static port now!"
            )
            return None

    return None


def get_ports(num, offset):
    if os.environ.get('FLAGS_START_PORT') is None:
        ports = find_free_ports(num)
        if ports is not None:
            ports = list(ports)
    else:
        start_port = os.environ.get('FLAGS_START_PORT')
        ports = range(start_port + offset, start_port + offset + num, 1)
    return ports


def pretty_print_envs(envs, header=None):
    spacing = 2
    max_k = 40
    max_v = 45

    for k, v in envs.items():
        max_k = max(max_k, len(k))

    h_format = "    " + "|{{:>{}s}}{}{{:^{}s}}|\n".format(max_k, " " * spacing,
                                                          max_v)
    l_format = "    " + "|{{:>{}s}}{{}}{{:^{}s}}|\n".format(max_k, max_v)
    length = max_k + max_v + spacing

    border = "    +" + "".join(["="] * length) + "+"
    line = "    +" + "".join(["-"] * length) + "+"

    draws = ""
    draws += border + "\n"

    if header:
        draws += h_format.format(header[0], header[1])
    else:
        draws += h_format.format("fleetrun Distributed Envs", "Value")

    draws += line + "\n"

    for k, v in envs.items():
        if isinstance(v, str) and len(v) >= max_v:
            str_v = "... " + v[-41:]
        else:
            str_v = v

        draws += l_format.format(k, " " * spacing, str(str_v))

    draws += border

    _str = "\n{}\n".format(draws)
    return _str


class TrainerProc(object):
    def __init__(self):
        self.proc = None
        self.log_fn = None
        self.log_offset = None
        self.rank = None
        self.local_rank = None
        self.cmd = None


def start_local_trainers(cluster,
                         pod,
                         training_script,
                         training_script_args,
                         log_dir=None,
                         envs=None):

    if envs is None:
        current_env = copy.copy(os.environ.copy())
    else:
        current_env = copy.copy(envs)

    # paddle broadcast ncclUniqueId use socket, and
    # proxy maybe make trainers unreachable, so delete them.
    # if we set them to "", grpc will log error message "bad uri"
    # so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    for idx, t in enumerate(pod.trainers):
        proc_env = {
            "FLAGS_selected_gpus": "%s" % ",".join([str(g) for g in t.gpus]),
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % t.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints())
        }

        current_env.update(proc_env)

        cmd = [sys.executable, "-u", training_script] + training_script_args

        logger.debug("start trainer proc{}  env:{}".format(cmd, current_env))

        if idx == 0:
            logger.info("Local start {} processes. First process distributed "
                        "environment info (Only For Debug): {}".format(
                            len(pod.trainers),
                            pretty_print_envs(proc_env, ("Distributed Envs",
                                                         "Value"))))
            logger.info(
                "details abouts PADDLE_TRAINER_ENDPOINTS can be found in {}/endpoints.log.".
                format(log_dir))
        fn = None
        if log_dir is not None:
            os.system("mkdir -p {}".format(log_dir))
            if os.path.exists("%s/endpoints.log" % log_dir):
                os.system("rm -f {}/endpoints.log".format(log_dir))
            with open("%s/endpoints.log" % log_dir, "w") as f:
                f.write("PADDLE_TRAINER_ENDPOINTS: \n")
                f.write("\n".join(cluster.trainers_endpoints()))
            fn = open("%s/workerlog.%d" % (log_dir, idx), "a")
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.local_rank = idx
        tp.log_fn = fn
        tp.log_offset = fn.tell() if fn else None
        tp.cmd = cmd

        procs.append(tp)

    return procs


def pull_worker_log(tp):
    if tp.log_fn:
        with open(tp.log_fn.name, 'r') as fin:
            fin.seek(tp.log_offset, 0)
            for line in fin:
                try:
                    sys.stdout.write(line)
                except UnicodeEncodeError:
                    sys.stdout.write(
                        'UnicodeEncodeError occurs at this line. '
                        'Please refer to the original log file "%s"\n' %
                        tp.log_fn.name)
            tp.log_offset = fin.tell()


def watch_local_trainers(procs, nranks):
    try:
        error = False
        error_rank = []
        # wait all process finish or one error
        alive = False
        for p in procs:
            if p.log_fn and p.local_rank == 0:
                pull_worker_log(p)

            ret = p.proc.poll()
            if ret is None:
                alive = True
            elif ret != 0:
                error = True
                error_rank.append(p.rank)

        if error:
            terminate_local_procs(procs)
            exit(1)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt, exit")
        terminate_local_procs(procs)
        raise
    except SystemExit:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_local_procs(procs)
        raise
    except:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_local_procs(procs)
        raise

    return alive


def direct_start(args):
    # run ps-cpu mode on paddlecloud, using given envs
    cmd = [sys.executable, "-u", args.training_script] + \
        args.training_script_args
    proc = subprocess.Popen(cmd)
    proc.wait()
    return


def get_custom_endpoints(origin_endpoints, offset=0):
    """
    origin_endpoint: ip:port
    user_define_endpoint: ip:(port+offset)
    """
    assert origin_endpoints != None
    paddle_user_define_endpoints_list = []
    for ip_port in origin_endpoints.split(","):
        ip = ip_port.split(":")[0]
        port = ip_port.split(":")[1]
        new_port = int(port) + offset
        paddle_user_define_endpoints_list.append(":".join((ip, str(new_port))))
    paddle_user_define_endpoints = ",".join(paddle_user_define_endpoints_list)
    return paddle_user_define_endpoints


def cloud_ps_heter_env_set(args):
    environs = {}

    paddle_trainer_endpoints = os.getenv("TRAINER_IP_PORT_LIST", "")
    assert paddle_trainer_endpoints != None
    environs["PADDLE_TRAINER_ENDPOINTS"] = paddle_trainer_endpoints

    paddle_pserver_endpoints = os.getenv("PSERVER_IP_PORT_LIST", "")
    assert paddle_pserver_endpoints != None
    environs["PADDLE_PSERVERS_IP_PORT_LIST"] = paddle_pserver_endpoints

    avilable_ports = os.getenv("TRAINER_PORTS", "").split(",")
    assert len(
        avilable_ports
    ) > 3, "set paddle_ports_num >= 2 in config.ini for paddlecloud job submit"

    # hard code for paddlecloud custom-framework
    trainers_num = len(paddle_pserver_endpoints.split(","))
    assert trainers_num != 0
    environs["PADDLE_TRAINERS_NUM"] = trainers_num
    environs["TRAINERS_NUM"] = trainers_num
    environs["PADDLE_HETER_TRAINER_DEVICE"] = args.heter_worker_device

    # hard code for paddlecloud custom-framework
    environs["PADDLE_HETER_TRAINER_IP_PORT_LIST"] = paddle_trainer_endpoints
    environs["PADDLE_PSERVERS_IP_PORT_LIST"] = paddle_pserver_endpoints
    environs["PADDLE_TRAINER_ENDPOINTS"] = get_custom_endpoints(
        paddle_pserver_endpoints, 1)

    for k, v in environs.items():
        os.environ[k] = str(v)
    logger.info("Set heter parameter server env: {}".format(
        pretty_print_envs(environs)))


class ParameterServerLauncher(object):
    def __init__(self, args):
        self.server_num = 0
        self.worker_num = 0
        self.heter_worker_num = 0

        self.server_endpoints = []
        self.server_endpoints_ips = []
        self.server_endpoints_port = []

        self.worker_endpoints = []
        self.worker_endpoints_ips = []
        self.worker_endpoints_port = []

        self.heter_worker_endpoints = []
        self.heter_worker_endpoints_ips = []
        self.heter_worker_endpoints_port = []

        self.is_local = True
        self.current_node_ip = ""

        self.get_role_endpoints(args)

    def get_role_endpoints(self, args):
        # get server envs
        if args.server_num:
            self.server_num = args.server_num
            if args.servers:
                assert len(
                    args.servers.split(",")
                ) == self.server_num, "The server_num and servers doesn't match. Expect servers endpoints num epual to server_num, but received servers enpoint num: {} and server_num {}".format(
                    len(args.servers.split(",")), self.server_num)
                self.server_endpoints = args.servers
            else:
                ports = get_ports(self.server_num, 0)
                self.server_endpoints = ",".join(
                    ["127.0.0.1:" + str(x) for x in ports])
        else:
            assert args.servers != "", "The setting of Parameter-Server must has server_num or servers."
            self.server_endpoints = args.servers
            self.server_num = len(self.server_endpoints.split(","))

        # get worker envs
        if args.worker_num:
            self.worker_num = args.worker_num
            if args.workers:
                assert len(
                    args.workers.split(",")
                ) == self.worker_num, "The worker_num and workers doesn't match. Expect workers endpoints num epual to worker_num, but received workers enpoint num: {} and worker_num {}".format(
                    len(args.workers.split(",")), self.worker_num)
                self.worker_endpoints = args.workers
            else:
                ports = get_ports(self.worker_num, self.server_num)
                self.worker_endpoints = ",".join(
                    ["127.0.0.1:" + str(x) for x in ports])
        else:
            assert args.workers != "", "The setting of Parameter-Server must has worker_num or workers."
            self.worker_endpoints = args.workers
            self.worker_num = len(self.worker_endpoints.split(","))

        # get heter worker envs
        if args.distributed_mode == "ps_heter":
            if args.heter_worker_num:
                self.heter_worker_num = args.heter_worker_num
                if args.heter_workers:
                    assert len(
                        args.heter_workers.split(",")
                    ) == self.heter_worker_num, "The heter_worker_num and heter_workers doesn't match. Expect heter_workers endpoints num epual to heter_worker_num, but received heter_workers enpoint num: {} and heter_worker_num {}".format(
                        len(args.heter_workers.split(",")),
                        self.heter_worker_num)
                    self.heter_worker_endpoints = args.heter_workers
                else:
                    ports = get_ports(self.heter_worker_num,
                                      self.server_num + self.worker_num)
                    self.heter_worker_endpoints = ",".join(
                        ["127.0.0.1:" + str(x) for x in ports])
            else:
                assert args.heter_workers != "", "The setting of Parameter-Server heter mode must has heter_worker_num or heter_workers."
                self.heter_worker_endpoints = args.heter_workers
                self.heter_worker_num = len(
                    self.heter_worker_endpoints.split(","))

        # check local or user define
        self.server_endpoints_ips = [
            x.strip().split(":")[0] for x in self.server_endpoints.split(",")
        ]
        self.worker_endpoints_ips = [
            x.strip().split(":")[0] for x in self.worker_endpoints.split(",")
        ]
        self.server_endpoints_port = [
            x.strip().split(":")[1] for x in self.server_endpoints.split(",")
        ]
        self.worker_endpoints_port = [
            x.strip().split(":")[1] for x in self.worker_endpoints.split(",")
        ]
        self.node_ips = list(
            set(self.server_endpoints_ips + self.worker_endpoints_ips))
        if args.distributed_mode == "ps_heter":
            self.heter_worker_endpoints_ips = [
                x.strip().split(":")[0]
                for x in self.heter_worker_endpoints.split(",")
            ]
            self.heter_worker_endpoints_port = [
                x.strip().split(":")[1]
                for x in self.heter_worker_endpoints.split(",")
            ]
            self.node_ips = list(
                set(self.node_ips + self.heter_worker_endpoints_ips))

        if len(set(self.node_ips)) == 1:
            self.is_local = True
            self.current_node_ip = self.node_ips[0]
        else:
            self.is_local = False
            _, self.current_node_ip = get_host_name_ip()
            assert self.current_node_ip in self.node_ips, "Can't find your local ip {%s} in args.servers and args.workers ips: {%s}" \
                % (self.current_node_ip, self.node_ips)
        self.node_rank = self.node_ips.index(self.current_node_ip)

        logger.debug(
            "parsed from args: node_ips:{} current_node_ip:{} node_rank:{}".
            format(self.node_ips, self.current_node_ip, self.node_rank))

    def start_ps(self, args):
        cluster = Cluster(hdfs=None)
        server_rank = 0
        worker_rank = 0
        heter_worker_rank = 0

        for node_rank, ip in enumerate(self.node_ips):
            pod = Pod()
            pod.rank = node_rank
            pod.addr = ip
            for i in range(len(self.server_endpoints_ips)):
                if ip == self.server_endpoints_ips[i]:
                    server = Trainer()
                    server.endpoint = "%s:%s" % (ip,
                                                 self.server_endpoints_port[i])
                    server.rank = server_rank
                    server_rank += 1
                    pod.servers.append(server)
            for j in range(len(self.worker_endpoints_ips)):
                if ip == self.worker_endpoints_ips[j]:
                    worker = Trainer()
                    worker.endpoint = "%s:%s" % (ip,
                                                 self.worker_endpoints_port[j])
                    worker.rank = worker_rank
                    worker_rank += 1
                    pod.workers.append(worker)
            for k in range(len(self.heter_worker_endpoints_ips)):
                if ip == self.heter_worker_endpoints_ips[k]:
                    heter_worker = Trainer()
                    heter_worker.endpoint = "%s:%s" % (
                        ip,
                        self.endpoints_dict["heter_worker_endpoints_port"][k])
                    heter_worker.rank = heter_worker_rank
                    heter_worker_rank += 1
                    pod.heter_workers.append(heter_worker)

            cluster.pods.append(pod)

        pod = cluster.pods[self.node_rank]
        self.gloo_rendezvous_dir = tempfile.mkdtemp()

        # 3. subproces start
        self.procs = []
        self.cmds = []
        self.log_fns = []

        self.start_pod_server(args, pod)
        self.start_pod_worker(args, pod)
        self.start_pod_heter_worker(args, pod)

        logger.info(
            "Please check servers, workers and heter_worker logs in {}/workerlog.*, {}/serverlog.* and {}/heterlog.*".
            format(args.log_dir, args.log_dir, args.log_dir))

        # only wait worker to finish here
        for i, proc in enumerate(self.procs):
            if i < len(pod.servers) and i > len(pod.servers) + len(pod.workers):
                continue
            self.procs[i].proc.wait()
            if len(self.log_fns) > 0:
                self.log_fns[i].close()
        print(
            "all workers exit, going to finish parameter server and heter_worker",
            file=sys.stderr)

        for i in range(
                len(pod.servers + pod.workers),
                len(pod.servers + pod.workers + pod.heter_workers)):
            if len(self.log_fns) > 0:
                self.log_fns[i].close()
            self.procs[i].proc.terminate()
        print("all heter worker are killed", file=sys.stderr)

        for i in range(len(pod.servers)):
            if len(self.log_fns) > 0:
                self.log_fns[i].close()
            self.procs[i].proc.terminate()
        print("all parameter server are killed", file=sys.stderr)

        if os.path.exists(self.gloo_rendezvous_dir):
            shutil.rmtree(self.gloo_rendezvous_dir)

    def start_pod_server(self, args, pod):
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)
        for idx, cur_server in enumerate(pod.servers):
            proc_env = {
                "PADDLE_PSERVERS_IP_PORT_LIST": self.server_endpoints,
                "PADDLE_TRAINER_ENDPOINTS": self.worker_endpoints,
                "PADDLE_HETER_TRAINER_IP_PORT_LIST":
                self.heter_worker_endpoints,
                "PADDLE_HETER_TRAINER_DEVICE": args.heter_worker_device,
                "PADDLE_PORT": cur_server.endpoint.split(":")[1],
                "TRAINING_ROLE": "PSERVER",
                "PADDLE_TRAINERS_NUM": str(self.worker_num),
                "POD_IP": cur_server.endpoint.split(":")[0],
                "PADDLE_WITH_GLOO": "1",
                "PADDLE_GLOO_RENDEZVOUS": "2",
                "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir
            }
            current_env.update(proc_env)

            cmd = [sys.executable, "-u", args.training_script
                   ] + args.training_script_args
            self.cmds.append(cmd)

            if idx == 0:
                logger.info(
                    "Local server start {} processes. First process distributed "
                    "environment info (Only For Debug): {}".format(
                        len(pod.servers),
                        pretty_print_envs(proc_env, ("Distributed Envs", "Value"
                                                     ))))

            if args.log_dir is not None:
                os.system("mkdir -p {}".format(args.log_dir))
                fn = open("%s/serverlog.%d" % (args.log_dir, idx), "w")
                self.log_fns.append(fn)
                proc = subprocess.Popen(
                    cmd, env=current_env, stdout=fn, stderr=fn)
            else:
                proc = subprocess.Popen(cmd, env=current_env)

            tp = TrainerProc()
            tp.proc = proc
            tp.rank = cur_server.rank
            tp.local_rank = idx
            tp.log_fn = fn
            tp.log_offset = fn.tell() if fn else None
            tp.cmd = cmd

            self.procs.append(tp)

    def start_pod_worker(self, args, pod):
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        heter_device_num = 0
        if args.heter_worker_device == "gpu":
            heter_device_num = fluid.core.get_cuda_device_count()
        elif args.heter_worker_device == "xpu":
            heter_device_num = fluid.core.get_xpu_device_count()

        for idx, cur_worker in enumerate(pod.workers):
            proc_env = {
                "PADDLE_PSERVERS_IP_PORT_LIST": self.server_endpoints,
                "PADDLE_TRAINER_ENDPOINTS": self.worker_endpoints,
                "PADDLE_TRAINERS_NUM": str(self.worker_num),
                "PADDLE_HETER_TRAINER_IP_PORT_LIST":
                self.heter_worker_endpoints,
                "PADDLE_HETER_TRAINER_DEVICE": args.heter_worker_device,
                "TRAINING_ROLE": "TRAINER",
                "PADDLE_TRAINER_ID": str(cur_worker.rank),
                "PADDLE_WITH_GLOO": "1",
                "PADDLE_GLOO_RENDEZVOUS": "2",
                "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                "FLAGS_selected_gpus": idx % heter_device_num,
                "FLAGS_selected_xpus": idx % heter_device_num,
                "CUDA_VISIBLE_DEVICES": idx % heter_device_num,
                "XPU_VISIBLE_DEVICES": idx % heter_device_num,
            }
            current_env.update(proc_env)

            cmd = [sys.executable, "-u", args.training_script
                   ] + args.training_script_args
            self.cmds.append(cmd)

            if idx == 0:
                logger.info(
                    "Local worker start {} processes. First process distributed "
                    "environment info (Only For Debug): {}".format(
                        len(pod.workers),
                        pretty_print_envs(proc_env, ("Distributed Envs", "Value"
                                                     ))))

            if args.log_dir is not None:
                os.system("mkdir -p {}".format(args.log_dir))
                fn = open("%s/workerlog.%d" % (args.log_dir, idx), "w")
                self.log_fns.append(fn)
                proc = subprocess.Popen(
                    cmd, env=current_env, stdout=fn, stderr=fn)
            else:
                proc = subprocess.Popen(cmd, env=current_env)

            tp = TrainerProc()
            tp.proc = proc
            tp.rank = cur_worker.rank
            tp.local_rank = idx
            tp.log_fn = fn
            tp.log_offset = fn.tell() if fn else None
            tp.cmd = cmd

            self.procs.append(tp)

    def start_pod_heter_worker(self, args, pod):
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        heter_device_num = 0
        if args.heter_worker_device == "gpu":
            heter_device_num = fluid.core.get_cuda_device_count()
        elif args.heter_worker_device == "xpu":
            heter_device_num = fluid.core.get_xpu_device_count()
        assert heter_device_num != 0

        for idx, cur_heter_worker in enumerate(pod.heter_workers):
            proc_env = {
                "PADDLE_PSERVERS_IP_PORT_LIST": self.server_endpoints,
                "PADDLE_TRAINER_ENDPOINTS": self.worker_endpoints,
                "PADDLE_HETER_TRAINER_IP_PORT_LIST":
                self.heter_worker_endpoints,
                "PADDLE_HETER_TRAINER_DEVICE": args.heter_worker_device,
                "PADDLE_PORT": cur_heter_worker.endpoint.split(":")[1],
                "TRAINING_ROLE": "HETER_TRAINER",
                "PADDLE_TRAINERS_NUM": str(self.worker_num),
                "POD_IP": cur_heter_worker.endpoint.split(":")[0],
                "PADDLE_WITH_GLOO": "1",
                "PADDLE_GLOO_RENDEZVOUS": "2",
                "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                "FLAGS_selected_gpus": idx % heter_device_num,
                "FLAGS_selected_xpus": idx % heter_device_num,
                "CUDA_VISIBLE_DEVICES": idx % heter_device_num,
                "XPU_VISIBLE_DEVICES": idx % heter_device_num,
            }
            current_env.update(proc_env)

            cmd = [sys.executable, "-u", args.training_script
                   ] + args.training_script_args
            self.cmds.append(cmd)

            if idx == 0:
                logger.info(
                    "Local server start {} processes. First process distributed "
                    "environment info (Only For Debug): {}".format(
                        len(pod.servers),
                        pretty_print_envs(proc_env, ("Distributed Envs", "Value"
                                                     ))))

            if args.log_dir is not None:
                os.system("mkdir -p {}".format(args.log_dir))
                fn = open("%s/heterlog.%d" % (args.log_dir, idx), "w")
                self.log_fns.append(fn)
                proc = subprocess.Popen(
                    cmd, env=current_env, stdout=fn, stderr=fn)
            else:
                proc = subprocess.Popen(cmd, env=current_env)

            tp = TrainerProc()
            tp.proc = proc
            tp.rank = cur_heter_worker.rank
            tp.local_rank = idx
            tp.log_fn = fn
            tp.log_offset = fn.tell() if fn else None
            tp.cmd = cmd

            self.procs.append(tp)
