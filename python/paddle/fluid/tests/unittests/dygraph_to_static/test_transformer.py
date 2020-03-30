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

import logging
import numpy as np
import time
import paddle.fluid as fluid

import unittest

import transformer_util as util

from transformer_dygraph_model import Transformer
from transformer_dygraph_model import CrossEntropyCriterion

trainer_count = 1
place = fluid.CPUPlace()
SEED = 10


def train_static(args, batch_generator):
    # set seed for CE
    train_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()
    random_seed = SEED
    if random_seed is not None:
        train_prog.random_seed = random_seed
        startup_prog.random_seed = random_seed

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            # define input and reader
            input_field_names = util.encoder_data_input_fields + \
                                util.decoder_data_input_fields[:-1] + util.label_data_input_fields
            input_descs = util.get_input_descs(args)
            input_slots = [{
                "name": name,
                "shape": input_descs[name][0],
                "dtype": input_descs[name][1]
            } for name in input_field_names]

            input_field = util.InputField(input_slots, build_pyreader=True)
            input_field.loader.set_batch_generator(batch_generator)

            # define model
            transformer = Transformer(
                args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
                args.n_layer, args.n_head, args.d_key, args.d_value,
                args.d_model, args.d_inner_hid, args.prepostprocess_dropout,
                args.attention_dropout, args.relu_dropout, args.preprocess_cmd,
                args.postprocess_cmd, args.weight_sharing, args.bos_idx,
                args.eos_idx)

            logits = transformer(*input_field.feed_list[:7])

            # define loss
            criterion = CrossEntropyCriterion(args.label_smooth_eps)
            lbl_word, lbl_weight = input_field.feed_list[7:]
            sum_cost, avg_cost, token_num = criterion(logits, lbl_word,
                                                      lbl_weight)

            # define optimizer
            learning_rate = fluid.layers.learning_rate_scheduler.noam_decay(
                args.d_model, args.warmup_steps, args.learning_rate)

            optimizer = fluid.optimizer.Adam(
                learning_rate=learning_rate,
                beta1=args.beta1,
                beta2=args.beta2,
                epsilon=float(args.eps))
            optimizer.minimize(avg_cost)

            # the best cross-entropy value with label smoothing
            loss_normalizer = -((1. - args.label_smooth_eps) * np.log(
                (1. - args.label_smooth_eps)) + args.label_smooth_eps * np.log(
                    args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))

    step_idx = 0
    total_batch_num = 0  # this is for benchmark
    # train loop
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    avg_loss = []
    for pass_id in range(args.epoch):
        batch_id = 0
        input_field.loader.start()
        while True:
            try:
                outs = exe.run(program=train_prog,
                               fetch_list=[sum_cost.name, token_num.name])

                if step_idx % args.print_step == 0:
                    sum_cost_val, token_num_val = np.array(outs[0]), np.array(
                        outs[1])
                    # sum the cost from multi-devices
                    total_sum_cost = sum_cost_val.sum()
                    total_token_num = token_num_val.sum()
                    total_avg_cost = total_sum_cost / total_token_num

                    avg_loss.append(total_avg_cost)

                    if step_idx == 0:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)])))
                        avg_batch_time = time.time()
                    else:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f, speed: %.2f step/s" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)]),
                             args.print_step / (time.time() - avg_batch_time)))
                        avg_batch_time = time.time()
                batch_id += 1
                step_idx += 1
                total_batch_num = total_batch_num + 1
                if step_idx == 10:
                    # if args.save_model:
                    #     model_path = os.path.join(args.save_model,
                    #                               "step_" + str(step_idx),
                    #                               "transformer")
                    #     fluid.save(train_prog, model_path)
                    break

            except fluid.core.EOFException:
                input_field.loader.reset()
                break
    return np.array(avg_loss)


def train_dygraph(args, batch_generator):

    with fluid.dygraph.guard(place):
        # set seed for CE
        random_seed = SEED
        if random_seed is not None:
            fluid.default_main_program().random_seed = random_seed
            fluid.default_startup_program().random_seed = random_seed

        # define data loader
        train_loader = fluid.io.DataLoader.from_generator(capacity=10)
        train_loader.set_batch_generator(batch_generator, places=place)

        # define model
        transformer = Transformer(
            args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
            args.n_layer, args.n_head, args.d_key, args.d_value, args.d_model,
            args.d_inner_hid, args.prepostprocess_dropout,
            args.attention_dropout, args.relu_dropout, args.preprocess_cmd,
            args.postprocess_cmd, args.weight_sharing, args.bos_idx,
            args.eos_idx)

        # define loss
        criterion = CrossEntropyCriterion(args.label_smooth_eps)

        # define optimizer
        learning_rate = fluid.layers.learning_rate_scheduler.noam_decay(
            args.d_model, args.warmup_steps, args.learning_rate)

        # define optimizer
        optimizer = fluid.optimizer.Adam(
            learning_rate=learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=float(args.eps),
            parameter_list=transformer.parameters())

        # the best cross-entropy value with label smoothing
        loss_normalizer = -(
            (1. - args.label_smooth_eps) * np.log(
                (1. - args.label_smooth_eps)) + args.label_smooth_eps *
            np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))

        ce_time = []
        ce_ppl = []
        avg_loss = []
        step_idx = 0
        # train loop
        for pass_id in range(args.epoch):
            pass_start_time = time.time()
            batch_id = 0
            for input_data in train_loader():
                (src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
                 trg_slf_attn_bias, trg_src_attn_bias, lbl_word,
                 lbl_weight) = input_data
                logits = transformer(src_word, src_pos, src_slf_attn_bias,
                                     trg_word, trg_pos, trg_slf_attn_bias,
                                     trg_src_attn_bias)

                sum_cost, avg_cost, token_num = criterion(logits, lbl_word,
                                                          lbl_weight)
                avg_cost.backward()

                optimizer.minimize(avg_cost)
                transformer.clear_gradients()

                if step_idx % args.print_step == 0:
                    total_avg_cost = avg_cost.numpy() * trainer_count

                    avg_loss.append(total_avg_cost[0])

                    if step_idx == 0:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)])))
                        avg_batch_time = time.time()
                    else:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f, speed: %.2f step/s" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)]),
                             args.print_step / (time.time() - avg_batch_time)))
                        ce_ppl.append(np.exp([min(total_avg_cost, 100)]))
                        avg_batch_time = time.time()

                batch_id += 1
                step_idx += 1
                if step_idx == 10:
                    # if args.save_model:
                    #     model_dir = os.path.join(args.save_model+'_dygraph',
                    #                              "step_" + str(step_idx))
                    #     if not os.path.exists(model_dir):
                    #         os.makedirs(model_dir)
                    #     fluid.save_dygraph(
                    #         transformer.state_dict(),
                    #         os.path.join(model_dir, "transformer"))
                    #     fluid.save_dygraph(
                    #         optimizer.state_dict(),
                    #         os.path.join(model_dir, "transformer"))
                    break

            time_consumed = time.time() - pass_start_time
            ce_time.append(time_consumed)

        return np.array(avg_loss)


class TestTransformer(unittest.TestCase):
    def prepare(self, mode='train'):
        args = util.ModelHyperParams()
        # define the data generator
        processor = util.DataProcessor(
            fpattern=args.training_file,
            src_vocab_fpath=args.src_vocab_fpath,
            trg_vocab_fpath=args.trg_vocab_fpath,
            token_delimiter=args.token_delimiter,
            use_token_batch=args.use_token_batch,
            batch_size=args.batch_size,
            device_count=trainer_count,
            pool_size=args.pool_size,
            sort_type=args.sort_type,
            shuffle=args.shuffle,
            shuffle_batch=args.shuffle_batch,
            start_mark=args.special_token[0],
            end_mark=args.special_token[1],
            unk_mark=args.special_token[2],
            max_length=args.max_length,
            n_head=args.n_head)
        batch_generator = processor.data_generator(phase=mode)

        args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = processor.get_vocab_summary()
        return args, batch_generator

    def test_train(self):
        args, batch_generator = self.prepare(mode='train')
        static_avg_loss = train_static(args, batch_generator)
        dygraph_avg_loss = train_dygraph(args, batch_generator)
        self.assertTrue(np.allclose(static_avg_loss, dygraph_avg_loss))


if __name__ == '__main__':
    unittest.main()
