paddle.fluid.Program ('paddle.fluid.framework.Program', ('document', '4f9e1829c89e0711355820e935d2b447'))

paddle.fluid.Program.__init__ (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.Program.block (ArgSpec(args=['self', 'index'], varargs=None, keywords=None, defaults=None), ('document', '28d066e432ceda86810b1e7deb8a4afa'))

paddle.fluid.Program.clone (ArgSpec(args=['self', 'for_test'], varargs=None, keywords=None, defaults=(False,)), ('document', '1e910e8c4186e8ff1afb62602f369033'))

paddle.fluid.Program.current_block (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '365e49ce9f346ac6d54265e29db447b5'))

paddle.fluid.Program.global_block (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', 'dd3f2b49147861d6ae48989a77482f05'))

paddle.fluid.Program.list_vars (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '757cf8d083dff9507676b17376ac5af1'))

paddle.fluid.Program.parse_from_string (ArgSpec(args=['binary_str'], varargs=None, keywords=None, defaults=None), ('document', '70e063a0a09d5a8ed322db0d5de9edb4'))

paddle.fluid.Program.to_string (ArgSpec(args=['self', 'throw_on_error', 'with_details'], varargs=None, keywords=None, defaults=(False,)), ('document', '6dfb00cd50eb515dcf2548a68ea94bfb'))

paddle.fluid.default_startup_program (ArgSpec(args=[], varargs=None, keywords=None, defaults=None), ('document', 'accb52b28228f8e93a26fabdc960f56c'))

paddle.fluid.default_main_program (ArgSpec(args=[], varargs=None, keywords=None, defaults=None), ('document', '853718df675e59aea7104f3d61bbf11d'))

paddle.fluid.program_guard (ArgSpec(args=['main_program', 'startup_program'], varargs=None, keywords=None, defaults=(None,)), ('document', '78fb5c7f70ef76bcf4a1862c3f6b8191'))

paddle.fluid.name_scope (ArgSpec(args=['prefix'], varargs=None, keywords=None, defaults=(None,)), ('document', '917d313881ff990de5fb18d98a9c7b42'))

paddle.fluid.cuda_places (ArgSpec(args=['device_ids'], varargs=None, keywords=None, defaults=(None,)), ('document', '1f2bb6ece651e44117652d2d7bedecf5'))

paddle.fluid.cpu_places (ArgSpec(args=['device_count'], varargs=None, keywords=None, defaults=(None,)), ('document', '956bab564ebc69ffd17195c08cc8ffa0'))

paddle.fluid.cuda_pinned_places (ArgSpec(args=['device_count'], varargs=None, keywords=None, defaults=(None,)), ('document', 'c2562241744aabe3fff1b59af22dd281'))

paddle.fluid.in_dygraph_mode (ArgSpec(args=[], varargs=None, keywords=None, defaults=None), ('document', 'df1f4d1ed7e1eefe04f6361efda6b75a'))

paddle.fluid.is_compiled_with_cuda (ArgSpec(args=[], varargs=None, keywords=None, defaults=None), ('document', '60c7f107a5050aeb58bb74eb175672b5'))

paddle.fluid.Variable ('paddle.fluid.framework.Variable', ('document', '65ff735c2b96673d7131f5ff6b0db40c'))

paddle.fluid.Variable.__init__ (ArgSpec(args=['self', 'block', 'type', 'name', 'shape', 'dtype', 'lod_level', 'capacity', 'persistable', 'error_clip', 'stop_gradient', 'is_data', 'need_check_feed'], varargs=None, keywords='kwargs', defaults=(VarType.LOD_TENSOR, None, None, None, None, None, None, None, False, False, False)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.Variable.astype (ArgSpec(args=['self', 'dtype'], varargs=None, keywords=None, defaults=None), ('document', '78541af4039262ed7ce3c447f8cc9cc1'))

paddle.fluid.Variable.backward (ArgSpec(args=['self', 'backward_strategy'], varargs=None, keywords=None, defaults=(None,)), ('document', 'cb928fa194da09694f4267f0a25268f1'))

paddle.fluid.Variable.clear_gradient (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '509a96d23c876fc5bfb10e1147e21d5f'))

paddle.fluid.Variable.detach (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '0730b2d310b014d9b0a903b2034757d7'))

paddle.fluid.Variable.gradient (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '86b246bfaf20f3058e91927abbcf9fb9'))

paddle.fluid.Variable.numpy (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '7536e8feb56d827875943e7f01d406fc'))

paddle.fluid.Variable.set_value (ArgSpec(args=['self', 'value'], varargs=None, keywords=None, defaults=None), ('document', 'c424b9e763ff51c38a6917f98026fe7d'))

paddle.fluid.Variable.to_string (ArgSpec(args=['self', 'throw_on_error', 'with_details'], varargs=None, keywords=None, defaults=(False,)), ('document', '31f359a2c074f26dc0ffff296fc3983f'))

paddle.fluid.load_op_library (ArgSpec(args=['lib_filename'], varargs=None, keywords=None, defaults=None), ('document', 'c009b2ea5fb6520f2d2f53aafec788e0'))

paddle.fluid.Executor ('paddle.fluid.executor.Executor', ('document', '34e8c1769313fbeff7817212dda6259e'))

paddle.fluid.Executor.__init__ (ArgSpec(args=['self', 'place'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.Executor.close (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '3a584496aa1343f36eebf3c46b323a74'))

paddle.fluid.Executor.infer_from_dataset (ArgSpec(args=['self', 'program', 'dataset', 'scope', 'thread', 'debug', 'fetch_list', 'fetch_info', 'print_period', 'fetch_handler'], varargs=None, keywords=None, defaults=(None, None, None, 0, False, None, None, 100, None)), ('document', '4ff256774ecaeee01c840a5fb5de8f7a'))

paddle.fluid.Executor.run (ArgSpec(args=['self', 'program', 'feed', 'fetch_list', 'feed_var_name', 'fetch_var_name', 'scope', 'return_numpy', 'use_program_cache'], varargs=None, keywords=None, defaults=(None, None, None, 'feed', 'fetch', None, True, False)), ('document', '4cfcd9c15b766a51b584cc46d38f1ad8'))

paddle.fluid.Executor.train_from_dataset (ArgSpec(args=['self', 'program', 'dataset', 'scope', 'thread', 'debug', 'fetch_list', 'fetch_info', 'print_period', 'fetch_handler'], varargs=None, keywords=None, defaults=(None, None, None, 0, False, None, None, 100, None)), ('document', '73024c79f46b4f14f1060edeaa4919c8'))

paddle.fluid.global_scope (ArgSpec(args=[], varargs=None, keywords=None, defaults=None), ('document', 'f65788d9ead293ada47551339df12203'))

paddle.fluid.scope_guard (ArgSpec(args=['scope'], varargs=None, keywords=None, defaults=None), ('document', '02fcfc1eda07c03a84ed62422366239c'))

paddle.fluid.DistributeTranspiler ('paddle.fluid.transpiler.distribute_transpiler.DistributeTranspiler', ('document', 'b2b19821c5dffcd11473d6a4eef089af'))

paddle.fluid.DistributeTranspiler.__init__ (ArgSpec(args=['self', 'config'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.DistributeTranspiler.get_pserver_program (ArgSpec(args=['self', 'endpoint'], varargs=None, keywords=None, defaults=None), ('document', 'b1951949c6d21698290aa8ac69afee32'))

paddle.fluid.DistributeTranspiler.get_pserver_programs (ArgSpec(args=['self', 'endpoint'], varargs=None, keywords=None, defaults=None), ('document', 'c89fc350f975ef827f5448d68af388cf'))

paddle.fluid.DistributeTranspiler.get_startup_program (ArgSpec(args=['self', 'endpoint', 'pserver_program', 'startup_program'], varargs=None, keywords=None, defaults=(None, None)), ('document', '90a40b80e0106f69262cc08b861c3e39'))

paddle.fluid.DistributeTranspiler.get_trainer_program (ArgSpec(args=['self', 'wait_port'], varargs=None, keywords=None, defaults=(True,)), ('document', '0e47f020304e2b824e87ff03475c17cd'))

paddle.fluid.DistributeTranspiler.transpile (ArgSpec(args=['self', 'trainer_id', 'program', 'pservers', 'trainers', 'sync_mode', 'startup_program', 'current_endpoint'], varargs=None, keywords=None, defaults=(None, '127.0.0.1:6174', 1, True, None, '127.0.0.1:6174')), ('document', '418c7e8b268e9be4104f2809e654c2f7'))

paddle.fluid.memory_optimize (ArgSpec(args=['input_program', 'skip_opt_set', 'print_log', 'level', 'skip_grads'], varargs=None, keywords=None, defaults=(None, False, 0, True)), ('document', '2be29dc8ecdec9baa7728fb0c7f80e24'))

paddle.fluid.release_memory (ArgSpec(args=['input_program', 'skip_opt_set'], varargs=None, keywords=None, defaults=(None,)), ('document', '2be29dc8ecdec9baa7728fb0c7f80e24'))

paddle.fluid.DistributeTranspilerConfig ('paddle.fluid.transpiler.distribute_transpiler.DistributeTranspilerConfig', ('document', 'beac6f89fe97eb8c66a25de5a09c56d2'))

paddle.fluid.DistributeTranspilerConfig.__init__ (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.ParallelExecutor ('paddle.fluid.parallel_executor.ParallelExecutor', ('document', '2b4d2e859f2e0c6161f4fed995f7956d'))

paddle.fluid.ParallelExecutor.__init__ (ArgSpec(args=['self', 'use_cuda', 'loss_name', 'main_program', 'share_vars_from', 'exec_strategy', 'build_strategy', 'num_trainers', 'trainer_id', 'scope'], varargs=None, keywords=None, defaults=(None, None, None, None, None, 1, 0, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.ParallelExecutor.drop_local_exe_scopes (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '77c739744ea5708b80fb1b37cc89db40'))

paddle.fluid.ParallelExecutor.run (ArgSpec(args=['self', 'fetch_list', 'feed', 'feed_dict', 'return_numpy'], varargs=None, keywords=None, defaults=(None, None, True)), ('document', '0af092676e5b1320bb4232396154ce4b'))

paddle.fluid.create_lod_tensor (ArgSpec(args=['data', 'recursive_seq_lens', 'place'], varargs=None, keywords=None, defaults=None), ('document', '0627369b86ff974f433f7078d1e78349'))

paddle.fluid.create_random_int_lodtensor (ArgSpec(args=['recursive_seq_lens', 'base_shape', 'place', 'low', 'high'], varargs=None, keywords=None, defaults=None), ('document', '4829bd8c4a4f1b19438500def321cb65'))

paddle.fluid.DataFeedDesc ('paddle.fluid.data_feed_desc.DataFeedDesc', ('document', '43877a0d9357db94d3dbc7359cbe8c73'))

paddle.fluid.DataFeedDesc.__init__ (ArgSpec(args=['self', 'proto_file'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.DataFeedDesc.desc (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '9c6615854b61caa5f0d3e6ccc5e51338'))

paddle.fluid.DataFeedDesc.set_batch_size (ArgSpec(args=['self', 'batch_size'], varargs=None, keywords=None, defaults=None), ('document', 'a34790bff4a2891713ddd644db56418d'))

paddle.fluid.DataFeedDesc.set_dense_slots (ArgSpec(args=['self', 'dense_slots_name'], varargs=None, keywords=None, defaults=None), ('document', 'fdd07ce63e72bed57f2c0db5bec5720f'))

paddle.fluid.DataFeedDesc.set_use_slots (ArgSpec(args=['self', 'use_slots_name'], varargs=None, keywords=None, defaults=None), ('document', 'c23a79dfa04edd014b477bd4b183da06'))

paddle.fluid.CompiledProgram ('paddle.fluid.compiler.CompiledProgram', ('document', '598d294107d44d7620bce76527a92c37'))

paddle.fluid.CompiledProgram.__init__ (ArgSpec(args=['self', 'program_or_graph', 'build_strategy'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.CompiledProgram.with_data_parallel (ArgSpec(args=['self', 'loss_name', 'build_strategy', 'exec_strategy', 'share_vars_from', 'places'], varargs=None, keywords=None, defaults=(None, None, None, None, None)), ('document', '1c7c6171bbf6d77f2fce0166aa0ec43b'))

paddle.fluid.ExecutionStrategy ('paddle.fluid.core_avx.ExecutionStrategy', ('document', '535ce28c4671176386e3cd283a764084'))

paddle.fluid.ExecutionStrategy.__init__ __init__(self: paddle.fluid.core_avx.ParallelExecutor.ExecutionStrategy) -> None

paddle.fluid.BuildStrategy ('paddle.fluid.core_avx.BuildStrategy', ('document', 'eec64b9b7cba58b0a63687b4c34ffe56'))

paddle.fluid.BuildStrategy.GradientScaleStrategy ('paddle.fluid.core_avx.GradientScaleStrategy', ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.BuildStrategy.GradientScaleStrategy.__init__ __init__(self: paddle.fluid.core_avx.ParallelExecutor.BuildStrategy.GradientScaleStrategy, arg0: int) -> None

paddle.fluid.BuildStrategy.ReduceStrategy ('paddle.fluid.core_avx.ReduceStrategy', ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.BuildStrategy.ReduceStrategy.__init__ __init__(self: paddle.fluid.core_avx.ParallelExecutor.BuildStrategy.ReduceStrategy, arg0: int) -> None

paddle.fluid.BuildStrategy.__init__ __init__(self: paddle.fluid.core_avx.ParallelExecutor.BuildStrategy) -> None

paddle.fluid.gradients (ArgSpec(args=['targets', 'inputs', 'target_gradients', 'no_grad_set'], varargs=None, keywords=None, defaults=(None, None)), ('document', 'e2097e1e0ed84ae44951437bfe269a1b'))

paddle.fluid.io.save_vars (ArgSpec(args=['executor', 'dirname', 'main_program', 'vars', 'predicate', 'filename'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', '869104f47e6fd21d897c3fcc426aa942'))

paddle.fluid.io.save_params (ArgSpec(args=['executor', 'dirname', 'main_program', 'filename'], varargs=None, keywords=None, defaults=(None, None)), ('document', '046d7c43d67e08c2660bb3bd7e081015'))

paddle.fluid.io.save_persistables (ArgSpec(args=['executor', 'dirname', 'main_program', 'filename'], varargs=None, keywords=None, defaults=(None, None)), ('document', 'ffcee38044975c29f2ab2fec0576f963'))

paddle.fluid.io.load_vars (ArgSpec(args=['executor', 'dirname', 'main_program', 'vars', 'predicate', 'filename'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', '1bb9454cf09d71f190bb51550c5a3ac9'))

paddle.fluid.io.load_params (ArgSpec(args=['executor', 'dirname', 'main_program', 'filename'], varargs=None, keywords=None, defaults=(None, None)), ('document', '116a9ed169e7ff0226faccff3c29364c'))

paddle.fluid.io.load_persistables (ArgSpec(args=['executor', 'dirname', 'main_program', 'filename'], varargs=None, keywords=None, defaults=(None, None)), ('document', 'cfa84ef7c5435625bff4cc132cb8a0e3'))

paddle.fluid.io.save_inference_model (ArgSpec(args=['dirname', 'feeded_var_names', 'target_vars', 'executor', 'main_program', 'model_filename', 'params_filename', 'export_for_deployment', 'program_only'], varargs=None, keywords=None, defaults=(None, None, None, True, False)), ('document', 'fc82bfd137a9b1ab8ebd1651bd35b6e5'))

paddle.fluid.io.load_inference_model (ArgSpec(args=['dirname', 'executor', 'model_filename', 'params_filename', 'pserver_endpoints'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', '2f54d7c206b62f8c10f4f9d78c731cfd'))

paddle.fluid.io.batch (ArgSpec(args=['reader', 'batch_size', 'drop_last'], varargs=None, keywords=None, defaults=(False,)), ('document', 'cf2869b408b39cadadd95206b4e03b39'))

paddle.fluid.io.PyReader ('paddle.fluid.reader.PyReader', ('document', 'b03399246f69cd6fc03b43e87af8bd4e'))

paddle.fluid.io.PyReader.__init__ (ArgSpec(args=['self', 'feed_list', 'capacity', 'use_double_buffer', 'iterable', 'return_list'], varargs=None, keywords=None, defaults=(None, None, True, True, False)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.io.PyReader.decorate_batch_generator (ArgSpec(args=['self', 'reader', 'places'], varargs=None, keywords=None, defaults=(None,)), ('document', '4364e836e3cb8ab5e68e411b763c50c7'))

paddle.fluid.io.PyReader.decorate_sample_generator (ArgSpec(args=['self', 'sample_generator', 'batch_size', 'drop_last', 'places'], varargs=None, keywords=None, defaults=(True, None)), ('document', 'efa4c8b90fe6d99dcbda637b70351bb1'))

paddle.fluid.io.PyReader.decorate_sample_list_generator (ArgSpec(args=['self', 'reader', 'places'], varargs=None, keywords=None, defaults=(None,)), ('document', '6c11980092720de304863de98074a64a'))

paddle.fluid.io.PyReader.next (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '08b2fd1463f3ea99d79d17303988349b'))

paddle.fluid.io.PyReader.reset (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '7432197701fdaab1848063860dc0b97e'))

paddle.fluid.io.PyReader.start (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', 'a0983fb21a0a51e6a31716009fe9a9c1'))

paddle.fluid.io.DataLoader ('paddle.fluid.reader.DataLoader', ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.io.DataLoader.__init__ 

paddle.fluid.io.DataLoader.from_dataset (ArgSpec(args=['dataset', 'places', 'drop_last'], varargs=None, keywords=None, defaults=(True,)), ('document', '58e8bffa033f26b00b256c8bb1daff11'))

paddle.fluid.io.DataLoader.from_generator (ArgSpec(args=['feed_list', 'capacity', 'use_double_buffer', 'iterable', 'return_list'], varargs=None, keywords=None, defaults=(None, None, True, True, False)), ('document', '8034bdb488fa18d60c4ffb0ba9658337'))

paddle.fluid.io.cache (ArgSpec(args=['reader'], varargs=None, keywords=None, defaults=None), ('document', '1676886070eb607cb608f7ba47be0d3c'))

paddle.fluid.io.map_readers (ArgSpec(args=['func'], varargs='readers', keywords=None, defaults=None), ('document', '77cbadb09df588e21e5cc0819b69c87d'))

paddle.fluid.io.buffered (ArgSpec(args=['reader', 'size'], varargs=None, keywords=None, defaults=None), ('document', '0d6186f109feceb99f60ec50a0a624cb'))

paddle.fluid.io.compose (ArgSpec(args=[], varargs='readers', keywords='kwargs', defaults=None), ('document', '884291104e1c3f37f33aae44b7deeb0d'))

paddle.fluid.io.chain (ArgSpec(args=[], varargs='readers', keywords=None, defaults=None), ('document', 'd22c34e379a53901ae67a6bca7f4def4'))

paddle.fluid.io.shuffle (ArgSpec(args=['reader', 'buf_size'], varargs=None, keywords=None, defaults=None), ('document', 'e42ea6fee23ce26b23cb142cd1d6522d'))

paddle.fluid.io.firstn (ArgSpec(args=['reader', 'n'], varargs=None, keywords=None, defaults=None), ('document', 'c5bb8f7dd4f917f1569a368aab5b8aad'))

paddle.fluid.io.xmap_readers (ArgSpec(args=['mapper', 'reader', 'process_num', 'buffer_size', 'order'], varargs=None, keywords=None, defaults=(False,)), ('document', '9c804a42f8a4dbaa76b3c98e0ab7f796'))

paddle.fluid.io.PipeReader ('paddle.reader.decorator.PipeReader', ('document', 'd3c250618f98c1a5fb646f869016a98e'))

paddle.fluid.io.PipeReader.__init__ (ArgSpec(args=['self', 'command', 'bufsize', 'file_type'], varargs=None, keywords=None, defaults=(8192, 'plain')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.io.PipeReader.get_line (ArgSpec(args=['self', 'cut_lines', 'line_break'], varargs=None, keywords=None, defaults=(True, '\n')), ('document', '9621ae612e595b6c34eb3bb5f3eb1a45'))

paddle.fluid.io.multiprocess_reader (ArgSpec(args=['readers', 'use_pipe', 'queue_size'], varargs=None, keywords=None, defaults=(True, 1000)), ('document', '7d8b3a96e592107c893d5d51ce968ba0'))

paddle.fluid.io.Fake ('paddle.reader.decorator.Fake', ('document', '0d8f4847b99bed6d456ade0d903202e1'))

paddle.fluid.io.Fake.__init__ (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.initializer.ConstantInitializer ('paddle.fluid.initializer.ConstantInitializer', ('document', '798f1fd87cbe9798d001ffb6e616415d'))

paddle.fluid.initializer.ConstantInitializer.__init__ (ArgSpec(args=['self', 'value', 'force_cpu'], varargs=None, keywords=None, defaults=(0.0, False)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.initializer.UniformInitializer ('paddle.fluid.initializer.UniformInitializer', ('document', '587b7035cd1d56f76f2ded617b92521d'))

paddle.fluid.initializer.UniformInitializer.__init__ (ArgSpec(args=['self', 'low', 'high', 'seed', 'diag_num', 'diag_step', 'diag_val'], varargs=None, keywords=None, defaults=(-1.0, 1.0, 0, 0, 0, 1.0)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.initializer.NormalInitializer ('paddle.fluid.initializer.NormalInitializer', ('document', '279a0d89bf01138fbf4c4ba14f22099b'))

paddle.fluid.initializer.NormalInitializer.__init__ (ArgSpec(args=['self', 'loc', 'scale', 'seed'], varargs=None, keywords=None, defaults=(0.0, 1.0, 0)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.initializer.TruncatedNormalInitializer ('paddle.fluid.initializer.TruncatedNormalInitializer', ('document', 'b8e90aad6ee5687cb5f2b6fd404370d1'))

paddle.fluid.initializer.TruncatedNormalInitializer.__init__ (ArgSpec(args=['self', 'loc', 'scale', 'seed'], varargs=None, keywords=None, defaults=(0.0, 1.0, 0)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.initializer.XavierInitializer ('paddle.fluid.initializer.XavierInitializer', ('document', '3d5676f1a5414aa0c815d793a795ccb3'))

paddle.fluid.initializer.XavierInitializer.__init__ (ArgSpec(args=['self', 'uniform', 'fan_in', 'fan_out', 'seed'], varargs=None, keywords=None, defaults=(True, None, None, 0)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.initializer.BilinearInitializer ('paddle.fluid.initializer.BilinearInitializer', ('document', '8a40b54fe33c19c3edcf6624ffae5d03'))

paddle.fluid.initializer.BilinearInitializer.__init__ (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', 'd389912dc079cbef432335a00017cec0'))

paddle.fluid.initializer.MSRAInitializer ('paddle.fluid.initializer.MSRAInitializer', ('document', 'b99e0ee95e2fd02640cb4b08a7ae80cc'))

paddle.fluid.initializer.MSRAInitializer.__init__ (ArgSpec(args=['self', 'uniform', 'fan_in', 'seed'], varargs=None, keywords=None, defaults=(True, None, 0)), ('document', '53c757bed9345f2ad3361902531e7cf5'))

paddle.fluid.initializer.force_init_on_cpu (ArgSpec(args=[], varargs=None, keywords=None, defaults=None), ('document', '5f55553caf939d270c7fe8dc418084b2'))

paddle.fluid.initializer.init_on_cpu (ArgSpec(args=[], varargs=None, keywords=None, defaults=None), ('document', 'eaa04fd68661a3af59abd0e19b3b6eda'))

paddle.fluid.initializer.NumpyArrayInitializer ('paddle.fluid.initializer.NumpyArrayInitializer', ('document', '7b0c371a233f9eb6feab75bbef8a74cc'))

paddle.fluid.initializer.NumpyArrayInitializer.__init__ (ArgSpec(args=['self', 'value'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.embedding (ArgSpec(args=['input', 'size', 'is_sparse', 'is_distributed', 'padding_idx', 'param_attr', 'dtype'], varargs=None, keywords=None, defaults=(False, False, None, None, 'float32')), ('document', 'd4ac047e0d5e6b7b1c5ff6ef7d7cfff5'))

paddle.fluid.one_hot (ArgSpec(args=['input', 'depth', 'allow_out_of_range'], varargs=None, keywords=None, defaults=(False,)), ('document', 'eef66730acc806088f9e8ba90252bda1'))

paddle.fluid.layers.fc (ArgSpec(args=['input', 'size', 'num_flatten_dims', 'param_attr', 'bias_attr', 'act', 'name'], varargs=None, keywords=None, defaults=(1, None, None, None, None)), ('document', 'e28421f1253a3545d9bfe81a8028ea68'))

paddle.fluid.layers.center_loss (ArgSpec(args=['input', 'label', 'num_classes', 'alpha', 'param_attr', 'update_center'], varargs=None, keywords=None, defaults=(True,)), ('document', '18112442f55b5862bbec8feee841c905'))

paddle.fluid.layers.embedding (ArgSpec(args=['input', 'size', 'is_sparse', 'is_distributed', 'padding_idx', 'param_attr', 'dtype'], varargs=None, keywords=None, defaults=(False, False, None, None, 'float32')), ('document', 'd8e405486a1e4e189b51d6ee28d67b1e'))

paddle.fluid.layers.dynamic_lstm (ArgSpec(args=['input', 'size', 'h_0', 'c_0', 'param_attr', 'bias_attr', 'use_peepholes', 'is_reverse', 'gate_activation', 'cell_activation', 'candidate_activation', 'dtype', 'name'], varargs=None, keywords=None, defaults=(None, None, None, None, True, False, 'sigmoid', 'tanh', 'tanh', 'float32', None)), ('document', '6d3ee14da70adfa36d85c40b18716ef2'))

paddle.fluid.layers.dynamic_lstmp (ArgSpec(args=['input', 'size', 'proj_size', 'param_attr', 'bias_attr', 'use_peepholes', 'is_reverse', 'gate_activation', 'cell_activation', 'candidate_activation', 'proj_activation', 'dtype', 'name', 'h_0', 'c_0', 'cell_clip', 'proj_clip'], varargs=None, keywords=None, defaults=(None, None, True, False, 'sigmoid', 'tanh', 'tanh', 'tanh', 'float32', None, None, None, None, None)), ('document', 'c37d51aad655c8a9f9b045c64717320a'))

paddle.fluid.layers.dynamic_gru (ArgSpec(args=['input', 'size', 'param_attr', 'bias_attr', 'is_reverse', 'gate_activation', 'candidate_activation', 'h_0', 'origin_mode'], varargs=None, keywords=None, defaults=(None, None, False, 'sigmoid', 'tanh', None, False)), ('document', '83617c165827e030636c80486d5de6f3'))

paddle.fluid.layers.gru_unit (ArgSpec(args=['input', 'hidden', 'size', 'param_attr', 'bias_attr', 'activation', 'gate_activation', 'origin_mode'], varargs=None, keywords=None, defaults=(None, None, 'tanh', 'sigmoid', False)), ('document', '33974b9bfa69f2f1eb85e6f956dff04e'))

paddle.fluid.layers.linear_chain_crf (ArgSpec(args=['input', 'label', 'param_attr', 'length'], varargs=None, keywords=None, defaults=(None, None)), ('document', 'b28bdb43160e9667be2a3457d19d9f5b'))

paddle.fluid.layers.crf_decoding (ArgSpec(args=['input', 'param_attr', 'label', 'length'], varargs=None, keywords=None, defaults=(None, None)), ('document', '933b7e268c4ffa3d5c3ef953a5ee9f0b'))

paddle.fluid.layers.cos_sim (ArgSpec(args=['X', 'Y'], varargs=None, keywords=None, defaults=None), ('document', '07bb25484c98d529fbe67338422724af'))

paddle.fluid.layers.cross_entropy (ArgSpec(args=['input', 'label', 'soft_label', 'ignore_index'], varargs=None, keywords=None, defaults=(False, -100)), ('document', '789a141e97fd0b37241f630935936d08'))

paddle.fluid.layers.bpr_loss (ArgSpec(args=['input', 'label', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'ae57e6e5136dade436f0df1f11770afa'))

paddle.fluid.layers.square_error_cost (ArgSpec(args=['input', 'label'], varargs=None, keywords=None, defaults=None), ('document', 'bbb9e708bab250359864fefbdf48e9d9'))

paddle.fluid.layers.chunk_eval (ArgSpec(args=['input', 'label', 'chunk_scheme', 'num_chunk_types', 'excluded_chunk_types', 'seq_length'], varargs=None, keywords=None, defaults=(None, None)), ('document', 'b02844e0ad4bd713c5fe6802aa13219c'))

paddle.fluid.layers.sequence_conv (ArgSpec(args=['input', 'num_filters', 'filter_size', 'filter_stride', 'padding', 'padding_start', 'bias_attr', 'param_attr', 'act', 'name'], varargs=None, keywords=None, defaults=(3, 1, True, None, None, None, None, None)), ('document', 'ebddcc5a1073ef065d22b4673e36b1d2'))

paddle.fluid.layers.conv2d (ArgSpec(args=['input', 'num_filters', 'filter_size', 'stride', 'padding', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act', 'name', 'data_format'], varargs=None, keywords=None, defaults=(1, 0, 1, None, None, None, True, None, None, 'NCHW')), ('document', 'b9be3712a46e196c7329eed52ed91d05'))

paddle.fluid.layers.conv3d (ArgSpec(args=['input', 'num_filters', 'filter_size', 'stride', 'padding', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act', 'name', 'data_format'], varargs=None, keywords=None, defaults=(1, 0, 1, None, None, None, True, None, None, 'NCDHW')), ('document', 'a7e4573745c40b8b1d726709f209b6e4'))

paddle.fluid.layers.sequence_pool (ArgSpec(args=['input', 'pool_type', 'is_test', 'pad_value'], varargs=None, keywords=None, defaults=(False, 0.0)), ('document', '5a709f7ef3fdb8fc819d09dc4fbada9a'))

paddle.fluid.layers.sequence_softmax (ArgSpec(args=['input', 'use_cudnn', 'name'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'eaa9d0bbd3d4e017c8bc4ecdac483711'))

paddle.fluid.layers.softmax (ArgSpec(args=['input', 'use_cudnn', 'name', 'axis'], varargs=None, keywords=None, defaults=(False, None, -1)), ('document', '7ccaea1b93fe4f7387a6036692986c6b'))

paddle.fluid.layers.pool2d (ArgSpec(args=['input', 'pool_size', 'pool_type', 'pool_stride', 'pool_padding', 'global_pooling', 'use_cudnn', 'ceil_mode', 'name', 'exclusive', 'data_format'], varargs=None, keywords=None, defaults=(-1, 'max', 1, 0, False, True, False, None, True, 'NCHW')), ('document', '630cae697d46b4b575b15d56cf8be25a'))

paddle.fluid.layers.pool3d (ArgSpec(args=['input', 'pool_size', 'pool_type', 'pool_stride', 'pool_padding', 'global_pooling', 'use_cudnn', 'ceil_mode', 'name', 'exclusive', 'data_format'], varargs=None, keywords=None, defaults=(-1, 'max', 1, 0, False, True, False, None, True, 'NCDHW')), ('document', 'db0035a3132b1dfb12e53c57591fb9f6'))

paddle.fluid.layers.adaptive_pool2d (ArgSpec(args=['input', 'pool_size', 'pool_type', 'require_index', 'name'], varargs=None, keywords=None, defaults=('max', False, None)), ('document', '52343203de40afe29607397e13aaf0d2'))

paddle.fluid.layers.adaptive_pool3d (ArgSpec(args=['input', 'pool_size', 'pool_type', 'require_index', 'name'], varargs=None, keywords=None, defaults=('max', False, None)), ('document', '55db6ae7275fb9678a6814aebab81a9c'))

paddle.fluid.layers.batch_norm (ArgSpec(args=['input', 'act', 'is_test', 'momentum', 'epsilon', 'param_attr', 'bias_attr', 'data_layout', 'in_place', 'name', 'moving_mean_name', 'moving_variance_name', 'do_model_average_for_mean_and_var', 'fuse_with_relu', 'use_global_stats'], varargs=None, keywords=None, defaults=(None, False, 0.9, 1e-05, None, None, 'NCHW', False, None, None, None, False, False, False)), ('document', 'b88a2a2d5de3e6d845d134782fb54857'))

paddle.fluid.layers.instance_norm (ArgSpec(args=['input', 'epsilon', 'param_attr', 'bias_attr', 'name'], varargs=None, keywords=None, defaults=(1e-05, None, None, None)), ('document', '5e2d18e85599ede7e71b06ed64d0f69e'))

paddle.fluid.layers.data_norm (ArgSpec(args=['input', 'act', 'epsilon', 'param_attr', 'data_layout', 'in_place', 'name', 'moving_mean_name', 'moving_variance_name', 'do_model_average_for_mean_and_var'], varargs=None, keywords=None, defaults=(None, 1e-05, None, 'NCHW', False, None, None, None, False)), ('document', '5ba4cdb4ea5c03382da545335ffc05b7'))

paddle.fluid.layers.beam_search_decode (ArgSpec(args=['ids', 'scores', 'beam_size', 'end_id', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '83e08f21af41ac8bac37aeab1f86fdd0'))

paddle.fluid.layers.conv2d_transpose (ArgSpec(args=['input', 'num_filters', 'output_size', 'filter_size', 'padding', 'stride', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act', 'name', 'data_format'], varargs=None, keywords=None, defaults=(None, None, 0, 1, 1, None, None, None, True, None, None, 'NCHW')), ('document', '0ca6c549ac2b63096bdc7832a08b4431'))

paddle.fluid.layers.conv3d_transpose (ArgSpec(args=['input', 'num_filters', 'output_size', 'filter_size', 'padding', 'stride', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act', 'name', 'data_format'], varargs=None, keywords=None, defaults=(None, None, 0, 1, 1, None, None, None, True, None, None, 'NCDHW')), ('document', '709d7ca3a94f52a253d15b06aafb1bd0'))

paddle.fluid.layers.sequence_expand (ArgSpec(args=['x', 'y', 'ref_level', 'name'], varargs=None, keywords=None, defaults=(-1, None)), ('document', '10e122eb755c2bd1f78ef2332b28f1a0'))

paddle.fluid.layers.sequence_expand_as (ArgSpec(args=['x', 'y', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '858c432e7cbd8bb952cc2eb555457d50'))

paddle.fluid.layers.sequence_pad (ArgSpec(args=['x', 'pad_value', 'maxlen', 'name'], varargs=None, keywords=None, defaults=(None, None)), ('document', 'df08b9c499ab3a90f95d08ab5b6c6c62'))

paddle.fluid.layers.sequence_unpad (ArgSpec(args=['x', 'length', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'e478180d5bc010a84f35af958cafa62c'))

paddle.fluid.layers.lstm_unit (ArgSpec(args=['x_t', 'hidden_t_prev', 'cell_t_prev', 'forget_bias', 'param_attr', 'bias_attr', 'name'], varargs=None, keywords=None, defaults=(0.0, None, None, None)), ('document', 'fe126c58e4339410e875ab1eba246d21'))

paddle.fluid.layers.reduce_sum (ArgSpec(args=['input', 'dim', 'keep_dim', 'name'], varargs=None, keywords=None, defaults=(None, False, None)), ('document', 'dd5f06fb7cf39ca06cbab4abd03e6893'))

paddle.fluid.layers.reduce_mean (ArgSpec(args=['input', 'dim', 'keep_dim', 'name'], varargs=None, keywords=None, defaults=(None, False, None)), ('document', 'a3024789eba11a70c2ef27c358173400'))

paddle.fluid.layers.reduce_max (ArgSpec(args=['input', 'dim', 'keep_dim', 'name'], varargs=None, keywords=None, defaults=(None, False, None)), ('document', '10023caec4d7f78c3b901f023a1feaa7'))

paddle.fluid.layers.reduce_min (ArgSpec(args=['input', 'dim', 'keep_dim', 'name'], varargs=None, keywords=None, defaults=(None, False, None)), ('document', '1a1c91625ce3c32646f69ca10d4d1da7'))

paddle.fluid.layers.reduce_prod (ArgSpec(args=['input', 'dim', 'keep_dim', 'name'], varargs=None, keywords=None, defaults=(None, False, None)), ('document', 'b386471f0476c80c61d8c8672278063d'))

paddle.fluid.layers.reduce_all (ArgSpec(args=['input', 'dim', 'keep_dim', 'name'], varargs=None, keywords=None, defaults=(None, False, None)), ('document', '8ab17ab51f68a6e76302b27f928cedf3'))

paddle.fluid.layers.reduce_any (ArgSpec(args=['input', 'dim', 'keep_dim', 'name'], varargs=None, keywords=None, defaults=(None, False, None)), ('document', '0483ac3b7a99e879ccde583ae8d7a60d'))

paddle.fluid.layers.sequence_first_step (ArgSpec(args=['input'], varargs=None, keywords=None, defaults=None), ('document', '227a75392ae194de0504f5c6812dade9'))

paddle.fluid.layers.sequence_last_step (ArgSpec(args=['input'], varargs=None, keywords=None, defaults=None), ('document', '34372f58331247749e8b0a1663cf233b'))

paddle.fluid.layers.sequence_slice (ArgSpec(args=['input', 'offset', 'length', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '39fbc5437be389f6c0c769f82fc1fba2'))

paddle.fluid.layers.dropout (ArgSpec(args=['x', 'dropout_prob', 'is_test', 'seed', 'name', 'dropout_implementation'], varargs=None, keywords=None, defaults=(False, None, None, 'downgrade_in_infer')), ('document', '4fd396b6cf16bb4ef2a56d695d0ad941'))

paddle.fluid.layers.split (ArgSpec(args=['input', 'num_or_sections', 'dim', 'name'], varargs=None, keywords=None, defaults=(-1, None)), ('document', '78cf3a7323d1a7697658242e13f63759'))

paddle.fluid.layers.ctc_greedy_decoder (ArgSpec(args=['input', 'blank', 'input_length', 'padding_value', 'name'], varargs=None, keywords=None, defaults=(None, 0, None)), ('document', '9abb7bb8d267e017620a39a146dc47ea'))

paddle.fluid.layers.edit_distance (ArgSpec(args=['input', 'label', 'normalized', 'ignored_tokens', 'input_length', 'label_length'], varargs=None, keywords=None, defaults=(True, None, None, None)), ('document', '77cbfb28cd2fc589f589c7013c5086cd'))

paddle.fluid.layers.l2_normalize (ArgSpec(args=['x', 'axis', 'epsilon', 'name'], varargs=None, keywords=None, defaults=(1e-12, None)), ('document', 'c1df110ea65998984f564c5c10abc54a'))

paddle.fluid.layers.matmul (ArgSpec(args=['x', 'y', 'transpose_x', 'transpose_y', 'alpha', 'name'], varargs=None, keywords=None, defaults=(False, False, 1.0, None)), ('document', '3720b4a386585094435993deb028b592'))

paddle.fluid.layers.topk (ArgSpec(args=['input', 'k', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'e50940f3ce5a08cc477b72f517491bf3'))

paddle.fluid.layers.warpctc (ArgSpec(args=['input', 'label', 'blank', 'norm_by_times', 'input_length', 'label_length'], varargs=None, keywords=None, defaults=(0, False, None, None)), ('document', 'a5be881ada816e47ea7a6ee4396da357'))

paddle.fluid.layers.sequence_reshape (ArgSpec(args=['input', 'new_dim'], varargs=None, keywords=None, defaults=None), ('document', 'eeb1591cfc854c6ffdac77b376313c44'))

paddle.fluid.layers.transpose (ArgSpec(args=['x', 'perm', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '8e72db173d4c082e27cb11f31d8c9bfa'))

paddle.fluid.layers.im2sequence (ArgSpec(args=['input', 'filter_size', 'stride', 'padding', 'input_image_size', 'out_stride', 'name'], varargs=None, keywords=None, defaults=(1, 1, 0, None, 1, None)), ('document', '33134416fc27dd65a767e5f15116ee16'))

paddle.fluid.layers.nce (ArgSpec(args=['input', 'label', 'num_total_classes', 'sample_weight', 'param_attr', 'bias_attr', 'num_neg_samples', 'name', 'sampler', 'custom_dist', 'seed', 'is_sparse'], varargs=None, keywords=None, defaults=(None, None, None, None, None, 'uniform', None, 0, False)), ('document', '83d4ca6dfb957912807f535756e76992'))

paddle.fluid.layers.sampled_softmax_with_cross_entropy (ArgSpec(args=['logits', 'label', 'num_samples', 'num_true', 'remove_accidental_hits', 'use_customized_samples', 'customized_samples', 'customized_probabilities', 'seed'], varargs=None, keywords=None, defaults=(1, True, False, None, None, 0)), ('document', 'd4435a63d34203339831ee6a86ef9242'))

paddle.fluid.layers.hsigmoid (ArgSpec(args=['input', 'label', 'num_classes', 'param_attr', 'bias_attr', 'name', 'path_table', 'path_code', 'is_custom', 'is_sparse'], varargs=None, keywords=None, defaults=(None, None, None, None, None, False, False)), ('document', 'b83e7dfa81059b39bb137922dc914f50'))

paddle.fluid.layers.beam_search (ArgSpec(args=['pre_ids', 'pre_scores', 'ids', 'scores', 'beam_size', 'end_id', 'level', 'is_accumulated', 'name', 'return_parent_idx'], varargs=None, keywords=None, defaults=(0, True, None, False)), ('document', '1270395ce97a4e1b556104abbb14f096'))

paddle.fluid.layers.row_conv (ArgSpec(args=['input', 'future_context_size', 'param_attr', 'act'], varargs=None, keywords=None, defaults=(None, None)), ('document', '1d8a1c8b686b55631ba1b77805e4eacf'))

paddle.fluid.layers.multiplex (ArgSpec(args=['inputs', 'index'], varargs=None, keywords=None, defaults=None), ('document', '2c4d1ae83da6ed35e3b36ba1b3b51d23'))

paddle.fluid.layers.layer_norm (ArgSpec(args=['input', 'scale', 'shift', 'begin_norm_axis', 'epsilon', 'param_attr', 'bias_attr', 'act', 'name'], varargs=None, keywords=None, defaults=(True, True, 1, 1e-05, None, None, None, None)), ('document', '79797f827d89ae72c77960e9696883a9'))

paddle.fluid.layers.group_norm (ArgSpec(args=['input', 'groups', 'epsilon', 'param_attr', 'bias_attr', 'act', 'data_layout', 'name'], varargs=None, keywords=None, defaults=(1e-05, None, None, None, 'NCHW', None)), ('document', '87dd4b818f102bc1a780e1804c28bd38'))

paddle.fluid.layers.spectral_norm (ArgSpec(args=['weight', 'dim', 'power_iters', 'eps', 'name'], varargs=None, keywords=None, defaults=(0, 1, 1e-12, None)), ('document', '9461e67095a6fc5d568fb2ce8fef66ff'))

paddle.fluid.layers.softmax_with_cross_entropy (ArgSpec(args=['logits', 'label', 'soft_label', 'ignore_index', 'numeric_stable_mode', 'return_softmax', 'axis'], varargs=None, keywords=None, defaults=(False, -100, True, False, -1)), ('document', '54e1675aa0364f4a78fa72804ec0f413'))

paddle.fluid.layers.smooth_l1 (ArgSpec(args=['x', 'y', 'inside_weight', 'outside_weight', 'sigma'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', 'cbe8940643ac80ef75e1abdfbdb09e88'))

paddle.fluid.layers.one_hot (ArgSpec(args=['input', 'depth', 'allow_out_of_range'], varargs=None, keywords=None, defaults=(False,)), ('document', 'ec4115591be842868c86b2e5334245c6'))

paddle.fluid.layers.autoincreased_step_counter (ArgSpec(args=['counter_name', 'begin', 'step'], varargs=None, keywords=None, defaults=(None, 1, 1)), ('document', '98e7927f09ee2270535b29f048e481ec'))

paddle.fluid.layers.reshape (ArgSpec(args=['x', 'shape', 'actual_shape', 'act', 'inplace', 'name'], varargs=None, keywords=None, defaults=(None, None, False, None)), ('document', 'ca73fdc4551c5765c92eb00f24874289'))

paddle.fluid.layers.squeeze (ArgSpec(args=['input', 'axes', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'ebbac07662a6e22e8e299ced880c7775'))

paddle.fluid.layers.unsqueeze (ArgSpec(args=['input', 'axes', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'b9bd3129d36a70e7c4385df51ff71c62'))

paddle.fluid.layers.lod_reset (ArgSpec(args=['x', 'y', 'target_lod'], varargs=None, keywords=None, defaults=(None, None)), ('document', '74498d37dd622ac472cb36887fce09ea'))

paddle.fluid.layers.lod_append (ArgSpec(args=['x', 'level'], varargs=None, keywords=None, defaults=None), ('document', '37663c7c179e920838a250ea0e28d909'))

paddle.fluid.layers.lrn (ArgSpec(args=['input', 'n', 'k', 'alpha', 'beta', 'name'], varargs=None, keywords=None, defaults=(5, 1.0, 0.0001, 0.75, None)), ('document', '73d297256da8954617996958d26ee93d'))

paddle.fluid.layers.pad (ArgSpec(args=['x', 'paddings', 'pad_value', 'name'], varargs=None, keywords=None, defaults=(0.0, None)), ('document', '36b6e58678956585e5b30aa3de123a60'))

paddle.fluid.layers.pad_constant_like (ArgSpec(args=['x', 'y', 'pad_value', 'name'], varargs=None, keywords=None, defaults=(0.0, None)), ('document', '95aa1972983f30fe9b5a3713e523e20f'))

paddle.fluid.layers.label_smooth (ArgSpec(args=['label', 'prior_dist', 'epsilon', 'dtype', 'name'], varargs=None, keywords=None, defaults=(None, 0.1, 'float32', None)), ('document', '214f1dfbe95a628600bbe99e836319cf'))

paddle.fluid.layers.roi_pool (ArgSpec(args=['input', 'rois', 'pooled_height', 'pooled_width', 'spatial_scale'], varargs=None, keywords=None, defaults=(1, 1, 1.0)), ('document', '49368d724023a66b41b0071be41c0ba5'))

paddle.fluid.layers.roi_align (ArgSpec(args=['input', 'rois', 'pooled_height', 'pooled_width', 'spatial_scale', 'sampling_ratio', 'name'], varargs=None, keywords=None, defaults=(1, 1, 1.0, -1, None)), ('document', 'dc2e2fa3d6e3d30de0a81e8ee70de733'))

paddle.fluid.layers.dice_loss (ArgSpec(args=['input', 'label', 'epsilon'], varargs=None, keywords=None, defaults=(1e-05,)), ('document', '7e8e4bf1f0f8612961ed113e8af8f0c5'))

paddle.fluid.layers.image_resize (ArgSpec(args=['input', 'out_shape', 'scale', 'name', 'resample', 'actual_shape', 'align_corners', 'align_mode', 'data_format'], varargs=None, keywords=None, defaults=(None, None, None, 'BILINEAR', None, True, 1, 'NCHW')), ('document', 'd29d829607b5ff12924197a3ba296c89'))

paddle.fluid.layers.image_resize_short (ArgSpec(args=['input', 'out_short_len', 'resample'], varargs=None, keywords=None, defaults=('BILINEAR',)), ('document', 'bd97ebfe4bdf5110a5fcb8ecb626a447'))

paddle.fluid.layers.resize_bilinear (ArgSpec(args=['input', 'out_shape', 'scale', 'name', 'actual_shape', 'align_corners', 'align_mode', 'data_format'], varargs=None, keywords=None, defaults=(None, None, None, None, True, 1, 'NCHW')), ('document', '44da7890c8a362a83a1c0902a1dc1e4d'))

paddle.fluid.layers.resize_trilinear (ArgSpec(args=['input', 'out_shape', 'scale', 'name', 'actual_shape', 'align_corners', 'align_mode', 'data_format'], varargs=None, keywords=None, defaults=(None, None, None, None, True, 1, 'NCDHW')), ('document', '5b4d0f823f94c260fe5e6f7eec60a797'))

paddle.fluid.layers.resize_nearest (ArgSpec(args=['input', 'out_shape', 'scale', 'name', 'actual_shape', 'align_corners', 'data_format'], varargs=None, keywords=None, defaults=(None, None, None, None, True, 'NCHW')), ('document', '0107a5cbae1aef3f381d3d769a6068eb'))

paddle.fluid.layers.gather (ArgSpec(args=['input', 'index', 'overwrite'], varargs=None, keywords=None, defaults=(True,)), ('document', 'f985c9b66e3aec96fa753a8eb44c991c'))

paddle.fluid.layers.gather_nd (ArgSpec(args=['input', 'index', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'a7d625028525167b138106f574dffdf9'))

paddle.fluid.layers.scatter (ArgSpec(args=['input', 'index', 'updates', 'name', 'overwrite'], varargs=None, keywords=None, defaults=(None, True)), ('document', '69b22affd4a6326502af166f04c095ab'))

paddle.fluid.layers.scatter_nd_add (ArgSpec(args=['ref', 'index', 'updates', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '2607b5c9369fbc52f208de066a80fc25'))

paddle.fluid.layers.scatter_nd (ArgSpec(args=['index', 'updates', 'shape', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'e43f1d3a938b35da246aea3e72a020ec'))

paddle.fluid.layers.sequence_scatter (ArgSpec(args=['input', 'index', 'updates', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'abe3f714120117a5a3d3e639853932bf'))

paddle.fluid.layers.random_crop (ArgSpec(args=['x', 'shape', 'seed'], varargs=None, keywords=None, defaults=(None,)), ('document', '042af0b8abea96b40c22f6e70d99e042'))

paddle.fluid.layers.mean_iou (ArgSpec(args=['input', 'label', 'num_classes'], varargs=None, keywords=None, defaults=None), ('document', 'e714b4aa7993dfe9c1a38886875dbaac'))

paddle.fluid.layers.relu (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '0942c174f4f6fb274976d4357356f6a2'))

paddle.fluid.layers.selu (ArgSpec(args=['x', 'scale', 'alpha', 'name'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', 'f93c61f5b0bf933cd425a64dca2c4fdd'))

paddle.fluid.layers.log (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '02f668664e3bfc4df6c00d7363467140'))

paddle.fluid.layers.crop (ArgSpec(args=['x', 'shape', 'offsets', 'name'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', 'ba3621917d5beffd3d022b88fbf6dc46'))

paddle.fluid.layers.crop_tensor (ArgSpec(args=['x', 'shape', 'offsets', 'name'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', 'd460aaf35afbbeb9beea4789aa6e4343'))

paddle.fluid.layers.rank_loss (ArgSpec(args=['label', 'left', 'right', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '8eb36596bb43d7a907d3397c7aedbdb3'))

paddle.fluid.layers.margin_rank_loss (ArgSpec(args=['label', 'left', 'right', 'margin', 'name'], varargs=None, keywords=None, defaults=(0.1, None)), ('document', '6fc86ed23b420c8a0f6c043563cf3937'))

paddle.fluid.layers.elu (ArgSpec(args=['x', 'alpha', 'name'], varargs=None, keywords=None, defaults=(1.0, None)), ('document', '9af1926c06711eacef9e82d7a9e4d308'))

paddle.fluid.layers.relu6 (ArgSpec(args=['x', 'threshold', 'name'], varargs=None, keywords=None, defaults=(6.0, None)), ('document', '538fc860b2a1734e118b94e4a1a3ee67'))

paddle.fluid.layers.pow (ArgSpec(args=['x', 'factor', 'name'], varargs=None, keywords=None, defaults=(1.0, None)), ('document', 'ca34f88ff61cf2a7f4c97a493d6000d0'))

paddle.fluid.layers.stanh (ArgSpec(args=['x', 'scale_a', 'scale_b', 'name'], varargs=None, keywords=None, defaults=(0.67, 1.7159, None)), ('document', 'd3f742178a7263adf5929153d104883d'))

paddle.fluid.layers.hard_sigmoid (ArgSpec(args=['x', 'slope', 'offset', 'name'], varargs=None, keywords=None, defaults=(0.2, 0.5, None)), ('document', '607d79ca873bee40eed1c79a96611591'))

paddle.fluid.layers.swish (ArgSpec(args=['x', 'beta', 'name'], varargs=None, keywords=None, defaults=(1.0, None)), ('document', 'e0dc7bc66cba939033bc028d7a62c5f4'))

paddle.fluid.layers.prelu (ArgSpec(args=['x', 'mode', 'param_attr', 'name'], varargs=None, keywords=None, defaults=(None, None)), ('document', '1fadca6622c70bd33cc260817f4ff191'))

paddle.fluid.layers.brelu (ArgSpec(args=['x', 't_min', 't_max', 'name'], varargs=None, keywords=None, defaults=(0.0, 24.0, None)), ('document', '49580538249a52c857fce75c94ad8af7'))

paddle.fluid.layers.leaky_relu (ArgSpec(args=['x', 'alpha', 'name'], varargs=None, keywords=None, defaults=(0.02, None)), ('document', '1eb3009c69060299ec87949ee0d4b9ae'))

paddle.fluid.layers.soft_relu (ArgSpec(args=['x', 'threshold', 'name'], varargs=None, keywords=None, defaults=(40.0, None)), ('document', '6455afd2498b00198f53f83d63d6c6a4'))

paddle.fluid.layers.flatten (ArgSpec(args=['x', 'axis', 'name'], varargs=None, keywords=None, defaults=(1, None)), ('document', '631a6d5d7a58ec80575993bf7c647212'))

paddle.fluid.layers.sequence_mask (ArgSpec(args=['x', 'maxlen', 'dtype', 'name'], varargs=None, keywords=None, defaults=(None, 'int64', None)), ('document', '6c3f916921b24edaad220f1fcbf039de'))

paddle.fluid.layers.stack (ArgSpec(args=['x', 'axis'], varargs=None, keywords=None, defaults=(0,)), ('document', 'a76f347bf27ffe21b990340d5d9524d5'))

paddle.fluid.layers.pad2d (ArgSpec(args=['input', 'paddings', 'mode', 'pad_value', 'data_format', 'name'], varargs=None, keywords=None, defaults=([0, 0, 0, 0], 'constant', 0.0, 'NCHW', None)), ('document', '3f3abdb795a5c2aad8c2312249551ce5'))

paddle.fluid.layers.unstack (ArgSpec(args=['x', 'axis', 'num'], varargs=None, keywords=None, defaults=(0, None)), ('document', 'b0c4ca08d4eb295189e1b107c920d093'))

paddle.fluid.layers.sequence_enumerate (ArgSpec(args=['input', 'win_size', 'pad_value', 'name'], varargs=None, keywords=None, defaults=(0, None)), ('document', 'b870fed41abd2aecf929ece65f555fa1'))

paddle.fluid.layers.unique (ArgSpec(args=['x', 'dtype'], varargs=None, keywords=None, defaults=('int32',)), ('document', 'cab0b06e5683875f12f0efc62fa230a9'))

paddle.fluid.layers.unique_with_counts (ArgSpec(args=['x', 'dtype'], varargs=None, keywords=None, defaults=('int32',)), ('document', '1cb59c65b41766116944b8ed1e6ad345'))

paddle.fluid.layers.expand (ArgSpec(args=['x', 'expand_times', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '7b97042c3ba55fb5fec6a06308523b73'))

paddle.fluid.layers.sequence_concat (ArgSpec(args=['input', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'f47f9d207ac60b6f294087bcb1b64ae8'))

paddle.fluid.layers.scale (ArgSpec(args=['x', 'scale', 'bias', 'bias_after_scale', 'act', 'name'], varargs=None, keywords=None, defaults=(1.0, 0.0, True, None, None)), ('document', '463e4713806e5adaa4d20a41e2218453'))

paddle.fluid.layers.elementwise_add (ArgSpec(args=['x', 'y', 'axis', 'act', 'name'], varargs=None, keywords=None, defaults=(-1, None, None)), ('document', '0c9c260e7738165a099f6a76da0b7814'))

paddle.fluid.layers.elementwise_div (ArgSpec(args=['x', 'y', 'axis', 'act', 'name'], varargs=None, keywords=None, defaults=(-1, None, None)), ('document', '4701ffd4eb4b7ee19756d3b90532c5f2'))

paddle.fluid.layers.elementwise_sub (ArgSpec(args=['x', 'y', 'axis', 'act', 'name'], varargs=None, keywords=None, defaults=(-1, None, None)), ('document', 'eab2518a801f3f393cf38fddc899c941'))

paddle.fluid.layers.elementwise_mul (ArgSpec(args=['x', 'y', 'axis', 'act', 'name'], varargs=None, keywords=None, defaults=(-1, None, None)), ('document', '56838de6feabe4e5eef4f58c841eed11'))

paddle.fluid.layers.elementwise_max (ArgSpec(args=['x', 'y', 'axis', 'act', 'name'], varargs=None, keywords=None, defaults=(-1, None, None)), ('document', '27e1e1604433f85cd7c4c320cb7b9a5f'))

paddle.fluid.layers.elementwise_min (ArgSpec(args=['x', 'y', 'axis', 'act', 'name'], varargs=None, keywords=None, defaults=(-1, None, None)), ('document', 'cdbfe2042da18b0e2a2fd79fde944dc2'))

paddle.fluid.layers.elementwise_pow (ArgSpec(args=['x', 'y', 'axis', 'act', 'name'], varargs=None, keywords=None, defaults=(-1, None, None)), ('document', '6fc5d7492830d60c7fa61b3bc8f0d7e7'))

paddle.fluid.layers.elementwise_mod (ArgSpec(args=['x', 'y', 'axis', 'act', 'name'], varargs=None, keywords=None, defaults=(-1, None, None)), ('document', '4101ee1f9280f00dce54054ccc434890'))

paddle.fluid.layers.elementwise_floordiv (ArgSpec(args=['x', 'y', 'axis', 'act', 'name'], varargs=None, keywords=None, defaults=(-1, None, None)), ('document', '67e6101c31314d4082621e8e443cfb68'))

paddle.fluid.layers.uniform_random_batch_size_like (ArgSpec(args=['input', 'shape', 'dtype', 'input_dim_idx', 'output_dim_idx', 'min', 'max', 'seed'], varargs=None, keywords=None, defaults=('float32', 0, 0, -1.0, 1.0, 0)), ('document', 'cfa120e583cd4a5bfa120c8a26f98a28'))

paddle.fluid.layers.gaussian_random (ArgSpec(args=['shape', 'mean', 'std', 'seed', 'dtype'], varargs=None, keywords=None, defaults=(0.0, 1.0, 0, 'float32')), ('document', 'ebbf399d4e03190ce5dc9488f05c92f4'))

paddle.fluid.layers.sampling_id (ArgSpec(args=['x', 'min', 'max', 'seed', 'dtype'], varargs=None, keywords=None, defaults=(0.0, 1.0, 0, 'float32')), ('document', 'c39b647b6cf08e058d96ee503d5284fe'))

paddle.fluid.layers.gaussian_random_batch_size_like (ArgSpec(args=['input', 'shape', 'input_dim_idx', 'output_dim_idx', 'mean', 'std', 'seed', 'dtype'], varargs=None, keywords=None, defaults=(0, 0, 0.0, 1.0, 0, 'float32')), ('document', 'b24d0b21361c4bb8ef2cec8c26fb12b2'))

paddle.fluid.layers.sum (ArgSpec(args=['x'], varargs=None, keywords=None, defaults=None), ('document', 'f4b60847cb0f1ae00823ba6fb1b11310'))

paddle.fluid.layers.slice (ArgSpec(args=['input', 'axes', 'starts', 'ends'], varargs=None, keywords=None, defaults=None), ('document', '315b4870f294e33a27ecbdf440bed3ff'))

paddle.fluid.layers.strided_slice (ArgSpec(args=['input', 'axes', 'starts', 'ends', 'strides'], varargs=None, keywords=None, defaults=None), ('document', '340d8d656272ea396b441aab848429a2'))

paddle.fluid.layers.shape (ArgSpec(args=['input'], varargs=None, keywords=None, defaults=None), ('document', 'bf61c8f79d795a8371bdb3b5468aa82b'))

paddle.fluid.layers.rank (ArgSpec(args=['input'], varargs=None, keywords=None, defaults=None), ('document', '096df0e0273145ab80ed119a4c294db3'))

paddle.fluid.layers.size (ArgSpec(args=['input'], varargs=None, keywords=None, defaults=None), ('document', 'cf2e156beae36378722666c4c33bebfe'))

paddle.fluid.layers.logical_and (ArgSpec(args=['x', 'y', 'out', 'name'], varargs=None, keywords=None, defaults=(None, None)), ('document', '12db97c6c459c0f240ec7006737174f2'))

paddle.fluid.layers.logical_or (ArgSpec(args=['x', 'y', 'out', 'name'], varargs=None, keywords=None, defaults=(None, None)), ('document', '15adbc561618b7db69671e02009bea67'))

paddle.fluid.layers.logical_xor (ArgSpec(args=['x', 'y', 'out', 'name'], varargs=None, keywords=None, defaults=(None, None)), ('document', '77ccf37b710c507dd97e03f08ce8bb29'))

paddle.fluid.layers.logical_not (ArgSpec(args=['x', 'out', 'name'], varargs=None, keywords=None, defaults=(None, None)), ('document', '6e2fe8a322ec69811f6507d22acf8f9f'))

paddle.fluid.layers.clip (ArgSpec(args=['x', 'min', 'max', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '0ce33756573c572da67302499455dbcd'))

paddle.fluid.layers.clip_by_norm (ArgSpec(args=['x', 'max_norm', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '1fc6e217c7a6128df31b806c1a8067ff'))

paddle.fluid.layers.mean (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '597257fb94d0597c404a6a5c91ab5258'))

paddle.fluid.layers.mul (ArgSpec(args=['x', 'y', 'x_num_col_dims', 'y_num_col_dims', 'name'], varargs=None, keywords=None, defaults=(1, 1, None)), ('document', '784b7e36cea88493f9e37a41b10fbf4d'))

paddle.fluid.layers.sigmoid_cross_entropy_with_logits (ArgSpec(args=['x', 'label', 'ignore_index', 'name', 'normalize'], varargs=None, keywords=None, defaults=(-100, None, False)), ('document', '7637c974f2d749d359acae9062c4d96f'))

paddle.fluid.layers.maxout (ArgSpec(args=['x', 'groups', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '169882eb87fb693198e0153629134c22'))

paddle.fluid.layers.space_to_depth (ArgSpec(args=['x', 'blocksize', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '26decdea9376b6b9a0d3432d82ca207b'))

paddle.fluid.layers.affine_grid (ArgSpec(args=['theta', 'out_shape', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'f85b263b7b6698d000977529a28f202b'))

paddle.fluid.layers.sequence_reverse (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '5b32ed21ab89140a8e758002923a0da3'))

paddle.fluid.layers.affine_channel (ArgSpec(args=['x', 'scale', 'bias', 'data_layout', 'name', 'act'], varargs=None, keywords=None, defaults=(None, None, 'NCHW', None, None)), ('document', '9f303c67538e468a36c5904a0a3aa110'))

paddle.fluid.layers.similarity_focus (ArgSpec(args=['input', 'axis', 'indexes', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '18ec2e3afeb90e70c8b73d2b71c40fdb'))

paddle.fluid.layers.hash (ArgSpec(args=['input', 'hash_size', 'num_hash', 'name'], varargs=None, keywords=None, defaults=(1, None)), ('document', 'a0b73c21be618cec0281e7903039e5e3'))

paddle.fluid.layers.grid_sampler (ArgSpec(args=['x', 'grid', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '5d16663e096d7f04954c70ce1cc5e195'))

paddle.fluid.layers.log_loss (ArgSpec(args=['input', 'label', 'epsilon', 'name'], varargs=None, keywords=None, defaults=(0.0001, None)), ('document', 'e3993a477c94729526040ff65d95728e'))

paddle.fluid.layers.add_position_encoding (ArgSpec(args=['input', 'alpha', 'beta', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'e399f9436fed5f7ff480d8532e42c937'))

paddle.fluid.layers.bilinear_tensor_product (ArgSpec(args=['x', 'y', 'size', 'act', 'name', 'param_attr', 'bias_attr'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', '45fc3652a8e1aeffbe4eba371c54f756'))

paddle.fluid.layers.merge_selected_rows (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'b2b0e5d5c155ce24bafc38b78cd0b164'))

paddle.fluid.layers.get_tensor_from_selected_rows (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '2c568321feb4d16c41a83df43f95089d'))

paddle.fluid.layers.lstm (ArgSpec(args=['input', 'init_h', 'init_c', 'max_len', 'hidden_size', 'num_layers', 'dropout_prob', 'is_bidirec', 'is_test', 'name', 'default_initializer', 'seed'], varargs=None, keywords=None, defaults=(0.0, False, False, None, None, -1)), ('document', 'baa7327ed89df6b7bdd32f9ffdb62f63'))

paddle.fluid.layers.shuffle_channel (ArgSpec(args=['x', 'group', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '276a1213dd431228cefa33c3146df34a'))

paddle.fluid.layers.temporal_shift (ArgSpec(args=['x', 'seg_num', 'shift_ratio', 'name'], varargs=None, keywords=None, defaults=(0.25, None)), ('document', '13b1cdcb01f5ffdc26591ff9a2ec4669'))

paddle.fluid.layers.py_func (ArgSpec(args=['func', 'x', 'out', 'backward_func', 'skip_vars_in_backward_input'], varargs=None, keywords=None, defaults=(None, None)), ('document', '8404e472ac12b4a30a505d3d3a3e5fdb'))

paddle.fluid.layers.psroi_pool (ArgSpec(args=['input', 'rois', 'output_channels', 'spatial_scale', 'pooled_height', 'pooled_width', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '42d5155374f69786300d90d751956998'))

paddle.fluid.layers.prroi_pool (ArgSpec(args=['input', 'rois', 'output_channels', 'spatial_scale', 'pooled_height', 'pooled_width', 'name'], varargs=None, keywords=None, defaults=(1.0, 1, 1, None)), ('document', '454c7ea8c73313dd41513929d7526303'))

paddle.fluid.layers.teacher_student_sigmoid_loss (ArgSpec(args=['input', 'label', 'soft_max_up_bound', 'soft_max_lower_bound'], varargs=None, keywords=None, defaults=(15.0, -15.0)), ('document', 'b0e07aa41caae04b07a8e8217cc96020'))

paddle.fluid.layers.huber_loss (ArgSpec(args=['input', 'label', 'delta'], varargs=None, keywords=None, defaults=None), ('document', '11bb8e62cc9256958eff3991fe4834da'))

paddle.fluid.layers.kldiv_loss (ArgSpec(args=['x', 'target', 'reduction', 'name'], varargs=None, keywords=None, defaults=('mean', None)), ('document', '18bc95c62d3300456c3c7da5278b47bb'))

paddle.fluid.layers.npair_loss (ArgSpec(args=['anchor', 'positive', 'labels', 'l2_reg'], varargs=None, keywords=None, defaults=(0.002,)), ('document', 'a41a93253c937697e900e19af172490d'))

paddle.fluid.layers.pixel_shuffle (ArgSpec(args=['x', 'upscale_factor'], varargs=None, keywords=None, defaults=None), ('document', '7e5cac851fd9bad344230e1044b6a565'))

paddle.fluid.layers.fsp_matrix (ArgSpec(args=['x', 'y'], varargs=None, keywords=None, defaults=None), ('document', 'd803767ef4fb885013a28c98634e0bc4'))

paddle.fluid.layers.continuous_value_model (ArgSpec(args=['input', 'cvm', 'use_cvm'], varargs=None, keywords=None, defaults=(True,)), ('document', 'c03490ffaa1b78258747157c313db4cd'))

paddle.fluid.layers.where (ArgSpec(args=['condition'], varargs=None, keywords=None, defaults=None), ('document', 'b1e1487760295e1ff55307b880a99e18'))

paddle.fluid.layers.sign (ArgSpec(args=['x'], varargs=None, keywords=None, defaults=None), ('document', 'b56afe9ae3fc553c95d907fd7ef6c314'))

paddle.fluid.layers.deformable_conv (ArgSpec(args=['input', 'offset', 'mask', 'num_filters', 'filter_size', 'stride', 'padding', 'dilation', 'groups', 'deformable_groups', 'im2col_step', 'param_attr', 'bias_attr', 'modulated', 'name'], varargs=None, keywords=None, defaults=(1, 0, 1, None, None, None, None, None, True, None)), ('document', '9b9c9d1282f994ccd4538201e0b6856f'))

paddle.fluid.layers.unfold (ArgSpec(args=['x', 'kernel_sizes', 'strides', 'paddings', 'dilations', 'name'], varargs=None, keywords=None, defaults=(1, 0, 1, None)), ('document', '3f884662ad443d9ecc2b3734b4f61ad6'))

paddle.fluid.layers.deformable_roi_pooling (ArgSpec(args=['input', 'rois', 'trans', 'no_trans', 'spatial_scale', 'group_size', 'pooled_height', 'pooled_width', 'part_size', 'sample_per_part', 'trans_std', 'position_sensitive', 'name'], varargs=None, keywords=None, defaults=(False, 1.0, [1, 1], 1, 1, None, 1, 0.1, False, None)), ('document', '47c5d1c890b36fa00ff3285c9398f613'))

paddle.fluid.layers.filter_by_instag (ArgSpec(args=['ins', 'ins_tag', 'filter_tag', 'is_lod'], varargs=None, keywords=None, defaults=None), ('document', '7703a2088af8de4128b143ff1164ca4a'))

paddle.fluid.layers.shard_index (ArgSpec(args=['input', 'index_num', 'nshards', 'shard_id', 'ignore_value'], varargs=None, keywords=None, defaults=(-1,)), ('document', 'c4969dd6bf164f9e6a90414ea4f4e5ad'))

paddle.fluid.layers.hard_swish (ArgSpec(args=['x', 'threshold', 'scale', 'offset', 'name'], varargs=None, keywords=None, defaults=(6.0, 6.0, 3.0, None)), ('document', '6a5152a7015c62cb8278fc24cb456459'))

paddle.fluid.layers.mse_loss (ArgSpec(args=['input', 'label'], varargs=None, keywords=None, defaults=None), ('document', 'd9ede6469288636e1b3233b461a165c9'))

paddle.fluid.layers.uniform_random (ArgSpec(args=['shape', 'dtype', 'min', 'max', 'seed'], varargs=None, keywords=None, defaults=('float32', -1.0, 1.0, 0)), ('document', '126ede8ce0e751244b1b54cd359c89d7'))

paddle.fluid.layers.data (ArgSpec(args=['name', 'shape', 'append_batch_size', 'dtype', 'lod_level', 'type', 'stop_gradient'], varargs=None, keywords=None, defaults=(True, 'float32', 0, VarType.LOD_TENSOR, True)), ('document', '9d7806e31bdf727c1a23b8782a09b545'))

paddle.fluid.layers.read_file (ArgSpec(args=['reader'], varargs=None, keywords=None, defaults=None), ('document', '88367daf9a30c9ab83adc5d7221e23ef'))

paddle.fluid.layers.double_buffer (ArgSpec(args=['reader', 'place', 'name'], varargs=None, keywords=None, defaults=(None, None)), ('document', '44724c493f41a124abc7531c2740e2e3'))

paddle.fluid.layers.py_reader (ArgSpec(args=['capacity', 'shapes', 'dtypes', 'lod_levels', 'name', 'use_double_buffer'], varargs=None, keywords=None, defaults=(None, None, True)), ('document', 'd78a1c7344955c5caed8dc13adb7beb6'))

paddle.fluid.layers.create_py_reader_by_data (ArgSpec(args=['capacity', 'feed_list', 'name', 'use_double_buffer'], varargs=None, keywords=None, defaults=(None, True)), ('document', '2edf37d57862b24a7a26aa19a3573f73'))

paddle.fluid.layers.load (ArgSpec(args=['out', 'file_path', 'load_as_fp16'], varargs=None, keywords=None, defaults=(None,)), ('document', '309f9e5249463e1b207a7347b2a91134'))

paddle.fluid.layers.create_tensor (ArgSpec(args=['dtype', 'name', 'persistable'], varargs=None, keywords=None, defaults=(None, False)), ('document', 'aaf0176c743c43e9bc684dd7dfac25c5'))

paddle.fluid.layers.create_parameter (ArgSpec(args=['shape', 'dtype', 'name', 'attr', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(None, None, False, None)), ('document', '021272f30e0cdf7503586815378abfb8'))

paddle.fluid.layers.create_global_var (ArgSpec(args=['shape', 'value', 'dtype', 'persistable', 'force_cpu', 'name'], varargs=None, keywords=None, defaults=(False, False, None)), ('document', '47ea8b8c91879e50c9036e418b00ef4a'))

paddle.fluid.layers.cast (ArgSpec(args=['x', 'dtype'], varargs=None, keywords=None, defaults=None), ('document', '1e44a534cf7d26ab230aa9f5e4e0525a'))

paddle.fluid.layers.tensor_array_to_tensor (ArgSpec(args=['input', 'axis', 'name'], varargs=None, keywords=None, defaults=(1, None)), ('document', '764c095ba4562ae740f979e970152d6e'))

paddle.fluid.layers.concat (ArgSpec(args=['input', 'axis', 'name'], varargs=None, keywords=None, defaults=(0, None)), ('document', 'b3f30feb5dec8f110d7393ffeb30dbd9'))

paddle.fluid.layers.sums (ArgSpec(args=['input', 'out'], varargs=None, keywords=None, defaults=(None,)), ('document', '5df743d578638cd2bbb9369499b44af4'))

paddle.fluid.layers.assign (ArgSpec(args=['input', 'output'], varargs=None, keywords=None, defaults=(None,)), ('document', '8bd94aef4e123986d9a8c29f67b5532b'))

paddle.fluid.layers.fill_constant_batch_size_like (ArgSpec(args=['input', 'shape', 'dtype', 'value', 'input_dim_idx', 'output_dim_idx'], varargs=None, keywords=None, defaults=(0, 0)), ('document', '3551aa494e88d0f271e40cd45d6e3020'))

paddle.fluid.layers.fill_constant (ArgSpec(args=['shape', 'dtype', 'value', 'force_cpu', 'out'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'd6b76c7d2c7129f8d713ca74f1c2c287'))

paddle.fluid.layers.argmin (ArgSpec(args=['x', 'axis'], varargs=None, keywords=None, defaults=(0,)), ('document', '3dd54487232d05df4d70fba94b7d0b79'))

paddle.fluid.layers.argmax (ArgSpec(args=['x', 'axis'], varargs=None, keywords=None, defaults=(0,)), ('document', '7f47cc9aa7531b6bd37c5c96bc7f0469'))

paddle.fluid.layers.argsort (ArgSpec(args=['input', 'axis', 'name'], varargs=None, keywords=None, defaults=(-1, None)), ('document', '9792371e3b66258531225a5551de8961'))

paddle.fluid.layers.ones (ArgSpec(args=['shape', 'dtype', 'force_cpu'], varargs=None, keywords=None, defaults=(False,)), ('document', '812c623ed52610b9773f9fc05413bc34'))

paddle.fluid.layers.zeros (ArgSpec(args=['shape', 'dtype', 'force_cpu'], varargs=None, keywords=None, defaults=(False,)), ('document', '95379f9288c2d05356ec0e2375c6bc57'))

paddle.fluid.layers.reverse (ArgSpec(args=['x', 'axis'], varargs=None, keywords=None, defaults=None), ('document', '628135603692137d52bcf5a8d8d6816d'))

paddle.fluid.layers.has_inf (ArgSpec(args=['x'], varargs=None, keywords=None, defaults=None), ('document', '51a0fa1cfaf2507c00a215adacdb8a63'))

paddle.fluid.layers.has_nan (ArgSpec(args=['x'], varargs=None, keywords=None, defaults=None), ('document', '129cf426e71452fe8276d616a6dc21ae'))

paddle.fluid.layers.isfinite (ArgSpec(args=['x'], varargs=None, keywords=None, defaults=None), ('document', 'b9fff4ffc8d11934cde099f4c39bf841'))

paddle.fluid.layers.range (ArgSpec(args=['start', 'end', 'step', 'dtype'], varargs=None, keywords=None, defaults=None), ('document', 'a45b42f21bc5a4e84b60981a3d629ab3'))

paddle.fluid.layers.linspace (ArgSpec(args=['start', 'stop', 'num', 'dtype'], varargs=None, keywords=None, defaults=None), ('document', '3663d1148946eed4c1c34c81be586b9e'))

paddle.fluid.layers.zeros_like (ArgSpec(args=['x', 'out'], varargs=None, keywords=None, defaults=(None,)), ('document', 'd88a23bcdc443719b3953593f7cef14a'))

paddle.fluid.layers.ones_like (ArgSpec(args=['x', 'out'], varargs=None, keywords=None, defaults=(None,)), ('document', 'd18d42059c6b189cbd3fab2fcb206c15'))

paddle.fluid.layers.diag (ArgSpec(args=['diagonal'], varargs=None, keywords=None, defaults=None), ('document', '88a15e15f0098d549f07a01eaebf9ce3'))

paddle.fluid.layers.eye (ArgSpec(args=['num_rows', 'num_columns', 'batch_shape', 'dtype'], varargs=None, keywords=None, defaults=(None, None, 'float32')), ('document', '60cdc70ae43ba69fae36d720ef3016a1'))

paddle.fluid.layers.While ('paddle.fluid.layers.control_flow.While', ('document', '50110155608a00f43d3d3fd1be41dcb4'))

paddle.fluid.layers.While.__init__ (ArgSpec(args=['self', 'cond', 'is_test', 'name'], varargs=None, keywords=None, defaults=(False, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.While.block (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.Switch ('paddle.fluid.layers.control_flow.Switch', ('document', 'a1c5ef8ff117d7d6ba8940ec104f02ce'))

paddle.fluid.layers.Switch.__init__ (ArgSpec(args=['self', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.Switch.case (ArgSpec(args=['self', 'condition'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.Switch.default (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.increment (ArgSpec(args=['x', 'value', 'in_place'], varargs=None, keywords=None, defaults=(1.0, True)), ('document', 'f88b5787bb80ae6b8bf513a70dabbdc1'))

paddle.fluid.layers.array_write (ArgSpec(args=['x', 'i', 'array'], varargs=None, keywords=None, defaults=(None,)), ('document', '3f913b5069ad40bd85d89b33e4aa5939'))

paddle.fluid.layers.create_array (ArgSpec(args=['dtype'], varargs=None, keywords=None, defaults=None), ('document', '556de793fdf24d515f3fc91260e2c048'))

paddle.fluid.layers.less_than (ArgSpec(args=['x', 'y', 'force_cpu', 'cond'], varargs=None, keywords=None, defaults=(None, None)), ('document', '04af32422c3a3d8f6040aeb406c82768'))

paddle.fluid.layers.less_equal (ArgSpec(args=['x', 'y', 'cond'], varargs=None, keywords=None, defaults=(None,)), ('document', '7b6d952a9f6340a044cfb91c16aad842'))

paddle.fluid.layers.greater_than (ArgSpec(args=['x', 'y', 'cond'], varargs=None, keywords=None, defaults=(None,)), ('document', '55710e2fafeda70cd1b53d7509712499'))

paddle.fluid.layers.greater_equal (ArgSpec(args=['x', 'y', 'cond'], varargs=None, keywords=None, defaults=(None,)), ('document', '14bff27b2be5e60eaa30e41925265beb'))

paddle.fluid.layers.equal (ArgSpec(args=['x', 'y', 'cond'], varargs=None, keywords=None, defaults=(None,)), ('document', '788aa651e8b9fec79d16931ef3a33e90'))

paddle.fluid.layers.not_equal (ArgSpec(args=['x', 'y', 'cond'], varargs=None, keywords=None, defaults=(None,)), ('document', '57adebb8858ffab6be2d86d0522b85dc'))

paddle.fluid.layers.array_read (ArgSpec(args=['array', 'i'], varargs=None, keywords=None, defaults=None), ('document', 'caf0d94349cdc28e1bda3b8a19411ac0'))

paddle.fluid.layers.array_length (ArgSpec(args=['array'], varargs=None, keywords=None, defaults=None), ('document', '6f24a9b872027634ad758ea2826c9727'))

paddle.fluid.layers.IfElse ('paddle.fluid.layers.control_flow.IfElse', ('document', 'a389f88e19c3b332c3afcbf7df4488a5'))

paddle.fluid.layers.IfElse.__init__ (ArgSpec(args=['self', 'cond', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.IfElse.false_block (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.IfElse.input (ArgSpec(args=['self', 'x'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.IfElse.output (ArgSpec(args=['self'], varargs='outs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.IfElse.true_block (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.DynamicRNN ('paddle.fluid.layers.control_flow.DynamicRNN', ('document', 'b71e87285dbd4a43a6cc4f8a473245e6'))

paddle.fluid.layers.DynamicRNN.__init__ (ArgSpec(args=['self', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.DynamicRNN.block (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6d3e0a5d9aa519a9773a36e1620ea9b7'))

paddle.fluid.layers.DynamicRNN.memory (ArgSpec(args=['self', 'init', 'shape', 'value', 'need_reorder', 'dtype'], varargs=None, keywords=None, defaults=(None, None, 0.0, False, 'float32')), ('document', '57cdd0a63747f4c670cdb9d250ceb7e1'))

paddle.fluid.layers.DynamicRNN.output (ArgSpec(args=['self'], varargs='outputs', keywords=None, defaults=None), ('document', 'b439a176a3328de8a75bdc5c08eece4a'))

paddle.fluid.layers.DynamicRNN.static_input (ArgSpec(args=['self', 'x'], varargs=None, keywords=None, defaults=None), ('document', '55ab9c562edd7dabec0bd6fd6c1a28cc'))

paddle.fluid.layers.DynamicRNN.step_input (ArgSpec(args=['self', 'x', 'level'], varargs=None, keywords=None, defaults=(0,)), ('document', '4b300851b5201891d0e11c406e4c7d07'))

paddle.fluid.layers.DynamicRNN.update_memory (ArgSpec(args=['self', 'ex_mem', 'new_mem'], varargs=None, keywords=None, defaults=None), ('document', '5d83987da13b98363d6a807a52d8024f'))

paddle.fluid.layers.StaticRNN ('paddle.fluid.layers.control_flow.StaticRNN', ('document', 'f73671cb98696a1962bf5deaf49dc2e9'))

paddle.fluid.layers.StaticRNN.__init__ (ArgSpec(args=['self', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.StaticRNN.memory (ArgSpec(args=['self', 'init', 'shape', 'batch_ref', 'init_value', 'init_batch_dim_idx', 'ref_batch_dim_idx'], varargs=None, keywords=None, defaults=(None, None, None, 0.0, 0, 1)), ('document', 'f1b60dc4194d0bb714d6c6f5921b227f'))

paddle.fluid.layers.StaticRNN.output (ArgSpec(args=['self'], varargs='outputs', keywords=None, defaults=None), ('document', 'df6ceab6e6c9bd31e97914d7e7538137'))

paddle.fluid.layers.StaticRNN.step (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6d3e0a5d9aa519a9773a36e1620ea9b7'))

paddle.fluid.layers.StaticRNN.step_input (ArgSpec(args=['self', 'x'], varargs=None, keywords=None, defaults=None), ('document', '903387ec11f3d0bf46821d31a68cffa5'))

paddle.fluid.layers.StaticRNN.step_output (ArgSpec(args=['self', 'o'], varargs=None, keywords=None, defaults=None), ('document', '252890d4c3199a7623ab8667e13fd837'))

paddle.fluid.layers.StaticRNN.update_memory (ArgSpec(args=['self', 'mem', 'var'], varargs=None, keywords=None, defaults=None), ('document', '7a0000520f179f35239956a5ba55119f'))

paddle.fluid.layers.reorder_lod_tensor_by_rank (ArgSpec(args=['x', 'rank_table'], varargs=None, keywords=None, defaults=None), ('document', '5b552a1f0f7eb4dacb768a975ba15d08'))

paddle.fluid.layers.Print (ArgSpec(args=['input', 'first_n', 'message', 'summarize', 'print_tensor_name', 'print_tensor_type', 'print_tensor_shape', 'print_tensor_lod', 'print_phase'], varargs=None, keywords=None, defaults=(-1, None, 20, True, True, True, True, 'both')), ('document', '3130bed32922b9fd84ce2dea6250f635'))

paddle.fluid.layers.is_empty (ArgSpec(args=['x', 'cond'], varargs=None, keywords=None, defaults=(None,)), ('document', '3011dc695f490afdf504dc24f628319a'))

paddle.fluid.layers.sigmoid (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'bdc9a71908d3c9748532ff44c2f31034'))

paddle.fluid.layers.logsigmoid (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '9a4c346630a042454f727ad5e0cffc11'))

paddle.fluid.layers.exp (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '92bec0a7fdec48ad78effdf30b02c6fa'))

paddle.fluid.layers.tanh (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'e7a81a4af62b6c6ce858c897f74a4f0f'))

paddle.fluid.layers.atan (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '2dde114018cbcaff9b24c566bf6704a5'))

paddle.fluid.layers.tanh_shrink (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '000a76652c8e59e21e7fb6d87cc7a668'))

paddle.fluid.layers.sqrt (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'e3dce5e892ce63cc9c6ed87a7e6206d5'))

paddle.fluid.layers.rsqrt (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '0b90c858d4d71a58896537c1bd7acb09'))

paddle.fluid.layers.abs (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '503f4d5723bbe1b6c9f24058078709ed'))

paddle.fluid.layers.ceil (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '5602b78da33c4b0ccaea0374411de423'))

paddle.fluid.layers.floor (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'a0977ab14448ba472e5c2e152f42a818'))

paddle.fluid.layers.cos (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'e94c8569179ffa3a0dca028a5b518dbf'))

paddle.fluid.layers.acos (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '5c9a00178c5c28bb824f7d6c25060d3b'))

paddle.fluid.layers.asin (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '20d1d49fe4d13430a63c57fc4b29a677'))

paddle.fluid.layers.sin (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '4441e4e5e9934eb98760e31330e7a13c'))

paddle.fluid.layers.round (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '40132ef34808ed621c63ed4fd886fd1c'))

paddle.fluid.layers.reciprocal (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '578106495166d0fb65ade2bb51cdf926'))

paddle.fluid.layers.square (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '728233aff902803f5f62e2d340c3bcbb'))

paddle.fluid.layers.softplus (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '74c4e6dfbdfc3453301ea11d722ad3d6'))

paddle.fluid.layers.softsign (ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'a70e9320b113ca33c1299bbc032f09d4'))

paddle.fluid.layers.softshrink (ArgSpec(args=['x', 'alpha'], varargs=None, keywords=None, defaults=(None,)), ('document', '958c7bfdfb0b5e92af6ca4a90d24e5ef'))

paddle.fluid.layers.hard_shrink (ArgSpec(args=['x', 'threshold'], varargs=None, keywords=None, defaults=(None,)), ('document', '386a4103d2884b2f1312ebc1e8ee6486'))

paddle.fluid.layers.cumsum (ArgSpec(args=['x', 'axis', 'exclusive', 'reverse'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', '5ab9d5721a6734fe127069e4314e1309'))

paddle.fluid.layers.thresholded_relu (ArgSpec(args=['x', 'threshold'], varargs=None, keywords=None, defaults=(None,)), ('document', '9a0464425426a9b9c1b7500ede2836c1'))

paddle.fluid.layers.prior_box (ArgSpec(args=['input', 'image', 'min_sizes', 'max_sizes', 'aspect_ratios', 'variance', 'flip', 'clip', 'steps', 'offset', 'name', 'min_max_aspect_ratios_order'], varargs=None, keywords=None, defaults=(None, [1.0], [0.1, 0.1, 0.2, 0.2], False, False, [0.0, 0.0], 0.5, None, False)), ('document', '0fdf82762fd0a5acb2578a72771b5b44'))

paddle.fluid.layers.density_prior_box (ArgSpec(args=['input', 'image', 'densities', 'fixed_sizes', 'fixed_ratios', 'variance', 'clip', 'steps', 'offset', 'flatten_to_2d', 'name'], varargs=None, keywords=None, defaults=(None, None, None, [0.1, 0.1, 0.2, 0.2], False, [0.0, 0.0], 0.5, False, None)), ('document', '7a484a0da5e993a7734867a3dfa86571'))

paddle.fluid.layers.multi_box_head (ArgSpec(args=['inputs', 'image', 'base_size', 'num_classes', 'aspect_ratios', 'min_ratio', 'max_ratio', 'min_sizes', 'max_sizes', 'steps', 'step_w', 'step_h', 'offset', 'variance', 'flip', 'clip', 'kernel_size', 'pad', 'stride', 'name', 'min_max_aspect_ratios_order'], varargs=None, keywords=None, defaults=(None, None, None, None, None, None, None, 0.5, [0.1, 0.1, 0.2, 0.2], True, False, 1, 0, 1, None, False)), ('document', 'fd58078fdfffd899b91f992ba224628f'))

paddle.fluid.layers.bipartite_match (ArgSpec(args=['dist_matrix', 'match_type', 'dist_threshold', 'name'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', '409c351dee8a4a4ea02771dc691b49cb'))

paddle.fluid.layers.target_assign (ArgSpec(args=['input', 'matched_indices', 'negative_indices', 'mismatch_value', 'name'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', 'e9685f32d21bec8c013626c0254502c5'))

paddle.fluid.layers.detection_output (ArgSpec(args=['loc', 'scores', 'prior_box', 'prior_box_var', 'background_label', 'nms_threshold', 'nms_top_k', 'keep_top_k', 'score_threshold', 'nms_eta', 'return_index'], varargs=None, keywords=None, defaults=(0, 0.3, 400, 200, 0.01, 1.0, False)), ('document', '5485bcaceb0cde2695565a2ffd5bbd40'))

paddle.fluid.layers.ssd_loss (ArgSpec(args=['location', 'confidence', 'gt_box', 'gt_label', 'prior_box', 'prior_box_var', 'background_label', 'overlap_threshold', 'neg_pos_ratio', 'neg_overlap', 'loc_loss_weight', 'conf_loss_weight', 'match_type', 'mining_type', 'normalize', 'sample_size'], varargs=None, keywords=None, defaults=(None, 0, 0.5, 3.0, 0.5, 1.0, 1.0, 'per_prediction', 'max_negative', True, None)), ('document', '14d1eeae0f41b6792be43c1c0be0589b'))

paddle.fluid.layers.rpn_target_assign (ArgSpec(args=['bbox_pred', 'cls_logits', 'anchor_box', 'anchor_var', 'gt_boxes', 'is_crowd', 'im_info', 'rpn_batch_size_per_im', 'rpn_straddle_thresh', 'rpn_fg_fraction', 'rpn_positive_overlap', 'rpn_negative_overlap', 'use_random'], varargs=None, keywords=None, defaults=(256, 0.0, 0.5, 0.7, 0.3, True)), ('document', '651d98d51879dfa1bc1cd40391786a41'))

paddle.fluid.layers.retinanet_target_assign (ArgSpec(args=['bbox_pred', 'cls_logits', 'anchor_box', 'anchor_var', 'gt_boxes', 'gt_labels', 'is_crowd', 'im_info', 'num_classes', 'positive_overlap', 'negative_overlap'], varargs=None, keywords=None, defaults=(1, 0.5, 0.4)), ('document', 'fa1d1c9d5e0111684c0db705f86a2595'))

paddle.fluid.layers.sigmoid_focal_loss (ArgSpec(args=['x', 'label', 'fg_num', 'gamma', 'alpha'], varargs=None, keywords=None, defaults=(2, 0.25)), ('document', 'aeac6aae100173b3fc7f102cf3023a3d'))

paddle.fluid.layers.anchor_generator (ArgSpec(args=['input', 'anchor_sizes', 'aspect_ratios', 'variance', 'stride', 'offset', 'name'], varargs=None, keywords=None, defaults=(None, None, [0.1, 0.1, 0.2, 0.2], None, 0.5, None)), ('document', 'd25e5e90f9a342764f32b5cd48657148'))

paddle.fluid.layers.roi_perspective_transform (ArgSpec(args=['input', 'rois', 'transformed_height', 'transformed_width', 'spatial_scale'], varargs=None, keywords=None, defaults=(1.0,)), ('document', 'a82016342789ba9d85737e405f824ff1'))

paddle.fluid.layers.generate_proposal_labels (ArgSpec(args=['rpn_rois', 'gt_classes', 'is_crowd', 'gt_boxes', 'im_info', 'batch_size_per_im', 'fg_fraction', 'fg_thresh', 'bg_thresh_hi', 'bg_thresh_lo', 'bbox_reg_weights', 'class_nums', 'use_random', 'is_cls_agnostic', 'is_cascade_rcnn'], varargs=None, keywords=None, defaults=(256, 0.25, 0.25, 0.5, 0.0, [0.1, 0.1, 0.2, 0.2], None, True, False, False)), ('document', '69def376b42ef0681d0cc7f53a2dac4b'))

paddle.fluid.layers.generate_proposals (ArgSpec(args=['scores', 'bbox_deltas', 'im_info', 'anchors', 'variances', 'pre_nms_top_n', 'post_nms_top_n', 'nms_thresh', 'min_size', 'eta', 'name'], varargs=None, keywords=None, defaults=(6000, 1000, 0.5, 0.1, 1.0, None)), ('document', 'b7d707822b6af2a586bce608040235b1'))

paddle.fluid.layers.generate_mask_labels (ArgSpec(args=['im_info', 'gt_classes', 'is_crowd', 'gt_segms', 'rois', 'labels_int32', 'num_classes', 'resolution'], varargs=None, keywords=None, defaults=None), ('document', 'b319b10ddaf17fb4ddf03518685a17ef'))

paddle.fluid.layers.iou_similarity (ArgSpec(args=['x', 'y', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '72fca4a39ccf82d5c746ae62d1868a99'))

paddle.fluid.layers.box_coder (ArgSpec(args=['prior_box', 'prior_box_var', 'target_box', 'code_type', 'box_normalized', 'name', 'axis'], varargs=None, keywords=None, defaults=('encode_center_size', True, None, 0)), ('document', '1d5144c3856673d05c29c752c7c8f821'))

paddle.fluid.layers.polygon_box_transform (ArgSpec(args=['input', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '4b6e1b6b113147ea7234448a6a271e3b'))

paddle.fluid.layers.yolov3_loss (ArgSpec(args=['x', 'gt_box', 'gt_label', 'anchors', 'anchor_mask', 'class_num', 'ignore_thresh', 'downsample_ratio', 'gt_score', 'use_label_smooth', 'name'], varargs=None, keywords=None, defaults=(None, True, None)), ('document', '400403175718d5a632402cdae88b01b8'))

paddle.fluid.layers.yolo_box (ArgSpec(args=['x', 'img_size', 'anchors', 'class_num', 'conf_thresh', 'downsample_ratio', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'ed56ff21536ca5c8ad418d0cfaf6a7b9'))

paddle.fluid.layers.box_clip (ArgSpec(args=['input', 'im_info', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '882c99ed2adad54f612a40275b881850'))

paddle.fluid.layers.multiclass_nms (ArgSpec(args=['bboxes', 'scores', 'score_threshold', 'nms_top_k', 'keep_top_k', 'nms_threshold', 'normalized', 'nms_eta', 'background_label', 'name'], varargs=None, keywords=None, defaults=(0.3, True, 1.0, 0, None)), ('document', 'f6e333d76922c6e564413b4d216c245c'))

paddle.fluid.layers.multiclass_nms2 (ArgSpec(args=['bboxes', 'scores', 'score_threshold', 'nms_top_k', 'keep_top_k', 'nms_threshold', 'normalized', 'nms_eta', 'background_label', 'return_index', 'name'], varargs=None, keywords=None, defaults=(0.3, True, 1.0, 0, False, None)), ('document', 'be156186ee7a2ee56ab30b964acb15e5'))

paddle.fluid.layers.retinanet_detection_output (ArgSpec(args=['bboxes', 'scores', 'anchors', 'im_info', 'score_threshold', 'nms_top_k', 'keep_top_k', 'nms_threshold', 'nms_eta'], varargs=None, keywords=None, defaults=(0.05, 1000, 100, 0.3, 1.0)), ('document', '078d28607ce261a0cba2b965a79f6bb8'))

paddle.fluid.layers.distribute_fpn_proposals (ArgSpec(args=['fpn_rois', 'min_level', 'max_level', 'refer_level', 'refer_scale', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', 'be432c9b5f19ccba7aca38789ead29e4'))

paddle.fluid.layers.box_decoder_and_assign (ArgSpec(args=['prior_box', 'prior_box_var', 'target_box', 'box_score', 'box_clip', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '5203935538d06a6d47b8630ad80cb2b0'))

paddle.fluid.layers.collect_fpn_proposals (ArgSpec(args=['multi_rois', 'multi_scores', 'min_level', 'max_level', 'post_nms_top_n', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '808fcca082e0040e2b77dbc53a0cf9d5'))

paddle.fluid.layers.accuracy (ArgSpec(args=['input', 'label', 'k', 'correct', 'total'], varargs=None, keywords=None, defaults=(1, None, None)), ('document', 'b691b7be425e281bd36897b514b2b064'))

paddle.fluid.layers.auc (ArgSpec(args=['input', 'label', 'curve', 'num_thresholds', 'topk', 'slide_steps'], varargs=None, keywords=None, defaults=('ROC', 4095, 1, 1)), ('document', 'c36ac7125da977c2bd1b192bee301f75'))

paddle.fluid.layers.exponential_decay (ArgSpec(args=['learning_rate', 'decay_steps', 'decay_rate', 'staircase'], varargs=None, keywords=None, defaults=(False,)), ('document', 'eaf430c5a0380fb11bfe9a8922cd6295'))

paddle.fluid.layers.natural_exp_decay (ArgSpec(args=['learning_rate', 'decay_steps', 'decay_rate', 'staircase'], varargs=None, keywords=None, defaults=(False,)), ('document', 'aa3146f64d5d508e4e50687603aa7b15'))

paddle.fluid.layers.inverse_time_decay (ArgSpec(args=['learning_rate', 'decay_steps', 'decay_rate', 'staircase'], varargs=None, keywords=None, defaults=(False,)), ('document', 'ea37a3a8a0b3ce2254e7bc49a0951dbe'))

paddle.fluid.layers.polynomial_decay (ArgSpec(args=['learning_rate', 'decay_steps', 'end_learning_rate', 'power', 'cycle'], varargs=None, keywords=None, defaults=(0.0001, 1.0, False)), ('document', 'a343254c36c2e89512cd8cd8a1960ead'))

paddle.fluid.layers.piecewise_decay (ArgSpec(args=['boundaries', 'values'], varargs=None, keywords=None, defaults=None), ('document', 'd9f654117542c6b702963dda107a247f'))

paddle.fluid.layers.noam_decay (ArgSpec(args=['d_model', 'warmup_steps'], varargs=None, keywords=None, defaults=None), ('document', 'fd57228fb76195e66bbcc8d8e42c494d'))

paddle.fluid.layers.cosine_decay (ArgSpec(args=['learning_rate', 'step_each_epoch', 'epochs'], varargs=None, keywords=None, defaults=None), ('document', '1062e487dd3b50a6e58b5703b4f594c9'))

paddle.fluid.layers.linear_lr_warmup (ArgSpec(args=['learning_rate', 'warmup_steps', 'start_lr', 'end_lr'], varargs=None, keywords=None, defaults=None), ('document', 'dc7292c456847ba41cfd318e9f7f4363'))

paddle.fluid.layers.Uniform ('paddle.fluid.layers.distributions.Uniform', ('document', 'af70e7003f437e7a8a9e28cded35c433'))

paddle.fluid.layers.Uniform.__init__ (ArgSpec(args=['self', 'low', 'high'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.Uniform.entropy (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', 'ba59f9ce77af3c93e2b4c8af1801a24e'))

paddle.fluid.layers.Uniform.kl_divergence (ArgSpec(args=['self', 'other'], varargs=None, keywords=None, defaults=None), ('document', '3baee52abbed82d47e9588d9dfe2f42f'))

paddle.fluid.layers.Uniform.log_prob (ArgSpec(args=['self', 'value'], varargs=None, keywords=None, defaults=None), ('document', 'b79091014ceaffb6a7372a198a341c23'))

paddle.fluid.layers.Uniform.sample (ArgSpec(args=['self', 'shape', 'seed'], varargs=None, keywords=None, defaults=(0,)), ('document', 'adac334af13f6984e991b3ecf12b8cb7'))

paddle.fluid.layers.Normal ('paddle.fluid.layers.distributions.Normal', ('document', '3265262d0d8b3b32c6245979a5cdced9'))

paddle.fluid.layers.Normal.__init__ (ArgSpec(args=['self', 'loc', 'scale'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.Normal.entropy (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', 'd2db47b1e62c037a2570fc526b93f518'))

paddle.fluid.layers.Normal.kl_divergence (ArgSpec(args=['self', 'other'], varargs=None, keywords=None, defaults=None), ('document', '2e8845cdf1129647e6fa6e816876cd3b'))

paddle.fluid.layers.Normal.log_prob (ArgSpec(args=['self', 'value'], varargs=None, keywords=None, defaults=None), ('document', 'b79091014ceaffb6a7372a198a341c23'))

paddle.fluid.layers.Normal.sample (ArgSpec(args=['self', 'shape', 'seed'], varargs=None, keywords=None, defaults=(0,)), ('document', 'adac334af13f6984e991b3ecf12b8cb7'))

paddle.fluid.layers.Categorical ('paddle.fluid.layers.distributions.Categorical', ('document', '865c9dac8af6190e05588486ba091ee8'))

paddle.fluid.layers.Categorical.__init__ (ArgSpec(args=['self', 'logits'], varargs=None, keywords=None, defaults=None), ('document', '933b96c9ebab8e2c1f6007a50287311e'))

paddle.fluid.layers.Categorical.entropy (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', 'b360a2a7a4da07c2d268b329e09c82c1'))

paddle.fluid.layers.Categorical.kl_divergence (ArgSpec(args=['self', 'other'], varargs=None, keywords=None, defaults=None), ('document', 'c2c4c37376584178025f0a4a61c4b862'))

paddle.fluid.layers.Categorical.log_prob (ArgSpec(args=['self', 'value'], varargs=None, keywords=None, defaults=None), ('document', 'c0edd2e2fc76711477b32dc4da9de768'))

paddle.fluid.layers.Categorical.sample (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '08a2bbcaa20ee176ee7ec3d05737a0f6'))

paddle.fluid.layers.MultivariateNormalDiag ('paddle.fluid.layers.distributions.MultivariateNormalDiag', ('document', 'f6ee0e8b2898796dcff2a68c9fda19f0'))

paddle.fluid.layers.MultivariateNormalDiag.__init__ (ArgSpec(args=['self', 'loc', 'scale'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.layers.MultivariateNormalDiag.entropy (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '3c679b573ba975c5067c8ebfd4354b02'))

paddle.fluid.layers.MultivariateNormalDiag.kl_divergence (ArgSpec(args=['self', 'other'], varargs=None, keywords=None, defaults=None), ('document', 'd9190d29dbd54c81f747a6436c35f062'))

paddle.fluid.layers.MultivariateNormalDiag.log_prob (ArgSpec(args=['self', 'value'], varargs=None, keywords=None, defaults=None), ('document', 'c0edd2e2fc76711477b32dc4da9de768'))

paddle.fluid.layers.MultivariateNormalDiag.sample (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '08a2bbcaa20ee176ee7ec3d05737a0f6'))

paddle.fluid.contrib.InitState ('paddle.fluid.contrib.decoder.beam_search_decoder.InitState', ('document', '3afd1f84232718e628e9e566941c5f05'))

paddle.fluid.contrib.InitState.__init__ (ArgSpec(args=['self', 'init', 'shape', 'value', 'init_boot', 'need_reorder', 'dtype'], varargs=None, keywords=None, defaults=(None, None, 0.0, None, False, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.StateCell ('paddle.fluid.contrib.decoder.beam_search_decoder.StateCell', ('document', 'ecd0066c02867d445d7b461e28220c50'))

paddle.fluid.contrib.StateCell.__init__ (ArgSpec(args=['self', 'inputs', 'states', 'out_state', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.StateCell.compute_state (ArgSpec(args=['self', 'inputs'], varargs=None, keywords=None, defaults=None), ('document', '92973b3f222081a1d17069c683cf4a99'))

paddle.fluid.contrib.StateCell.get_input (ArgSpec(args=['self', 'input_name'], varargs=None, keywords=None, defaults=None), ('document', '6f24a007cfa184e32f01a960703bfd70'))

paddle.fluid.contrib.StateCell.get_state (ArgSpec(args=['self', 'state_name'], varargs=None, keywords=None, defaults=None), ('document', '630a4945cfe659ea4f307598fbbce5d2'))

paddle.fluid.contrib.StateCell.out_state (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '7ad681dff0393ddf13a724194e720f28'))

paddle.fluid.contrib.StateCell.set_state (ArgSpec(args=['self', 'state_name', 'state_value'], varargs=None, keywords=None, defaults=None), ('document', 'd4e0e08cd5d9d9a571cbc52d114f5ae9'))

paddle.fluid.contrib.StateCell.state_updater (ArgSpec(args=['self', 'updater'], varargs=None, keywords=None, defaults=None), ('document', 'd5afe1b7665d94fb023b15cf913ca510'))

paddle.fluid.contrib.StateCell.update_states (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', 'fe0b0f1338723516a35a30247899c81b'))

paddle.fluid.contrib.TrainingDecoder ('paddle.fluid.contrib.decoder.beam_search_decoder.TrainingDecoder', ('document', 'cec7e190c2bdd3b17e8178bc3f177799'))

paddle.fluid.contrib.TrainingDecoder.__init__ (ArgSpec(args=['self', 'state_cell', 'name'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.TrainingDecoder.block (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '98d88fa1c989748410a12517c6a585bf'))

paddle.fluid.contrib.TrainingDecoder.output (ArgSpec(args=['self'], varargs='outputs', keywords=None, defaults=None), ('document', 'f0a457dee586559036202087ce2eff69'))

paddle.fluid.contrib.TrainingDecoder.static_input (ArgSpec(args=['self', 'x'], varargs=None, keywords=None, defaults=None), ('document', 'a024c72664fe815068423ba630b7658a'))

paddle.fluid.contrib.TrainingDecoder.step_input (ArgSpec(args=['self', 'x'], varargs=None, keywords=None, defaults=None), ('document', '4659db7a888a2495e71c1838a0483909'))

paddle.fluid.contrib.BeamSearchDecoder ('paddle.fluid.contrib.decoder.beam_search_decoder.BeamSearchDecoder', ('document', '102da4a2d2002fbb12d44b8ea36121ed'))

paddle.fluid.contrib.BeamSearchDecoder.__init__ (ArgSpec(args=['self', 'state_cell', 'init_ids', 'init_scores', 'target_dict_dim', 'word_dim', 'input_var_dict', 'topk_size', 'sparse_emb', 'max_len', 'beam_size', 'end_id', 'name'], varargs=None, keywords=None, defaults=({}, 50, True, 100, 1, 1, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BeamSearchDecoder.block (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '98d88fa1c989748410a12517c6a585bf'))

paddle.fluid.contrib.BeamSearchDecoder.decode (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '1e47c60f080c1343ebb6ceaef89656b2'))

paddle.fluid.contrib.BeamSearchDecoder.early_stop (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '3a84a7454ed6707f79b9e954d92a7575'))

paddle.fluid.contrib.BeamSearchDecoder.read_array (ArgSpec(args=['self', 'init', 'is_ids', 'is_scores'], varargs=None, keywords=None, defaults=(False, False)), ('document', 'aa89eb8fd5e4cabaf5cc1bcae14665a4'))

paddle.fluid.contrib.BeamSearchDecoder.update_array (ArgSpec(args=['self', 'array', 'value'], varargs=None, keywords=None, defaults=None), ('document', '5754e9b3212b7c09497151516a0de5a7'))

paddle.fluid.contrib.memory_usage (ArgSpec(args=['program', 'batch_size'], varargs=None, keywords=None, defaults=None), ('document', '8fcb2f93bb743693baa8d4860a5ccc47'))

paddle.fluid.contrib.op_freq_statistic (ArgSpec(args=['program'], varargs=None, keywords=None, defaults=None), ('document', '4d43687113c4bf5b29d15aee2f4e4afa'))

paddle.fluid.contrib.QuantizeTranspiler ('paddle.fluid.contrib.quantize.quantize_transpiler.QuantizeTranspiler', ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.QuantizeTranspiler.__init__ (ArgSpec(args=['self', 'weight_bits', 'activation_bits', 'activation_quantize_type', 'weight_quantize_type', 'window_size', 'moving_rate'], varargs=None, keywords=None, defaults=(8, 8, 'abs_max', 'abs_max', 10000, 0.9)), ('document', '14b39f1fcd5667ff556b1aad94357d1d'))

paddle.fluid.contrib.QuantizeTranspiler.convert_to_int8 (ArgSpec(args=['self', 'program', 'place', 'scope'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.QuantizeTranspiler.freeze_program (ArgSpec(args=['self', 'program', 'place', 'scope'], varargs=None, keywords=None, defaults=(None,)), ('document', '909675a1ab055c69b436a7893fcae4fd'))

paddle.fluid.contrib.QuantizeTranspiler.training_transpile (ArgSpec(args=['self', 'program', 'startup_program'], varargs=None, keywords=None, defaults=(None, None)), ('document', '6dd9909f10b283ba2892a99058a72884'))

paddle.fluid.contrib.distributed_batch_reader (ArgSpec(args=['batch_reader'], varargs=None, keywords=None, defaults=None), ('document', 'b60796eb0a481484dd34e345f0eaa4d5'))

paddle.fluid.contrib.Compressor ('paddle.fluid.contrib.slim.core.compressor.Compressor', ('document', 'a5417774a94aa9ae5560a42b96527e7d'))

paddle.fluid.contrib.Compressor.__init__ (ArgSpec(args=['self', 'place', 'scope', 'train_program', 'train_reader', 'train_feed_list', 'train_fetch_list', 'eval_program', 'eval_reader', 'eval_feed_list', 'eval_fetch_list', 'eval_func', 'save_eval_model', 'prune_infer_model', 'teacher_programs', 'checkpoint_path', 'train_optimizer', 'distiller_optimizer', 'search_space', 'log_period'], varargs=None, keywords=None, defaults=(None, None, None, None, None, None, None, None, True, None, [], None, None, None, None, 20)), ('document', '26261076fa2140a1367cb4fbf3ac03fa'))

paddle.fluid.contrib.Compressor.config (ArgSpec(args=['self', 'config_file'], varargs=None, keywords=None, defaults=None), ('document', '780d9c007276ccbb95b292400d7807b0'))

paddle.fluid.contrib.Compressor.run (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', 'c6e43d6a078d307672283c1f36e04fe9'))

paddle.fluid.contrib.load_persistables_for_increment (ArgSpec(args=['dirname', 'executor', 'program', 'lookup_table_var', 'lookup_table_var_path'], varargs=None, keywords=None, defaults=None), ('document', '2ab36d4f7a564f5f65e455807ad06c67'))

paddle.fluid.contrib.load_persistables_for_inference (ArgSpec(args=['dirname', 'executor', 'program', 'lookup_table_var_name'], varargs=None, keywords=None, defaults=None), ('document', '59066bac9db0ac6ce414d05780b7333f'))

paddle.fluid.contrib.convert_dist_to_sparse_program (ArgSpec(args=['program'], varargs=None, keywords=None, defaults=None), ('document', '74c39c595dc70d6be2f16d8e462d282b'))

paddle.fluid.contrib.HDFSClient ('paddle.fluid.contrib.utils.hdfs_utils.HDFSClient', ('document', '31207aa18424eab2249c54fe11724798'))

paddle.fluid.contrib.HDFSClient.__init__ (ArgSpec(args=['self', 'hadoop_home', 'configs'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.HDFSClient.delete (ArgSpec(args=['self', 'hdfs_path'], varargs=None, keywords=None, defaults=None), ('document', 'c3721aa2d4d9ef5a857dd47b2681c03e'))

paddle.fluid.contrib.HDFSClient.download (ArgSpec(args=['self', 'hdfs_path', 'local_path', 'overwrite', 'unzip'], varargs=None, keywords=None, defaults=(False, False)), ('document', 'ca55bde92184d3fd0f9f5c963b25e634'))

paddle.fluid.contrib.HDFSClient.is_dir (ArgSpec(args=['self', 'hdfs_path'], varargs=None, keywords=None, defaults=(None,)), ('document', '45bde1bae02605a205c8245b58b9156d'))

paddle.fluid.contrib.HDFSClient.is_exist (ArgSpec(args=['self', 'hdfs_path'], varargs=None, keywords=None, defaults=(None,)), ('document', 'be9c94bccff7ba0c1d95883ac62b5864'))

paddle.fluid.contrib.HDFSClient.ls (ArgSpec(args=['self', 'hdfs_path'], varargs=None, keywords=None, defaults=None), ('document', '808acac504870c7e46594b95674f8a86'))

paddle.fluid.contrib.HDFSClient.lsr (ArgSpec(args=['self', 'hdfs_path', 'only_file', 'sort'], varargs=None, keywords=None, defaults=(True, True)), ('document', 'fae835aa3354eb6a0434c0f9ba3c2747'))

paddle.fluid.contrib.HDFSClient.make_local_dirs (ArgSpec(args=['local_path'], varargs=None, keywords=None, defaults=None), ('document', 'e76b89c8e7f019b5da576c0026fcf689'))

paddle.fluid.contrib.HDFSClient.makedirs (ArgSpec(args=['self', 'hdfs_path'], varargs=None, keywords=None, defaults=None), ('document', '44d9972aae390aedf40aaea731a37e4b'))

paddle.fluid.contrib.HDFSClient.rename (ArgSpec(args=['self', 'hdfs_src_path', 'hdfs_dst_path', 'overwrite'], varargs=None, keywords=None, defaults=(False,)), ('document', '0eb133644d9a9f4da45bb39261ff0955'))

paddle.fluid.contrib.HDFSClient.upload (ArgSpec(args=['self', 'hdfs_path', 'local_path', 'overwrite', 'retry_times'], varargs=None, keywords=None, defaults=(False, 5)), ('document', '7d053b4bfd6dcfdd2c9dda0e0dbd9665'))

paddle.fluid.contrib.multi_download (ArgSpec(args=['client', 'hdfs_path', 'local_path', 'trainer_id', 'trainers', 'multi_processes'], varargs=None, keywords=None, defaults=(5,)), ('document', '100927be598ed8f9eaa1f3ef1b23568a'))

paddle.fluid.contrib.multi_upload (ArgSpec(args=['client', 'hdfs_path', 'local_path', 'multi_processes', 'overwrite', 'sync'], varargs=None, keywords=None, defaults=(5, False, True)), ('document', '183f34c83d30dbe16e09e8716c41958a'))

paddle.fluid.contrib.extend_with_decoupled_weight_decay (ArgSpec(args=['base_optimizer'], varargs=None, keywords=None, defaults=None), ('document', 'a1095dfd4ec725747f662d69cd7659d4'))

paddle.fluid.contrib.mixed_precision.decorate (ArgSpec(args=['optimizer', 'amp_lists', 'init_loss_scaling', 'incr_every_n_steps', 'decr_every_n_nan_or_inf', 'incr_ratio', 'decr_ratio', 'use_dynamic_loss_scaling'], varargs=None, keywords=None, defaults=(None, 1.0, 1000, 2, 2.0, 0.8, True)), ('document', '5f118631fc8632afb981b3a26daae731'))

paddle.fluid.contrib.mixed_precision.AutoMixedPrecisionLists ('paddle.fluid.contrib.mixed_precision.fp16_lists.AutoMixedPrecisionLists', ('document', 'c116ec6bb5d30998792daea8db21ee40'))

paddle.fluid.contrib.mixed_precision.AutoMixedPrecisionLists.__init__ (ArgSpec(args=['self', 'custom_white_list', 'custom_black_list'], varargs=None, keywords=None, defaults=(None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.fused_elemwise_activation (ArgSpec(args=['x', 'y', 'functor_list', 'axis', 'scale', 'save_intermediate_out'], varargs=None, keywords=None, defaults=(-1, 0.0, True)), ('document', '1c4b247a2858cea8d9d8750693688270'))

paddle.fluid.contrib.sequence_topk_avg_pooling (ArgSpec(args=['input', 'row', 'col', 'topks', 'channel_num'], varargs=None, keywords=None, defaults=None), ('document', '5218c85dd4122b626da9bb92f3b50042'))

paddle.fluid.contrib.var_conv_2d (ArgSpec(args=['input', 'row', 'col', 'input_channel', 'output_channel', 'filter_size', 'stride', 'param_attr', 'act', 'dtype', 'name'], varargs=None, keywords=None, defaults=(1, None, None, 'float32', None)), ('document', 'f52a6edf6d3e970568788604da3329c2'))

paddle.fluid.contrib.match_matrix_tensor (ArgSpec(args=['x', 'y', 'channel_num', 'act', 'param_attr', 'dtype', 'name'], varargs=None, keywords=None, defaults=(None, None, 'float32', None)), ('document', '3bdc4b2891c1460bc630fdcd22766b21'))

paddle.fluid.contrib.tree_conv (ArgSpec(args=['nodes_vector', 'edge_set', 'output_size', 'num_filters', 'max_depth', 'act', 'param_attr', 'bias_attr', 'name'], varargs=None, keywords=None, defaults=(1, 2, 'tanh', None, None, None)), ('document', '7c727562ebdda38274106d1a9b338e5b'))

paddle.fluid.contrib.BasicGRUUnit ('paddle.fluid.contrib.layers.rnn_impl.BasicGRUUnit', ('document', '2aed2540ed1540f081be9f4d08f2a65e'))

paddle.fluid.contrib.BasicGRUUnit.__init__ (ArgSpec(args=['self', 'name_scope', 'hidden_size', 'param_attr', 'bias_attr', 'gate_activation', 'activation', 'dtype'], varargs=None, keywords=None, defaults=(None, None, None, None, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicGRUUnit.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.contrib.BasicGRUUnit.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.contrib.BasicGRUUnit.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicGRUUnit.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicGRUUnit.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.contrib.BasicGRUUnit.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.contrib.BasicGRUUnit.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicGRUUnit.forward (ArgSpec(args=['self', 'input', 'pre_hidden'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicGRUUnit.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.contrib.BasicGRUUnit.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicGRUUnit.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.contrib.BasicGRUUnit.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicGRUUnit.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.contrib.BasicGRUUnit.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.basic_gru (ArgSpec(args=['input', 'init_hidden', 'hidden_size', 'num_layers', 'sequence_length', 'dropout_prob', 'bidirectional', 'batch_first', 'param_attr', 'bias_attr', 'gate_activation', 'activation', 'dtype', 'name'], varargs=None, keywords=None, defaults=(1, None, 0.0, False, True, None, None, None, None, 'float32', 'basic_gru')), ('document', '0afcbe4fbe1b8c35eda58b4efe48f9fd'))

paddle.fluid.contrib.BasicLSTMUnit ('paddle.fluid.contrib.layers.rnn_impl.BasicLSTMUnit', ('document', '3d0b2e3172ce58e1304199efee066c99'))

paddle.fluid.contrib.BasicLSTMUnit.__init__ (ArgSpec(args=['self', 'name_scope', 'hidden_size', 'param_attr', 'bias_attr', 'gate_activation', 'activation', 'forget_bias', 'dtype'], varargs=None, keywords=None, defaults=(None, None, None, None, 1.0, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicLSTMUnit.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.contrib.BasicLSTMUnit.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.contrib.BasicLSTMUnit.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicLSTMUnit.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicLSTMUnit.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.contrib.BasicLSTMUnit.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.contrib.BasicLSTMUnit.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicLSTMUnit.forward (ArgSpec(args=['self', 'input', 'pre_hidden', 'pre_cell'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicLSTMUnit.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.contrib.BasicLSTMUnit.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicLSTMUnit.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.contrib.BasicLSTMUnit.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.BasicLSTMUnit.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.contrib.BasicLSTMUnit.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.contrib.basic_lstm (ArgSpec(args=['input', 'init_hidden', 'init_cell', 'hidden_size', 'num_layers', 'sequence_length', 'dropout_prob', 'bidirectional', 'batch_first', 'param_attr', 'bias_attr', 'gate_activation', 'activation', 'forget_bias', 'dtype', 'name'], varargs=None, keywords=None, defaults=(1, None, 0.0, False, True, None, None, None, None, 1.0, 'float32', 'basic_lstm')), ('document', 'fe4d0c3c55a162b8cfe10b05fabb7ce4'))

paddle.fluid.contrib.ctr_metric_bundle (ArgSpec(args=['input', 'label'], varargs=None, keywords=None, defaults=None), ('document', 'b68d12366896c41065fc3738393da2aa'))

paddle.fluid.data (ArgSpec(args=['name', 'shape', 'dtype', 'lod_level'], varargs=None, keywords=None, defaults=('float32', 0)), ('document', '28ee6b35836449f44ce53467f0616137'))

paddle.fluid.dygraph.Layer ('paddle.fluid.dygraph.layers.Layer', ('document', 'a889d5affd734ede273e94d4257163ab'))

paddle.fluid.dygraph.Layer.__init__ (ArgSpec(args=['self', 'name_scope', 'dtype'], varargs=None, keywords=None, defaults=(VarType.FP32,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Layer.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.Layer.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.Layer.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Layer.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Layer.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.Layer.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.Layer.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Layer.forward (ArgSpec(args=['self'], varargs='inputs', keywords='kwargs', defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Layer.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.Layer.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Layer.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.Layer.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Layer.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.Layer.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.__impl__ (ArgSpec(args=['func'], varargs=None, keywords=None, defaults=()), ('document', '75d1d3afccc8b39cdebf05cb1f5969f9'))

paddle.fluid.dygraph.guard (ArgSpec(args=['place'], varargs=None, keywords=None, defaults=(None,)), ('document', '7071320ffe2eec9aacdae574951278c6'))

paddle.fluid.dygraph.to_variable (ArgSpec(args=['value', 'block', 'name'], varargs=None, keywords=None, defaults=(None, None)), ('document', '0e69fa3666f15dd01b6e3e270b9371cd'))

paddle.fluid.dygraph.Conv2D ('paddle.fluid.dygraph.nn.Conv2D', ('document', '0b6acb9cc7fbb4f5b129e1f6dd985581'))

paddle.fluid.dygraph.Conv2D.__init__ (ArgSpec(args=['self', 'name_scope', 'num_filters', 'filter_size', 'stride', 'padding', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act', 'dtype'], varargs=None, keywords=None, defaults=(1, 0, 1, None, None, None, True, None, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2D.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.Conv2D.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.Conv2D.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2D.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2D.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.Conv2D.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.Conv2D.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2D.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2D.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.Conv2D.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2D.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.Conv2D.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2D.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.Conv2D.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3D ('paddle.fluid.dygraph.nn.Conv3D', ('document', '50412bd3fbf3557a8ef48e25c6517025'))

paddle.fluid.dygraph.Conv3D.__init__ (ArgSpec(args=['self', 'name_scope', 'num_filters', 'filter_size', 'stride', 'padding', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act'], varargs=None, keywords=None, defaults=(1, 0, 1, None, None, None, True, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3D.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.Conv3D.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.Conv3D.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3D.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3D.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.Conv3D.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.Conv3D.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3D.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3D.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.Conv3D.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3D.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.Conv3D.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3D.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.Conv3D.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Pool2D ('paddle.fluid.dygraph.nn.Pool2D', ('document', '50e6fd200e42859daf2924ecb0561ada'))

paddle.fluid.dygraph.Pool2D.__init__ (ArgSpec(args=['self', 'name_scope', 'pool_size', 'pool_type', 'pool_stride', 'pool_padding', 'global_pooling', 'use_cudnn', 'ceil_mode', 'exclusive', 'dtype'], varargs=None, keywords=None, defaults=(-1, 'max', 1, 0, False, True, False, True, VarType.FP32)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Pool2D.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.Pool2D.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.Pool2D.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Pool2D.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Pool2D.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.Pool2D.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.Pool2D.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Pool2D.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Pool2D.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.Pool2D.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Pool2D.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.Pool2D.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Pool2D.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.Pool2D.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.FC ('paddle.fluid.dygraph.nn.FC', ('document', '2f73ae00e57c67454c6aa7e911d9bfd6'))

paddle.fluid.dygraph.FC.__init__ (ArgSpec(args=['self', 'name_scope', 'size', 'num_flatten_dims', 'param_attr', 'bias_attr', 'act', 'is_test', 'dtype'], varargs=None, keywords=None, defaults=(1, None, None, None, False, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.FC.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.FC.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.FC.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.FC.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.FC.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.FC.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.FC.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.FC.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.FC.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.FC.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.FC.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.FC.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.FC.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.FC.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BatchNorm ('paddle.fluid.dygraph.nn.BatchNorm', ('document', '390fb9b986423ec6680731ffc7cf24ab'))

paddle.fluid.dygraph.BatchNorm.__init__ (ArgSpec(args=['self', 'name_scope', 'num_channels', 'act', 'is_test', 'momentum', 'epsilon', 'param_attr', 'bias_attr', 'dtype', 'data_layout', 'in_place', 'moving_mean_name', 'moving_variance_name', 'do_model_average_for_mean_and_var', 'fuse_with_relu', 'use_global_stats', 'trainable_statistics'], varargs=None, keywords=None, defaults=(None, False, 0.9, 1e-05, None, None, 'float32', 'NCHW', False, None, None, False, False, False, False)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BatchNorm.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.BatchNorm.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.BatchNorm.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BatchNorm.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BatchNorm.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.BatchNorm.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.BatchNorm.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BatchNorm.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BatchNorm.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.BatchNorm.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BatchNorm.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.BatchNorm.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BatchNorm.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.BatchNorm.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Embedding ('paddle.fluid.dygraph.nn.Embedding', ('document', 'b1b1ed9dc2125c3e16ee08113605fcb4'))

paddle.fluid.dygraph.Embedding.__init__ (ArgSpec(args=['self', 'name_scope', 'size', 'is_sparse', 'is_distributed', 'padding_idx', 'param_attr', 'dtype'], varargs=None, keywords=None, defaults=(False, False, None, None, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Embedding.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.Embedding.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.Embedding.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Embedding.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Embedding.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.Embedding.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.Embedding.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Embedding.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Embedding.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.Embedding.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Embedding.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.Embedding.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Embedding.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.Embedding.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GRUUnit ('paddle.fluid.dygraph.nn.GRUUnit', ('document', '389e860e455b67aab1f4d472ac9d7e49'))

paddle.fluid.dygraph.GRUUnit.__init__ (ArgSpec(args=['self', 'name_scope', 'size', 'param_attr', 'bias_attr', 'activation', 'gate_activation', 'origin_mode', 'dtype'], varargs=None, keywords=None, defaults=(None, None, 'tanh', 'sigmoid', False, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GRUUnit.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.GRUUnit.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.GRUUnit.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GRUUnit.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GRUUnit.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.GRUUnit.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.GRUUnit.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GRUUnit.forward (ArgSpec(args=['self', 'input', 'hidden'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GRUUnit.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.GRUUnit.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GRUUnit.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.GRUUnit.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GRUUnit.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.GRUUnit.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.LayerNorm ('paddle.fluid.dygraph.nn.LayerNorm', ('document', '8bc39f59fe2d3713bc143fdf1222a63b'))

paddle.fluid.dygraph.LayerNorm.__init__ (ArgSpec(args=['self', 'name_scope', 'scale', 'shift', 'begin_norm_axis', 'epsilon', 'param_attr', 'bias_attr', 'act'], varargs=None, keywords=None, defaults=(True, True, 1, 1e-05, None, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.LayerNorm.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.LayerNorm.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.LayerNorm.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.LayerNorm.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.LayerNorm.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.LayerNorm.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.LayerNorm.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.LayerNorm.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.LayerNorm.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.LayerNorm.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.LayerNorm.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.LayerNorm.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.LayerNorm.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.LayerNorm.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NCE ('paddle.fluid.dygraph.nn.NCE', ('document', '993aeea9be436e9c709a758795cb23e9'))

paddle.fluid.dygraph.NCE.__init__ (ArgSpec(args=['self', 'name_scope', 'num_total_classes', 'sample_weight', 'param_attr', 'bias_attr', 'num_neg_samples', 'sampler', 'custom_dist', 'seed', 'is_sparse'], varargs=None, keywords=None, defaults=(None, None, None, None, 'uniform', None, 0, False)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NCE.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.NCE.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.NCE.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NCE.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NCE.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.NCE.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.NCE.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NCE.forward (ArgSpec(args=['self', 'input', 'label', 'sample_weight'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NCE.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.NCE.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NCE.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.NCE.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NCE.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.NCE.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PRelu ('paddle.fluid.dygraph.nn.PRelu', ('document', 'da956af1676b08bf15553751a3643b55'))

paddle.fluid.dygraph.PRelu.__init__ (ArgSpec(args=['self', 'name_scope', 'mode', 'param_attr'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PRelu.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.PRelu.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.PRelu.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PRelu.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PRelu.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.PRelu.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.PRelu.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PRelu.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PRelu.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.PRelu.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PRelu.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.PRelu.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PRelu.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.PRelu.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BilinearTensorProduct ('paddle.fluid.dygraph.nn.BilinearTensorProduct', ('document', 'be70d0f6d43729d9cb80c9a34ed5f26b'))

paddle.fluid.dygraph.BilinearTensorProduct.__init__ (ArgSpec(args=['self', 'name_scope', 'size', 'name', 'act', 'param_attr', 'bias_attr'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BilinearTensorProduct.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.BilinearTensorProduct.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.BilinearTensorProduct.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BilinearTensorProduct.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BilinearTensorProduct.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.BilinearTensorProduct.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.BilinearTensorProduct.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BilinearTensorProduct.forward (ArgSpec(args=['self', 'x', 'y'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BilinearTensorProduct.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.BilinearTensorProduct.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BilinearTensorProduct.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.BilinearTensorProduct.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BilinearTensorProduct.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.BilinearTensorProduct.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2DTranspose ('paddle.fluid.dygraph.nn.Conv2DTranspose', ('document', 'cf23c905abc00b07603dfa71a432d6f7'))

paddle.fluid.dygraph.Conv2DTranspose.__init__ (ArgSpec(args=['self', 'name_scope', 'num_filters', 'output_size', 'filter_size', 'padding', 'stride', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act'], varargs=None, keywords=None, defaults=(None, None, 0, 1, 1, None, None, None, True, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2DTranspose.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.Conv2DTranspose.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.Conv2DTranspose.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2DTranspose.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2DTranspose.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.Conv2DTranspose.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.Conv2DTranspose.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2DTranspose.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2DTranspose.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.Conv2DTranspose.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2DTranspose.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.Conv2DTranspose.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv2DTranspose.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.Conv2DTranspose.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3DTranspose ('paddle.fluid.dygraph.nn.Conv3DTranspose', ('document', '91ba132bc690eaf76eabdbde8f87e4a0'))

paddle.fluid.dygraph.Conv3DTranspose.__init__ (ArgSpec(args=['self', 'name_scope', 'num_filters', 'output_size', 'filter_size', 'padding', 'stride', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act', 'name'], varargs=None, keywords=None, defaults=(None, None, 0, 1, 1, None, None, None, True, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3DTranspose.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.Conv3DTranspose.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.Conv3DTranspose.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3DTranspose.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3DTranspose.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.Conv3DTranspose.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.Conv3DTranspose.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3DTranspose.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3DTranspose.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.Conv3DTranspose.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3DTranspose.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.Conv3DTranspose.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Conv3DTranspose.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.Conv3DTranspose.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GroupNorm ('paddle.fluid.dygraph.nn.GroupNorm', ('document', '72c125b07bdd1e612607dc77039b2722'))

paddle.fluid.dygraph.GroupNorm.__init__ (ArgSpec(args=['self', 'name_scope', 'groups', 'epsilon', 'param_attr', 'bias_attr', 'act', 'data_layout'], varargs=None, keywords=None, defaults=(1e-05, None, None, None, 'NCHW')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GroupNorm.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.GroupNorm.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.GroupNorm.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GroupNorm.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GroupNorm.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.GroupNorm.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.GroupNorm.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GroupNorm.forward (ArgSpec(args=['self', 'input'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GroupNorm.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.GroupNorm.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GroupNorm.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.GroupNorm.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.GroupNorm.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.GroupNorm.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.SpectralNorm ('paddle.fluid.dygraph.nn.SpectralNorm', ('document', '8f5cfbc431a8b4b44b605cde8b0381ef'))

paddle.fluid.dygraph.SpectralNorm.__init__ (ArgSpec(args=['self', 'name_scope', 'dim', 'power_iters', 'eps', 'name'], varargs=None, keywords=None, defaults=(0, 1, 1e-12, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.SpectralNorm.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.SpectralNorm.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.SpectralNorm.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.SpectralNorm.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.SpectralNorm.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.SpectralNorm.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.SpectralNorm.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.SpectralNorm.forward (ArgSpec(args=['self', 'weight'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.SpectralNorm.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.SpectralNorm.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.SpectralNorm.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.SpectralNorm.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.SpectralNorm.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.SpectralNorm.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.TreeConv ('paddle.fluid.dygraph.nn.TreeConv', ('document', '6e175a7bf2a43ae6c0f3a8a54bd69afe'))

paddle.fluid.dygraph.TreeConv.__init__ (ArgSpec(args=['self', 'name_scope', 'output_size', 'num_filters', 'max_depth', 'act', 'param_attr', 'bias_attr', 'name'], varargs=None, keywords=None, defaults=(1, 2, 'tanh', None, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.TreeConv.add_parameter (ArgSpec(args=['self', 'name', 'parameter'], varargs=None, keywords=None, defaults=None), ('document', 'f35ab374c7d5165c3daf3bd64a5a2ec1'))

paddle.fluid.dygraph.TreeConv.add_sublayer (ArgSpec(args=['self', 'name', 'sublayer'], varargs=None, keywords=None, defaults=None), ('document', '839ff3c0534677ba6ad8735c3fd4e995'))

paddle.fluid.dygraph.TreeConv.backward (ArgSpec(args=['self'], varargs='inputs', keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.TreeConv.clear_gradients (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.TreeConv.create_parameter (ArgSpec(args=['self', 'attr', 'shape', 'dtype', 'is_bias', 'default_initializer'], varargs=None, keywords=None, defaults=(False, None)), ('document', 'a6420ca1455366eaaf972191612de0b6'))

paddle.fluid.dygraph.TreeConv.create_variable (ArgSpec(args=['self', 'name', 'persistable', 'dtype', 'type'], varargs=None, keywords=None, defaults=(None, None, None, VarType.LOD_TENSOR)), ('document', '171cccfceba636d5bbf7bbae672945d8'))

paddle.fluid.dygraph.TreeConv.eval (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.TreeConv.forward (ArgSpec(args=['self', 'nodes_vector', 'edge_set'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.TreeConv.full_name (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '23ce4f961f48ed0f79cadf93a3938ed2'))

paddle.fluid.dygraph.TreeConv.load_dict (ArgSpec(args=['self', 'stat_dict', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.TreeConv.parameters (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '5aec25a854eb57abc798dccccbb507d5'))

paddle.fluid.dygraph.TreeConv.state_dict (ArgSpec(args=['self', 'destination', 'include_sublayers'], varargs=None, keywords=None, defaults=(None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.TreeConv.sublayers (ArgSpec(args=['self', 'include_sublayers'], varargs=None, keywords=None, defaults=(True,)), ('document', '00a881005ecbc96578faf94513bf0d62'))

paddle.fluid.dygraph.TreeConv.train (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Tracer ('paddle.fluid.dygraph.tracer.Tracer', ('document', '28d72409112111274c33e1f07229d5da'))

paddle.fluid.dygraph.Tracer.__init__ (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Tracer.all_parameters (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Tracer.eval_mode (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Tracer.trace 1. trace(self: paddle.fluid.core_avx.Tracer, arg0: unicode, arg1: Dict[unicode, handle], arg2: Dict[unicode, handle], arg3: Dict[unicode, Variant], arg4: paddle::platform::CUDAPlace, arg5: bool) -> None  2. trace(self: paddle.fluid.core_avx.Tracer, arg0: unicode, arg1: Dict[unicode, handle], arg2: Dict[unicode, handle], arg3: Dict[unicode, Variant], arg4: paddle::platform::CPUPlace, arg5: bool) -> None

paddle.fluid.dygraph.Tracer.trace_op (ArgSpec(args=['self', 'type', 'inputs', 'outputs', 'attrs', 'stop_gradient'], varargs=None, keywords=None, defaults=(False,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Tracer.trace_var (ArgSpec(args=['self', 'name', 'var'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.Tracer.train_mode (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.prepare_context (ArgSpec(args=['strategy'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.save_persistables (ArgSpec(args=['model_dict', 'dirname', 'optimizers'], varargs=None, keywords=None, defaults=('save_dir', None)), ('document', 'b0b2ec2a502214a737300fb648cb9dc7'))

paddle.fluid.dygraph.load_persistables (ArgSpec(args=['dirname'], varargs=None, keywords=None, defaults=('save_dir',)), ('document', 'e0709f8259620fdcfd2c0c1b23348852'))

paddle.fluid.dygraph.NoamDecay ('paddle.fluid.dygraph.learning_rate_scheduler.NoamDecay', ('document', '9ccfea97dbf15134d406a23aae1e1fa2'))

paddle.fluid.dygraph.NoamDecay.__init__ (ArgSpec(args=['self', 'd_model', 'warmup_steps', 'begin', 'step', 'dtype'], varargs=None, keywords=None, defaults=(1, 1, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NoamDecay.create_lr_var (ArgSpec(args=['self', 'lr'], varargs=None, keywords=None, defaults=None), ('document', '013bc233558149d0757b3df57845b866'))

paddle.fluid.dygraph.NoamDecay.step (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PiecewiseDecay ('paddle.fluid.dygraph.learning_rate_scheduler.PiecewiseDecay', ('document', '8f4d37eaad4e2f5b12850f3663856758'))

paddle.fluid.dygraph.PiecewiseDecay.__init__ (ArgSpec(args=['self', 'boundaries', 'values', 'begin', 'step', 'dtype'], varargs=None, keywords=None, defaults=(1, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PiecewiseDecay.create_lr_var (ArgSpec(args=['self', 'lr'], varargs=None, keywords=None, defaults=None), ('document', '013bc233558149d0757b3df57845b866'))

paddle.fluid.dygraph.PiecewiseDecay.step (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NaturalExpDecay ('paddle.fluid.dygraph.learning_rate_scheduler.NaturalExpDecay', ('document', '94bed58b392a5a71b6d1abd39eed7111'))

paddle.fluid.dygraph.NaturalExpDecay.__init__ (ArgSpec(args=['self', 'learning_rate', 'decay_steps', 'decay_rate', 'staircase', 'begin', 'step', 'dtype'], varargs=None, keywords=None, defaults=(False, 0, 1, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.NaturalExpDecay.create_lr_var (ArgSpec(args=['self', 'lr'], varargs=None, keywords=None, defaults=None), ('document', '013bc233558149d0757b3df57845b866'))

paddle.fluid.dygraph.NaturalExpDecay.step (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.ExponentialDecay ('paddle.fluid.dygraph.learning_rate_scheduler.ExponentialDecay', ('document', 'a259689c649c5f82636536386ce2ef19'))

paddle.fluid.dygraph.ExponentialDecay.__init__ (ArgSpec(args=['self', 'learning_rate', 'decay_steps', 'decay_rate', 'staircase', 'begin', 'step', 'dtype'], varargs=None, keywords=None, defaults=(False, 0, 1, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.ExponentialDecay.create_lr_var (ArgSpec(args=['self', 'lr'], varargs=None, keywords=None, defaults=None), ('document', '013bc233558149d0757b3df57845b866'))

paddle.fluid.dygraph.ExponentialDecay.step (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.InverseTimeDecay ('paddle.fluid.dygraph.learning_rate_scheduler.InverseTimeDecay', ('document', '6a868b2c7cc0f09f57ef71902bbc93ca'))

paddle.fluid.dygraph.InverseTimeDecay.__init__ (ArgSpec(args=['self', 'learning_rate', 'decay_steps', 'decay_rate', 'staircase', 'begin', 'step', 'dtype'], varargs=None, keywords=None, defaults=(False, 0, 1, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.InverseTimeDecay.create_lr_var (ArgSpec(args=['self', 'lr'], varargs=None, keywords=None, defaults=None), ('document', '013bc233558149d0757b3df57845b866'))

paddle.fluid.dygraph.InverseTimeDecay.step (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PolynomialDecay ('paddle.fluid.dygraph.learning_rate_scheduler.PolynomialDecay', ('document', 'bb90314cee58952f13522dcd571ca832'))

paddle.fluid.dygraph.PolynomialDecay.__init__ (ArgSpec(args=['self', 'learning_rate', 'decay_steps', 'end_learning_rate', 'power', 'cycle', 'begin', 'step', 'dtype'], varargs=None, keywords=None, defaults=(0.0001, 1.0, False, 0, 1, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.PolynomialDecay.create_lr_var (ArgSpec(args=['self', 'lr'], varargs=None, keywords=None, defaults=None), ('document', '013bc233558149d0757b3df57845b866'))

paddle.fluid.dygraph.PolynomialDecay.step (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.CosineDecay ('paddle.fluid.dygraph.learning_rate_scheduler.CosineDecay', ('document', '46dadadee1a8a92d70bd277d9345bfb0'))

paddle.fluid.dygraph.CosineDecay.__init__ (ArgSpec(args=['self', 'learning_rate', 'step_each_epoch', 'epochs', 'begin', 'step', 'dtype'], varargs=None, keywords=None, defaults=(0, 1, 'float32')), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.CosineDecay.create_lr_var (ArgSpec(args=['self', 'lr'], varargs=None, keywords=None, defaults=None), ('document', '013bc233558149d0757b3df57845b866'))

paddle.fluid.dygraph.CosineDecay.step (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph.BackwardStrategy ('paddle.fluid.core_avx.BackwardStrategy', ('document', '5d9496052ec793810c9f12ffad5c73ce'))

paddle.fluid.dygraph.BackwardStrategy.__init__ __init__(self: paddle.fluid.core_avx.BackwardStrategy) -> None

paddle.fluid.transpiler.DistributeTranspiler ('paddle.fluid.transpiler.distribute_transpiler.DistributeTranspiler', ('document', 'b2b19821c5dffcd11473d6a4eef089af'))

paddle.fluid.transpiler.DistributeTranspiler.__init__ (ArgSpec(args=['self', 'config'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.transpiler.DistributeTranspiler.get_pserver_program (ArgSpec(args=['self', 'endpoint'], varargs=None, keywords=None, defaults=None), ('document', 'b1951949c6d21698290aa8ac69afee32'))

paddle.fluid.transpiler.DistributeTranspiler.get_pserver_programs (ArgSpec(args=['self', 'endpoint'], varargs=None, keywords=None, defaults=None), ('document', 'c89fc350f975ef827f5448d68af388cf'))

paddle.fluid.transpiler.DistributeTranspiler.get_startup_program (ArgSpec(args=['self', 'endpoint', 'pserver_program', 'startup_program'], varargs=None, keywords=None, defaults=(None, None)), ('document', '90a40b80e0106f69262cc08b861c3e39'))

paddle.fluid.transpiler.DistributeTranspiler.get_trainer_program (ArgSpec(args=['self', 'wait_port'], varargs=None, keywords=None, defaults=(True,)), ('document', '0e47f020304e2b824e87ff03475c17cd'))

paddle.fluid.transpiler.DistributeTranspiler.transpile (ArgSpec(args=['self', 'trainer_id', 'program', 'pservers', 'trainers', 'sync_mode', 'startup_program', 'current_endpoint'], varargs=None, keywords=None, defaults=(None, '127.0.0.1:6174', 1, True, None, '127.0.0.1:6174')), ('document', '418c7e8b268e9be4104f2809e654c2f7'))

paddle.fluid.transpiler.memory_optimize (ArgSpec(args=['input_program', 'skip_opt_set', 'print_log', 'level', 'skip_grads'], varargs=None, keywords=None, defaults=(None, False, 0, True)), ('document', '2be29dc8ecdec9baa7728fb0c7f80e24'))

paddle.fluid.transpiler.release_memory (ArgSpec(args=['input_program', 'skip_opt_set'], varargs=None, keywords=None, defaults=(None,)), ('document', '2be29dc8ecdec9baa7728fb0c7f80e24'))

paddle.fluid.transpiler.HashName ('paddle.fluid.transpiler.ps_dispatcher.HashName', ('document', '8190ddc66ee412441f5d97fd3f702bdd'))

paddle.fluid.transpiler.HashName.__init__ (ArgSpec(args=['self', 'pserver_endpoints'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.transpiler.HashName.dispatch (ArgSpec(args=['self', 'varlist'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.transpiler.HashName.reset (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.transpiler.RoundRobin ('paddle.fluid.transpiler.ps_dispatcher.RoundRobin', ('document', 'c124359054923c614758c3fbbf666290'))

paddle.fluid.transpiler.RoundRobin.__init__ (ArgSpec(args=['self', 'pserver_endpoints'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.transpiler.RoundRobin.dispatch (ArgSpec(args=['self', 'varlist'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.transpiler.RoundRobin.reset (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.transpiler.DistributeTranspilerConfig ('paddle.fluid.transpiler.distribute_transpiler.DistributeTranspilerConfig', ('document', 'beac6f89fe97eb8c66a25de5a09c56d2'))

paddle.fluid.transpiler.DistributeTranspilerConfig.__init__ (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.nets.simple_img_conv_pool (ArgSpec(args=['input', 'num_filters', 'filter_size', 'pool_size', 'pool_stride', 'pool_padding', 'pool_type', 'global_pooling', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'param_attr', 'bias_attr', 'act', 'use_cudnn'], varargs=None, keywords=None, defaults=(0, 'max', False, 1, 0, 1, 1, None, None, None, True)), ('document', '13f01ff80e8dfbd3427d90cf49bc62eb'))

paddle.fluid.nets.sequence_conv_pool (ArgSpec(args=['input', 'num_filters', 'filter_size', 'param_attr', 'act', 'pool_type', 'bias_attr'], varargs=None, keywords=None, defaults=(None, 'sigmoid', 'max', None)), ('document', 'd6a1e527b53f5cc15594fee307dfc5cf'))

paddle.fluid.nets.glu (ArgSpec(args=['input', 'dim'], varargs=None, keywords=None, defaults=(-1,)), ('document', 'b87bacfc70dd3477ed25ef14aa01389a'))

paddle.fluid.nets.scaled_dot_product_attention (ArgSpec(args=['queries', 'keys', 'values', 'num_heads', 'dropout_rate'], varargs=None, keywords=None, defaults=(1, 0.0)), ('document', 'b1a07a0000eb9103e3a143ca8c13de5b'))

paddle.fluid.nets.img_conv_group (ArgSpec(args=['input', 'conv_num_filter', 'pool_size', 'conv_padding', 'conv_filter_size', 'conv_act', 'param_attr', 'conv_with_batchnorm', 'conv_batchnorm_drop_rate', 'pool_stride', 'pool_type', 'use_cudnn'], varargs=None, keywords=None, defaults=(1, 3, None, None, False, 0.0, 1, 'max', True)), ('document', '591a48aa9d871896aa8548c977c4c120'))

paddle.fluid.optimizer.SGDOptimizer ('paddle.fluid.optimizer.SGDOptimizer', ('document', 'c3c8dd3193d991adf8bda505560371d6'))

paddle.fluid.optimizer.SGDOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'regularization', 'name'], varargs=None, keywords=None, defaults=(None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.SGDOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.SGDOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.SGDOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.SGDOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.SGDOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.SGDOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.MomentumOptimizer ('paddle.fluid.optimizer.MomentumOptimizer', ('document', 'a72bd02e5459e64596897d190413d449'))

paddle.fluid.optimizer.MomentumOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'momentum', 'use_nesterov', 'regularization', 'name'], varargs=None, keywords=None, defaults=(False, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.MomentumOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.MomentumOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.MomentumOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.MomentumOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.MomentumOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.MomentumOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.AdagradOptimizer ('paddle.fluid.optimizer.AdagradOptimizer', ('document', 'a1d4f0682cde43ad34432b1338aadf04'))

paddle.fluid.optimizer.AdagradOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'epsilon', 'regularization', 'name', 'initial_accumulator_value'], varargs=None, keywords=None, defaults=(1e-06, None, None, 0.0)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.AdagradOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.AdagradOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.AdagradOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.AdagradOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.AdagradOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.AdagradOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.AdamOptimizer ('paddle.fluid.optimizer.AdamOptimizer', ('document', '6fe871b955cab6e267422d5af666dafa'))

paddle.fluid.optimizer.AdamOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'beta1', 'beta2', 'epsilon', 'regularization', 'name', 'lazy_mode'], varargs=None, keywords=None, defaults=(0.001, 0.9, 0.999, 1e-08, None, None, False)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.AdamOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.AdamOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.AdamOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.AdamOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.AdamOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.AdamOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.AdamaxOptimizer ('paddle.fluid.optimizer.AdamaxOptimizer', ('document', '883fc4541214e8343d3a89711936e15d'))

paddle.fluid.optimizer.AdamaxOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'beta1', 'beta2', 'epsilon', 'regularization', 'name'], varargs=None, keywords=None, defaults=(0.001, 0.9, 0.999, 1e-08, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.AdamaxOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.AdamaxOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.AdamaxOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.AdamaxOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.AdamaxOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.AdamaxOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.DpsgdOptimizer ('paddle.fluid.optimizer.DpsgdOptimizer', ('document', '71113c30b66c0f4035b10ebd8af8c5ad'))

paddle.fluid.optimizer.DpsgdOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'clip', 'batch_size', 'sigma'], varargs=None, keywords=None, defaults=(0.001, 0.9, 0.999, 1e-08)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.DpsgdOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.DpsgdOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.DpsgdOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.DpsgdOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.DpsgdOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.DpsgdOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.DecayedAdagradOptimizer ('paddle.fluid.optimizer.DecayedAdagradOptimizer', ('document', 'e76838a8586bf2e58e6b5cdd2f67f780'))

paddle.fluid.optimizer.DecayedAdagradOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'decay', 'epsilon', 'regularization', 'name'], varargs=None, keywords=None, defaults=(0.95, 1e-06, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.DecayedAdagradOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.DecayedAdagradOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.DecayedAdagradOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.DecayedAdagradOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.DecayedAdagradOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.DecayedAdagradOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.FtrlOptimizer ('paddle.fluid.optimizer.FtrlOptimizer', ('document', 'cba8aae0a267b9a4d8833ae79a00fc55'))

paddle.fluid.optimizer.FtrlOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'l1', 'l2', 'lr_power', 'regularization', 'name'], varargs=None, keywords=None, defaults=(0.0, 0.0, -0.5, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.FtrlOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.FtrlOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.FtrlOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.FtrlOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.FtrlOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.FtrlOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.RMSPropOptimizer ('paddle.fluid.optimizer.RMSPropOptimizer', ('document', '5217bc4fc399010021d6b70541005780'))

paddle.fluid.optimizer.RMSPropOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'rho', 'epsilon', 'momentum', 'centered', 'regularization', 'name'], varargs=None, keywords=None, defaults=(0.95, 1e-06, 0.0, False, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.RMSPropOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.RMSPropOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.RMSPropOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.RMSPropOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.RMSPropOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.RMSPropOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.AdadeltaOptimizer ('paddle.fluid.optimizer.AdadeltaOptimizer', ('document', 'f4354aef5e3b9134fa68919b75a3a097'))

paddle.fluid.optimizer.AdadeltaOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'epsilon', 'rho', 'regularization', 'name'], varargs=None, keywords=None, defaults=(1e-06, 0.95, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.AdadeltaOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.AdadeltaOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.AdadeltaOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.AdadeltaOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.AdadeltaOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.AdadeltaOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.ModelAverage ('paddle.fluid.optimizer.ModelAverage', ('document', '0a0adcd60230630e21fe1ef46362dbc0'))

paddle.fluid.optimizer.ModelAverage.__init__ (ArgSpec(args=['self', 'average_window_rate', 'min_average_window', 'max_average_window', 'regularization', 'name'], varargs=None, keywords=None, defaults=(10000, 10000, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.ModelAverage.apply (ArgSpec(args=['self', 'executor', 'need_restore'], varargs=None, keywords=None, defaults=(True,)), ('document', '648010d0ac1fa707dac0b89f74b0e35c'))

paddle.fluid.optimizer.ModelAverage.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.ModelAverage.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.ModelAverage.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.ModelAverage.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.ModelAverage.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.ModelAverage.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.ModelAverage.restore (ArgSpec(args=['self', 'executor'], varargs=None, keywords=None, defaults=None), ('document', '5f14ea4adda2791e1c3b37ff327f6a83'))

paddle.fluid.optimizer.LarsMomentumOptimizer ('paddle.fluid.optimizer.LarsMomentumOptimizer', ('document', '030b9092a96a409b1bf5446bf45d0659'))

paddle.fluid.optimizer.LarsMomentumOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'momentum', 'lars_coeff', 'lars_weight_decay', 'regularization', 'name'], varargs=None, keywords=None, defaults=(0.001, 0.0005, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.LarsMomentumOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.LarsMomentumOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.LarsMomentumOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.LarsMomentumOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.LarsMomentumOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.LarsMomentumOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.DGCMomentumOptimizer ('paddle.fluid.optimizer.DGCMomentumOptimizer', ('document', 'facdbef1b4871d0cf74c736ff2e94720'))

paddle.fluid.optimizer.DGCMomentumOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'momentum', 'rampup_begin_step', 'rampup_step', 'sparsity', 'use_nesterov', 'local_grad_clip_norm', 'num_trainers', 'regularization', 'name'], varargs=None, keywords=None, defaults=(1, [0.999], False, None, None, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.DGCMomentumOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.DGCMomentumOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.DGCMomentumOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.DGCMomentumOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.DGCMomentumOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.DGCMomentumOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.LambOptimizer ('paddle.fluid.optimizer.LambOptimizer', ('document', '7dd8b270156a52f1f6b4663336960893'))

paddle.fluid.optimizer.LambOptimizer.__init__ (ArgSpec(args=['self', 'learning_rate', 'lamb_weight_decay', 'beta1', 'beta2', 'epsilon', 'regularization', 'exclude_from_weight_decay_fn', 'name'], varargs=None, keywords=None, defaults=(0.001, 0.01, 0.9, 0.999, 1e-06, None, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.LambOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '80ea99c9af7ef5fac7e57fb302103610'))

paddle.fluid.optimizer.LambOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '5c46d1926a40f1f873ffe9f37ac89dae'))

paddle.fluid.optimizer.LambOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'ba3a113d0229ff7bc9d39bda0a6d947f'))

paddle.fluid.optimizer.LambOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.LambOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '649a92cf7f1ea28666fd00c4ea01acde'))

paddle.fluid.optimizer.LambOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', 'b15cffad0903fc81af77a0580ceb2a9b'))

paddle.fluid.optimizer.ExponentialMovingAverage ('paddle.fluid.optimizer.ExponentialMovingAverage', ('document', 'a38b7d5b9f17a295ed15d4c1b9ab4cd0'))

paddle.fluid.optimizer.ExponentialMovingAverage.__init__ (ArgSpec(args=['self', 'decay', 'thres_steps', 'name'], varargs=None, keywords=None, defaults=(0.999, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.ExponentialMovingAverage.apply (ArgSpec(args=['self', 'executor', 'need_restore'], varargs=None, keywords=None, defaults=(True,)), ('document', '30f494752ac8921dc5835a63637f453a'))

paddle.fluid.optimizer.ExponentialMovingAverage.restore (ArgSpec(args=['self', 'executor'], varargs=None, keywords=None, defaults=None), ('document', '8c8a1791608b02a1ede53d6dd3a4fcec'))

paddle.fluid.optimizer.ExponentialMovingAverage.update (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', 'ea10f08af6d7aac3b7974aa976e4085f'))

paddle.fluid.optimizer.PipelineOptimizer ('paddle.fluid.optimizer.PipelineOptimizer', ('document', '6f85382abedb922387b08d98e8d0b69c'))

paddle.fluid.optimizer.PipelineOptimizer.__init__ (ArgSpec(args=['self', 'optimizer', 'cut_list', 'place_list', 'concurrency_list', 'queue_size', 'sync_steps', 'start_cpu_core_id'], varargs=None, keywords=None, defaults=(None, None, None, 30, 1, 0)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.PipelineOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set'], varargs=None, keywords=None, defaults=(None, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.LookaheadOptimizer ('paddle.fluid.optimizer.LookaheadOptimizer', ('document', 'c291cadfa7452c7bf58b9e2f900a3511'))

paddle.fluid.optimizer.LookaheadOptimizer.__init__ (ArgSpec(args=['self', 'inner_optimizer', 'alpha', 'k'], varargs=None, keywords=None, defaults=(0.5, 5)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.LookaheadOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.RecomputeOptimizer ('paddle.fluid.optimizer.RecomputeOptimizer', ('document', '05769ba1182270f808f85488a50c8caa'))

paddle.fluid.optimizer.RecomputeOptimizer.__init__ (ArgSpec(args=['self', 'optimizer'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.RecomputeOptimizer.apply_gradients (ArgSpec(args=['self', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '7838e157ec5ff4f835f814adf3a2b9cc'))

paddle.fluid.optimizer.RecomputeOptimizer.apply_optimize (ArgSpec(args=['self', 'loss', 'startup_program', 'params_grads'], varargs=None, keywords=None, defaults=None), ('document', '89c5348bfd78ad21f90b1da4af4b3cd1'))

paddle.fluid.optimizer.RecomputeOptimizer.backward (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'callbacks', 'checkpoints'], varargs=None, keywords=None, defaults=(None, None, None, None, None)), ('document', 'a26b3dbb0f63ee81d847d92e9fb942dc'))

paddle.fluid.optimizer.RecomputeOptimizer.get_opti_var_name_list (ArgSpec(args=['self'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.optimizer.RecomputeOptimizer.load (ArgSpec(args=['self', 'stat_dict'], varargs=None, keywords=None, defaults=None), ('document', '7b2b8ae72011bc4decb67e97623f2c56'))

paddle.fluid.optimizer.RecomputeOptimizer.minimize (ArgSpec(args=['self', 'loss', 'startup_program', 'parameter_list', 'no_grad_set', 'grad_clip'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.backward.append_backward (ArgSpec(args=['loss', 'parameter_list', 'no_grad_set', 'callbacks', 'checkpoints'], varargs=None, keywords=None, defaults=(None, None, None, None)), ('document', '52488008103886c793843a3828bacd5e'))

paddle.fluid.backward.gradients (ArgSpec(args=['targets', 'inputs', 'target_gradients', 'no_grad_set'], varargs=None, keywords=None, defaults=(None, None)), ('document', 'e2097e1e0ed84ae44951437bfe269a1b'))

paddle.fluid.regularizer.L1DecayRegularizer ('paddle.fluid.regularizer.L1DecayRegularizer', ('document', '34603757e70974d2fcc730643b382925'))

paddle.fluid.regularizer.L1DecayRegularizer.__init__ (ArgSpec(args=['self', 'regularization_coeff'], varargs=None, keywords=None, defaults=(0.0,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.regularizer.L2DecayRegularizer ('paddle.fluid.regularizer.L2DecayRegularizer', ('document', 'b94371c3434d7f695bc5b2d6fb5531fd'))

paddle.fluid.regularizer.L2DecayRegularizer.__init__ (ArgSpec(args=['self', 'regularization_coeff'], varargs=None, keywords=None, defaults=(0.0,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.LoDTensor ('paddle.fluid.core_avx.LoDTensor', ('document', '25e8432ed1b9a375868bc8911359aa0d'))

paddle.fluid.LoDTensor.__init__ 1. __init__(self: paddle.fluid.core_avx.LoDTensor, arg0: List[List[int]]) -> None  2. __init__(self: paddle.fluid.core_avx.LoDTensor) -> None

paddle.fluid.LoDTensor.has_valid_recursive_sequence_lengths has_valid_recursive_sequence_lengths(self: paddle.fluid.core_avx.LoDTensor) -> bool

paddle.fluid.LoDTensor.lod lod(self: paddle.fluid.core_avx.LoDTensor) -> List[List[int]]

paddle.fluid.LoDTensor.recursive_sequence_lengths recursive_sequence_lengths(self: paddle.fluid.core_avx.LoDTensor) -> List[List[int]]

paddle.fluid.LoDTensor.set_lod set_lod(self: paddle.fluid.core_avx.LoDTensor, lod: List[List[int]]) -> None

paddle.fluid.LoDTensor.set_recursive_sequence_lengths set_recursive_sequence_lengths(self: paddle.fluid.core_avx.LoDTensor, recursive_sequence_lengths: List[List[int]]) -> None

paddle.fluid.LoDTensor.shape shape(self: paddle.fluid.core_avx.Tensor) -> List[int]

paddle.fluid.LoDTensorArray ('paddle.fluid.core_avx.LoDTensorArray', ('document', 'e9895b67ba54438b9c0f7053e18966f5'))

paddle.fluid.LoDTensorArray.__init__ __init__(self: paddle.fluid.core_avx.LoDTensorArray) -> None

paddle.fluid.LoDTensorArray.append append(self: paddle.fluid.core_avx.LoDTensorArray, tensor: paddle.fluid.core_avx.LoDTensor) -> None

paddle.fluid.CPUPlace ('paddle.fluid.core_avx.CPUPlace', ('document', '6014005ef2649045b77d502aeb6cd7f9'))

paddle.fluid.CPUPlace.__init__ __init__(self: paddle.fluid.core_avx.CPUPlace) -> None

paddle.fluid.CUDAPlace ('paddle.fluid.core_avx.CUDAPlace', ('document', '6a6cd8ed607beb951692c4b066d08c94'))

paddle.fluid.CUDAPlace.__init__ __init__(self: paddle.fluid.core_avx.CUDAPlace, arg0: int) -> None

paddle.fluid.CUDAPinnedPlace ('paddle.fluid.core_avx.CUDAPinnedPlace', ('document', 'afd58ea5d390b5ea06ca70291a266d45'))

paddle.fluid.CUDAPinnedPlace.__init__ __init__(self: paddle.fluid.core_avx.CUDAPinnedPlace) -> None

paddle.fluid.ParamAttr ('paddle.fluid.param_attr.ParamAttr', ('document', 'a4d4d13ce9eeb86bbaa7ab935c207577'))

paddle.fluid.ParamAttr.__init__ (ArgSpec(args=['self', 'name', 'initializer', 'learning_rate', 'regularizer', 'trainable', 'gradient_clip', 'do_model_average'], varargs=None, keywords=None, defaults=(None, None, 1.0, None, True, None, True)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.WeightNormParamAttr ('paddle.fluid.param_attr.WeightNormParamAttr', ('document', 'b5ae1698ea72d5a9428000b916a67379'))

paddle.fluid.WeightNormParamAttr.__init__ (ArgSpec(args=['self', 'dim', 'name', 'initializer', 'learning_rate', 'regularizer', 'trainable', 'gradient_clip', 'do_model_average'], varargs=None, keywords=None, defaults=(None, None, None, 1.0, None, True, None, False)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.DataFeeder ('paddle.fluid.data_feeder.DataFeeder', ('document', 'd9e64be617bd5f49dbb08ac2bc8665e6'))

paddle.fluid.DataFeeder.__init__ (ArgSpec(args=['self', 'feed_list', 'place', 'program'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.DataFeeder.decorate_reader (ArgSpec(args=['self', 'reader', 'multi_devices', 'num_places', 'drop_last'], varargs=None, keywords=None, defaults=(None, True)), ('document', 'a0ed5ce816b5d603cb595aacb922335a'))

paddle.fluid.DataFeeder.feed (ArgSpec(args=['self', 'iterable'], varargs=None, keywords=None, defaults=None), ('document', 'ce65fe1d81dcd7067d5092a5667f35cc'))

paddle.fluid.DataFeeder.feed_parallel (ArgSpec(args=['self', 'iterable', 'num_places'], varargs=None, keywords=None, defaults=(None,)), ('document', '334c6af750941a4397a2dd2ea8a4d76f'))

paddle.fluid.clip.set_gradient_clip (ArgSpec(args=['clip', 'param_list', 'program'], varargs=None, keywords=None, defaults=(None, None)), ('document', 'a0b00ccc8584b4a1cf4ec5aa74780e77'))

paddle.fluid.clip.ErrorClipByValue ('paddle.fluid.clip.ErrorClipByValue', ('document', 'e6f815a03be88dee2537707d9e6b9209'))

paddle.fluid.clip.ErrorClipByValue.__init__ (ArgSpec(args=['self', 'max', 'min'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.clip.GradientClipByValue ('paddle.fluid.clip.GradientClipByValue', ('document', 'b7a22f687269cae0c338ef3866322db7'))

paddle.fluid.clip.GradientClipByValue.__init__ (ArgSpec(args=['self', 'max', 'min'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.clip.GradientClipByNorm ('paddle.fluid.clip.GradientClipByNorm', ('document', 'a5c23d96a3d8c8c1183e9469a5d0d52e'))

paddle.fluid.clip.GradientClipByNorm.__init__ (ArgSpec(args=['self', 'clip_norm'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.clip.GradientClipByGlobalNorm ('paddle.fluid.clip.GradientClipByGlobalNorm', ('document', 'ef50acbe212101121d4b82f693ec1733'))

paddle.fluid.clip.GradientClipByGlobalNorm.__init__ (ArgSpec(args=['self', 'clip_norm', 'group_name'], varargs=None, keywords=None, defaults=('default_group',)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph_grad_clip.GradClipByValue ('paddle.fluid.dygraph_grad_clip.GradClipByValue', ('document', '6971a42222de0387a7ee9c59671dd2e3'))

paddle.fluid.dygraph_grad_clip.GradClipByValue.__init__ (ArgSpec(args=['self', 'min_value', 'max_value'], varargs=None, keywords=None, defaults=(None,)), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph_grad_clip.GradClipByNorm ('paddle.fluid.dygraph_grad_clip.GradClipByNorm', ('document', '2039274ea09987ba48eded67999dc280'))

paddle.fluid.dygraph_grad_clip.GradClipByNorm.__init__ (ArgSpec(args=['self', 'clip_norm'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.dygraph_grad_clip.GradClipByGlobalNorm ('paddle.fluid.dygraph_grad_clip.GradClipByGlobalNorm', ('document', 'd1872377e7d7a5fe0dd2e8c42e4c9656'))

paddle.fluid.dygraph_grad_clip.GradClipByGlobalNorm.__init__ (ArgSpec(args=['self', 'max_global_norm'], varargs=None, keywords=None, defaults=None), ('document', '6adf97f83acf6453d4a6a4b1070f3754'))

paddle.fluid.profiler.cuda_profiler (ArgSpec(args=['output_file', 'output_mode', 'config'], varargs=None, keywords=None, defaults=(None, None)), ('document', '4053b45953807a24e28027dc86829d6c'))

paddle.fluid.profiler.reset_profiler (ArgSpec(args=[], varargs=None, keywords=None, defaults=None), ('document', 'fd1f25a7a06516ca9a1f4ab0783a4d70'))

paddle.fluid.profiler.profiler (ArgSpec(args=['state', 'sorted_key', 'profile_path'], varargs=None, keywords=None, defaults=(None, '/tmp/profile')), ('document', 'a2be24e028dffa06ab28cc55a27c59e4'))

paddle.fluid.profiler.start_profiler (ArgSpec(args=['state'], varargs=None, keywords=None, defaults=None), ('document', '4c192ea399e6e80b1ab47a8265b022a5'))

paddle.fluid.profiler.stop_profiler (ArgSpec(args=['sorted_key', 'profile_path'], varargs=None, keywords=None, defaults=(None, '/tmp/profile')), ('document', 'bc8628b859b04242200e48a458c971c4'))

paddle.fluid.unique_name.generate (ArgSpec(args=['key'], varargs=None, keywords=None, defaults=None), ('document', '4d68cde4c4df8f1b8018620b4dc19b42'))

paddle.fluid.unique_name.switch (ArgSpec(args=['new_generator'], varargs=None, keywords=None, defaults=(None,)), ('document', '695a6e91afbcdbafac69a069038811be'))

paddle.fluid.unique_name.guard (ArgSpec(args=['new_generator'], varargs=None, keywords=None, defaults=(None,)), ('document', 'ead717d6d440a1eb11971695cd1727f4'))

paddle.fluid.Scope Scope() -> paddle.fluid.core_avx._Scope

paddle.fluid.install_check.run_check (ArgSpec(args=[], varargs=None, keywords=None, defaults=None), ('document', '66b7c84a17ed32fec2df9628367be2b9'))

