import os
import sys
import argparse
import importlib

abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
os.system(f"pip install -r {abs_path}/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system(f"pip install mindpet -i https://pypi.tuna.tsinghua.edu.cn/simple")

import time
import albumentations
import mindspore as ms
from mindspore import Model, context
from mindspore.nn import DynamicLossScaleUpdateCell
from mindspore.nn import TrainOneStepWithLossScaleCell
from mindspore import load_checkpoint, load_param_into_net
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint

from omegaconf import OmegaConf

from ldm.data.dataset import load_data
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.callback import OverflowMonitor
from ldm.modules.train.learningrate import LearningRate
from ldm.modules.train.parallel_config import ParallelConfig
from ldm.models.clip_zh.simple_tokenizer import WordpieceTokenizer
from ldm.modules.train.tools import parse_with_config, set_random_seed
from ldm.modules.train.cell_wrapper import ParallelTrainOneStepWithLossScaleCell

os.environ['HCCL_CONNECT_TIMEOUT'] = '6000'


def init_env(opts):
    """ init_env """
    set_random_seed(opts.seed)
    if opts.use_parallel:
        init()
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = get_group_size()
        ParallelConfig.dp = device_num
        rank_id = get_rank()
        opts.rank = rank_id
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank_id, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num)
    else:
        device_num = 1
        device_id = int(os.getenv('DEVICE_ID', 0))
        rank_id = 0
        opts.rank = rank_id

    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id,
                        )

    """ create dataset"""
    tokenizer = WordpieceTokenizer()
    dataset = load_data(
                data_path=opts.data_path,
                batch_size=opts.train_batch_size,
                tokenizer=tokenizer,
                image_size=opts.image_size,
                image_filter_size=opts.image_filter_size,
                device_num=device_num,
                rank_id = rank_id, 
                random_crop = opts.random_crop,
                filter_small_size = opts.filter_small_size,
                sample_num=-1
                )
    print(f"rank id {rank_id}, sample num is {dataset.get_dataset_size()}")

    return dataset, rank_id, device_id, device_num


def instantiate_from_config(config):
    config = OmegaConf.load(config).model
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)
    if os.path.exists(ckpt):
        param_dict = ms.load_checkpoint(ckpt)
        if param_dict:
            param_not_load = ms.load_param_into_net(model, param_dict)
            print("param not load:", param_not_load)
    else:
        print(f"{ckpt} not exist:")

    return model


def load_pretrained_model(pretrained_ckpt, net):
    print(f"start loading pretrained_ckpt {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)
        param_not_load = load_param_into_net(net, param_dict)
        print("param not load:", param_not_load)
    else:
        print("ckpt file not exist!")

    print("end loading ckpt")


def load_pretrained_model_clip_and_vae(pretrained_ckpt, net):
    new_param_dict = {}
    print(f"start loading pretrained_ckpt {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)
        for key in param_dict:
            if key.startswith("first") or key.startswith("cond"):
                new_param_dict[key] = param_dict[key]
        param_not_load = load_param_into_net(net, new_param_dict)
        print("param not load:")
        for param in param_not_load:
            print(param)
    else:
        print("ckpt file not exist!")

    print("end loading ckpt")


def main(opts):
    dataset, rank_id, device_id, device_num = init_env(opts)
    LatentDiffusionWithLoss = instantiate_from_config(opts.model_config)
    pretrained_ckpt = os.path.join(opts.pretrained_model_path, opts.pretrained_model_file)
    load_pretrained_model(pretrained_ckpt, LatentDiffusionWithLoss)

    if opts.enable_lora:
        from tk.graph import freeze_delta
        # 适配lora算法后，冻结lora模块之外的参数
        freeze_delta(LatentDiffusionWithLoss, 'lora')

    if not opts.decay_steps:
        dataset_size = dataset.get_dataset_size()
        opts.decay_steps = opts.epochs * dataset_size
    lr = LearningRate(opts.start_learning_rate, opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(LatentDiffusionWithLoss, opts, lr, enable_lora=opts.enable_lora)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=opts.init_loss_scale,
                                             scale_factor=opts.loss_scale_factor,
                                             scale_window=opts.scale_window)

    if opts.use_parallel:
        net_with_grads = ParallelTrainOneStepWithLossScaleCell(LatentDiffusionWithLoss, optimizer=optimizer,          
                                                               scale_sense=update_cell, parallel_config=ParallelConfig)
    else:
        net_with_grads = TrainOneStepWithLossScaleCell(LatentDiffusionWithLoss, optimizer=optimizer,
                                                       scale_sense=update_cell)
    model = Model(net_with_grads)
    callback = [TimeMonitor(opts.callback_size), LossMonitor(opts.callback_size)]

    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if opts.use_zhisuan or opts.use_qizhi:
        from openi import EnvToOpenIEpochEnd
        
        local_rank=int(os.getenv('RANK_ID'))
        #非必选，每个epoch结束后，都手动上传训练结果到启智平台，注意这样使用会占用很多内存，只有在部分特殊需要手动上传的任务才需要使用
        uploadOutput = EnvToOpenIEpochEnd(opts.output_path,opts.model_url)
        callback.append(uploadOutput) 
        

    if rank_id == 0:
        dataset_size = dataset.get_dataset_size()
        if not opts.save_checkpoint_steps:
            opts.save_checkpoint_steps = dataset_size
        ckpt_dir = os.path.join(opts.output_path, "ckpt", f"rank_{str(rank_id)}")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir) 

        if not opts.enable_lora:
            config_ck = CheckpointConfig(save_checkpoint_steps=opts.save_checkpoint_steps,
                                         keep_checkpoint_max=10,
                                         integrated_save=False)
            ckpoint_cb = ModelCheckpoint(prefix="wkhh_txt2img",
                                         directory=ckpt_dir,
                                         config=config_ck)
        else:
            from tk.graph.ckpt_util import TrainableParamsCheckPoint

            config_ck = CheckpointConfig(save_checkpoint_steps=opts.save_checkpoint_steps,
                                         keep_checkpoint_max=10,
                                         integrated_save=False,
                                         saved_network=LatentDiffusionWithLoss)
            ckpoint_cb = TrainableParamsCheckPoint(prefix="wkhh_txt2img_lora",
                                         directory=ckpt_dir,
                                         config=config_ck)
        callback.append(ckpoint_cb)

    print("start_training...")
    model.train(opts.epochs, dataset, callbacks=callback, dataset_sink_mode=False)


if __name__ == "__main__":
    print('process id:', os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_parallel', default=False, type=str2bool, help='use parallel')
    parser.add_argument('--data_path', default="dataset", type=str, help='data path')
    parser.add_argument('--output_path', default="output/", type=str, help='use audio out')
    parser.add_argument('--train_config', default="configs/train_config.json", type=str, help='train config path')
    parser.add_argument('--model_config', default="configs/v1-train-chinese.yaml", type=str, help='model config path')
    parser.add_argument('--pretrained_model_path', default="", type=str, help='pretrained model directory')
    parser.add_argument('--pretrained_model_file', default="", type=str, help='pretrained model file name')
    
    parser.add_argument('--optim', default="adamw", type=str, help='optimizer')
    parser.add_argument('--seed', default=3407, type=int, help='data path')
    parser.add_argument('--warmup_steps', default=1000, type=int, help='warmup steps')
    parser.add_argument('--train_batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--callback_size', default=1, type=int, help='callback size.')
    parser.add_argument("--start_learning_rate", default=1e-5, type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int,help="lr decay steps.")
    parser.add_argument("--epochs", default=10, type=int, help="epochs")
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--save_checkpoint_steps", default=0, type=int, help="save checkpoint steps")
    parser.add_argument('--random_crop', default=False, type=str2bool, help='random crop')
    parser.add_argument('--filter_small_size', default=True, type=str2bool, help='filter small images')
    parser.add_argument('--image_size', default=512, type=int, help='images size')
    parser.add_argument('--image_filter_size', default=256, type=int, help='image filter size')

    parser.add_argument('--enable_lora', default=False, type=str2bool, help='enable lora')
    
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_url', metavar='DIR', default='', help='path to dataset')
    parser.add_argument('--train_url', metavar='DIR', default='', help='save output')
    parser.add_argument('--result_url', metavar='DIR', default='', help='save result url')
    parser.add_argument('--multi_data_url',help='path to multi dataset', default= '/cache/data/')
    parser.add_argument('--ckpt_url', type=str, default=None,help='load ckpt file path')
    parser.add_argument('--ckpt_path', type=str, default='/cache/pretrain/',help='load ckpt file path')
    parser.add_argument('--pretrain_url', type=str, default=None, help='load ckpt file path')
    parser.add_argument('--use_qizhi', type=bool, default=False,help='use qizhi')
    parser.add_argument('--use_zhisuan', type=bool, default=True, help='use zhisuan')

    parser.add_argument('--model_url', type=str, default='',help='load ckpt file path')
    parser.add_argument('--grampus_code_url', type=str, default='',help='load ckpt file path')
    parser.add_argument('--grampus_code_file_name', type=str, default='',help='load ckpt file path')

    
    args = parser.parse_args()
    args = parse_with_config(args)
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    args.model_config = os.path.join(abs_path, args.model_config)
    print(args)
        
    if args.use_qizhi:
        from openi import openi_multidataset_to_env as DatasetToEnv  
        from openi import pretrain_to_env as PretrainToEnv
        from openi import env_to_openi as EnvToOpeni

        data_dir = '/cache/data'  
        train_dir = '/cache/output'
        pretrain_dir = '/cache/pretrain'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)      
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir)
        DatasetToEnv(args.multi_data_url,data_dir)
        PretrainToEnv(args.pretrain_url,pretrain_dir)


    if args.use_zhisuan:
        from openi import c2net_multidataset_to_env as DatasetToEnv  
        from openi import pretrain_to_env as PretrainToEnv
        from openi import env_to_openi as EnvToOpeni

        data_dir = '/cache/data'  
        train_dir = '/cache/output'
        pretrain_dir = '/cache/pretrain'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)      
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir)
        DatasetToEnv(args.multi_data_url,data_dir)
        PretrainToEnv(args.pretrain_url,pretrain_dir)

    start = time.time()
    main(args)
    end = time.time()
    print("training time: ", end-start)
    
    if args.use_qizhi:
        EnvToOpeni(train_dir,args.train_url)