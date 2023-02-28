# """run_infer.py
#
# Usage:
#   run_infer.py [options] [--help] <command> [<args>...]
#   run_infer.py --version
#   run_infer.py (-h | --help)
#
# Options:
#   -h --help                   Show this string.
#   --version                   Show version.
#
#   --gpu=<id>                  GPU list. [default: 0]
#   --nr_types=<n>              Number of nuclei types to predict. [default: 0]
#   --type_info_path=<path>     Path to a json define mapping between type id, type name,
#                               and expected overlaid color. [default: '']
#
#   --model_path=<path>         Path to saved checkpoint.
#   --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC,
#                               'original' or 'fast'. [default: fast]
#   --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
#   --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
#   --batch_size=<n>            Batch size. [default: 128]
#
# Two command mode are `tile` and `wsi` to enter corresponding inference mode
#     tile  run the inference on tile
#     wsi   run the inference on wsi
#
# Use `run_infer.py <command> --help` to show their options and usage.
# """
#
# tile_cli = """
# Arguments for processing tiles.
#
# usage:
#     tile (--input_dir=<path>) (--output_dir=<path>) \
#          [--draw_dot] [--save_qupath] [--save_raw_map]
#
# options:
#    --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
#    --output_dir=<path>    Path to output directory..
#
#    --draw_dot             To draw nuclei centroid on overlay. [default: False]
#    --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
#    --save_raw_map         To save raw prediction or not. [default: False]
# """
#
# wsi_cli = """
# Arguments for processing wsi
#
# usage:
#     wsi (--input_dir=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
#         [--cache_path=<path>] [--input_mask_dir=<path>] \
#         [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
#         [--save_thumb] [--save_mask]
#
# options:
#     --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
#     --output_dir=<path>     Path to output directory.
#     --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
#     --mask_dir=<path>       Path to directory containing tissue masks.
#                             Should have the same name as corresponding WSIs. [default: '']
#
#     --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
#     --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
#     --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
#     --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
#     --save_thumb            To save thumb. [default: False]
#     --save_mask             To save mask. [default: False]
# """
#
# import logging
# import os
# import copy
# from docopt import docopt
#
# os.environ['MKL_THREADING_LAYER'] = 'GNU'  # 添加
#
# # -------------------------------------------------------------------------------------------------------
#
# if __name__ == '__main__':
#     sub_cli_dict = {'tile': tile_cli, 'wsi': wsi_cli}
#     args = docopt(__doc__, help=False, options_first=True,
#                   version='HoVer-Net Pytorch Inference v1.0')
#     sub_cmd = args.pop('<command>')
#     sub_cmd_args = args.pop('<args>')
#
#     if args['--help'] and sub_cmd is not None:
#         if sub_cmd in sub_cli_dict:
#             print(sub_cli_dict[sub_cmd])
#         else:
#             print(__doc__)
#         exit()
#     if args['--help'] or sub_cmd is None:
#         print(__doc__)
#         exit()
#
#     sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)
#
#     args.pop('--version')
#     gpu_list = args.pop('--gpu')
#     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
#
#     args = {k.replace('--', ''): v for k, v in args.items()}
#     sub_args = {k.replace('--', ''): v for k, v in sub_args.items()}
#     if args['model_path'] == None:
#         raise Exception('A model path must be supplied as an argument with --model_path.')
#
#     nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
#     method_args = {
#         'method': {
#             'model_args': {
#                 'nr_types': nr_types,
#                 'mode': args['model_mode'],
#             },
#             'model_path': args['model_path'],
#         },
#         'type_info_path': None if args['type_info_path'] == '' \
#             else args['type_info_path'],
#     }
#
#     # ***
#     run_args = {
#         'batch_size': int(args['batch_size']),
#
#         'nr_inference_workers': int(args['nr_inference_workers']),
#         'nr_post_proc_workers': int(args['nr_post_proc_workers']),
#     }
#
#     if args['model_mode'] == 'fast':
#         run_args['patch_input_shape'] = 256
#         run_args['patch_output_shape'] = 164
#     else:
#         run_args['patch_input_shape'] = 270
#         run_args['patch_output_shape'] = 80
#
#     if sub_cmd == 'tile':
#         run_args.update({
#             'input_dir': sub_args['input_dir'],
#             'output_dir': sub_args['output_dir'],
#
#             'draw_dot': sub_args['draw_dot'],
#             'save_qupath': sub_args['save_qupath'],
#             'save_raw_map': sub_args['save_raw_map'],
#         })
#
#     if sub_cmd == 'wsi':
#         run_args.update({
#             'input_dir': sub_args['input_dir'],
#             'output_dir': sub_args['output_dir'],
#             'input_mask_dir': sub_args['input_mask_dir'],
#             'cache_path': sub_args['cache_path'],
#
#             'proc_mag': int(sub_args['proc_mag']),
#             'ambiguous_size': int(sub_args['ambiguous_size']),
#             'chunk_shape': int(sub_args['chunk_shape']),
#             'tile_shape': int(sub_args['tile_shape']),
#             'save_thumb': sub_args['save_thumb'],
#             'save_mask': sub_args['save_mask'],
#         })
#     # ***
#
#     # ! TODO: where to save logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s', datefmt='%Y-%m-%d|%H:%M:%S',
#         handlers=[
#             logging.FileHandler("debug.log"),
#             logging.StreamHandler()
#         ]
#     )
#
#     if sub_cmd == 'tile':
#         from infer.tile import InferManager
#
#         infer = InferManager(**method_args)
#         infer.process_file_list(run_args)
#     else:
#         from infer.wsi import InferManager
#
#         infer = InferManager(**method_args)
#         infer.process_wsi_list(run_args)

"""run_infer.py

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name,
                              and expected overlaid color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC,
                              'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size. [default: 128]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""

tile_cli = """
Arguments for processing tiles.

usage:
    tile (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map]

options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
"""

wsi_cli = """
Arguments for processing wsi

usage:
    wsi (--input_dir=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
        [--cache_path=<path>] [--input_mask_dir=<path>] \
        [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
        [--save_thumb] [--save_mask]

options:
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks.
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
"""

import logging
import os
import copy
from docopt import docopt
import argparse
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='hover inference Script')
    parser.add_argument('--nr_types', default=5, type=int,
                        help='class for inferenceing')
    parser.add_argument('--model_mode', default='original', type=str,
                        help='model_mode')
    parser.add_argument('--model_path', default='./logs/ulsam/01/net_epoch=83.tar', type=str,
                        help='model_path')
    parser.add_argument('--type_info_path', default='./type_info.json', type=str,
                        help='type_info_path')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch_size')
    parser.add_argument('--nr_inference_workers', default=8, type=int,  
                    help='nr_inference_workers')
    parser.add_argument('--nr_post_proc_workers', default=16, type=int,
                        help='nr_post_proc_workers')

    sub_parser = argparse.ArgumentParser(
        description='hover inference Script')
    sub_parser.add_argument('--input_dir', default='./dataset/CoNSeP/Test/Images/', type=str,
                            help='input_dir')
    sub_parser.add_argument('--output_dir', default='./dataset/sample_tiles/pred/', type=str,
                            help='input_dir')
    sub_parser.add_argument('--draw_dot', default=False, type=str,
                            help='draw_dot')
    sub_parser.add_argument('--save_qupath', default=True, type=str,
                            help='save_qupath')
    sub_parser.add_argument('--save_raw_map', default=True, type=str,
                            help='save_raw_map')

    args = parser.parse_args()
    sub_args = sub_parser.parse_args()
    sub_cmd = 'tile'


    if args.model_path == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')

    # nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
    nr_types = int(args.nr_types) if int(args.nr_types) > 0 else None

    method_args = {
        'method': {
            'model_args': {
                'nr_types': nr_types,
                # 'mode'       : args['model_mode'],
                'mode': args.model_mode,

            },
            # 'model_path' : args['model_path'],
            'model_path': args.model_path,
            # 'model_path':'./logs/pannuke/01/net_epoch=50.tar',

        },
        # 'type_info_path'  : None if args['type_info_path'] == '' \
        #                     else args['type_info_path'],
        'type_info_path': 'type_info.json',
    }

    # ***
    run_args = {
        # 'batch_size' : int(args['batch_size']),
        'batch_size': 64,

        # 'nr_inference_workers' : int(args['nr_inference_workers']),
        # 'nr_post_proc_workers' : int(args['nr_post_proc_workers']),
        'nr_inference_workers': 8,
        'nr_post_proc_workers': 16,
    }

    if args.model_mode == 'original':
        run_args['patch_input_shape'] = 270
        run_args['patch_output_shape'] = 80

    if sub_cmd == 'tile':
        run_args.update({

            # 'input_dir'      : sub_args['input_dir'],
            # 'output_dir'     : sub_args['output_dir'],
            # 'draw_dot'    : sub_args['draw_dot'],
            # 'save_qupath' : sub_args['save_qupath'],
            # 'save_raw_map': sub_args['save_raw_map'],
            'input_dir': sub_args.input_dir,
            'output_dir': sub_args.output_dir,
            'draw_dot': sub_args.draw_dot,
            'save_qupath': sub_args.save_qupath,
            'save_raw_map': sub_args.save_raw_map,
        })

    # ***

    # ! TODO: where to save logging
    logging.basicConfig(
        level=logging.INFO,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s', datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    if sub_cmd == 'tile':
        from infer.tile import InferManager

        infer = InferManager(**method_args)
        infer.process_file_list(run_args)


