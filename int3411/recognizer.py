import os
from gooey import Gooey, GooeyParser

from src.dtw import run_dtw
from src.gmmhmm import run_gmmhmm


@Gooey(default_size=(800, 600), required_cols=1, optional_cols=2, use_cmd_args=True)
def parser():
    parser = GooeyParser(description='Speech Recognition')
    parser.add_argument('mat_dir', type=str, help='Mat files directory', widget='DirChooser')
    parser.add_argument('-i', '--input_mat', type=str, default='.', help='Path to input mat file', widget='FileChooser')
    parser.add_argument('-r', '--recognition_type', type=str, choices=['dtw', 'gmm-hmm'], default='dtw', help='Recognition type')
    parser.add_argument('-m', '--pretrained_model', type=str, default='.', help='Path to pretrained model (GMMHMM only)', widget='FileChooser')
    args = parser.parse_args()

    if not os.path.isdir(args.mat_dir):
        raise FileNotFoundError(f'Mat files directory \"{args.mat_dir}\" not found')

    return args


if __name__ == '__main__':
    args = parser()
    if args.recognition_type == 'dtw':
        run_dtw(args.mat_dir, args.input_mat)
    elif args.recognition_type == 'gmm-hmm':
        run_gmmhmm(args.mat_dir, args.pretrained_model, args.input_mat)
