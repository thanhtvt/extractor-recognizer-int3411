import os
import argparse
import src
from gooey import Gooey, GooeyParser


@Gooey(default_size=(800, 600), required_cols=1, optional_cols=3, use_cmd_args=True)
def parser():
    parser = GooeyParser(description='Extract MFCC features from audio files')
    # parser = argparse.ArgumentParser(description='Extract MFCC features from audio files')
    parser.add_argument('audio_path', type=str, help='Path to audio file', widget='FileChooser')
    parser.add_argument('label_path', type=str, help='Path to label file', widget='FileChooser')
    parser.add_argument('-s', '--show_fig', action='store_true', help='Show figure')
    parser.add_argument('-sf', '--save_fig', action='store_true', help='Save figure')
    parser.add_argument('-sm', '--save_mat', action='store_true', help='Save mat file')
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        raise FileNotFoundError(f'Audio file \"{args.audio_path}\" not found.')
    if not os.path.isfile(args.label_path):
        raise FileNotFoundError(f'Label file \"{args.label_path}\" not found.')

    return args


def preprocess_label(text, audio, sr):
    start_time, end_time, label = text.strip().split('\t')
    if label == 'sil':
        return None, label
    start = int(float(start_time) * sr)
    end = int(float(end_time) * sr)
    audio_segment = audio[start:end + 1].astype(float)
    return audio_segment, label


if __name__ == '__main__':
    args = parser()
    sr, audio = src.utils.load_audio(args.audio_path)
    with open(args.label_path, 'r') as f:
        lines = f.read().splitlines()
        mfcc = src.mfcc.MFCC()
        for line in lines:
            segment, label = preprocess_label(line, audio, sr)
            if label == 'sil':
                continue
            feats = mfcc(segment, sr)
            if args.save_mat:
                src.utils.save_mat(feats, label)
            src.utils.plotfig(feats, label, save=args.save_fig, show=args.show_fig)
    print('Done.')
