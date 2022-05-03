import os
import argparse
import src


def parser():
    parser = argparse.ArgumentParser(description='Extract MFCC features from audio files')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('label_path', type=str, help='Path to label file')
    parser.add_argument('-sf', '--save_fig', action='store_true', help='Whether to save figure')
    parser.add_argument('-sm', '--save_mat', action='store_true', help='Whether to save mat file')
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
            src.utils.plotfig(feats, label, save=args.save_fig)
    print('Done.')
