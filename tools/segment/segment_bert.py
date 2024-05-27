import argparse
import sys
import tokenization
from tqdm import tqdm
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(dir_path, "vocab.txt"), do_lower_case=False)


def main(args):
    filename = args.input_file
    with open(filename, 'r', encoding='utf-8') as f:
        with open(f'{filename}.char', 'w', encoding='utf-8') as f1:
            for line in tqdm(f.readlines()):
                line = line.strip()
                line = line.replace(" ", "")
                if not line:
                    continue
                line1 = []
                for l in line.split('\t'):
                    l = tokenization.convert_to_unicode(l)
                    tokens = tokenizer.tokenize(l)
                    tokens = ' '.join(tokens)
                    line1.append(tokens)
                line = '\t'.join(line1)
                f1.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input_file',
                        help='Path to the input file',
                        required=True)
    args = parser.parse_args()
    main(args)
