import sys
import argparse
sys.path.append('..')
from df.nview import analyze_single_file


def test(args):
    srcfile = args.file
    svgfile = srcfile.replace('.score', '.svg')
    analyze_single_file(srcfile, svgfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='../../analysis/logs/ref_27_cur_28.score')
    args = parser.parse_args()
    test(args)

