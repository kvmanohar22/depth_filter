# Given a directory structure with the generated log files,
#   this script generates histogram distributions and two-view stats

import sys
import argparse
sys.path.append('..')
from df.nview import analyze
from df.nview import generate_nview_stats

def test(args):
    analyze(args.dir)
    generate_nview_stats(args.dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='../../analysis/logs/df_upper_stats_nview/idx_6_point_706')
    args = parser.parse_args()
    test(args)

