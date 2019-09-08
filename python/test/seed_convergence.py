import sys
import argparse
sys.path.append('..')
from df.seed_convergence import convergence
from df.seed_convergence import compare_convergence

# convergence('../../analysis/forward/log3/depth_filter_rpg_synthetic_forward_convergence.txt', 'forward camera', 214637.0)
# convergence('../../analysis/forward/log4/depth_filter_rpg_synthetic_forward_convergence.txt', 'forward camera', 217384.0)
# convergence('../../analysis/downward/log2/depth_filter_sin2_tex2_h1_v8_d_convergence.txt', 'downward camera',275384.0)

compare_convergence(
        ['../../analysis/forward/log3/depth_filter_rpg_synthetic_forward_convergence.txt',
         '../../analysis/forward/log4/depth_filter_rpg_synthetic_forward_convergence.txt'],
        ['forward mean = 23', 'forward mean = 50'], 217384.0
        )

compare_convergence(
        ['../../analysis/downward/log2/depth_filter_sin2_tex2_h1_v8_d_convergence.txt',
         '../../analysis/forward/log3/depth_filter_rpg_synthetic_forward_convergence.txt'],
        ['downward', 'forward'], 217384.0
        )

