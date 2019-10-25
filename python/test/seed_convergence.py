import sys
import argparse
sys.path.append('..')
from df.seed_convergence import convergence
from df.seed_convergence import compare_convergence

# convergence('../../analysis/forward/log5/depth_filter_rpg_synthetic_forward_convergence.txt', 'forward camera', 214637.0)
# convergence('../../analysis/forward/log10/depth_filter_rpg_synthetic_forward_convergence.txt', 'forward camera', 219478.0)
# convergence('../../analysis/forward/log6/depth_filter_rpg_synthetic_forward_convergence.txt', 'forward camera', 217384.0)
# convergence('../../analysis/downward/log2/depth_filter_sin2_tex2_h1_v8_d_convergence.txt', 'downward camera',275384.0)
# convergence('../../analysis/downward/log3/depth_filter_sin2_tex2_h1_v8_d_convergence.txt', 'downward camera',275384.0)

compare_convergence(
        ['../../analysis/forward/log8/depth_filter_rpg_synthetic_forward_convergence.txt',
         '../../analysis/forward/log9/depth_filter_rpg_synthetic_forward_convergence.txt',
         '../../analysis/forward/log10/depth_filter_rpg_synthetic_forward_convergence.txt'],
        ['forward mean = 23', 'forward mean = 50', 'forward mean = 100'], 219478.0
        )

compare_convergence(
        ['../../analysis/downward/log2/depth_filter_sin2_tex2_h1_v8_d_convergence.txt',
         '../../analysis/downward/log3/depth_filter_sin2_tex2_h1_v8_d_convergence.txt'],
        ['downward mean = 5m', 'downward mean = 2m'], 277125.0
        )

compare_convergence(
        ['../../analysis/downward/log2/depth_filter_sin2_tex2_h1_v8_d_convergence.txt',
         '../../analysis/forward/log9/depth_filter_rpg_synthetic_forward_convergence.txt'],
        ['downward [mean=5m]', 'forward [mean=50m]'], 217384.0
        )

