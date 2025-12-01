import sys
sys.path.insert(0, '.')

from examples.sgemm_kernel_1 import *
from examples.sgemm_kernel_2 import *
from examples.sgemm_kernel_3 import *
from examples.sgemm_kernel_4 import *
from examples.sgemm_kernel_5 import *
from examples.sgemm_shared_mem_block import *
from examples.scan import *
from examples.h100_matmul import *
from prism import generate

kernels = [
    'my_sgemm_kernel_1',
    'my_sgemm_kernel_2',
    'my_sgemm_kernel_3',
    'my_sgemm_kernel_4',
    'my_sgemm_kernel_5',
    'sgemm_shared_mem_block',
]

h100_kernels = [
    'cutlass_my_h100_match_no_tail',
]

scan_kernels = [
    'shfl_up_sync_wrapper',
    'scan_warps',
    'get_warp_ends_inclusive',
    'naive_scan_warp_ends',
    'update_with_new_data_and_writeback',
    'write_block_aggregate',
    'write_block_prefix',
    'lookback_to_prev_blocks',
    'get_block_idx_from_global_counter',
    'scan_kernel',
]

import os
os.makedirs('generated', exist_ok=True)

for name in kernels + h100_kernels + scan_kernels:
    code = generate(name, auto_sync=True)
    filename = f'generated/{name}.cu'
    with open(filename, 'w') as f:
        f.write(code)
    print(f'  {filename}')
