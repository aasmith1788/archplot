import os
from pycore.tikzeng import *

arch = [
    to_head('Example', scale=1.15),
    to_cor(),
    to_begin(),
    to_Conv('conv1', 256, 48, to='(0,0)', caption='Conv'),
    to_Pool('pool1', to='(2,0)', caption='Pool'),
    to_FullyConnected('fc1', 10, to='(4,0)', caption='FC'),
    to_SoftMax('out', 10, to='(6,0)', caption='Output'),
    to_end()
]

if __name__ == '__main__':
    to_generate(arch, 'pyramid_cnn.tex')
    os.system('pdflatex pyramid_cnn.tex')
