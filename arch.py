import sys
import os

# Add path to PlotNeuralNet
sys.path.append('../')
from pycore.tikzeng import *

# spacing and caption configuration
GAPX = 4.5
GAPE = 3.5
CAPTION_SIZE = '\\footnotesize'

# customise head with figure scale and caption shift
head = to_head('..')
head = head.replace('\\begin{tikzpicture}',
                    '\\begin{tikzpicture}[scale=1.15]\n\\tikzstyle{caption}=[yshift=-8pt]')

# Define your PyramidAttnCNN architecture with better spacing
arch = [
    head,
    to_cor(),
    to_begin(),
    
    # ============ INPUT LAYER ============
    to_Conv("input", 512, 42, offset="(0,0,0)", to="(0,0,0)",
            height=40, depth=40, width=3,
            caption=f"{CAPTION_SIZE} Input\\\\42x101\\\\IMU Data"),
    
    # ============ BRANCH 1: KERNEL SIZE 3 (Top Branch) ============
    # Conv Layer 1
    to_Conv("conv1_k3", 256, 24, offset=f"({GAPX + 1},0,4)", to="(input-east)",
            height=35, depth=35, width=2.5,
            caption=f"{CAPTION_SIZE} Conv3\\\\24ch"),
    
    # Conv Layer 2  
    to_Conv("conv2_k3", 256, 48, offset=f"({GAPX},0,0)", to="(conv1_k3-east)",
            height=30, depth=30, width=3,
            caption=f"{CAPTION_SIZE} Conv3\\\\48ch"),
    
    # Conv Layer 3
    to_Conv("conv3_k3", 256, 96, offset=f"({GAPX},0,0)", to="(conv2_k3-east)",
            height=25, depth=25, width=3.5,
            caption=f"{CAPTION_SIZE} Conv3\\\\96ch"),
    
    # Temporal Downsampling
    to_Pool("down1_k3", offset=f"({GAPX},0,0)", to="(conv3_k3-east)",
            height=18, depth=18, width=2,
            caption=f"{CAPTION_SIZE} Down\\\\s=4"),
    to_Pool("down2_k3", offset=f"({GAPE + 0.5},0,0)", to="(down1_k3-east)",
            height=12, depth=12, width=2,
            caption=f"{CAPTION_SIZE} Down\\\\s=3"),
    
    # ============ BRANCH 2: KERNEL SIZE 7 (Bottom Branch) ============
    # Conv Layer 1
    to_Conv("conv1_k7", 256, 24, offset=f"({GAPX + 1},0,-4)", to="(input-east)",
            height=35, depth=35, width=2.5,
            caption=f"{CAPTION_SIZE} Conv7\\\\24ch"),
    
    # Conv Layer 2
    to_Conv("conv2_k7", 256, 48, offset=f"({GAPX},0,0)", to="(conv1_k7-east)",
            height=30, depth=30, width=3,
            caption=f"{CAPTION_SIZE} Conv7\\\\48ch"),
    
    # Conv Layer 3
    to_Conv("conv3_k7", 256, 96, offset=f"({GAPX},0,0)", to="(conv2_k7-east)",
            height=25, depth=25, width=3.5,
            caption=f"{CAPTION_SIZE} Conv7\\\\96ch"),
    
    # Temporal Downsampling
    to_Pool("down1_k7", offset=f"({GAPX},0,0)", to="(conv3_k7-east)",
            height=18, depth=18, width=2,
            caption=f"{CAPTION_SIZE} Down\\\\s=4"),
    to_Pool("down2_k7", offset=f"({GAPE + 0.5},0,0)", to="(down1_k7-east)",
            height=12, depth=12, width=2,
            caption=f"{CAPTION_SIZE} Down\\\\s=3"),
    
    # ============ MERGE & ATTENTION (Centered) ============
    # Concatenation - positioned between the two branches
    to_Conv("concat", 128, 192, offset=f"({GAPX + 1},0,0)", to="(down2_k3-east)",
            height=20, depth=20, width=4,
            caption=f"{CAPTION_SIZE} Concat\\\\192ch"),
    
    # Channel Reduction
    to_Conv("reduce", 128, 96, offset=f"({GAPX},0,0)", to="(concat-east)",
            height=18, depth=18, width=2.5,
            caption=f"{CAPTION_SIZE} 1x1 Conv\\\\96ch"),
    
    # Multi-Head Attention Block
    to_Conv("attention", 96, 96, offset=f"({GAPX},0,0)", to="(reduce-east)",
            height=16, depth=16, width=3.5,
            caption=f"{CAPTION_SIZE} MH-Attn\\\\4heads"),
    
    # ============ OUTPUT HEADS (Spread out vertically) ============
    to_SoftMax("out_x", 101, f"({GAPX + 1},0,5)", "(attention-east)",
               caption=f"{CAPTION_SIZE} X angles"),
    to_SoftMax("out_y", 101, f"({GAPX + 1},0,0)", "(attention-east)",
               caption=f"{CAPTION_SIZE} Y angles"),
    to_SoftMax("out_z", 101, f"({GAPX + 1},0,-5)", "(attention-east)",
               caption=f"{CAPTION_SIZE} Z angles"),
    
    # ============ CONNECTIONS ============
    # Input to branches
    to_connection("input", "conv1_k3"),
    to_connection("input", "conv1_k7"),
    
    # Branch 1 flow
    to_connection("conv1_k3", "conv2_k3"),
    to_connection("conv2_k3", "conv3_k3"),
    to_connection("conv3_k3", "down1_k3"),
    to_connection("down1_k3", "down2_k3"),
    
    # Branch 2 flow
    to_connection("conv1_k7", "conv2_k7"),
    to_connection("conv2_k7", "conv3_k7"),
    to_connection("conv3_k7", "down1_k7"),
    to_connection("down1_k7", "down2_k7"),
    
    # Merge connections
    to_connection("down2_k3", "concat"),
    to_connection("down2_k7", "concat"),
    to_connection("concat", "reduce"),
    to_connection("reduce", "attention"),
    
    # Output connections
    to_connection("attention", "out_x"),
    to_connection("attention", "out_y"),
    to_connection("attention", "out_z"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')
    print(f"✅ Generated {namefile}.tex")
    print(f"Next: Run 'pdflatex {namefile}.tex' to create PDF")

if __name__ == '__main__':
    main()
