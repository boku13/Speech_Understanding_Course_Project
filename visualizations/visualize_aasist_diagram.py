import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as path_effects
import os

# Create visualizations directory if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

# Set figure size and style
plt.figure(figsize=(16, 12))
plt.style.use('default')

# Define pastel colors (keeping these as requested)
PINK = "#FFB6C1"       # Light pink
LAVENDER = "#E6E6FA"   # Lavender
MINT = "#98FB98"       # Mint green
PEACH = "#FFDAB9"      # Peach
SKY = "#87CEEB"        # Sky blue
LILAC = "#DDA0DD"      # Lilac
LEMON = "#FFFACD"      # Lemon chiffon
CORAL = "#FFA07A"      # Light coral

# Background color - changed to white as requested
bg_color = "#FFFFFF"   # Plain white background

def add_box(ax, x, y, width, height, color, alpha=0.9, label=""):
    # Draw a rounded box with subtle shadow for professional look
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.4,rounding_size=0.2",
        facecolor=color, alpha=alpha,
        edgecolor='gray', linewidth=1,
        path_effects=[path_effects.withSimplePatchShadow(
            offset=(1, -1), shadow_rgbFace='gray', alpha=0.2)]
    )
    ax.add_patch(box)
    
    # Add label with professional font (Arial/Helvetica)
    text = ax.text(x + width/2, y + height/2, label, 
             ha='center', va='center', 
             fontsize=11, 
             fontfamily='sans-serif',
             fontweight='bold', 
             color='#333333')
    
    # Add subtle outline to make text pop against colored backgrounds
    text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    return x + width/2, y + height/2  # Return center coordinates

def add_arrow(ax, start, end, color='#555555', style='->', width=1.2):
    arrow = FancyArrowPatch(
        start, end, 
        arrowstyle=style, 
        color=color, 
        linewidth=width,
        connectionstyle="arc3,rad=0.1",
        alpha=0.8
    )
    ax.add_patch(arrow)

# Create plot with professional white background
fig, ax = plt.subplots(figsize=(16, 12), facecolor=bg_color)
ax.set_facecolor(bg_color)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Add title with professional styling
title = ax.text(50, 95, "Deception Detection in Real life Trials using Audio with Graph Convolutional\nNeural Networks", 
         ha='center', va='center', 
         fontsize=20, 
         fontfamily='sans-serif',
         fontweight='bold',
         color='#333333')  # Dark gray for professional look

# Input
input_center = add_box(ax, 10, 80, 15, 6, PINK, label="Input Audio")

# First processing stage - CONV
conv_center = add_box(ax, 30, 80, 15, 6, LAVENDER, label="CONV (SincFilters)")
add_arrow(ax, input_center, conv_center)

# Encoder
encoder_center = add_box(ax, 50, 80, 15, 6, MINT, label="Residual Encoder")
add_arrow(ax, conv_center, encoder_center)

# Split to Spectral and Temporal
spec_center = add_box(ax, 40, 70, 15, 6, PEACH, label="Spectral Features")
temp_center = add_box(ax, 60, 70, 15, 6, PEACH, label="Temporal Features")
add_arrow(ax, encoder_center, spec_center)
add_arrow(ax, encoder_center, temp_center)

# GAT layers
gat_s_center = add_box(ax, 40, 60, 15, 6, SKY, label="GAT-S")
gat_t_center = add_box(ax, 60, 60, 15, 6, SKY, label="GAT-T")
add_arrow(ax, spec_center, gat_s_center)
add_arrow(ax, temp_center, gat_t_center)

# Pool layers
pool_s_center = add_box(ax, 40, 50, 15, 6, LILAC, label="Pool-S")
pool_t_center = add_box(ax, 60, 50, 15, 6, LILAC, label="Pool-T")
add_arrow(ax, gat_s_center, pool_s_center)
add_arrow(ax, gat_t_center, pool_t_center)

# Heterogeneous GAT layers - Branch 1
master1_center = add_box(ax, 20, 40, 10, 5, CORAL, alpha=0.9, label="Master1")
hgat_st11_center = add_box(ax, 50, 40, 20, 5, LEMON, alpha=0.9, label="HtrgGAT-ST11")
add_arrow(ax, pool_s_center, hgat_st11_center)
add_arrow(ax, pool_t_center, hgat_st11_center)
add_arrow(ax, master1_center, hgat_st11_center)

# Heterogeneous GAT layers - Branch 2
master2_center = add_box(ax, 20, 30, 10, 5, CORAL, alpha=0.9, label="Master2")
hgat_st21_center = add_box(ax, 50, 30, 20, 5, LEMON, alpha=0.9, label="HtrgGAT-ST21")
add_arrow(ax, pool_s_center, hgat_st21_center)
add_arrow(ax, pool_t_center, hgat_st21_center)
add_arrow(ax, master2_center, hgat_st21_center)

# Pool layers after first HtrgGAT
pool_hs1_center = add_box(ax, 40, 20, 15, 5, MINT, alpha=0.9, label="Pool-hS1")
pool_ht1_center = add_box(ax, 60, 20, 15, 5, MINT, alpha=0.9, label="Pool-hT1")
add_arrow(ax, hgat_st11_center, pool_hs1_center)
add_arrow(ax, hgat_st11_center, pool_ht1_center)

# Pool layers after second HtrgGAT
pool_hs2_center = add_box(ax, 40, 10, 15, 5, SKY, alpha=0.9, label="Pool-hS2")
pool_ht2_center = add_box(ax, 60, 10, 15, 5, SKY, alpha=0.9, label="Pool-hT2")
add_arrow(ax, hgat_st21_center, pool_hs2_center)
add_arrow(ax, hgat_st21_center, pool_ht2_center)

# Output layer
output_center = add_box(ax, 50, 3, 25, 5, PINK, label="Truth or Lie Classification")
add_arrow(ax, pool_hs1_center, output_center)
add_arrow(ax, pool_ht1_center, output_center)
add_arrow(ax, pool_hs2_center, output_center)
add_arrow(ax, pool_ht2_center, output_center)
add_arrow(ax, master1_center, output_center)
add_arrow(ax, master2_center, output_center)

# Add explanatory text in professional font
footer = ax.text(50, 1, "The model uses Graph Attention Networks to model spectral-temporal relationships for detecting deception in speech.", 
           ha='center', va='center', 
           fontsize=10, 
           fontfamily='sans-serif',
           fontweight='normal',
           color='#555555')

# Save figure with higher quality
output_file = "visualizations/aasist_diagram.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=bg_color)
print(f"Professional diagram saved to: {os.path.abspath(output_file)}")

plt.close() 