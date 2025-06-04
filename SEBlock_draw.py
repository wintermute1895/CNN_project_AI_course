import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_se_block_final():
    fig, ax = plt.subplots(figsize=(10, 7)) # Increased height slightly for better spacing
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5) # Adjusted y-limit
    ax.axis('off')

    # Base style for text alignment ONLY
    text_alignment_style = dict(ha="center", va="center")
    arrow_style = dict(facecolor='gray', edgecolor='gray', shrink=0.05, width=0.5, headwidth=8, linestyle="-")
    connector_arrow_style = dict(arrowstyle="-|>", connectionstyle="arc3,rad=0", color="gray", lw=1.5)


    # Input Feature Map
    ax.text(1.5, 6.5, r"Input: $X \in \mathbb{R}^{H \times W \times C}$", **text_alignment_style, fontsize=11, color="black")

    # --- Squeeze Operation ---
    y_op_center = 5.0 # Unified y-center for Squeeze and Excitation for a clearer horizontal flow
    squeeze_x = 1.5
    squeeze_w = 2.0
    squeeze_h = 1.2
    squeeze_box = patches.FancyBboxPatch((squeeze_x - squeeze_w/2, y_op_center - squeeze_h/2), squeeze_w, squeeze_h,
                                         boxstyle="round,pad=0.3", fc="lightcoral", ec="darkred", lw=1.5)
    ax.add_patch(squeeze_box)
    ax.text(squeeze_x, y_op_center, "Squeeze\n(Global Avg Pool)", **text_alignment_style, fontsize=10, color="black")
    ax.text(squeeze_x, y_op_center - squeeze_h/2 - 0.3, r"$z \in \mathbb{R}^{1 \times 1 \times C}$",
            **text_alignment_style, fontsize=9, color="darkred")

    # Arrow from Input to Squeeze
    ax.annotate("", xy=(squeeze_x, y_op_center + squeeze_h/2), xytext=(1.5, 6.2),
                arrowprops=connector_arrow_style)


    # --- Excitation Operation ---
    excitation_title_x = 5.0
    fc1_x = excitation_title_x - 1.0
    fc2_x = excitation_title_x + 1.0
    fc_w = 1.5
    fc_h = 1.0

    # FC1 (Reduce)
    fc1_box = patches.FancyBboxPatch((fc1_x - fc_w/2, y_op_center - fc_h/2), fc_w, fc_h,
                                     boxstyle="round,pad=0.3", fc="lightgreen", ec="darkgreen", lw=1.5)
    ax.add_patch(fc1_box)
    ax.text(fc1_x, y_op_center, "FC1 (ReLU)\n(C -> C/r)", **text_alignment_style, fontsize=9, color="black")

    # FC2 (Expand)
    fc2_box = patches.FancyBboxPatch((fc2_x - fc_w/2, y_op_center - fc_h/2), fc_w, fc_h,
                                     boxstyle="round,pad=0.3", fc="lightgreen", ec="darkgreen", lw=1.5)
    ax.add_patch(fc2_box)
    ax.text(fc2_x, y_op_center, "FC2 (Sigmoid)\n(C/r -> C)", **text_alignment_style, fontsize=9, color="black")

    ax.text(excitation_title_x, y_op_center + fc_h/2 + 0.5, "Excitation",
            **text_alignment_style, fontsize=11, fontweight="bold", color="black")
    ax.text(excitation_title_x, y_op_center - fc_h/2 - 0.3, r"Channel Weights: $s \in \mathbb{R}^{1 \times 1 \times C}$",
            **text_alignment_style, fontsize=9, color="darkgreen")


    # Arrows for Excitation
    ax.annotate("", xy=(fc1_x - fc_w/2, y_op_center), xytext=(squeeze_x + squeeze_w/2, y_op_center),
                arrowprops=connector_arrow_style) # Squeeze to FC1
    ax.annotate("", xy=(fc2_x - fc_w/2, y_op_center), xytext=(fc1_x + fc_w/2, y_op_center),
                arrowprops=connector_arrow_style) # FC1 to FC2


    # --- Scale Operation (Channel-wise Multiplication) ---
    y_scale = 2.5
    x_scale_mult = 5.0

    scale_circle = patches.Circle((x_scale_mult, y_scale), radius=0.3, fc="gold", ec="orange", lw=1.5)
    ax.add_patch(scale_circle)
    ax.text(x_scale_mult, y_scale, r"$\otimes$", **text_alignment_style, fontsize=16, fontweight="bold", color="black")
    ax.text(x_scale_mult, y_scale + 0.7, "Scale (Reweight)", **text_alignment_style, fontsize=11, fontweight="bold", color="black")

    # Arrow from Excitation output (s) to Scale
    ax.annotate("", xy=(x_scale_mult, y_scale + 0.3), xytext=(fc2_x, y_op_center - fc_h/2), # Point from bottom-center of FC2
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.2", color="gray", lw=1.5))


    # Arrow from original Input X to Scale (for multiplication)
    line = patches.FancyArrowPatch((1.5, 6.2), (x_scale_mult - 0.35, y_scale + 0.05), # Adjusted end point slightly
                                   connectionstyle=f"angle3,angleA=-90,angleB=170",
                                   arrowstyle="-|>", mutation_scale=15, lw=1.5, color="gray")
    ax.add_patch(line)


    # Output Feature Map
    output_x = x_scale_mult + 3.0
    ax.text(output_x, y_scale, r"Output: $\tilde{X} = s \cdot X \in \mathbb{R}^{H \times W \times C}$",
            **text_alignment_style, fontsize=11, color="black")
    ax.annotate("", xy=(output_x - 1.5, y_scale), xytext=(x_scale_mult + 0.3, y_scale),
                arrowprops=connector_arrow_style) # Scale to Output

    fig.suptitle("SEBlock (Squeeze-and-Excitation Block)", fontsize=14, fontweight="bold", color="black")
    plt.subplots_adjust(top=0.90) # Adjust top to make space for suptitle
    plt.savefig("se_block_diagram_final.png", dpi=300, bbox_inches='tight')
    plt.show()

# Call the function
plot_se_block_final()