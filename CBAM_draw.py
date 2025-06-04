import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_cbam_block_final():
    fig, ax = plt.subplots(figsize=(16, 11)) # Increased figure size for clarity
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12.5)
    ax.axis('off')

    # Base style for text alignment ONLY
    text_alignment_style = dict(ha="center", va="center")
    arrow_style = dict(facecolor='gray', edgecolor='gray', shrink=0.05, width=0.5, headwidth=8, linestyle="-")
    connector_arrow_style = dict(arrowstyle="-|>", connectionstyle="arc3,rad=0", color="gray", lw=1.5)
    main_flow_arrow_style = dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=15) # Thicker main flow
    sub_block_style = dict(boxstyle="round,pad=0.3", ec="olive", lw=1)

    # Input Feature Map F
    input_x = 2.0
    input_y = 11.0
    ax.text(input_x, input_y, r"Input: $F \in \mathbb{R}^{H \times W \times C}$",
            **text_alignment_style, fontsize=12, color="black")

    # --- 1. Channel Attention Module (CAM) ---
    cam_center_x = 3.5
    cam_center_y = 8.0
    cam_width = 6.0
    cam_height = 3.5
    cam_rect = patches.Rectangle((cam_center_x - cam_width/2, cam_center_y - cam_height/2),
                                 cam_width, cam_height, fill=False, edgecolor='darkblue', lw=1.5, linestyle='--')
    ax.add_patch(cam_rect)
    ax.text(cam_center_x, cam_center_y + cam_height/2 + 0.4, "Channel Attention Module (CAM)",
            **text_alignment_style, fontsize=11, fontweight="bold", color="darkblue")

    # AvgPool & MaxPool (Spatial for CAM)
    pool_cam_x = cam_center_x - cam_width/2 + 1.0
    pool_cam_w = 1.8
    pool_cam_h = 0.9
    gap_cam_y = cam_center_y + 0.7
    gmp_cam_y = cam_center_y - 0.7

    gap_cam = patches.FancyBboxPatch((pool_cam_x, gap_cam_y - pool_cam_h/2), pool_cam_w, pool_cam_h,
                                     **sub_block_style, fc="lightyellow")
    ax.add_patch(gap_cam)
    ax.text(pool_cam_x + pool_cam_w/2, gap_cam_y, "Global\nAvgPool", **text_alignment_style, fontsize=8, color="black")

    gmp_cam = patches.FancyBboxPatch((pool_cam_x, gmp_cam_y - pool_cam_h/2), pool_cam_w, pool_cam_h,
                                     **sub_block_style, fc="lightyellow")
    ax.add_patch(gmp_cam)
    ax.text(pool_cam_x + pool_cam_w/2, gmp_cam_y, "Global\nMaxPool", **text_alignment_style, fontsize=8, color="black")

    # Arrows from F to CAM Pools
    ax.annotate("", xy=(pool_cam_x, gap_cam_y), xytext=(input_x, input_y - 0.5),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="angle3,angleA=180,angleB=90", color="gray", lw=1))
    ax.annotate("", xy=(pool_cam_x, gmp_cam_y), xytext=(input_x, input_y - 0.5),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="angle3,angleA=180,angleB=-90", color="gray", lw=1))

    # Shared MLP for CAM
    mlp_cam_x = pool_cam_x + pool_cam_w + 0.5
    mlp_cam_w = 1.5
    mlp_cam_h = 0.8
    mlp_box_upper = patches.FancyBboxPatch((mlp_cam_x, gap_cam_y - mlp_cam_h/2), mlp_cam_w, mlp_cam_h,
                                           **sub_block_style, fc="lightgreen")
    ax.add_patch(mlp_box_upper)
    ax.text(mlp_cam_x + mlp_cam_w/2, gap_cam_y, "Shared\nMLP", **text_alignment_style, fontsize=8, color="black")
    mlp_box_lower = patches.FancyBboxPatch((mlp_cam_x, gmp_cam_y - mlp_cam_h/2), mlp_cam_w, mlp_cam_h,
                                           **sub_block_style, fc="lightgreen")
    ax.add_patch(mlp_box_lower)
    ax.text(mlp_cam_x + mlp_cam_w/2, gmp_cam_y, "Shared\nMLP", **text_alignment_style, fontsize=8, color="black")

    ax.annotate("", xy=(mlp_cam_x, gap_cam_y), xytext=(pool_cam_x + pool_cam_w, gap_cam_y), arrowprops=connector_arrow_style)
    ax.annotate("", xy=(mlp_cam_x, gmp_cam_y), xytext=(pool_cam_x + pool_cam_w, gmp_cam_y), arrowprops=connector_arrow_style)

    # Element-wise Sum & Sigmoid for Mc
    sum_sig_cam_x = mlp_cam_x + mlp_cam_w + 0.7
    sum_circle_cam = patches.Circle((sum_sig_cam_x, cam_center_y), radius=0.25, fc="pink", ec="purple", lw=1)
    ax.add_patch(sum_circle_cam)
    ax.text(sum_sig_cam_x, cam_center_y, r"$+$", **text_alignment_style, fontsize=12, color="black")
    ax.text(sum_sig_cam_x, cam_center_y - 0.6, "Sigmoid", **text_alignment_style, fontsize=8, color="black")
    ax.text(sum_sig_cam_x, cam_center_y + 0.6, r"$M_c(F)$", **text_alignment_style, fontsize=10, color="purple")

    ax.annotate("", xy=(sum_sig_cam_x - 0.25, cam_center_y + 0.05), xytext=(mlp_cam_x + mlp_cam_w, gap_cam_y), arrowprops=connector_arrow_style)
    ax.annotate("", xy=(sum_sig_cam_x - 0.25, cam_center_y - 0.05), xytext=(mlp_cam_x + mlp_cam_w, gmp_cam_y), arrowprops=connector_arrow_style)
    ax.annotate("", xy=(sum_sig_cam_x, cam_center_y - 0.25), xytext=(sum_sig_cam_x, cam_center_y - 0.35), # Sum to Sigmoid arrow
                arrowprops=dict(arrowstyle="-", color="purple", lw=1)) # Conceptual arrow
    # (Actual Sigmoid is after sum, Mc is the final output of this sub-path)


    # Output F' (after CAM) - Multiplication Point
    f_prime_mult_x = cam_center_x # Align with CAM center
    f_prime_mult_y = cam_center_y - cam_height/2 - 0.8 # Below CAM
    mult_circle_cam = patches.Circle((f_prime_mult_x, f_prime_mult_y), radius=0.3, fc="gold", ec="orange", lw=1.5)
    ax.add_patch(mult_circle_cam)
    ax.text(f_prime_mult_x, f_prime_mult_y, r"$\otimes$", **text_alignment_style, fontsize=16, fontweight="bold", color="black")
    ax.text(f_prime_mult_x, f_prime_mult_y - 0.7, r"$F' = M_c(F) \otimes F$", **text_alignment_style, fontsize=10, color="black")

    # Arrow from F to CAM multiplication (main data flow)
    ax.annotate("", xy=(f_prime_mult_x, f_prime_mult_y + 0.3), xytext=(input_x, input_y - 0.5),
                arrowprops=main_flow_arrow_style)
    # Arrow from Mc to CAM multiplication
    ax.annotate("", xy=(f_prime_mult_x + 0.25, f_prime_mult_y + 0.1), xytext=(sum_sig_cam_x + 0.2, cam_center_y - 0.4), # Adjusted start
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", color="purple", lw=1))


    # --- 2. Spatial Attention Module (SAM) ---
    sam_center_x = f_prime_mult_x + 5.5 # Shift SAM to the right
    sam_center_y = f_prime_mult_y # Align y with F' output for cleaner flow
    sam_width = 6.5
    sam_height = 3.5

    sam_rect = patches.Rectangle((sam_center_x - sam_width/2, sam_center_y - sam_height/2),
                                 sam_width, sam_height, fill=False, edgecolor='darkgreen', lw=1.5, linestyle='--')
    ax.add_patch(sam_rect)
    ax.text(sam_center_x, sam_center_y + sam_height/2 + 0.4, "Spatial Attention Module (SAM)",
            **text_alignment_style, fontsize=11, fontweight="bold", color="darkgreen")

    # Input to SAM is F'
    ax.text(f_prime_mult_x + 1.2, f_prime_mult_y, r"Input: $F'$", **text_alignment_style, fontsize=10, color="gray")
    ax.annotate("", xy=(sam_center_x - sam_width/2, sam_center_y), xytext=(f_prime_mult_x + 0.3, f_prime_mult_y),
                arrowprops=connector_arrow_style)


    # AvgPool & MaxPool (Channel-wise for SAM)
    pool_sam_x = sam_center_x - sam_width/2 + 1.0
    gap_sam_y = sam_center_y + 0.7
    gmp_sam_y = sam_center_y - 0.7

    gap_sam = patches.FancyBboxPatch((pool_sam_x, gap_sam_y - pool_cam_h/2), pool_cam_w, pool_cam_h,
                                     **sub_block_style, fc="lightyellow")
    ax.add_patch(gap_sam)
    ax.text(pool_sam_x + pool_cam_w/2, gap_sam_y, "AvgPool\n(Channel)", **text_alignment_style, fontsize=8, color="black")

    gmp_sam = patches.FancyBboxPatch((pool_sam_x, gmp_sam_y - pool_cam_h/2), pool_cam_w, pool_cam_h,
                                     **sub_block_style, fc="lightyellow")
    ax.add_patch(gmp_sam)
    ax.text(pool_sam_x + pool_cam_w/2, gmp_sam_y, "MaxPool\n(Channel)", **text_alignment_style, fontsize=8, color="black")

    # Arrows from F' (conceptually, it's already inside SAM box) to SAM Pools
    ax.annotate("", xy=(pool_sam_x, gap_sam_y), xytext=(sam_center_x - sam_width/2 + 0.1, sam_center_y + 0.1),
                arrowprops=dict(arrowstyle="-", connectionstyle="angle3,angleA=180,angleB=90", color="gray", lw=1))
    ax.annotate("", xy=(pool_sam_x, gmp_sam_y), xytext=(sam_center_x - sam_width/2 + 0.1, sam_center_y - 0.1),
                arrowprops=dict(arrowstyle="-", connectionstyle="angle3,angleA=180,angleB=-90", color="gray", lw=1))
    # Dummy invisible arrows to make the line start from center of F' input to SAM
    ax.plot([sam_center_x - sam_width/2 + 0.05, sam_center_x - sam_width/2 + 0.05], [sam_center_y, gap_sam_y], color="gray", lw=1, linestyle="-")
    ax.plot([sam_center_x - sam_width/2 + 0.05, sam_center_x - sam_width/2 + 0.05], [sam_center_y, gmp_sam_y], color="gray", lw=1, linestyle="-")
    ax.arrow(sam_center_x - sam_width/2 + 0.05, gap_sam_y, 0.1, 0, head_width=0.1, head_length=0.15, fc='gray', ec='gray', lw=1)
    ax.arrow(sam_center_x - sam_width/2 + 0.05, gmp_sam_y, 0.1, 0, head_width=0.1, head_length=0.15, fc='gray', ec='gray', lw=1)


    # Concatenate for SAM
    concat_sam_x = pool_sam_x + pool_cam_w + 0.5
    concat_box = patches.FancyBboxPatch((concat_sam_x - 0.6, sam_center_y - 0.4), 1.2, 0.8,
                                        **sub_block_style, fc="lightblue")
    ax.add_patch(concat_box)
    ax.text(concat_sam_x, sam_center_y, "Concat", **text_alignment_style, fontsize=8, color="black")

    ax.annotate("", xy=(concat_sam_x - 0.6, sam_center_y + 0.05), xytext=(pool_sam_x + pool_cam_w, gap_sam_y), arrowprops=connector_arrow_style)
    ax.annotate("", xy=(concat_sam_x - 0.6, sam_center_y - 0.05), xytext=(pool_sam_x + pool_cam_w, gmp_sam_y), arrowprops=connector_arrow_style)


    # Convolution Layer for SAM
    conv_sam_x = concat_sam_x + 1.2 + 0.5
    conv_box = patches.FancyBboxPatch((conv_sam_x - 0.9, sam_center_y - 0.6), 1.8, 1.2,
                                      **sub_block_style, fc="lightcoral")
    ax.add_patch(conv_box)
    ax.text(conv_sam_x, sam_center_y, "Conv ($7 \\times 7$)\nSigmoid", **text_alignment_style, fontsize=8, color="black")
    ax.text(conv_sam_x, sam_center_y - 0.9, r"$M_s(F')$", **text_alignment_style, fontsize=10, color="darkred")

    ax.annotate("", xy=(conv_sam_x - 0.9, sam_center_y), xytext=(concat_sam_x + 0.6, sam_center_y), arrowprops=connector_arrow_style)


    # Output F'' - Multiplication point for SAM
    f_double_prime_mult_x = sam_center_x # Align with SAM center
    f_double_prime_mult_y = sam_center_y - sam_height/2 - 0.8 # Below SAM
    mult_circle_sam = patches.Circle((f_double_prime_mult_x, f_double_prime_mult_y), radius=0.3, fc="gold", ec="orange", lw=1.5)
    ax.add_patch(mult_circle_sam)
    ax.text(f_double_prime_mult_x, f_double_prime_mult_y, r"$\otimes$", **text_alignment_style, fontsize=16, fontweight="bold", color="black")
    ax.text(f_double_prime_mult_x, f_double_prime_mult_y - 0.7, r"$F'' = M_s(F') \otimes F'$",
            **text_alignment_style, fontsize=10, color="black")

    # Arrow from F' (output of CAM) to SAM multiplication
    ax.annotate("", xy=(f_double_prime_mult_x, f_double_prime_mult_y + 0.3), xytext=(f_prime_mult_x, f_prime_mult_y-0.3),
                arrowprops=main_flow_arrow_style)
    # Arrow from Ms to SAM multiplication
    ax.annotate("", xy=(f_double_prime_mult_x + 0.25, f_double_prime_mult_y + 0.1), xytext=(conv_sam_x + 0.3, sam_center_y - 0.7),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", color="darkred", lw=1))

    # Final Output of CBAM
    final_output_x = f_double_prime_mult_x + 3.5
    ax.text(final_output_x, f_double_prime_mult_y, r"Output: $F'' \in \mathbb{R}^{H \times W \times C}$",
            **text_alignment_style, fontsize=12, color="black")
    ax.annotate("", xy=(final_output_x - 1.8, f_double_prime_mult_y), xytext=(f_double_prime_mult_x + 0.3, f_double_prime_mult_y),
                arrowprops=main_flow_arrow_style)


    fig.suptitle("CBAM (Convolutional Block Attention Module)", fontsize=14, fontweight="bold", color="black")
    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95)
    plt.savefig("cbam_block_diagram_final.png", dpi=300, bbox_inches='tight')
    plt.show()

# Call the function
plot_cbam_block_final()