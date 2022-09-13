"""
==========
Bar of pie
==========

Make a "bar of pie" chart where the first slice of the pie is
"exploded" into a bar chart with a further breakdown of said slice's
characteristics. The example demonstrates using a figure with multiple
sets of axes and using the axes patches list to add two ConnectionPatches
to link the subplot charts.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np




Color1 = "#490727"
Color1a = "#490727c5"
Color2 = "#b34a1d"
Color3 = "#0e253e"
Color3a = "#0e253ec5"
Color4 = "#119a54"
Color5 = "#3b6d0c"


# Plot parameters
# C IV non
line = 'CIV_non'
j = 0
explode = [0.1, 0, 0, 0]
age_ratios = [13/31, 10/31, 4/31, 4/31]
Color_p = Color1

# C IV ps
# line = 'CIV_ps'
# j = 1
# explode = [0, 0.1, 0, 0]
# age_ratios = [45/79, 22/79, 1/79, 11/79]
# Color_p = Color3


# # Mg II non
# line = 'MgII_non'
# j = 2
# explode = [0, 0, 0.1, 0]
# age_ratios = [7/32, 7/32, 11/32, 7/32]
# Color_p = Color2


# # Mg II non
# line = 'MgII_ps'
# j = 3
# explode = [0, 0, 0, 0.1]
# age_ratios = [22/69, 21/69, 14/69, 12/69]
# Color_p = Color4


# make figure and assign axis objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
fig.subplots_adjust(wspace=0)

# pie chart parameters
overall_ratios = [31/211, 79/211, 32/211, 69/211]
angle = -180 * overall_ratios[1]
labels = ['C IV (nonDetected)', 'C IV (psDetected)', 'Mg II (nonDetected)', \
          'Mg II (psDetected)']
wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                     labels=labels, explode=explode,\
                         colors=[Color1a, Color3a, Color2, Color4])

# bar chart parameters
age_labels = ['3', '2', '1', '0']
bottom = 1
width = .2

# Adding from the top matches the legend.
for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    # CIV ps
    bc = ax2.bar(0, height, width, bottom=bottom, color=Color_p, label=label,
                 alpha=0.1 + 0.25 * j)
    ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

ax2.set_title('Number of fitted Gaussians')
ax2.legend()
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

# use ConnectionPatch to draw lines between the two plots
# theta1, theta2 = wedges[0].theta1, wedges[0].theta2
# center, r = wedges[0].center, wedges[0].r
# bar_height = sum(age_ratios)

# # draw top connecting line
# x = r * np.cos(np.pi / 180 * theta2) + center[0]
# y = r * np.sin(np.pi / 180 * theta2) + center[1]
# con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
#                       xyB=(x, y), coordsB=ax1.transData)
# con.set_color([0, 0, 0])
# con.set_linewidth(4)
# ax2.add_artist(con)

# # draw bottom connecting line
# x = r * np.cos(np.pi / 180 * theta1) + center[0]
# y = r * np.sin(np.pi / 180 * theta1) + center[1]
# con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
#                       xyB=(x, y), coordsB=ax1.transData)
# con.set_color([0, 0, 0])
# ax2.add_artist(con)
# con.set_linewidth(4)

plt.show()
fig.savefig(line+'_pie_1.pdf',format="pdf",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )