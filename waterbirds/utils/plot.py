import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_context(context='paper', font_scale=2)


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def next_color(ax):
    return next(ax._get_lines.prop_cycler)['color']


def plot_image(ax, image):
    image = image.transpose((1, 2, 0))
    ax.imshow(image)