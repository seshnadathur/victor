import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, colorConverter

ryb_colors = np.array(['#3130ff', '#3366ff', '#9DAFFF', '#A6BDD7', '#F4C800', '#FFB300',
                       '#FF8E00', '#F13A13', '#C10020'])
ryg_colors = np.array(['#007D34', '#93AA00', '#F4C800', '#FFB300', '#FF8E00', '#F13A13',
                       '#C10020', '#7F180D'])

def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
    """

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def plot_2D_ccf(xi_sp, rs, rp=None, even=True, cmap=mpl.cm.RdYlBu_r, vmin=-1, vmax=0.5,
                contours=None, contour_colors='white', clabel=False, clabel_precision=2,
                linewidths=1.2, shift=True, colorbar=True, axis_label='r', xlabel=None,
                ylabel=None, cbar_label=None):
    """
    """
    if shift:
        mid = 1 - vmax/(vmax + abs(vmin))
        cmap = shifted_color_map(cmap, midpoint=mid)
    if colorbar:
        plt.figure(figsize=(7.5,6))
    else:
        plt.figure(figsize=(6.2,6))
    if rp is None:
        rp = rs
        even = True
    im = plt.pcolormesh(rs, rp, xi_sp(rs, rp), vmin=vmin, vmax=vmax, cmap=cmap, shading='gouraud')
    plt.pcolormesh(-rs, rp, xi_sp(rs, rp), vmin=vmin, vmax=vmax, cmap=cmap, shading='gouraud')
    if even:
        plt.pcolormesh(rs, -rp, xi_sp(rs, rp), vmin=vmin, vmax=vmax, cmap=cmap, shading='gouraud')
        plt.pcolormesh(-rs, -rp, xi_sp(rs, rp), vmin=vmin, vmax=vmax, cmap=cmap, shading='gouraud')
    plt.tick_params(labelsize=16)
    if colorbar:
        cb = plt.colorbar(im)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=18)

    if contours:
        cs = plt.contour(rs, rp, xi_sp(rs, rp), contours, colors=contour_colors, linestyles='solid', linewidths=linewidths)
        plt.contour(-rs, rp, xi_sp(rs, rp), contours, colors=contour_colors, linestyles='solid', linewidths=linewidths)
        if even:
            plt.contour(rs, -rp, xi_sp(rs, rp), contours, colors=contour_colors, linestyles='solid', linewidths=linewidths)
            plt.contour(-rs, -rp, xi_sp(rs, rp), contours, colors=contour_colors, linestyles='solid', linewidths=linewidths)
        if clabel:
            def fmt(x):
                precision = f'{0.1*clabel_precision}f'
                return f"{x:.2f}"
            plt.clabel(cs, inline=True, fontsize=10, fmt=fmt)

    if axis_label is not None:
        xlabel = r'$%s_\perp\;[h^{-1}\mathrm{Mpc}]$' % axis_label
        ylabel = r'$%s_{||}\;[h^{-1}\mathrm{Mpc}]$' % axis_label
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    print(plt.xticks()[0])
    plt.yticks(ticks=plt.xticks()[0])
    plt.xlim(-np.max(rs), np.max(rs))
    plt.ylim(-np.max(rp), np.max(rp))
