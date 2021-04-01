import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, colorConverter
from scipy.ndimage import gaussian_filter
import os

class FigureUtilities:

    def __init__(self):

        self.wad_colours = ['#2A4BD7', '#1D6914', '#814A19', '#8126C0',
                            '#9DAFFF', '#81C57A', '#E9DEBB', '#AD2323',
                            '#29D0D0', '#FFEE1F', '#FF9233', '#FFCDF3',
                            '#000000', '#575757', '#A0A0A0']

        self.kelly_colours = ['#FFB300',  # 0 Vivid Yellow
                              '#803E75',  # 1 Strong Purple
                              '#FF6800',  # 2 Vivid Orange
                              '#A6BDD7',  # 3 Very Light Blue
                              '#C10020',  # 4 Vivid Red
                              '#CEA262',  # 5 Grayish Yellow
                              '#817066',  # 6 Medium Gray
                              '#007D34',  # 7 Vivid Green
                              '#F6768E',  # 8 Strong Purplish Pink
                              '#00538A',  # 9 Strong Blue
                              '#FF7A5C',  # 10 Strong Yellowish Pink
                              '#53377A',  # 11 Strong Violet
                              '#FF8E00',  # 12 Vivid Orange Yellow
                              '#B32851',  # 13 Strong Purplish Red
                              '#F4C800',  # 14 Vivid Greenish Yellow
                              '#7F180D',  # 15 Strong Reddish Brown
                              '#93AA00',  # 16 Vivid Yellowish Green
                              '#593315',  # 17 Deep Yellowish Brown
                              '#F13A13',  # 18 Vivid Reddish Orange
                              '#232C16',  # 19 Dark Olive Green
                              ]

        self.RdYlGn = np.asarray(self.kelly_colours)[[7, 16, 14, 0, 12, 2, 18, 4, 15]]
        self.basic_RdYlBu = np.asarray(self.kelly_colours)[[11, 9, 3, 0, 12, 2, 18, 4, 15]]
        self.RdYlBu = np.array(['#3130ff', '#3366ff', self.wad_colours[4], self.kelly_colours[3],
                                self.kelly_colours[14], self.kelly_colours[0], self.kelly_colours[12],
                                self.kelly_colours[18], self.kelly_colours[4]])
        self.pointstyles = np.array(['o', '^', 'v', 's', 'D', '<', '>', 'p', 'd', '*', 'h', 'H', '8'])

        parchment_file = "/Users/seshadri/Workspace/colormaps/PlanckParchment.txt"
        if os.access(parchment_file, os.F_OK):
            self.parchment_cmap = ListedColormap(np.loadtxt(parchment_file)/255.)

    @staticmethod
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

    @staticmethod
    def contour_2d(vectorx, vectory, weights=None, bins=20, levels=[0.683, 0.952]):

        H, X, Y = np.histogram2d(vectorx, vectory, bins=bins, weights=weights)
        H = gaussian_filter(H, 1)
        Hflat = H.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except:
                V[i] = Hflat[0]
        V.sort()
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        X2 = np.concatenate([
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ])
        Y2 = np.concatenate([
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ])

        return X2, Y2, H2, V, H.max(), levels

    @staticmethod
    def plot_contour_2d(X2, Y2, H2, V, Hmax, levels, color1='cornflowerblue', color2='navy'):

        rgba_color = colorConverter.to_rgba(color1)
        contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
        for i, l in enumerate(levels):
            contour_cmap[i][-1] *= float(i) / (len(levels) + 1)
        contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        # contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased", False)
        plt.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [Hmax * (1 + 1e-4)]]), **contourf_kwargs)
        contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color2)
        plt.contour(X2, Y2, H2.T, V, **contour_kwargs)
