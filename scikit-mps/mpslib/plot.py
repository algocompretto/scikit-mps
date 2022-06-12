#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPS plotting utilities
"""
from typing import Tuple


def module_exists(module_name: str, show_info=0):
    """
    Function to check whether a module exists or not.

    Parameters
    ----------
    module_name : str
        String for module name.
    show_info : int
        If bigger than zero (0), then it will print information to the user console.

    Returns
    --------
    x : bool
    """
    try:
        __import__(module_name)
    except ImportError:
        if show_info > 0:
            print(f'{module_name} could not be loaded. Please install it using e.g')
            print(f'\tpip install {module_name}')
        return False
    else:
        return True


def plot_3d_reals(mps_object, show: int = 4):
    """
    Wrapper function for `plot_3d_reals_pyvista`.

    Parameters
    ----------
    mps_object : MPS object
        MPS object after performing the simulations operations.
    show : int
        Number of realizations to show.

    See also
    -------
    plot_3d_reals_pyvista
    """
    plot_3d_reals_pyvista(mps_object, show)


def plot_3d_reals_pyvista(mps_object, show: int = 4):
    """
    Plot realizations in mps_object.sim in 3D using pyvista

    Parameters
    ----------
    mps_object : MPS object
        MPS object.
    show : int
        Show a maximum of `show` realizations.
    """
    import numpy as np
    import pyvista

    if not (hasattr(mps_object, 'sim')):
        print('No data to plot (no "sim" attribute)')
        return -1
    if mps_object.sim is None:
        print('No data to plot ("sim" attribute i "None")')
        return -1

    nr = mps_object.par['n_real']
    n_show = np.min((show, nr))

    nxy = np.ceil(np.sqrt(n_show)).astype('int')

    plotter = pyvista.Plotter(shape=(nxy, nxy))

    i = -1
    for ix in range(nxy):
        for iy in range(nxy):
            i = i + 1
            plotter.subplot(iy, ix)

            data = mps_object.sim[i]
            grid = pyvista.UniformGrid()
            grid.dimensions = np.array(data.shape) + 1

            grid.origin = mps_object.par['origin']
            grid.spacing = mps_object.par['grid_cell_size']
            # Flattens the array
            grid.cell_arrays['values'] = data.flatten(order='F')

            plotter.add_mesh(grid.slice_orthogonal())
            plotter.add_text('#%d' % (i + 1))
            plotter.show_grid()
    plotter.show()


def plot_3d(data, slice: int = 0,
            origin: Tuple[int, int, int] = (0, 0, 0),
            spacing: Tuple[int, int, int] = (1, 1, 1),
            threshold: Tuple = (), filename: str = "", header: str = ""):
    """
    Wrapper function for `plot_3d_reals_pyvista`.

    Parameters
    ----------
    data :
        Data of MPS object.
    slice : int
        Slice integer
    origin : Tuple[int, int, int]
        Origin tuple for plotting.
    spacing : Tuple[int, int, int]
        Spacing tuple.
    threshold : Tuple
        Threshold tuple
    filename : str
        Filename for plot
    header : str
        Header for plot

    See also
    -------
    plot_3d_pyvista
    """
    plot_3d_pyvista(data, slice, origin, spacing, threshold, filename, header)


def numpy_to_pvgrid(data, origin: Tuple[int, int, int] = (0, 0, 0),
                    spacing: Tuple[int, int, int] = (1, 1, 1)):
    """
    Convert 3D numpy array to pyvista uniform grid.

    Parameters
    ----------
    data :
        Data of MPS object.
    origin : Tuple[int, int, int]
        Origin tuple for plotting.
    spacing : Tuple[int, int, int]
        Spacing tuple.
    """
    import numpy as np

    if module_exists('pyvista', 1):
        import pyvista
    else:
        return 1
    # Create the spatial reference
    grid = pyvista.UniformGrid()
    # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
    grid.dimensions = np.array(data.shape) + 1
    # Edit the spatial reference
    grid.origin = origin  # The bottom left corner of the data set
    grid.spacing = spacing  # These are the cell sizes along each axis
    # Add the data values to the cell data
    grid.cell_arrays['values'] = data.flatten(order='F')  # Flatten the array!

    return grid


def plot_3d_pyvista(data, slice: int = 0,
                    threshold: Tuple = (), filename: str = "", header: str = ""):
    """
    Plot 3D cube using 'pyvista'

    Parameters
    ----------
    data :
        Data of MPS object.
    slice : Tuple[int, int, int]
        Origin tuple for plotting.
    threshold : Tuple[int, int, int]
        Spacing tuple.
    filename : str
        Filename for plot
    header : str
        Header for plot
    """

    if module_exists('pyvista', 1):
        import pyvista
    else:
        return 1
    print(filename)

    # create uniform grid
    grid = numpy_to_pvgrid(data, origin=(0, 0, 0), spacing=(1, 1, 1))

    # Now plot the grid!
    if len(threshold) == 2:
        plot = pyvista.BackgroundPlotter()
        grid_threshold = grid.threshold(threshold)
        plot.add_mesh(grid_threshold)
        if len(filename) > 0:
            plot.screenshot(filename)
        plot.show()

    elif slice == 1:
        plot = pyvista.BackgroundPlotter()

        grid_slice = grid.slice_orthogonal()
        plot.add_mesh(grid_slice)

        plot.add_text(header)
        plot.show_grid()
        if len(filename) > 0:
            plot.screenshot(filename)
        plot.show()

    else:
        grid.plot(show_edges=True)


def plot_3d_mpl(data):
    """
    Plot 3D numpy array with matplotlib

    Parameters
    ----------
    data :
        Data of MPS object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    print('USE THIS WITH CAUTION... ONLY SUITABLE FOR SMALL 3D MODELS.'
          ' Give preference to the pyvista interface instead')

    cat0 = data < .5
    cat1 = data >= .5

    # set the colors of each object
    colors = np.empty(data.shape, dtype=object)
    colors[cat0] = 'white'
    colors[cat1] = 'red'

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(data, facecolors=colors, edgecolor='k')
    plt.show()


def plot_3d_real(mps_object, ireal: int = 0,
                 slice: int = 0):
    """
    Plot 3D realizations using pyvista.

    Parameters
    ----------
    mps_object : MPS object
        MPS object with simulations.
    ireal: int
        Number of realizations
    slice: int
        Use 1 for slices and 0 for a 3D cube
    """
    plot_3d_pyvista(mps_object.sim[ireal], slice=slice,
                    origin=mps_object.par['origin'],
                    spacing=mps_object.par['grid_cell_size'])


def plot_eas(Deas):
    """
    Plot data directly form EAS
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import squeeze

    # check for matrix/cube
    if 'Dmat' in Deas:
        if Deas['dim']['nz'] == 1:
            if Deas['dim']['ny'] == 1:
                plt.plot(np.arange(Deas['dim']['nx']), Deas['Dmat'])
            elif Deas['dim']['nx'] == 1:
                plt.plot(np.arange(Deas['dim']['ny']), Deas['Dmat'])
            else:
                # X-Y
                plt.imshow(np.transpose(Deas['Dmat'][:, :, 0]))
                plt.xlabel('X')
                plt.ylabel('Y')
        elif Deas['dim']['ny'] == 1:
            if \
                    Deas['dim']['nz'] == 1:
                plt.plot(np.arange(Deas['dim']['nx']), Deas['Dmat'])
            elif Deas['dim']['nx'] == 1:
                plt.plot(np.arange(Deas['dim']['nz']), Deas['Dmat'])
            else:
                # X-Z
                plt.imshow(squeeze(Deas['Dmat'][:, 0, :]))
                plt.xlabel('X')
                plt.xlabel('Z')
        elif Deas['dim']['nx'] == 1:
            if Deas['dim']['ny'] == 1:
                plt.plot(np.arange(Deas['dim']['nz']), Deas['Dmat'])
            elif Deas['dim']['nz'] == 1:
                plt.plot(np.arange(Deas['dim']['ny']), Deas['Dmat'])
            else:
                # Y-Z
                plt.imshow(squeeze(Deas['Dmat'][0, :, :]))
                plt.xlabel('Y')
                plt.xlabel('Z')
        else:
            plot_3d_pyvista(Deas['Dmat'])
    else:
        # scatter plot
        print('EAS scatter plot not yet implemented')


def margin_1d(mps_object, plot=1, hardcopy=0, hardcopy_filename='marg1D'):
    """Plot 1D marginal probabilities from realizations and compares to the
    1D marginal from the training image

    Parameters
    ----------
    mps_object : mpslib object

    plot : int (def=0)
        plot the output


    Output
    ------
    mps_object.marg1D_sim : np.array [nr,ncat]
    mps_object.marg1D_ti : np.array [ncat]

    """
    import numpy as np
    import matplotlib.pyplot as plt

    # %%
    # D = np.array(mps_object.sim)
    # cat = np.sort(np.unique(mps_object.ti))
    D = np.array(mps_object.sim)
    cat = np.sort(np.unique(mps_object.ti))

    ncat = len(cat)
    dim = D.shape
    # dim_xyz = dim[1:4]
    nr = dim[0]
    # H = np.zeros(dim_xyz)
    # P = np.zeros((ncat,dim_xyz[0],dim_xyz[1],dim_xyz[2]))

    # %% 1D marginal stats
    margin_1d = []
    for ir in range(nr):
        c = np.zeros(ncat)
        for icat in range(ncat):
            c[icat] = np.count_nonzero(mps_object.sim[ir] == cat[icat])
            p = c / np.sum(c)
        margin_1d.append(p)
    # %%
    mps_object.marg1D_sim = np.array(margin_1d)
    u, c = np.unique(mps_object.ti, return_counts=True)
    p_ti = c / np.sum(c)
    mps_object.marg1D_ti = p_ti

    if plot:
        plt.figure(1)
        plt.clf()
        plt.hist(mps_object.marg1D_sim)
        plt.plot(mps_object.marg1D_ti, np.zeros(len(mps_object.marg1D_ti)), '*', markersize=50)
        plt.xlabel('1D marginal Probability of category form simulations and ti')
        if hardcopy:
            plt.savefig(hardcopy_filename)

        plt.figure(2)
        plt.clf()
        for icat in range(ncat):
            plt.plot(mps_object.marg1D_sim[:, icat], label='Cat=%d' % (cat[icat]))
        plt.legend()
        tmp = mps_object.marg1D_sim
        for icat in range(ncat):
            tmp[:, icat] = mps_object.marg1D_ti[icat]
        plt.plot(tmp, 'k-')
        plt.xlabel('Realization number')
        plt.ylabel('Prob (cat|realization)')
        plt.show()
        if hardcopy:
            plt.savefig(hardcopy_filename + '_2')

    return True

    # %% Probability map
    # for icat in range(ncat):
