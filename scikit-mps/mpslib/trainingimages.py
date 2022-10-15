import os
import numpy as np
from . import eas
from . import plot

try:
    from urllib.request import urlretrieve as urlretrieve
except ImportError:
    from urllib import urlretrieve as urlretrieve


def get_remote(url='https://www.trainingimages.org/uploads/3/4/7/0/34703305/ti_strebelle.sgems', local_file='ti.dat',
               is_zip=0, filename_in_zip=''):
    if is_zip == 1:
        local_file_zip = local_file + '.zip'

    if not (os.path.exists(local_file)):
        if is_zip == 1:
            import zipfile
            # Download zip file
            print(f'Beginning download of {url} to {local_file_zip}')
            urlretrieve(url, local_file_zip)

            # Unzip file
            print(f'Unzipping {local_file_zip} to {local_file}')
            zip_ref = zipfile.ZipFile(local_file_zip, 'r')
            zip_ref.extractall('.')
            zip_ref.close()
            # Rename unzipped file
            if len(filename_in_zip) > 0:
                os.rename(filename_in_zip, local_file)
        else:
            print(f'Beginning download of {url} to {local_file}')
            urlretrieve(url, local_file)
    return local_file


def coarsen_2d_ti(ti_2d, di=2):
    """
    Takes a 2D Training Image and makes it coarser, by constructing a 3D TI
    based on all coarsened 2D images
    """
    from scipy import squeeze

    nx, ny, nz = ti_2d.shape
    ndim3 = di * di
    x = np.arange(nx)
    y = np.arange(ny)
    ix = x[0:(nx - di):di]
    iy = y[0:(ny - di):di]

    # Filling empty array
    ti = np.zeros((len(ix), len(iy), nz, ndim3))
    current_col = -1
    for j in range(di):
        for k in range(di):
            current_col = current_col + 1
            ti_small = ti_2d[(0 + j)::di, (0 + k)::di, 0]
            ti[:, :, 0, current_col] = ti_small[0:len(ix), 0:len(iy)]
    ti = squeeze(ti)
    return ti


def ti_list(show=1):
    ti_name = []
    ti_desc = []

    ti_name.append('checkerboard')
    ti_desc.append('2D checkerboard')

    ti_name.append('checkerboard2')
    ti_desc.append('2D checkerboard - alternative')

    ti_name.append('strebelle')
    ti_desc.append('2D discrete channels from Strebelle')

    ti_name.append('lines')
    ti_desc.append('2D discrete lines')

    ti_name.append('stones')
    ti_desc.append('2D continuous stones')

    ti_name.append('bangladesh')
    ti_desc.append('2D discrete Bangladesh')

    ti_name.append('maze')
    ti_desc.append('2D discrete maze')

    ti_name.append('rot90')
    ti_desc.append('3D rotation 90')

    ti_name.append('rot20')
    ti_desc.append('3D rotation 20')

    ti_name.append('horizons')
    ti_desc.append('3D continuous horizons')

    ti_name.append('fluvsim')
    ti_desc.append('3D discrete fluvsim')

    if show == 1:
        print('Available training images:')
        for i in range(len(ti_name)):
            print('%15s - %s' % (ti_name[i], ti_desc[i]))
    return ti_name, ti_desc


def ti_plot_all():
    """
    Plot all training images
    """

    import sys
    this_mod = sys.modules[__name__]

    ti_fnames, d = ti_list(1)

    for i in range(len(ti_fnames)):
        print('Loading %s' % ti_fnames[i])
        ti, ti_fname = getattr(this_mod, ti_fnames[i])()
        print(ti.shape)
        plot.plot_3d(ti, 1)


"""
Getting Training Images from the GAIA repository
"""


def fluvsim():
    local_file = 'ti_fluvsim.dat'
    filename_in_zip = 'ti_fluvsim_big_channels3D.SGEMS'
    url = 'https://github.com/GAIA-UNIL/trainingimages/raw/master/MPS_book_data/Part2/ti_fluvsim_big_channels3D.zip'
    is_zip = 1
    local_file = get_remote(url, local_file, is_zip=is_zip, filename_in_zip=filename_in_zip)
    deas = eas.read(local_file)
    ti = deas['Dmat']
    return ti, local_file


def horizons():
    local_file = 'ti_horizons.dat'
    url = 'https://github.com/GAIA-UNIL/trainingimages/raw/master/MPS_book_data/Part2/TI_horizons_continuous.SGEMS'
    filename_in_zip = 'TI_horizons_continuous.SGEMS'
    is_zip = 0
    local_file = get_remote(url, local_file, is_zip=is_zip, filename_in_zip=filename_in_zip)
    deas = eas.read(local_file)
    ti = deas['Dmat']
    return ti, local_file


def rot90():
    local_file = 'ti_tot90.dat'
    url = 'https://github.com/GAIA-UNIL/trainingimages/raw/master/MPS_book_data/Part2/checker_rtoinvariant_90.zip'
    filename_in_zip = 'checker_rtoinvariant_90.SGEMS'
    is_zip = 1
    local_file = get_remote(url, local_file, is_zip=is_zip, filename_in_zip=filename_in_zip)
    deas = eas.read(local_file)
    ti = deas['Dmat']
    return ti, local_file


def rot20():
    local_file = 'ti_rot20.dat'
    url = 'https://github.com/GAIA-UNIL/trainingimages/blob/master/MPS_book_data/Part2/checker_rtoinvariant_20.zip'
    filename_in_zip = 'checker_rtoinvariant_20.SGEMS'
    is_zip = 1
    local_file = get_remote(url, local_file, is_zip=is_zip, filename_in_zip=filename_in_zip)
    deas = eas.read(local_file)
    ti = deas['Dmat']

    return ti, local_file


def strebelle(di=1, coarse3d=0):
    url = 'https://github.com/GAIA-UNIL/trainingimages/raw/master/MPS_book_data/Part2/ti_strebelle.sgems'
    local_file = get_remote(url, 'ti_strebelle.dat')
    deas = eas.read(local_file)
    local_file = 'ti_strebelle_%d.dat' % di

    ti = deas['Dmat']
    if di > 1:
        if coarse3d == 0:
            dmat = ti
            ti = dmat[::di, ::di, :]
        else:
            dmat = ti
            ti = coarsen_2d_ti(dmat, di)
    eas.write_mat(ti, local_file)
    return ti, local_file


def lines(di=1, coarse3d=0):
    local_file = 'ti_lines.dat'
    url = 'https://github.com/GAIA-UNIL/trainingimages/raw/master/MPS_book_data/Part2/ti_lines_arrows.sgems'
    get_remote(url, local_file)
    deas = eas.read(local_file)
    ti = deas['Dmat']

    if di > 1:
        local_file = "ti_lines_%d.dat" % di
        if coarse3d == 0:
            dmat = ti
            ti = dmat[::di, ::di]
        else:
            dmat = ti
            ti = coarsen_2d_ti(dmat, di)

    return ti, local_file


def stones():
    local_file = 'ti_stones.dat'
    url = 'https://github.com/GAIA-UNIL/trainingimages/raw/master/MPS_book_data/Part2/ti_stonewall.sgems'
    get_remote(url, local_file)
    deas = eas.read(local_file)
    ti = deas['Dmat']
    return ti, local_file


def bangladesh(di=1, coarse3d=0):
    local_file = 'ti_bangladesh.dat'
    url = 'https://github.com/GAIA-UNIL/trainingimages/raw/master/MPS_book_data/Part2/bangladesh.sgems'
    get_remote(url, local_file)
    deas = eas.read(local_file)
    ti = deas['Dmat']

    if di > 1:
        if coarse3d == 0:
            dmat = ti
            ti = dmat[::di, ::di, ::di]
        else:
            dmat = ti
            ti = coarsen_2d_ti(dmat, di)

    return ti, local_file


def maze():
    local_file = 'ti_maze.dat'
    url = 'https://raw.githubusercontent.com/cultpenguin/mGstat/master/ti/maze.gslib'
    get_remote(url, local_file)
    deas = eas.read(local_file)
    ti = deas['Dmat']
    return ti, local_file


def checkerboard(nx=40, ny=40, cellsize=4):
    import numpy as np
    ti = np.kron([[1, 0] * cellsize, [0, 1] * cellsize] * cellsize, np.ones((nx, ny)))
    ti = ti[:, :, np.newaxis]

    local_file = 'ti_checkerboard.dat'

    eas.write_mat(ti, local_file)

    deas = eas.read(local_file)
    ti = deas['Dmat']

    return ti, local_file


def checkerboard2(nx=40, ny=50, cell_x=8, cell_y=4, cell_2=10):
    ti = np.zeros((ny, nx))

    for ix in range(nx):  # Note that i ranges from 0 through 7, inclusive.
        for iy in range(ny):  # So does j.
            if (ix % cell_x < (cell_x / 2)) == (iy % cell_y < (cell_y / 2)):  # The checkerboard
                ti[iy, ix] = 0
            else:
                ti[iy, ix] = 1

            if (ix % (cell_2 + 1) < (cell_2 / 2)) & (
                    iy % (cell_2 + 1) < (cell_2 / 2)):  # some more 'checks' a little off
                ti[iy, ix] = 2

            if (ix + iy) % cell_x == 0:
                ti[iy, ix] = 0

    local_file = 'ti_checkerboard2_%d_%d__%d_%d__%d.dat' % (nx, ny, cell_x, cell_y, cell_2)  # a diagonal

    eas.write_mat(ti, local_file)

    deas = eas.read(local_file)
    ti = deas['Dmat']
    return ti, local_file
