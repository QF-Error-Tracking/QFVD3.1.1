# -*- coding: utf-8 -*-
# Version date: March 13, 2019
# @author: Sara Brambilla

import math
import numpy as np
import pylab
import shutil
import os
import sys
import copy
import inspect
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK

# wait = raw_input("PRESS ENTER TO CONTINUE.")


class FbClass:
    def __init__(self):
        self.i = None
        self.j = None
        self.k = None
        self.state = None
        self.time = None


class ImgClass:
    def __init__(self):
        self.figure_size = None
        self.axis_font = None
        self.title_font = None
        self.colorbar_font = None


class LinIndexClass:
    def __init__(self):
        self.ijk = None
        self.num_cells = None


class FlagsClass:
    def __init__(self):
        self.firebrands = None
        self.en2atm = None
        self.perc_mass_burnt = None
        self.fuel_density = None
        self.emissions = None
        self.thermal_rad = None
        self.qf_winds = None
        self.qu_qwinds_inst = None
        self.qu_qwinds_ave = None
        self.react_rate = None
        self.moisture = None


class IgnitionClass:
    def __init__(self):
        self.hor_plane = None
        self.flag = None


class SimField:
    def __init__(self):
        self.isfire = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.z = None
        self.zm = None
        self.sim_time = None
        self.ntimes = None
        self.time = None
        self.ntimes_ave = None
        self.time_ave = None
        self.dt = None
        self.dt_print = None
        self.dt_print_ave = None
        self.horizontal_extent = None
        self.indexing = LinIndexClass()


def split_string(s, datatype):
    # http://stackoverflow.com/questions/4289331/python-extract-numbers-from-a-string
    s = s.strip()
    out = []
    for t in s.split():
        try:
            if datatype == 1:
                out.append(int(t))
            else:
                out.append(float(t))
        except ValueError:
            pass    
    return out[0]


def get_line(fid, datatype):
    return split_string(fid.readline(), datatype)


def generate_jet_colorbar(m):
    n = int(math.ceil(m / 4.))
    u = np.concatenate((np.arange(1, n + 1) / n, np.ones(n - 1), np.arange(n, 0, -1) / n))
    g = math.ceil(n * 0.5) - ((m % 4.) == 1) + np.arange(1, len(u) + 1)
    r = g + n
    b = g - n

    iremove = np.where(g > m)
    iremove = iremove[0]
    g = np.delete(g, iremove, None)
    g = g.astype(int) - 1

    iremove = np.where(r > m)
    iremove = iremove[0]
    r = np.delete(r, iremove, None)
    r = r.astype(int) - 1

    iremove = np.where(b < 1)
    iremove = iremove[0]
    b = np.delete(b, iremove, None)
    b = b.astype(int) - 1

    j = np.zeros((int(m), 3))
    for i in range(0, len(r)):
        j[r[i], 0] = u[i]
    for i in range(0, len(g)):
        j[g[i], 1] = u[i]
    for i in range(0, len(b)):
        j[b[i], 2] = u[len(u) - len(b) + i]

    return j


def plot_percmassburnt(qf, myextent, plotvar, fuel_dens, ystr, savestr, save_dir, img_specs):
    myvmin = -100./64. - 1e-6
    myvmax = 100

    var_dim = plotvar[0].squeeze().shape
    currval0 = np.zeros((var_dim[0], var_dim[1]))
    f0 = np.sum(fuel_dens, axis=2)
    k0 = np.where(f0 == 0)
    if len(k0) > 0:
        if0 = k0[1]
        jf0 = k0[0]
        currval0[[jf0, if0]] = myvmin

    my_cmap = generate_jet_colorbar(65)
    my_cmap[0][::1] = 1.
    my_cmap = pylab.matplotlib.colors.ListedColormap(my_cmap, 'my_colormap', N=None)

    plane_str = ''
    for i in range(0, qf.ntimes):
        print("     * time %d/%d" % (i + 1, qf.ntimes))

        currval = copy.deepcopy(currval0)
        loc_var = plotvar[i].squeeze()
        k_loc = np.where(loc_var > 0.)
        if len(k_loc) > 0:
            i_loc = k_loc[1]
            j_loc = k_loc[0]
            currval[[j_loc, i_loc]] = loc_var[[j_loc, i_loc]]

        fig = pylab.figure(figsize=(img_specs.figure_size[0], img_specs.figure_size[1]))
        ax = fig.add_subplot(111)

        pylab.imshow(currval, cmap=my_cmap, interpolation='none', origin='lower',
                     extent=myextent, vmin=myvmin, vmax=myvmax)
        cbar = pylab.colorbar()
        cbar.set_label(ystr, size=img_specs.axis_font["size"], fontname=img_specs.axis_font["fontname"])
        cbar.ax.tick_params(labelsize=img_specs.colorbar_font["size"])
        pylab.xlabel('X [m]', **img_specs.axis_font)
        pylab.ylabel('Y [m]', **img_specs.axis_font)
        pylab.title('Time = %s s' % qf.time[i], **img_specs.title_font)
        set_ticks_font(img_specs.axis_font, ax)
        time_str = '_Time_%d_s' % qf.time[i]
        pylab.savefig(save_dir + os.sep + savestr + time_str + plane_str + '.png')
        pylab.close()
        del currval
        del loc_var


def plot_2d_field(is_ave_time, q, plane, plotvar, ystr,
                  savestr, cblim, save_dir, img_specs):

    if is_ave_time is True:
        ntimes = q.ntimes_ave
        times = q.time_ave
    else:
        ntimes = q.ntimes
        times = q.time

    if not cblim:
        myvmin = 1e8
        myvmax = -1e8
        for i in range(0, ntimes):
            if plane:
                currval = plotvar[i][::1, ::1, plane - 1]
            else:
                currval = plotvar[i]

            myvmin = min(myvmin, np.amin(currval))
            myvmax = max(myvmax, np.amax(currval))

        myvmin = math.floor(myvmin)
        myvmax = math.ceil(myvmax)
        if myvmin == myvmax:
            if myvmin == 0:
                myvmin = -1
                myvmax = +1
            else:
                myvmin *= 0.5
                myvmax *= 2.
    else:
        myvmin = cblim[0]
        myvmax = cblim[1]

    for i in range(0, ntimes):
        print("     * time %d/%d" % (i + 1, ntimes))

        if plane!=0:
            currval = plotvar[i][::1, ::1, plane - 1]
            plane_str = '_Plane_%d' % plane
        else:
            currval = plotvar[i]
            plane_str = ''

        currval = currval.squeeze()

        fig = pylab.figure(figsize=(img_specs.figure_size[0], img_specs.figure_size[1]))
        ax = fig.add_subplot(111)

        pylab.imshow(currval, cmap='jet', interpolation='none', origin='lower',
                     extent=q.horizontal_extent, vmin=myvmin, vmax=myvmax)
        cbar = pylab.colorbar()
        cbar.set_label(ystr, size=img_specs.axis_font["size"], fontname=img_specs.axis_font["fontname"])
        cbar.ax.tick_params(labelsize=img_specs.colorbar_font["size"])
        pylab.xlabel('X [m]', **img_specs.axis_font)
        pylab.ylabel('Y [m]', **img_specs.axis_font)
        pylab.title('Time = %s s' % times[i], **img_specs.title_font)
        time_str = '_Time_%d_s' % times[i]
        set_ticks_font(img_specs.axis_font, ax)

        pylab.savefig(save_dir + os.sep + savestr + time_str + plane_str + '.png')
        pylab.close()


def set_ticks_font(axis_font, ax):
    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(axis_font["fontname"])
        label.set_fontsize(axis_font["size"])


def set_image_specifications(img_specs):
    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    img_specs.figure_size = np.array([12, 8.5])
    fs = 22
    fs_str = str(fs)
    img_specs.axis_font = {'fontname': 'Arial', 'size': fs_str}
    fs_str = str(fs + 2)
    img_specs.title_font = {'fontname': 'Arial', 'size': fs_str, 'fontweight': 'bold'}
    fs_str = str(fs - 2)
    img_specs.colorbar_font = {'fontname': 'Arial', 'size': fs_str}


def create_plots_folder():
    save_dir = "Plots"
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True, onerror=None)
    os.mkdir(save_dir)
    return save_dir


def plot_outputs(qu, qf, ignitions, flags, fb):
    print("Plotting output files")

    # Setting image specs
    img_specs = ImgClass()
    set_image_specifications(img_specs)

    # Create folder to save images
    save_dir = create_plots_folder()
    out_dir = "Output/"

    if flags.fuel_density == 1:
        print("  - fuel density field")
        temp_ntimes = qf.ntimes
        temp_time = qf.time
    else:
        temp_ntimes = 1
        temp_time = [qf.time[0]]

    if flags.fuel_density == 1:
        fuel_dens = read_fireca_field("fuels-dens-", temp_ntimes, temp_time, qf, 0)

        # ------- Firetech ignitions
        print("  - initial ignitions")
        plot_ignitions(fuel_dens[0], ignitions.hor_plane, qf.horizontal_extent, save_dir, img_specs)

    #plot topograpy
    print("  - topo")
    plot_topo("h", qf.horizontal_extent, save_dir, img_specs,qu,qf)
    # ------- Firebrands
    # if flags.firebrands == 1:
    #     print("  - firebrands")
    #     plot_firebrands(fuel_dens, ignitions.hor_plane, qf, fb,
    #                     save_dir, img_specs)

    # ------- % mass burnt (vertically-integrated)
    if flags.perc_mass_burnt == 1:
        print("  - % mass burnt")
        perc_mass_burnt = read_fireca_field("mburnt_integ-", qf.ntimes, qf.time, qf, 1)
        plot_percmassburnt(qf, qf.horizontal_extent, perc_mass_burnt, fuel_dens[0],
                           "Mass burnt (vertically-integ.) [%]", "perc_mass_burnt", save_dir, img_specs)
        del perc_mass_burnt

    plane = 1
    # ------- Fuel mass
    if flags.fuel_density == 1:
        print("  - fuel mass")
        fuel_dens = read_fireca_field("fuels-dens-", qf.ntimes, qf.time, qf, 0)
        plot_2d_field(False, qf, plane, fuel_dens, "Fuel density [kg/m^3]", "fuel_dens",
                      [0., np.amax(fuel_dens[0][::1, ::1, plane-1], axis=None)], save_dir, img_specs)
        del fuel_dens

    # ------- Emissions
    if flags.emissions == 1:
        print("  - pm emissions")
        emiss = read_fireca_field("pm_emissions-", qf.ntimes_ave, qf.time_ave, qf, 0)
        minval = +1e8
        maxval = -1.
        for it in range(0, qf.ntimes_ave):
            for k in range(0, qf. nz):
                for i in range(0, qf.nx):
                    for j in range(0, qf.ny):
                        if emiss[it][i, j, k] > 0:
                            emiss[it][i, j, k] = math.log10(emiss[it][i, j, k])
                            minval = min(minval, emiss[it][i, j, k])
                            maxval = max(maxval, emiss[it][i, j, k])
        minval = math.floor(minval)
        maxval = math.ceil(maxval)
        plot_2d_field(True, qf, plane, emiss, "Soot (log10) [g]", "pm_emissions_",
                      [minval, maxval], save_dir, img_specs)
        del emiss

        print("  - co emissions")
        emiss = read_fireca_field("co_emissions-", qf.ntimes_ave, qf.time_ave, qf, 0)
        minval = +1e8
        maxval = -1.
        for it in range(0, qf.ntimes_ave):
            for k in range(0, qf. nz):
                for i in range(0, qf.nx):
                    for j in range(0, qf.ny):
                        if emiss[it][i, j, k] > 0:
                            emiss[it][i, j, k] = math.log10(emiss[it][i, j, k])
                            minval = min(minval, emiss[it][i, j, k])
                            maxval = max(maxval, emiss[it][i, j, k])
        minval = math.floor(minval)
        maxval = math.ceil(maxval)
        plot_2d_field(True, qf, plane, emiss, "CO (log10) [g]", "co_emissions_",
                      [minval, maxval], save_dir, img_specs)
        del emiss
        
    # ------- Radiation
    if flags.thermal_rad == 1:
        print("  - radiation")
        conv = read_fireca_field("thermalradiation-", qf.ntimes_ave, qf.time_ave, qf, 0)
        plot_2d_field(True, qf, plane, conv, "Convective heat to human [kW/m^2 skin]", "conv_heat_",
                      [], save_dir, img_specs)
        del conv

    # ------- Energy to atmosphere
    if flags.en2atm == 1:
        print("  - energy to atmosphere")
        en_to_atm = read_fireca_field("fire-energy_to_atmos-", qf.ntimes, qf.time, qf, 1)
        plot_2d_field(False, qf, plane, en_to_atm, "Energy to atmosphere [kW/m^3]", "en_to_atm",
                      [], save_dir, img_specs)
        del en_to_atm

    # -------  Fireca winds
    if flags.qf_winds == 1:
        plane=1
        print("  - fireca winds")
        print("    * u")
        windu_qf = read_fireca_field("windu", qf.ntimes, qf.time, qf, 1)
        plot_2d_field(False, qf, plane, windu_qf, "U [m/s]", "u", [], save_dir, img_specs)

        print("    * v")
        windv_qf = read_fireca_field("windv", qf.ntimes, qf.time, qf, 1)
        plot_2d_field(False, qf, plane, windv_qf, "V [m/s]", "v", [], save_dir, img_specs)
        del windv_qf

        print("    * w")
        windw_qf = read_fireca_field("windw", qf.ntimes, qf.time, qf, 1)
        plot_2d_field(False, qf, plane, windw_qf, "W [m/s]", "w", [], save_dir, img_specs)

        # print("    * sigma")
        # wsgma = read_fireca_field("wsgma", qf.ntimes, qf.time, qf, 0)
        # plot_2d_field(False, qf, plane, wsgma, "wsgma [??]", "wsgma", [], save_dir, img_specs)
        # del wsgma

    # -------  QU winds (instantaneous)
    if flags.qu_qwinds_inst == 1:
        if(qu.isfire ==1):
            plane = 1
            print("  - QU winds")
            print("    * u")
            print(qu.ntimes)
            print(qu.time)
            windu_qu = read_fireca_field("qu_windu", qu.ntimes, qu.time, qu, 1)
            plot_2d_field(False, qu, plane, windu_qu, "U_inst [m/s]", "u_qu", [], save_dir, img_specs)
            del windu_qu

            print("    * v")
            windv_qu = read_fireca_field("qu_windv", qu.ntimes, qu.time, qu, 1)
            plot_2d_field(False, qu, plane, windv_qu, "V_inst [vm/s]", "v_qu", [], save_dir, img_specs)
            del windv_qu

            print("    * w")
            windw_qu = read_fireca_field("qu_windw", qu.ntimes, qu.time, qu, 1)
            plot_2d_field(False, qu, plane, windw_qu, "W_inst [m/s]", "w_qu", [], save_dir, img_specs)
            del windw_qu
        else:
            plane = 1
            print("  - QU winds")
            print("    * u")
            #qu.ntimes= qu.ntimes-1
            windu_qu = read_fireca_field("qu_windu", qu.ntimes, qu.time, qu, 1)
            plot_2d_field(False, qu, plane, windu_qu, "U_inst [m/s]", "u_qu", [], save_dir, img_specs)
            #del windu_qu

            print("    * v")
            windv_qu = read_fireca_field("qu_windv", qu.ntimes, qu.time, qu, 1)
            plot_2d_field(False, qu, plane, windv_qu, "V_inst [vm/s]", "v_qu", [], save_dir, img_specs)
            #del windv_qu

            print("    * w")
            windw_qu = read_fireca_field("qu_windw", qu.ntimes, qu.time, qu, 1)
            plot_2d_field(False, qu, plane, windw_qu, "W_inst [m/s]", "w_qu", [], save_dir, img_specs)
            #del windw_qu


    # -------  QU winds (average)
    if flags.qu_qwinds_ave == 1:
        plane = 2
        print("  - QU winds")
        print("    * u")
        windu_qu = read_fireca_field("qu_windu_ave", qu.ntimes_ave, qu.time_ave, qu, 1)
        plot_2d_field(True, qu, plane, windu_qu, "U_ave [m/s]", "u_qu_ave", [], save_dir, img_specs)

        print("    * v")
        windv_qu = read_fireca_field("qu_windv_ave", qu.ntimes_ave, qu.time_ave, qu, 1)
        plot_2d_field(True, qu, plane, windv_qu, "V_ave [m/s]", "v_qu_ave", [], save_dir, img_specs)
        del windv_qu

        print("    * w")
        windw_qu = read_fireca_field("qu_windw_ave", qu.ntimes_ave, qu.time_ave, qu, 1)
        plot_2d_field(True, qu, plane, windw_qu, "W_ave [m/s]", "w_qu_ave", [], save_dir, img_specs)

    # ------- Reaction rate
    plane = 1
    if flags.react_rate == 1:
        print("  - reaction rate")
        react_rate = read_fireca_field("fire-reaction_rate-", qf.ntimes, qf.time, qf, 0)
        plot_2d_field(False, qf, plane, react_rate,
                      "Reaction rate [??]", "react_rate", [], save_dir, img_specs)
        del react_rate

    # ------- Fuel moisture
    if flags.moisture == 1:
        print("  - moisture")
        fuels_moist = read_fireca_field("fuels-moist-", qf.ntimes, qf.time, qf, 0)
        plot_2d_field(False, qf, plane, fuels_moist,
                      "Fuel moisture [-]", "fuels_moist", [], save_dir, img_specs)
        del fuels_moist


def plot_firebrands(fuel_dens, ignitions, qf, fb, save_dir, img_specs):
    # Define colormap
    # http://matplotlib.org/api/colors_api.html
    mycol = [
        [1., 1., 1.],  # white = no fuel
        [0.765, 0.765, 0.765],  # gray = fuel
        [0.6, 0.8, 1.],  # light blue = initial ignitions
        [0., 1., 0.],  # green = fb on cells already on fire
        [1., 0., 0.],  # red = fb on cells not on fire
        [0., 0., 0.]]  # black = fb on cells without fuel
    my_cmap = pylab.matplotlib.colors.ListedColormap(mycol, 'my_colormap', N=None)

    f0 = np.sum(fuel_dens[0], axis=2)
    currval0 = f0
    currval0[np.where(currval0 > 0)] = 1

    currval0[np.where(ignitions > 0)] = 2
    currval = np.zeros((qf.ny, qf.nx))
    for j in range(1, 6):
        nel = np.where(currval0 == j)
        nel = nel[0]

    for it in range(0, qf.ntimes - 1):
        print("     * time %d/%d" % (it + 1, qf.ntimes))

        np.copyto(currval, currval0)
                
        # Firebrands launched during this time perios
        kt = np.where(np.logical_and(fb.time > qf.time[it], fb.time <= qf.time[it + 1]))
        kt = kt[0]

        ii = fb.i[kt]
        jj = fb.j[kt]
        kk = fb.k[kt]

        fb.state = fb.state[kt]

        m = ii * qf.ny + jj
        mu, index = np.unique(m, return_index=True)

        nnew = 0

        for i in range(0, len(mu)):
            j = np.where(m == mu[i])
            j = j[0]

            im = ii[j]
            jm = jj[j]
            km = kk[j]

            if len(fuel_dens) == 1:
                itm = 0
            else:
                itm = it + 1

            elem1 = np.where((fb.state[j] == 0) & (fuel_dens[itm][[jm, im, km]] > 0))
            elem2 = np.where((fb.state[j] == 1) | (fb.state[j] == 2))

            if len(elem1[0]) > 0:
                val_assign = 4
                nnew += 1
            elif len(elem2[0]) > 0:
                val_assign = 3
            else:
                val_assign = 5

            currval[jm[0], im[0]] = val_assign

        fig = pylab.figure(figsize=(img_specs.figure_size[0], img_specs.figure_size[1]))
        ax = fig.add_subplot(111)

        pylab.imshow(currval, cmap=my_cmap, interpolation='none', origin='lower',
                     extent=qf.horizontal_extent, vmin=-0.5, vmax=5.5)
        cbar = pylab.colorbar(ticks=[0, 1, 2, 3, 4, 5])
        cbar.ax.set_yticklabels(['No fuel', 'Fuel', 'Init. ign.', 'FB exist. fire', 'FB no fire', 'FB no fuel'])
        cbar.ax.tick_params(labelsize=img_specs.colorbar_font["size"])
        pylab.xlabel('X [m]', **img_specs.axis_font)
        pylab.ylabel('Y [m]', **img_specs.axis_font)
        pylab.title('Time = %s-%s s - # new fires = %d' % (qf.time[it], qf.time[it+1], nnew), **img_specs.title_font)
        time_str = '_Time_%d_s' % (qf.time[it+1])
        set_ticks_font(img_specs.axis_font, ax)
        pylab.savefig(save_dir + os.sep + 'FirebrandsOnFuel' + time_str + '.png')
        # pylab.show()
        pylab.close()
        

def plot_ignitions(fuel_dens0, ignitions, myextent, save_dir, img_specs):
    # Define colormap
    # http://matplotlib.org/api/colors_api.html
    mycol = [[1., 1., 1.], [0.765, 0.765, 0.765], [1., 0., 0.]]
    my_cmap = pylab.matplotlib.colors.ListedColormap(mycol, 'my_colormap', N=None)

    currval = np.sum(fuel_dens0, axis=2)
    currval[np.where(currval > 0)] = 1
    inds = np.where(ignitions > 0)
    currval[inds] = 2

    fig = pylab.figure(figsize=(img_specs.figure_size[0], img_specs.figure_size[1]))
    ax = fig.add_subplot(111)

    pylab.imshow(currval, cmap=my_cmap, interpolation='none', origin='lower',
                 extent=myextent, vmin=-0.5, vmax=2.5)
    cbar = pylab.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['No fuel', 'Fuel', 'Ignitions'])
    cbar.ax.tick_params(labelsize=img_specs.colorbar_font["size"])

    pylab.xlabel('X [m]', **img_specs.axis_font)
    pylab.ylabel('Y [m]', **img_specs.axis_font)
    pylab.title('Selected ignitions # %d' % len(inds[0]), **img_specs.title_font)

    set_ticks_font(img_specs.axis_font, ax)

    pylab.savefig(save_dir + os.sep + 'InitialIgnitions.png')
    pylab.close()

def plot_topo(filestr, myextent, save_dir, img_specs,qu,qf):

    fname = 'Output/'+filestr + ".bin"
    fid = open_file(fname, 'rb')
    var = np.fromfile(fid, dtype=np.float32, count=(qu.nx+2)*(qu.ny+2))
    currval = np.reshape(var,(qu.ny+2,qu.nx+2))
    myvmin = 1e8
    myvmax = -1e8
    myvmin = min(myvmin, np.amin(currval))
    myvmax = max(myvmax, np.amax(currval))
    myvmin = math.floor(myvmin)
    myvmax = math.ceil(myvmax)
    if myvmin == myvmax:
        if myvmin == 0:
            myvmin = -1
            myvmax = +1
        else:
            myvmin *= 0.5
            myvmax *= 2.
    fig = pylab.figure(figsize=(img_specs.figure_size[0], img_specs.figure_size[1]))
    ax = fig.add_subplot(111)

    pylab.imshow(currval,cmap="jet", interpolation='none', origin='lower',
                 extent=myextent, vmin=myvmin, vmax=myvmax)

    pylab.colorbar()
    pylab.xlabel('X [m]', **img_specs.axis_font)
    pylab.ylabel('Y [m]', **img_specs.axis_font)
    pylab.title('Topography', **img_specs.title_font)

    set_ticks_font(img_specs.axis_font, ax)

    pylab.savefig(save_dir + os.sep + 'topography.png')
    pylab.close()
    del var, currval

def read_fireca_field(filestr, ntimes, times, qf, is_3d, *args, **kwargs):
    outvar = []
    if (filestr == "mburnt_integ-" or filestr =="h"):
        nvert = 1
    else:
        c = kwargs.get('c', None)
        layers = kwargs.get('layers', None)
        if(layers == None):
            nvert = qf.nz
        else:
            nvert = layers

    for i in range(0, ntimes):
        fname = 'Output/'+filestr + '%05d.bin' % (times[i])
        # Open file
        fid = open_file(fname, 'rb')
        temp = np.zeros((qf.ny, qf.nx, nvert))
        # Read header
        np.fromfile(fid, dtype=np.float32, count=1)
        if is_3d == 0:
            var = np.fromfile(fid, dtype=np.float32, count=qf.indexing.num_cells)
            # http://scipy-cookbook.readthedocs.io/items/Indexing.html
            index = [qf.indexing.ijk[::1, 1], qf.indexing.ijk[::1, 0], qf.indexing.ijk[::1, 2]]
            temp[index] = var
        else:
            temp = np.zeros((qf.ny, qf.nx, nvert))
            for k in range(0, nvert):
                t = np.fromfile(fid, dtype=np.float32, count=qf.nx * qf.ny)
                temp[::1, ::1, k] = np.reshape(t, (qf.ny, qf.nx))
        outvar.append(temp)
        fid.close()

    return outvar


def open_file(filename, howto):
    try:
        fid = open(filename, howto)
        return fid
    except IOError:
        print("Error while opening " + filename)
        input("PRESS ENTER TO CONTINUE.")
        sys.exit()


def read_vertical_grid(qu):
    out_dir = 'Output/'
    fid = open_file(out_dir+'z_qu.bin', 'rb')

    # Header
    np.fromfile(fid, dtype=np.int32, count=1)
    # Read z
    qu.z = np.fromfile(fid, dtype=np.float32, count=qu.nz + 2)

    # Header
    np.fromfile(fid, dtype=np.int32, count=2)

    # Read zm
    qu.zm = np.fromfile(fid, dtype=np.float32, count=qu.nz + 2)

    fid.close()


def read_qu_grid(qu):

    # ------- QU_simparams
    fid = open_file('QU_simparams.inp', 'r')
    fid.readline()  # header
    qu.nx = get_line(fid, 1)
    qu.ny = get_line(fid, 1)
    qu.nz = get_line(fid, 1)
    qu.dx = get_line(fid, 2)
    qu.dy = get_line(fid, 2)
    grid_flag = get_line(fid, 1)
    if grid_flag == 0:
        temp = get_line(fid, 2)
        qu.dz = np.ones(qu.nz) * temp
    else:
        fid.readline()
        fid.readline()
        fid.readline()
        qu.dz = []
        for i in range(0, qu.nz):
            qu.dz.append(get_line(fid, 2))
    qu.ntimes = get_line(fid,1)
    qu.time = []
    for i in range(qu.ntimes):
        qu.time.append(i) 
    fid.close()

    read_vertical_grid(qu)

    qu.horizontal_extent = [0., qu.dx * float(qu.nx), 0., qu.dy * float(qu.ny)]


def read_times(fid, qu, qf):
    # Fire times
    qu.isfire = int(get_line(fid, 1))
    for i in range(0, 3):
        fid.readline()

    # Simulation time
    qu.sim_time = int(get_line(fid, 1))
    qf.sim_time = qu.sim_time

    # Fire time step
    qf.dt = int(get_line(fid, 1))

    # QU time step
    qu.dt = int(get_line(fid, 1) * qf.dt)

    # Print time for FireCA variables
    qf.dt_print = int(get_line(fid, 1) * qf.dt)

    # Print time for QUIC variables
    qu.dt_print = int(get_line(fid, 1) * qu.dt)

    # Print time for emission and rad variables
    qf.dt_print_ave = int(get_line(fid, 1) * qf.dt)

    # Print time for averaged QUIC variables
    qu.dt_print_ave = int(get_line(fid, 1) * qu.dt)

    qf.ntimes = int(qf.sim_time / qf.dt_print + 1)
    qf.time = np.zeros((qf.ntimes,), dtype=np.int)

    if(qu.isfire == 1):
        qu.ntimes = int(qu.sim_time / qu.dt_print + 1)
        qu.time = np.zeros((qu.ntimes,), dtype=np.int)
        for i in range(0, qu.ntimes):
            qu.time[i] = qu.dt_print * i
        qu.ntimes_ave = int(qu.sim_time / qu.dt_print_ave + 1)
        qu.time_ave = np.zeros((qu.ntimes_ave,), dtype=np.int)
        for i in range(0, qu.ntimes_ave):
            qu.time_ave[i] = qu.dt_print_ave * i

    for i in range(0, qf.ntimes):
        qf.time[i] = qf.dt_print * i


    qf.ntimes_ave = int(qf.sim_time / qf.dt_print_ave + 1)
    qf.time_ave = np.zeros((qf.ntimes_ave,), dtype=np.int)


    for i in range(0, qf.ntimes_ave):
        qf.time_ave[i] = qf.dt_print_ave * i



def read_fire_grid(fid, qu, qf):
    fid.readline()  # ! FIRE GRID
    qf.nz = get_line(fid, 1)
    ratiox = get_line(fid, 1)
    ratioy = get_line(fid, 1)
    qf.nx = qu.nx * ratiox
    qf.ny = qu.ny * ratioy
    qf.dx = qu.dx / float(ratiox)
    qf.dy = qu.dy / float(ratioy)
    dz_flag = get_line(fid, 1)
    if dz_flag == 0:
        dztemp = get_line(fid, 2)
        qf.dz = dztemp * np.ones((qf.nz,), dtype=np.float32)
    else:
        qf.dz = np.zeros((qf.nz,), dtype=np.float32)
        for i in range(0, qf.nz):
            qf.dz[i] = get_line(fid, 2)

    qf.z = np.zeros((qf.nz + 1,))
    for k in range(1, qf.nz + 1):
        qf.z[k] = qf.z[k - 1] + qf.dz[k - 1]

    qf.zm = np.zeros((qf.nz,))
    for k in range(0, qf.nz):
        qf.zm[k] = qf.z[k] + qf.dz[k] * 0.5

    qf.horizontal_extent = [0., qf.dx * float(qf.nx), 0., qf.dy * float(qf.ny)]

    if(qu.isfire==1):
        read_fire_grid_indexing(qf)


def read_fuel(fid):
    fid.readline()  # ! firetec fuel type
    fid.readline()  # ! stream type
    fid.readline()  # ! FUEL
    # - fuel density flag
    dens_flag = get_line(fid, 1)
    if dens_flag == 1:
        fid.readline()  # read density

    # - moisture flag
    if get_line(fid, 1) == 1:
        fid.readline()

    # - topo flag and file string
    #fid.readline() #topo comment
    #fid.readline() #topo flag
    #fid.readline() #topo filepath
    if(dens_flag ==1):
        # - height flag
        fid.readline()
        fid.readline()
    
    


def set_line_fire(fid, ignitions, qf):
    x0 = get_line(fid, 2)
    y0 = get_line(fid, 2)
    len_x = get_line(fid, 2)
    len_y = get_line(fid, 2)

    iis = math.ceil(x0 / qf.dx)
    if x0 % qf.dx == 0:
        iis += 1
    iie = math.ceil((x0 + len_x) / qf.dx)
    jjs = math.ceil(y0 / qf.dy)
    if y0 % qf.dy == 0:
        jjs += 1
    jje = math.ceil((y0 + len_y) / qf.dy)

    iis = int(iis - 1)
    iie = int(iie - 1)
    jjs = int(jjs - 1)
    jje = int(jje - 1)

    ignitions.hor_plane[jjs:jje:1, iis:iie:1] = 1


def set_square_circle(fid, ignitions, qf):
    x0 = get_line(fid, 2)
    y0 = get_line(fid, 2)
    len_x = get_line(fid, 2)
    len_y = get_line(fid, 2)
    width_x = get_line(fid, 2)
    width_y = get_line(fid, 2)

    idelta = math.ceil(width_x / qf.dx)
    jdelta = math.ceil(width_y / qf.dy)
    iis = math.ceil(x0 / qf.dx)
    if x0 % qf.dx == 0:
        iis += 1
    iie = math.ceil((x0 + len_x) / qf.dx)
    jjs = math.ceil(y0 / qf.dy)
    if y0 % qf.dy == 0:
        jjs += 1
    jje = math.ceil((y0 + len_y) / qf.dy)

    iis = int(iis - 1)
    iie = int(iie - 1)
    jjs = int(jjs - 1)
    jje = int(jje - 1)
    idelta = int(idelta)
    jdelta = int(jdelta)

    for i in range(iis, iie):
        # bottom
        for j in range(jjs, jjs + jdelta - 1):
            ignitions.hor_plane[j, i] = 1
        # top
        for j in range(jje - jdelta + 1, jje):
            ignitions.hor_plane[j, i] = 1
    for j in range(jjs, jje):
        # right
        for i in range(iis, iis + idelta - 1):
            ignitions.hor_plane[j, i] = 1
        # left
        for i in range(iie - idelta + 1, iie):
            ignitions.hor_plane[j, i] = 1


def set_circle(fid, ignitions, qf):
    x0 = get_line(fid, 2)
    y0 = get_line(fid, 2)
    len_x = get_line(fid, 2)
    len_y = len_x
    width_x = get_line(fid, 2)

    iis = math.ceil(x0 / qf.dx)
    if x0 % qf.dx == 0:
        iis += 1
    iie = math.ceil((x0 + len_x) / qf.dx)
    jjs = math.ceil(y0 / qf.dy)
    if y0 % qf.dy == 0:
        jjs += 1
    jje = math.ceil((y0 + len_y) / qf.dy)

    iis = int(iis - 1)
    iie = int(iie - 1)
    jjs = int(jjs - 1)
    jje = int(jje - 1)

    radius = len_x * 0.5
    xringcenter = x0 + radius
    yringcenter = y0 + radius
    for j in range(jjs, jje):
        y = (float(j) - 0.5) * qf.dy
        for i in range(iis, iie):
            x = (float(i) - 0.5) * qf.dx
            dist = math.sqrt(math.pow(x - xringcenter, 2) + math.pow(y - yringcenter, 2))
            if radius - width_x <= dist <= radius:
                ignitions.hor_plane[j, i] = 1


def set_firetech_ignitions(ignitions, qf):
    sel_ign = np.zeros((qf.ny, qf.nx, qf.nz))
    fname = 'ignite_selected.dat'
    nelem = int(os.path.getsize(fname) / (5 * 4))
    fid = open_file(fname, 'rb')
    var = np.fromfile(fid, dtype=np.int32, count=nelem * 5)
    var = np.reshape(var, (5, nelem), order='F')
    var -= 1
    myindex = [var[2], var[1], var[3]]
    sel_ign[myindex] = 1
    fid.close()
    ignitions.hor_plane = np.sum(sel_ign, axis=2)


def read_ignitions(fid, qf, ignitions):
    print(fid.readline())  # ! IGNITION LOCATIONS
    ignitions.flag = get_line(fid, 1)

    # Specify 2D array of ignitions
    ignitions.hor_plane = np.zeros((qf.ny, qf.nx))

    if ignitions.flag == 1:  # line
        set_line_fire(fid, ignitions, qf)
        n = 1

    elif ignitions.flag == 2:  # square circle
        set_square_circle(fid, ignitions, qf)
        n = 1

    elif ignitions.flag == 3:  # circle
        set_circle(fid, ignitions, qf)
        n = 1

    elif ignitions.flag == 4:  # QF_Ignitions.inp
        n = 0

    elif ignitions.flag == 5:  # QF_IgnitionPattern.inp
        n = 0

    elif ignitions.flag == 6:  # ignite.dat
        set_firetech_ignitions(ignitions, qf)
        n = 1

    elif ignitions.flag == 7:  # ignite.dat
        #set_firetech_ignitions(ignitions, qf)
        n = 1

    else:
        n = 0

    # Read uninteresting lines
    for i in range(0, n):
        fid.readline()


def read_firebrands(fb):
    out_dir = 'Output/'
    fname = out_dir+'firebrands.bin'
    nelem = int(os.path.getsize(fname) / 4)
    fid = open_file(fname, 'rb')
    var = np.fromfile(fid, dtype=np.int32, count=nelem)
    var = np.reshape(var, ((5+2), int(nelem / (5+2))), order='F')
    fb.time = var[1]
    fb.i = var[2] - 1
    fb.j = var[3] - 1
    fb.k = var[4] - 1
    fb.state = var[5]


def read_file_flags(fid, flags):
    # Firebrands
    fid.readline()  # FIREBRANDS
    flags.firebrands = get_line(fid, 1)

    # Out files
    fid.readline()  # OUTPUT FILES
    flags.en2atm = get_line(fid, 1)
    flags.react_rate = get_line(fid, 1)
    flags.fuel_density = get_line(fid, 1)
    flags.qf_winds = get_line(fid, 1)
    flags.qu_qwinds_inst = get_line(fid, 1)
    flags.qu_qwinds_ave = get_line(fid, 1)
    fid.readline()
    flags.moisture = get_line(fid, 1)
    flags.perc_mass_burnt = get_line(fid, 1)
    fid.readline()
    flags.emissions = get_line(fid, 1)
    flags.thermal_rad = get_line(fid, 1)


def read_path(fid, qf):
    fid.readline()  # ! PATH LABEL
    temp = fid.readline()  # ! PATH
    qf.path = temp[1:-1]
    

def read_qfire_file(qf, qu, ignitions, flags, fb):
    fid = open_file('QUIC_fire.inp', 'r')

    read_times(fid, qu, qf)
    read_fire_grid(fid, qu, qf)
    read_path(fid, qf)
    read_fuel(fid)
    read_ignitions(fid, qf, ignitions)
    read_file_flags(fid, flags)

    #check if run finished or if times need to be altered
    if(qu.isfire == 1):
        containedFiles = os.listdir('Output/')
        #Check quaverage winds first
        if(flags.qu_qwinds_ave ==1):
            count, maxNum = detNumberAndTimeOfFile('qu_windu_ave',containedFiles)
            if(maxNum < max(qu.time_ave)):
                qu.time_ave[count-1]=maxNum
                qu.ntimes_ave = count
        if(flags.qu_qwinds_inst ==1):
            count, maxNum = detNumberAndTimeOfFile('qu_windu',containedFiles)
            if(maxNum < max(qu.time)):
                qu.time[count-1]=maxNum
                qu.ntimes = count
        if(flags.fuel_density + flags.perc_mass_burnt + flags.en2atm + flags.qf_winds   \
            + flags.react_rate + flags.moisture > 0):
            foundFlag = 0
            if(flags.fuel_density):
                rName = 'fuels-dens-'
                count, maxNum = detNumberAndTimeOfFile(rName,containedFiles)
                foundFlag = 1
            elif(flags.perc_mass_burnt and foundFlag != 1):
                rName = 'mburnt_integ-'
                count, maxNum = detNumberAndTimeOfFile(rName,containedFiles)
                foundFlag = 1
            elif(flags.en2atm and foundFlag != 1):
                rName = 'fire-energy_to_atmos-'
                count, maxNum = detNumberAndTimeOfFile(rName,containedFiles)
                foundFlag = 1
            elif(flags.qf_winds and foundFlag != 1):
                rName = 'windu'
                count, maxNum = detNumberAndTimeOfFile(rName,containedFiles)
                foundFlag = 1
            elif(flags.react_rate and foundFlag != 1):
                rName = 'fire-reaction_rate-'
                count, maxNum = detNumberAndTimeOfFile(rName,containedFiles)
                foundFlag = 1
            elif(flags.moisture and foundFlag != 1):
                rName = 'fuels-moist-'
                count, maxNum = detNumberAndTimeOfFile(rName,containedFiles)
                foundFlag = 1
            if(maxNum < max(qf.time)):
                qf.ntimes = count
                qf.time[count-1]=maxNum
        if(flags.emissions + flags.thermal_rad  > 0):
            count=0
            foundFlag = 0
            if(flags.emissions):
                rName = 'pm_emisssions-'
                count, maxNum = detNumberAndTimeOfFile(rName,containedFiles)
                foundFlag = 1
            elif(flags.thermal_rad and foundFlag != 1):
                rName = 'thermalradiation-'
                count, maxNum = detNumberAndTimeOfFile(rName,containedFiles)
                foundFlag = 1
            if(maxNum < max(qf.time_ave)):
                qf.ntimes_ave = count
                qf.time_ave[count-1]=maxNum
        

    if flags.firebrands == 1:
        read_firebrands(fb)

    fid.close()

def detNumberAndTimeOfFile(fString,containedFiles):
    count=0
    tempname=''
    maxNum = 0
    nameList = []
    for name in containedFiles:
        if(fString in name):
            count+=1
            tempname=name
            maxNum=max(int(tempname.split('.')[0][-5::]),maxNum)
            nameList.append(tempname)
    for name in nameList:
        containedFiles.remove(name)
    return count, maxNum

def read_fire_grid_indexing(qf):
    fid = open_file('Output/'+'fire_indexes.bin', 'rb')
    np.fromfile(fid, dtype=np.int32, count=1)
    temp = np.fromfile(fid, dtype=np.int32, count=1)
    qf.indexing.num_cells = temp[0]
    np.fromfile(fid, dtype=np.int32, count=7 + qf.indexing.num_cells)
    qf.indexing.ijk = np.zeros((qf.indexing.num_cells, 3))
    for i in range(0, 3):
        qf.indexing.ijk[::1, i] = np.fromfile(fid, dtype=np.int32, count=qf.indexing.num_cells)

    qf.indexing.ijk = qf.indexing.ijk.astype(int)
    qf.indexing.ijk -= 1
    fid.close()


def import_inputs():
    print("Importing input data")

    # Initialization
    qu = SimField()
    qf = SimField()
    ignitions = IgnitionClass()
    flags = FlagsClass()
    fb = FbClass()

    # Read input files
    read_qu_grid(qu)
    read_qfire_file(qf, qu, ignitions, flags, fb)

    return qu, qf, ignitions, flags, fb


def export_vtk(qf, qu, flags):

    windLayer = qu.nz-1
    
    fOffset = 1
    # QF
    x = np.arange(0, qf.nx*qf.dx, qf.dx)
    y = np.arange(0, qf.ny*qf.dy, qf.dy)
    z = qf.zm

    xm,ym = np.meshgrid(x,y)
    Xs = np.repeat(xm[:, :, np.newaxis], z.size, axis=2)
    Ys = np.repeat(ym[:, :, np.newaxis], z.size, axis=2)
    z1 = np.zeros((qu.ny,qu.nx,z.size))
    my_data = {}

    print("Exporting VTK files (fuel)")

    #Import Topo
    print("- add topo")
    fid = open_file('Output/'+"h.bin", 'rb')
    var = np.fromfile(fid, dtype=np.float32, count=(qf.nx)*(qf.ny))
    topo = np.reshape(var,(qf.ny,qf.nx))
    zg = read_fireca_field("zg", 1, [0], qu, 1)
    zg = zg[0].copy()
    
    for i in range(z.size):
        z1[:,:,i]=zg[:,:,0]+z[i]
    zg = zg[:,:,fOffset:windLayer+fOffset].copy()
    zp = zg[:,:,0]
    ztemp = np.repeat(zp[:,:,np.newaxis],len(zg[0,0,:]),axis=2)
    Xs = np.repeat(xm[:, :, np.newaxis], z.size, axis=2)
    #Scale WindFieldGrid by 2
    #zg = 4.0*(zg-ztemp)+ztemp
    # pylab.imshow(zg[:,:,0])
    # pylab.colorbar()
    # pylab.show()


    xm, ym = np.meshgrid(x,y)
    #my_data['h'] = topo
    if flags.fuel_density == 1:
        print("\t- add fuel density")
        fuel_dens = read_fireca_field("fuels-dens-", qf.ntimes, qf.time, qf, 0)
        my_data['fuel_density'] = fuel_dens
    #     print(fuel_dens[0].shape)
    # for ii in range(qu.ny):
    #     plt.figure(1,figsize=(8,8))
    #     plt.imshow(np.transpose(fuel_dens[0][ii,:,:]))
    #     ax = plt.gca()
    #     ax.set_aspect(aspect=40.0)
    #     #plt.colorbar()
    #     plt.title('%d'%ii)
    #     plt.pause(0.0001)
    #     plt.waitforbuttonpress()
    #     plt.cla()

    if flags.react_rate == 1:
        print("\t- add reaction rate")
        react_rate = read_fireca_field("fire-reaction_rate-", qf.ntimes, qf.time, qf, 0)
        my_data['reaction_rate'] = react_rate

    if flags.en2atm == 1:
        print("\t- add energy to atmos")
        en_to_atm = read_fireca_field("fire-energy_to_atmos-", qf.ntimes, qf.time, qf, 1)
        my_data['energy_to_atmos'] = en_to_atm
        
    if flags.moisture == 1:
        print("\t- add fuel moisture")
        react_rate = read_fireca_field("fuels-moist-", qf.ntimes, qf.time, qf, 0)
        my_data['moisture'] = react_rate

    if bool(my_data):
        print('Write fuel VTK files (fuel)')
        print(z1.shape)
        print(Xs.shape)
        print(Ys.shape)
        for t in range(qf.ntimes):
            print('\ttime: ', qf.time[t])
            out_data = {}
            for k, v in my_data.items():
                out_data[k] = v[t]
            #gridToVTK("./fuels-%05d" % qf.time[t], y, x, z, pointData=out_data)
            #pointsToVTK("./fuels-%05d" % qf.time[t],Xs,Ys,z1,data=out_data)
            gridToVTK("./fuels-%05d" % qf.time[t],Xs,Ys,z1,pointData=out_data)

    # QU
    x = np.arange(0, qu.nx * qu.dx, qu.dx)
    y = np.arange(0, qu.ny * qu.dy, qu.dy)
    z = qu.zm[1:-1]
    xm,ym = np.meshgrid(x,y)
    x1 = np.zeros((1,qu.ny,qu.nx))
    y1 = np.zeros((1,qu.ny,qu.nx))
    z1 = np.zeros((1,qu.ny,qu.nx))
    #Xs = np.repeat(xm[:, :, np.newaxis], z.size-fOffset, axis=2)
    #Ys = np.repeat(ym[:, :, np.newaxis], z.size-fOffset, axis=2)
    Xs = np.repeat(xm[:, :, np.newaxis], windLayer-fOffset+1, axis=2)
    Ys = np.repeat(ym[:, :, np.newaxis], windLayer-fOffset+1, axis=2)
    x1[0]=Xs[:,:,0]
    y1[0]=Ys[:,:,0]
    z1[0]=zg[:,:,0]
    my_data = {}
    print('Writing Topo VTK')
    out_data = {'topo': z1}
    pointsToVTK("./topography",x1,y1,z1,data=out_data)

    if flags.qu_qwinds_inst == 1:
        print("\t- Winds")
        print("\t\t* u")
        windu_qu = read_fireca_field("qu_windu", qu.ntimes, qu.time, qu, 1, layers = windLayer+fOffset)

        print("\t\t* v")
        windv_qu = read_fireca_field("qu_windv", qu.ntimes, qu.time, qu, 1, layers = windLayer+fOffset)

        print("\t\t* w")
        windw_qu = read_fireca_field("qu_windw", qu.ntimes, qu.time, qu, 1, layers = windLayer+fOffset)

        my_data['u'] = windu_qu
        my_data['v'] = windv_qu
        my_data['w'] = windw_qu

    if bool(my_data):
        print('Write Wind VTK files (QU winds)')
        for t in range(qu.ntimes):
            print('\ttime: ', qu.time[t])
            out_data = {
                'winds': (my_data['u'][t][:,:,fOffset::].copy(), my_data['v'][t][:,:,fOffset::].copy(), my_data['w'][t][:,:,fOffset::].copy()),
                'wind_speed': np.sqrt(np.power(my_data['u'][t][:,:,fOffset::], 2) +
                                      np.power(my_data['v'][t][:,:,fOffset::], 2) +
                                      np.power(my_data['w'][t][:,:,fOffset::], 2))}
            out_data['wind_speed'] = out_data['wind_speed']-out_data['wind_speed'][0]
            for k,v in my_data.items():
                if k=='w':
                    out_data[k] =abs(v[t])-abs(v[0])
                else:
                    out_data[k] = v[t]
            #gridToVTK("./quwinds-%05d" % qu.time[t], y, x, z, pointData=out_data)
            print(Ys.shape)
            print(Xs.shape)
            print(zg.shape)
            print(my_data['u'][0].shape)

            #pointsToVTK("./quwinds-%05d" % qu.time[t], Xs, Ys, zg+5.0, data=out_data)
            gridToVTK("./quwindsFull-%05d" % qu.time[t], Xs, Ys, zg, pointData=out_data)

def main(gen_vtk):

    offset = 1000
    # read input files
    qu, qf, ignitions, flags, fb = import_inputs()
    print(qf.nx)
    print(qf.ny)
    #qu.time = qu.time+offsetsdf
    #qf.time = qf.time+offset
    # plot outputs
    #   -if no fire don't plot fire files
    if(qu.isfire==0):
        flags.firebrands = 0
        flags.en2atm = 0
        flags.perc_mass_burnt = 0
        flags.fuel_density = 0
        flags.emissions = 0
        flags.thermal_rad = 0
        flags.qf_winds = 0
        flags.react_rate = 0
        flags.moisture = 0
        flags.qu_qwinds_ave = 0

    plot_outputs(qu, qf, ignitions, flags, fb)

    # VTK    
    if gen_vtk == 1:
        export_vtk(qf, qu, flags)

    print("Program terminated")


if __name__ == '__main__':    
    main(int(sys.argv[1]))
