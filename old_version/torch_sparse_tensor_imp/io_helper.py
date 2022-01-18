import numpy as np
import partio
import pandas as pd
from Particles import Particles
from Constants import *
import partio


def load_df_grid(filename):
    pos = []
    ptype = []
    grid_df = pd.read_csv(filename)
    for i in range(len(grid_df.index)):
        x = grid_df.iloc[i, 0]
        y = grid_df.iloc[i, 1]
        z = grid_df.iloc[i, 2]
        particle_type = grid_df.iloc[i, 3]
        pos.append([x, y, z])
        t = np.int64(particle_type)
        ptype.append(t)
    pos_arr = np.array(pos).astype(np.float32)
    ptype_arr = np.array(ptype)
    vel_arr = np.zeros_like(pos_arr)
    case = Particles(pos_arr, vel_arr, ptype_arr, G, NU, RHO, DT)
    return case


def load_grid(filename):
    pos = np.empty((0, 3), dtype=np.float32)
    vel = np.empty_like(pos)
    ptype = np.empty((0, ), dtype=np.int64)
    with open(filename) as f:
        for line in f.readlines():
            info = line.strip().split(',')
            x, y, z = float(info[0]), float(info[1]), float(info[2])
            u, v, w = float(info[3]), float(info[4]), float(info[5])
            tpe = int(info[6])
            pos = np.concatenate((pos, np.array([[x, y, z]])), axis=0)
            vel = np.concatenate((vel, np.array([[u, v, w]])), axis=0)
            ptype = np.concatenate((ptype, np.array([tpe])))
    case = Particles(pos, vel, ptype, G, NU, RHO, DT)
    return case


def load_npz_grid(filename):
    f = np.load(filename)
    pos, vel, ptype = f['pos'], f['vel'], f['ptype']
    case = Particles(pos, vel, ptype, G, NU, RHO, DT)
    return case


def p2bgeo(fluid_path, output_path, use_vel=False):

    f = np.load(fluid_path)
    pos = f['pos']

    n = pos.shape[0]
    particleSet = partio.create()
    p = particleSet.addAttribute("position", partio.VECTOR, 3)
    if use_vel:
        vel = f['vel']
        v = particleSet.addAttribute("velocity", partio.VECTOR, 3)
    for _ in range(n):
        i = particleSet.addParticle()
        particleSet.set(p, i, pos[i])
        if use_vel:
            particleSet.set(v, i, vel[i])
        partio.write(output_path + '.bgeo', particleSet)  # write uncompressed


def batch2bgeo(input_path, output_path, step_range=(0, 1000), use_vel=False):
    for step in range(step_range[0], step_range[1]):
        fluid_path = input_path + str(step) + '.npz'
        output_bgeo_path = output_path + str(step)
        p2bgeo(fluid_path, output_bgeo_path, use_vel)
