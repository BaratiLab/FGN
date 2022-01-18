import partio
import numpy as np
import os
import argparse


def p2bgeo(fluid_path, output_path, use_vel=False):
    f = np.load(fluid_path)
    pos = f['pos'].astype(np.single)
    #pos = f
    n = pos.shape[0]
    particleSet = partio.create()
    p = particleSet.addAttribute("position", partio.VECTOR, 3)
    v = particleSet.addAttribute("velocity", partio.VECTOR, 3)

    if use_vel:
        vel = f['vel'].astype(np.single)
    for _ in range(n):
        i = particleSet.addParticle()
        particleSet.set(p, i, (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
        if use_vel:
            particleSet.set(v, i, (float(vel[i,0]),float(vel[i,1]), float(vel[i,2])))
        else:
            particleSet.set(v, i, (0., 0., 0.))
    partio.write(output_path + '.bgeo', particleSet)  # write uncompressed


def batch2bgeo(input_path, output_path, step_range=(1, 10000), use_vel=False):
    for step in range(step_range[0], step_range[1]+1):
        fluid_path = input_path + str(step) + '.npz'
        output_bgeo_path = output_path + str(step)
        p2bgeo(fluid_path, output_bgeo_path, use_vel)


def main(input_path, output_path):
    # parser = argparse.ArgumentParser(
    #     description=
    #     "Transform numpy array into bgeo"
    # )
    #
    # parser.add_argument("--input_dir",
    #                     type=str,
    #                     required=True,
    #                     help="Input directory")
    # parser.add_argument("--output_dir",
    #                     type=str,
    #                     required=True,
    #                     help="Output directory")
    # args = parser.parse_args()
    #
    # input_path = args.input_dir
    # output_path = args.output_dir
    batch2bgeo(input_path, output_path)


if __name__ == '__main__':
    main('./output/canyon_FGN/sim_info', './bgeo_file/canyon_newFGN10000/scene')

    # pos = np.load('./grid/bunny_obs.npy')
    # n = pos.shape[0]
    # particleSet = partio.create()
    # p = particleSet.addAttribute("position", partio.VECTOR, 3)
    # v = particleSet.addAttribute("velocity", partio.VECTOR, 3)
    #
    # for _ in range(n):
    #     i = particleSet.addParticle()
    #     particleSet.set(p, i, (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
    #     particleSet.set(v, i, (0., 0., 0.))
    # os.mkdir('bgeo_file/bunny_bgeo/')
    # for t in range(1, 1001):
    #     partio.write('./bgeo_file/bunny_bgeo/scene' +str(t) + '.bgeo', particleSet)