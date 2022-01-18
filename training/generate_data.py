import os
to_run = 'run_physics_model.py'
method = 'MPS'
max_step = '1000'

# for training
for i in range(20):
    grid_path = '../grid/train_scene/case'+str(i)+'_grid.npz'
    output_dir = '../dataset/training/'

    # run physics models
    os.system("python ../run_physics_model.py"
              + " --method " + method
              + " --max_step " + max_step
              + " --grid_path " + grid_path
              + " --output_dir " + output_dir
              + f" --output_prefix seed{i}_"
              + " --write_output"
              + " --write_train")

# for testing
for i in range(5):
    grid_path = '../grid/testing_scene/case'+str(i)+'_grid.txt'
    output_dir = '../dataset/testing/'

    # run physics models
    os.system("python ../run_physics_model.py"
              + " --method " + method
              + " --max_step " + max_step
              + " --grid_path " + grid_path
              + " --output_dir " + output_dir
              + f" --output_prefix seed{i}_"
              + " --write_output"
              + " --write_train")

