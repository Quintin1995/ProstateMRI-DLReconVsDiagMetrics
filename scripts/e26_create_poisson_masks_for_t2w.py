import SimpleITK as sitk
import argparse
from umcglib.utils import print_, apply_parallel
from umcglib.kspace import get_poisson_mask_3d


# OLD
def get_shapes():
    train_path_list = 'data/path_lists/pirads_4plus/current_t2w.txt'
    t2_files = [l.strip() for l in open(train_path_list)]
    shapes = []
    for t2_file in t2_files:
        x_true_s = sitk.ReadImage(t2_file, sitk.sitkFloat32)
        shape = (x_true_s.GetSize()[0], x_true_s.GetSize()[1], x_true_s.GetSize()[2])
        print_(f"> {t2_file}")
        print_(f">with shape{shape}")
        if shape not in shapes:
            shapes.append(shape)
    return shapes


# OLD
def get_unique_shapes():
    shapes = [(384, 384, 19), (384, 384, 21), (384, 384, 23), (384, 384, 20), (384, 384, 27), (640, 640, 19), (384, 384, 25), (384, 384, 22), (320, 320, 31), (320, 320, 23), (384, 384, 26), (320, 320, 21), (384, 384, 31), (512, 512, 28), (320, 320, 33), (384, 384, 28), (384, 384, 18), (384, 384, 24), (768, 768, 33), (1024, 1024, 27), (768, 768, 31), (640, 640, 31), (768, 768, 27), (1024, 1024, 29), (256, 256, 33), (384, 384, 29), (320, 320, 25), (320, 320, 24), (512, 512, 31), (320, 320, 35), (256, 256, 28), (576, 576, 35), (384, 384, 35), (384, 384, 33), (1024, 1024, 31), (640, 640, 35), (640, 640, 33), (768, 768, 29), (432, 512, 23), (640, 640, 37), (768, 768, 23), (1024, 1024, 30), (640, 640, 45), (256, 256, 35), (256, 256, 25), (768, 768, 35), (320, 320, 26), (256, 256, 24), (512, 512, 33), (1024, 1024, 28), (256, 256, 43), (640, 640, 41), (320, 320, 37), (320, 320, 27)]
    unique_shapes = []
    for shape in shapes:
        sh = (shape[0], shape[1], 1)
        if sh not in unique_shapes:
            unique_shapes.append(sh)

    return unique_shapes


# NEW
def get_unique_acquistion_matrices():
    # I querried the DICOM databases to give me all unique acquisition matrices
    # and pasted them here for simplicity (replaced / with a ,)
    db_results = ['0,768,585,0', '0,656,513,0', '0,768,582,0', '0,768,608,0', '0,656,512,0', '0,878,576,0', '320,0,0,320', '0,701,536,0', '0,780,629,0', '0,762,583,0', '0,768,606,0', '0,768,566,0', '568,0,0,430', '0,768,590,0', '0,628,487,0', '0,768,580,0', '0,768,605,0', '0,368,283,0', '0,256,243,0', '0,384,307,0', '0,320,275,0', '0,320,298,0', '0,256,205,0', '256,0,0,256', '0,320,256,0', '0,452,302,0', '0,768,579,0', '0,320,300,0', '0,628,584,0', '0,628,488,0', '0,768,593,0', '320,0,0,275', '0,512,302,0', '0,240,216,0', '0,224,216,0', '0,256,252,0', '0,384,330,0', '0,320,240,0', '0,256,257,0', '0,756,577,0', '0,384,384,0', '0,320,320,0', '0,256,256,0', '0,320,262,0', '256,0,0,205', '0,128,122,0', '0,192,127,0', '0,320,288,0', '0,320,272,0', '0,448,265,0', '0,256,192,0', '0,320,146,0']
    ac_matrices = []
    for db_res in db_results:
        parts = db_res.split(',')
        # Multi-valued: {freq_rows, freq_columns, phase_rows, phase_columns}
        freq_rows = int(parts[0])
        freq_cols = int(parts[1])
        phase_rows = int(parts[2])
        phase_cols = int(parts[3])
        ac_tup = (max(freq_rows, phase_rows), max(freq_cols, phase_cols), 1)
        ac_matrices.append(ac_tup)

    return ac_matrices


def parse_input_args():
    parser = argparse.ArgumentParser(description='Argument parser for the reconstruction model.')

    parser.add_argument(
        '-a',
        '--acceleration',
        type=float,
        required=True,
        help='Acceleration',
    )
    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help="Use this if the code should be run in debug mode. So paramters will be scaled down so that it runs much faster."
    )

    parser.add_argument(
        '-w',
        '--n_workers',
        type=int,
        required=True,
        help='number of workers on the task. Choose between 1 and 12',
    )

    args = parser.parse_args()
    return args


################################################################################


if __name__ == '__main__':

    args = parse_input_args()
    DEBUG = args.debug

    # get the relevant shapes
    shapes = get_unique_acquistion_matrices()

    accel        = args.acceleration
    n_workers    = args.n_workers
    n_neighbours = 42           # emperically determined
    seed         = 12345
    centre_samp  = 0.2          # Value from paper that Frank Simonis mentioned
    tolerance    = 0.02         # Emperically determined and seemed acceptable

    if DEBUG:
        shapes = shapes[:n_workers]

    for idx, shape in enumerate(shapes):
        if shape[0] % 2 == 1:
            shapes[idx] = (shape[0]-1, shape[1], shape[2])
        if shape[1] % 2 == 1:
            shapes[idx] = (shape[0], shape[1]-1, shape[2])

    if n_workers > 1:
        masks = apply_parallel(
            item_list    = shapes,
            function     = get_poisson_mask_3d,
            num_workers  = n_workers,
            accel        = accel,
            n_neighbours = n_neighbours,
            seed         = seed,
            centre_samp  = centre_samp,
            tol          = tolerance,
            mask_dir     = 'new_gen',
        )
    else:
        masks = []
        for idx, shape in enumerate(shapes):

            mask = get_poisson_mask_3d(
                shape        = shape,
                accel        = accel,
                n_neighbours = n_neighbours,
                seed         = seed,
                centre_samp  = centre_samp,
                tol          = tolerance,
                mask_dir     = 'new_gen', 
            )
            masks.append(mask)

    print_("--Completed--")