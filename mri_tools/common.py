import os
import subprocess
import numpy as np
import nibabel as nib
from mri_tools.shell_utils import get_fsl_command
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl

__author__ = 'Robbert Harms'
__date__ = "2015-05-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def collect_mdt_output_maps(input_dir, model_name='Tensor', map_name='Tensor.FA'):
    """Collect maps from MDT batch fitting output.

    This assumes that all the subjects have their own directory and in that directory are for each model the results
    of the model fitting:
        <input_dir>/<subject_id>/<model_name>/<map_name>.nii.gz

    Such a directory structure is generally obtained by running 'mdt.collect_batch_fit_output()' after batch fitting.

    Args:
        input_dir (str): the input dir to use (see above).
        model_name (str): the model to use per subject
        map_name (str): the map to use in the given model dir as FA map

    Returns:
        dict: mapping subject id to map name
    """
    maps = {}
    for subject in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, subject)
        if os.path.isdir(full_path):
            maps.update({subject: os.path.join(full_path, model_name, map_name + '.nii.gz')})
    return maps


def create_mask_from_mean_unweighted(item_dir, item):
    """Creates a brain mask from the mean of the unweighted volumes.

    This supposes that the following exists:
        - item_dir/item + '.nii.gz'
        - item_dir/item + '.bvec'
        - item_dir/item + '.bval'

    And will create:
        - item_dir/item + '_mask.nii.gz'

    Args:
        item_dir (str): the directory containing all the relevant files
        item (str): the item to generate the mask for
    """
    dwi = os.path.join(item_dir, item + '.nii.gz')
    bvec = os.path.join(item_dir, item + '.bvec')
    bval = os.path.join(item_dir, item + '.bval')

    out_file = os.path.join(item_dir, item + '_mask')
    b0_file = out_file + '_b0s.nii.gz'
    mean_b0_file = out_file + '_mean_b0s.nii.gz'

    write_unweighted(dwi, bvec, bval, b0_file)
    create_mean_volumes(b0_file, mean_b0_file)

    bet = pe.Node(fsl.BET(mask=True, robust=True, in_file=mean_b0_file, out_file=out_file), name='bet')
    bet.run()

    os.remove(b0_file)
    os.remove(mean_b0_file)
    os.remove(out_file + '.nii.gz')
    os.rename(out_file + '_mask.nii.gz', out_file + '.nii.gz')


def find_unweighted_indices(bvec_file, bval_file, column_based='auto', unweighted_threshold=300.0):
    """Find the unweighted indices from a bvec and bval file.

    If column_based
    This supposes that the bvec (the vector file) has 3 rows (gx, gy, gz) and is space or tab seperated.
    The bval file (the b values) are one one single line with space or tab separated b values.

    Args:
        bvec_file (str): The filename of the bvec file
        bval_file (str): The filename of the bval file
        column_based (boolean): If true, this supposes that the bvec (the vector file) has 3 rows (gx, gy, gz)
            and is space or tab seperated and that the bval file (the b values) are one one single line
            with space or tab separated b values.
            If false, the vectors and b values are each one a different line.
            If 'auto' it is autodetected, this is the default.
        unweighted_threshold (double): The threshold under which we call a direction unweighted (in mm^2)

    Returns:
        list of int: A list of indices to the unweighted volumes.
    """
    bvec = np.genfromtxt(bvec_file)
    bval = np.expand_dims(np.genfromtxt(bval_file), axis=1)

    if len(bvec.shape) < 2:
        raise ValueError('Bvec file does not have enough dimensions.')

    if column_based == 'auto':
        if bvec.shape[1] > bvec.shape[0]:
            bvec = bvec.transpose()
    elif column_based:
        bvec = bvec.transpose()

    if bvec.shape[0] != bval.shape[0]:
        raise ValueError('Columns not of same length.')

    b = bval
    g = bvec

    g_limit = np.sqrt(g[:, 0]**2 + g[:, 1]**2 + g[:, 2]**2) < 0.99
    b_limit = b[:, 0] < unweighted_threshold

    return np.unique(np.argwhere(g_limit + b_limit))


def write_unweighted(dwi_fname, bvec_fname, bval_fname, out_fname):
    """Write an image with all the the unweighted volumes from the given dwi file.

    Args:
        dwi_fname (str): Full filename of the dwi file to get the unweighted volumes from
        bvec_fname (str): Full filename of the bvec file.
        bval_fname (str): Full filename of the bval file.
        out_fname (str): Full filename of the output file with the unweighted images.

    Returns:
        str: the path to the output filename
    """
    image_nifti = nib.load(dwi_fname)
    image = image_nifti.get_data()
    header = image_nifti.get_header()

    unweighted_indices = find_unweighted_indices(bvec_fname, bval_fname)
    unweighted_volume = image[..., unweighted_indices]

    nib.Nifti1Image(unweighted_volume, None, header).to_filename(out_fname)

    return out_fname


def create_mean_volumes(unweighted_volumes_fname, out_fname):
    """Write the mean of the volumes of the given input file to the output file.

    Args:
        unweighted_volumes_fname (str): The input file from which to calculate the mean of all the volumes
        out_fname (str): The path to the output file to which to write the mean of all the volumes

    Returns:
        str: the path to the out file
    """
    fslmaths = get_fsl_command('fslmaths')
    bash_command = fslmaths + ' {0} -Tmean {1}'.format(unweighted_volumes_fname, out_fname)
    subprocess.call(bash_command.split())


def combine_volumes(item_list, out_dwi_fname, out_bvec_fname, out_bval_fname):
    """Combine the images, bvals and bvecs of the given input items.

    Args:
        item_list (list of str): The list with paths to the image, bval and bvec files
        out_dwi_fname (str): The filename to store the output dwi file with all the combined volumes
        out_bvec_fname (str): The filename to store the output bvec
        out_bval_fname (str): The filename to store the output bval
    """
    bvals = []
    bvecs = []
    images = []
    header = None

    for path in item_list:
        bvecs.append(np.genfromtxt(path + '.bvec'))
        bvals.append(np.genfromtxt(path + '.bval'))

        nib_container = nib.load(path + '.nii.gz')
        header = header or nib_container.get_header()
        images.append(nib_container.get_data())

    np.savetxt(out_bvec_fname, np.concatenate(bvecs, axis=1))
    np.savetxt(out_bval_fname, np.expand_dims(np.concatenate(bvals), axis=1).transpose())

    combined_image = np.concatenate(images, axis=3)
    nib.Nifti1Image(combined_image, None, header).to_filename(out_dwi_fname)


def multiply_volumes(volumes, out_dwi_fname, recalculate=True):
    """Multiplies the given volumes and puts the results in the output image.

    Args:
        item_list (list of str): The list with paths to the images we would like to multiply
        out_dwi_fname (str): The filename to store the output dwi file with multiplied result volume
        recalculate (boolean): if False we will not recalculate if the output file exists
    """
    if not recalculate and os.path.isfile(out_dwi_fname):
        return

    header = None
    result_img = None
    for volume in volumes:
        nib_container = nib.load(volume)
        header = header or nib_container.get_header()

        if result_img is None:
            result_img = nib_container.get_data()
        else:
            result_img *= nib_container.get_data()

    if not os.path.isdir(os.path.dirname(out_dwi_fname)):
        os.makedirs(os.path.dirname(out_dwi_fname))

    nib.Nifti1Image(result_img, None, header).to_filename(out_dwi_fname)


def merge_csv(csv_region_files, output_file, delimiter=',', recalculate=True):
    """Per region of interest a CSV file which we will concatenate row based.

    This is meant to be used on the output of 'extract_regions()'. This merges all the regions into one large file.

    The columns are the subjects, the rows are the voxels of all the ROIs.

    Args:
        csv_region_files (list): the roi files we will concatenate
        output_file (str): the location of the output file
        delimeter (str): the delimiter to use for reading and writing the csv
        recalculate (boolean): if False we return if all the files exist.
    """
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    if not recalculate and os.path.isfile(output_file):
        return

    regions = np.hstack([np.genfromtxt(roi, delimiter=delimiter) for roi in csv_region_files])
    np.savetxt(output_file, regions, delimiter=delimiter)
