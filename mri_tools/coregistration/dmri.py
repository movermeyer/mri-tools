import subprocess
import os
import nipype.pipeline.engine as pe
import numpy as np
import nibabel as nib
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as niu


__author__ = 'Robbert Harms'
__date__ = "2015-05-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def coregister(items, input_dir, tmp_dir, output_dir, reference_item=None):
    """Coregisters several corrected dMRI datasets to each other.

    This will output all the volumes corrected separately and one with all items combined.

    Args:
        items (list of str): The list of base names of the items, should contain at least an image, bval and bvec.
        input_dir (str): the input directory
        tmp_dir (str): the temporary files location
        output_dir (str): the location of the output files
        reference_item (str): Which one of the given items to use as the reference image for the co-registration.
    """
    # create a brain mask using those b0's with options frac=0.35, mask=True
    # coregister the b0's one for one to the reference image
    # apply the transformation to the original volume
    # combine the volumes
    # write output

    reference_item = reference_item or items[0]

    for d in [tmp_dir, output_dir]:
        if not os.path.isdir(d):
            os.mkdir(d)

    for item in items:
        dwi = os.path.join(input_dir, item + '.nii.gz')
        bvec = os.path.join(input_dir, item + '.bvec')
        bval = os.path.join(input_dir, item + '.bval')
        b0_file = os.path.join(tmp_dir, item + '_b0s.nii.gz')
        mean_b0_file = os.path.join(tmp_dir, item + '_mean_b0s.nii.gz')

        write_unweighted(dwi, bvec, bval, b0_file)
        create_mean_volumes(b0_file, mean_b0_file)


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
    bash_command = 'fslmaths {0} -Tmean {1}'.format(unweighted_volumes_fname, out_fname)
    subprocess.call(bash_command.split())



def combine_volumes(epi_list, out_dwi_fname=None, out_bvec_fname=None, out_bval_fname=None,
                    out_split_ind_fname=None):
    """Combine the images, bvals and bvecs of the given input items.

    Args:
        epi_list (list of str): The list with paths to the image, bval and bvec files
        out_dwi_fname (str): The filename to store the output dwi file with all the combined volumes
        out_bvec_fname (str): The filename to store the output bvec
        out_bval_fname (str): The filename to store the output bval
        out_split_ind_fname (str): The filename with the indices to be used to split the volumes again

    Returns:
        list of str: a three element list with at
            0) path to dwi file
            1) path to bvec file
            2) path to bval file
            3) out_split_ind_fname
    """
    import numpy as np
    import nibabel as nib
    import os

    out_dwi_fname = out_dwi_fname or os.path.abspath('dwi_combined.nii.gz')
    out_bvec_fname = out_bvec_fname or os.path.abspath('bvec_combined.bvec')
    out_bval_fname = out_bval_fname or os.path.abspath('bval_combined.bval')
    out_split_ind_fname = out_split_ind_fname or os.path.abspath('split_indices.txt')

    bvals = []
    bvecs = []
    images = []
    header = None

    for path in epi_list:
        bvecs.append(np.genfromtxt(path + '.bvec'))
        bvals.append(np.genfromtxt(path + '.bval'))

        nib_container = nib.load(path + '.nii.gz')
        header = header or nib_container.get_header()
        images.append(nib_container.get_data())

    split_indices = []
    ind_sum = 0
    for i in range(len(images) - 1):
        split_indices.append(images[i].shape[3] + 1 + ind_sum)
        ind_sum += images[i].shape[3]

    np.savetxt(out_bvec_fname, np.concatenate(bvecs, axis=1))
    np.savetxt(out_bval_fname, np.expand_dims(np.concatenate(bvals), axis=1).transpose())
    np.savetxt(out_split_ind_fname, split_indices)

    combined_image = np.concatenate(images, axis=3)
    nib.Nifti1Image(combined_image, None, header).to_filename(out_dwi_fname)

    return [out_dwi_fname, out_bvec_fname, out_bval_fname, out_split_ind_fname]


def find_unweighted_indices(bvec_file, bval_file, column_based='auto', unweighted_threshold=25.0):
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
        raise ValueError('Bval file does not have enough dimensions.')

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