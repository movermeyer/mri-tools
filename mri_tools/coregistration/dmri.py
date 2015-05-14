import os
import multiprocessing
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import shutil
from mri_tools.common import create_mean_volumes, write_unweighted, combine_volumes


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
    reference_item = reference_item or items[0]
    register_items = [e for e in items if e != reference_item]

    for d in [tmp_dir, output_dir]:
        if not os.path.isdir(d):
            os.mkdir(d)

    pool = multiprocessing.Pool()
    pool.map(RunPrepareCoregistration(input_dir, tmp_dir), items)
    pool.map(RunApplyCoRegistration(reference_item, input_dir, tmp_dir, output_dir), register_items)

    shutil.copy(os.path.join(input_dir, reference_item + '.nii.gz'),
                os.path.join(output_dir, reference_item + '.nii.gz'))
    for item in items:
        for ext in ['.bval', '.bvec']:
            shutil.copy(os.path.join(input_dir, item + ext),
                        os.path.join(output_dir, item + ext))

    to_combine = [os.path.join(output_dir, e) for e in items]
    combine_volumes(to_combine, os.path.join(output_dir, 'combined.nii.gz'),
                    os.path.join(output_dir, 'combined.bvec'), os.path.join(output_dir, 'combined.bval'))


class RunApplyCoRegistration(object):

    def __init__(self, reference_item, input_dir, tmp_dir, output_dir):
        self._reference_item = reference_item
        self._input_dir = input_dir
        self._tmp_dir = tmp_dir
        self._output_dir = output_dir

    def __call__(self, item):
        return _apply_coregistration(item, self._reference_item, self._input_dir, self._tmp_dir, self._output_dir)


def _apply_coregistration(item, reference_item, input_dir, tmp_dir, output_dir):
    """Coregisters all items to the reference item. The reference item is skipped in the list of items.

    Args:
        item (str): the items base_name to register to the reference item
        reference_item (str): the name of the reference item in the items list
        input_dir (str): the input directory in which the items reside
        tmp_dir (str): the working dir. Outputs reference volume and reference matrix
    """
    reference_mean_b0 = os.path.join(tmp_dir, reference_item + '_mask.nii.gz')
    reference_dwi = os.path.join(input_dir, reference_item + '.nii.gz')

    input_mean_b0 = os.path.join(tmp_dir, item + '_mask.nii.gz')
    input_dwi = os.path.join(input_dir, item + '.nii.gz')

    out_b0_vol = os.path.join(tmp_dir, item + '_to_' + reference_item + '_mean_b0.nii.gz')
    out_img = os.path.join(output_dir, item + '.nii.gz')
    out_mat = os.path.join(tmp_dir, item + '_to_' + reference_item + '_mean_b0.mat')

    flirt_b0 = pe.Node(fsl.FLIRT(in_file=input_mean_b0, reference=reference_mean_b0,
                                 out_file=out_b0_vol, out_matrix_file=out_mat, dof=6),
                       name='flirt_b0')
    flirt_b0.run()

    flirt_apply = pe.Node(fsl.FLIRT(in_file=input_dwi, reference=reference_dwi,
                                    out_file=out_img, in_matrix_file=out_mat,
                                    apply_xfm=True),
                          name='flirt')
    flirt_apply.run()


class RunPrepareCoregistration(object):

    def __init__(self, input_dir, tmp_dir):
        self._input_dir = input_dir
        self._tmp_dir = tmp_dir

    def __call__(self, item):
        return _prepare_coregistration(item, self._input_dir, self._tmp_dir)


def _prepare_coregistration(item, input_dir, tmp_dir):
    """Prepare coregistering by generating the mean b0's and the corresponding masks.

    Args:
        item (str): the item basename to prepare
        input_dir (str): the input directory in which the items reside
        tmp_dir (str): the output dir. The output is:
            - basename_b0s.nii.gz
            - basename_mean_b0s.nii.gz
            - basename_mask.nii.gz (the skull stripped image)
            - basename_mask_mask.nii.gz (the binary mask).
    """
    dwi = os.path.join(input_dir, item + '.nii.gz')
    bvec = os.path.join(input_dir, item + '.bvec')
    bval = os.path.join(input_dir, item + '.bval')
    b0_file = os.path.join(tmp_dir, item + '_b0s.nii.gz')
    mean_b0_file = os.path.join(tmp_dir, item + '_mean_b0s.nii.gz')
    masked_file = os.path.join(tmp_dir, item + '_mask.nii.gz')

    write_unweighted(dwi, bvec, bval, b0_file)
    create_mean_volumes(b0_file, mean_b0_file)

    bet = pe.Node(fsl.BET(frac=0.35, mask=True, robust=True, in_file=mean_b0_file, out_file=masked_file),
                  name='bet')
    bet.run()