import glob
import os
import shutil
from mri_tools.topup_eddy.nipype_overwrite.all_peb_pipeline import all_peb_pipeline
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio


__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def run_pre_processing(epi_name, alt_epi_name, output_name, input_dir, tmp_dir, output_dir):
    epi_bname = os.path.join(input_dir, epi_name)
    alt_epi_bname = os.path.join(input_dir, alt_epi_name)

    wf = pe.Workflow(name=output_name, base_dir=tmp_dir)

    correction_wf = all_peb_pipeline(epi_params=read_epi_params(epi_bname),
                                     altepi_params=read_epi_params(alt_epi_bname))
    correction_wf.inputs.inputnode.in_file = epi_bname + '.nii.gz'
    correction_wf.inputs.inputnode.alt_file = alt_epi_bname + '.nii.gz'
    correction_wf.inputs.inputnode.in_bval = epi_bname + '.bval'
    correction_wf.inputs.inputnode.in_bvec = epi_bname + '.bvec'

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = os.path.join(output_dir, output_name)
    wf.connect([(correction_wf, datasink, [('outputnode.out_bvec', 'bvec'),
                                           ('outputnode.out_file', 'image'),
                                           ('outputnode.out_mask', 'mask')])])
    wf.run()

    _write_output_files(epi_bname, output_dir, output_name)


def read_epi_params(item_basename):
    """Read all the necessary EPI parameters from files with the given basename.

    Args:
        basename (str): The basename for all the files we need to read.
            In particular the following files should exist:
                - basename + '.acc_fac_pe.txt' (with the acceleration factor)
                - basename + '.epi_factor.txt' (with the epi factor)
                - basename + '.echo_spacing.txt' (with the echo spacing in seconds)
                - basename + '.phase_enc_dir.txt' (with the phase encode direction, like AP, PA, LR, RL, SI, IS, ...)

    Returns:
        dict: A dictionary for use in the nipype workflow 'all_peb_pipeline'. It contains the keys:
            - echospacing (the echo spacing)
            - acc_factor (the acceleration factor)
            - enc_dir (the phase encode direction, converted to nipype standards (x, -x, y, -y, ...))
            - epi_factor (the epi factor)
    """
    with open(item_basename + '.acc_fac_pe.txt', 'r') as f:
        acc_factor = float(f.read())

    with open(item_basename + '.epi_factor.txt', 'r') as f:
        epi_factor = float(f.read())

    with open(item_basename + '.echo_spacing.txt', 'r') as f:
        echo_spacing = float(f.read())

    with open(item_basename + '.phase_enc_dir.txt', 'r') as f:
        phase_encoding = f.read()

    phase_enc_dirs_translate = {'AP': 'y-', 'PA': 'y',
                                'LR': 'x-', 'RL': 'x', 'SD': 'x-', 'DS': 'x',
                                'SI': 'z-', 'IS': 'z', 'HF': 'z-', 'FH': 'z'}

    return {'echospacing': echo_spacing,
            'acc_factor': acc_factor,
            'enc_dir': phase_enc_dirs_translate[phase_encoding],
            'epi_factor': epi_factor}


def _write_output_files(input_bname, output_dir, output_name):
    """Write all the output files from various places to the output directory.

    This will write the bvec, bval, image, and mask file to the given output directory with the given output name as
    basename:
        - basename + '.bvec'
        - basename + '.bvec'
        - basename + '.nii.gz'
        - basename + '_mask.nii.gz'

    Args:
        input_bname (str): the original input basename
        output (str): The output directory for the resulting files
        output_name (str): The output name.
    """
    output_bname = os.path.join(output_dir, output_name)

    os.rename(glob.glob(os.path.join(output_dir, output_name, 'bvec', '*.bvec'))[0],
              output_bname + '.bvec')

    os.rename(glob.glob(os.path.join(output_dir, output_name, 'image', '*.nii.gz'))[0],
              output_bname + '.nii.gz')

    os.rename(glob.glob(os.path.join(output_dir, output_name, 'mask', '*.nii.gz'))[0],
              output_bname + '_mask.nii.gz')

    shutil.rmtree(os.path.join(output_dir, output_name))
    shutil.copy(input_bname + '.bval', output_bname + '.bval')