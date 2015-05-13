import glob
import os
import shutil
from mri_tools.dwi_pre_processing.nipype_overwrite.all_peb_pipeline import all_peb_pipeline, hmc_pipeline
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import re
import numpy as np
import nibabel as nib
import subprocess
from mri_tools.shell_utils import get_fsl_command
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as niu


__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def apply_corrections(epis, alt_epis, input_dir, tmp_dir, output_dir):
    epis = [os.path.join(input_dir, e) for e in epis]
    alt_epis = [os.path.join(input_dir, e) for e in alt_epis]

    pipeline = get_correction_pipeline(base_dir=tmp_dir)
    pipeline.inputs.inputnode.epi_list = epis
    pipeline.inputs.inputnode.alt_epi_list = alt_epis
    # pipeline.run(plugin='MultiProc')
    pipeline.run()


def get_correction_pipeline(name='all_peb_correction_pipeline', base_dir=None):
    inputnode = pe.Node(niu.IdentityInterface(fields=['epi_list', 'alt_epi_list']),
                        name='inputnode')

    combine_epi = combine_image_items_node('combine_epi')
    unweighted_epi = get_join_unweighted_node('unweighted_epi')

    avg_b0_0 = pe.Node(niu.Function(input_names=['unweighted_volumes_fname'],
                                    output_names=['out_fname'],
                                    function=create_mean_volumes),
                       name='avg_b0_0')

    bet_dwi_0 = pe.Node(fsl.BET(frac=0.3, mask=True, robust=True),
                        name='bet_dwi_0')

    combine_alt_epi = combine_image_items_node('combine_alt_epi')
    unweighted_alt_epi = get_join_unweighted_node('unweighted_alt_epi')

    create_acq_params_epi = create_acq_params_node('create_acq_params_epi')
    create_acq_params_alt_epi = create_acq_params_node('create_acq_params_alt_epi')

    unweighted_epi_post_hmc = get_join_unweighted_node('unweighted_epi_post_hmc')

    prep_topup = pe.Node(niu.Function(input_names=['unweighted_epi_fname', 'unweighted_alt_epi_fname',
                                                   'acq_epi_fname', 'acq_alt_epi_fname'],
                                      output_names=['out_topup_image_fname',
                                                    'out_topup_acq_fname'],
                                      function=prepare_topup),
                         name='prepare_topup')


    hmc = hmc_pipeline()
    topup = pe.Node(fsl.TOPUP(), name='topup')

    # sdc = sdc_peb_multi_shell(epi_params=epi_params, altepi_params=altepi_params)

    wf = pe.Workflow(name=name, base_dir=base_dir)
    wf.connect([
        # todo acq params read out time is not correct yet
        (inputnode, create_acq_params_epi, [('epi_list', 'epi_list')]),
        (inputnode, create_acq_params_alt_epi, [('alt_epi_list', 'epi_list')]),

        (inputnode, combine_epi, [('epi_list', 'epi_list')]),
        (combine_epi, unweighted_epi, [('out_dwi_fname', 'dwi_fname'),
                                             ('out_bvec_fname', 'bvec_fname'),
                                             ('out_bval_fname', 'bval_fname')]),
        (unweighted_epi, avg_b0_0, [('out_fname', 'unweighted_volumes_fname')]),

        (inputnode, combine_alt_epi, [('alt_epi_list', 'epi_list')]),
        (combine_alt_epi, unweighted_alt_epi, [('out_dwi_fname', 'dwi_fname'),
                                                     ('out_bvec_fname', 'bvec_fname'),
                                                     ('out_bval_fname', 'bval_fname')]),

        (avg_b0_0,  bet_dwi_0, [('out_fname', 'in_file')]),

        (combine_epi, hmc, [('out_dwi_fname', 'inputnode.in_file'),
                            ('out_bvec_fname', 'inputnode.in_bvec'),
                            ('out_bval_fname', 'inputnode.in_bval')]),
        (bet_dwi_0,  hmc, [('mask_file', 'inputnode.in_mask')]),

        (hmc, unweighted_epi_post_hmc, [('outputnode.out_file', 'dwi_fname')]),
        (combine_epi, unweighted_epi_post_hmc, [('out_bvec_fname', 'bvec_fname'),
                                                ('out_bval_fname', 'bval_fname')]),

        (unweighted_epi_post_hmc, prep_topup, [('out_fname', 'unweighted_epi_fname')]),
        (unweighted_alt_epi, prep_topup, [('out_fname', 'unweighted_alt_epi_fname')]),
        (create_acq_params_epi, prep_topup, [('out_fname', 'acq_epi_fname')]),
        (create_acq_params_alt_epi, prep_topup, [('out_fname', 'acq_alt_epi_fname')]),

        (prep_topup, topup, [('out_topup_image_fname', 'in_file'),
                             ('out_topup_acq_fname', 'encoding_file')])

        # run apply_topup

        # apply vsm2warp
        # phase_enc_dirs_translate = {'AP': 'y-', 'PA': 'y',
        #                             'LR': 'x-', 'RL': 'x', 'SD': 'x-', 'DS': 'x',
        #                             'SI': 'z-', 'IS': 'z', 'HF': 'z-', 'FH': 'z'}


        # add the ecc pipeline


        # add the unwarp pipeline

        # join the output nodes
    ])
    return wf


def get_join_unweighted_node(name):
    """Get the node for joining the unweighted volumes from an input volume."""
    return pe.Node(niu.Function(input_names=['dwi_fname', 'bvec_fname', 'bval_fname'],
                                output_names=['out_fname'],
                                function=join_unweighted),
                   name=name)


def combine_image_items_node(name):
    """Get the node for combining image items (images, bvec, bval)"""
    return pe.Node(niu.Function(input_names=['epi_list'],
                                output_names=['out_dwi_fname', 'out_bvec_fname',
                                              'out_bval_fname', 'out_split_ind_fname'],
                                function=combine_volumes),
                   name=name)


def create_acq_params_node(name):
    """The node for creating an acquisitions parameters file for topup"""
    return pe.Node(niu.Function(input_names=['epi_list'],
                                output_names=['out_fname'],
                                function=create_acq_params_file),
                   name=name)


def sdc_peb_multi_shell_pipeline():
    """Get the susceptibility distortion correction pipeline phase-encoding-based.

    The phase-encoding-based (PEB) method implements SDC by acquiring
    diffusion images with two different enconding directions [Andersson2003]_.
    The most typical case is acquiring with opposed phase-gradient blips
    (e.g. *A>>>P* and *P>>>A*, or equivalently, *-y* and *y*)
    as in [Chiou2000]_, but it is also possible to use orthogonal
    configurations [Cordes2000]_ (e.g. *A>>>P* and *L>>>R*,
    or equivalently *-y* and *x*).
    This workflow uses the implementation of FSL
    (`TOPUP <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP>`_).

    Inputnodes:
        combined_dwi: The combined image with the combined epi items
        epi_list: The list with the different shell paths
        alt_epi_list: The list with the alternative phase encoding epi paths

    Outputnodes:



    Returns:
        The workflow for sdc correction using peb.
    """

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_bval',
                        'in_mask', 'alt_file', 'ref_num']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_vsm',
                         'out_warp']), name='outputnode')

    b0_ref = pe.Node(fsl.ExtractROI(t_size=1), name='b0_ref')
    b0_alt = pe.Node(fsl.ExtractROI(t_size=1), name='b0_alt')
    b0_comb = pe.Node(niu.Merge(2), name='b0_list')
    b0_merge = pe.Node(fsl.Merge(dimension='t'), name='b0_merged')

    topup = pe.Node(fsl.TOPUP(), name='topup')
    topup.inputs.encoding_direction = [epi_params['enc_dir'],
                                       altepi_params['enc_dir']]

    readout = compute_readout(epi_params)
    topup.inputs.readout_times = [readout,
                                  compute_readout(altepi_params)]

    unwarp = pe.Node(fsl.ApplyTOPUP(in_index=[1], method='jac'), name='unwarp')

    # scaling = pe.Node(niu.Function(input_names=['in_file', 'enc_dir'],
    #                   output_names=['factor'], function=_get_zoom),
    #                   name='GetZoom')
    # scaling.inputs.enc_dir = epi_params['enc_dir']
    vsm2dfm = vsm2warp()
    vsm2dfm.inputs.inputnode.enc_dir = epi_params['enc_dir']
    vsm2dfm.inputs.inputnode.scaling = readout

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,  b0_ref,     [('in_file', 'in_file'),
                                  (('ref_num', _checkrnum), 't_min')]),
        (inputnode,  b0_alt,     [('alt_file', 'in_file'),
                                  (('ref_num', _checkrnum), 't_min')]),
        (b0_ref,     b0_comb,    [('roi_file', 'in1')]),
        (b0_alt,     b0_comb,    [('roi_file', 'in2')]),
        (b0_comb,    b0_merge,   [('out', 'in_files')]),
        (b0_merge,   topup,      [('merged_file', 'in_file')]),
        (topup,      unwarp,     [('out_fieldcoef', 'in_topup_fieldcoef'),
                                  ('out_movpar', 'in_topup_movpar'),
                                  ('out_enc_file', 'encoding_file')]),
        (inputnode,  unwarp,     [('in_file', 'in_files')]),
        (unwarp,     outputnode, [('out_corrected', 'out_file')]),
        # (b0_ref,      scaling,    [('roi_file', 'in_file')]),
        # (scaling,     vsm2dfm,    [('factor', 'inputnode.scaling')]),
        (b0_ref,      vsm2dfm,    [('roi_file', 'inputnode.in_ref')]),
        (topup,       vsm2dfm,    [('out_field', 'inputnode.in_vsm')]),
        (topup,       outputnode, [('out_field', 'out_vsm')]),
        (vsm2dfm,     outputnode, [('outputnode.out_warp', 'out_warp')])
    ])
    return wf


def prepare_topup(unweighted_epi_fname, unweighted_alt_epi_fname, acq_epi_fname, acq_alt_epi_fname,
                  out_topup_image_fname=None, out_topup_acq_fname=None):
    """This will prepare the unweighted images and the acq params file for use in topup.

    This will concatenate both the unweighted volumes and the acq params files.

    Args:
        unweighted_epi_fname (str): the file with the unweighted volumes from the epi data
        unweighted_alt_epi_fname (str): the file with the unweighted volumes from the alternative epi data
        acq_epi_fname (str): the acq params file for the epi data
        acq_alt_epi_fname (str): the acq params file for the alternative epi data
        out_topup_image_fname (str): the filename for the output image file
        out_topup_acq_fname (str): the output file for the output acq params file

    Returns:
        list of str: [out_topup_image_fname, out_topup_acq_fname]
    """
    import os
    import nibabel as nib
    import numpy as np

    out_topup_image_fname = out_topup_image_fname or os.path.abspath('combined_unweighted.nii.gz')
    out_topup_acq_fname = out_topup_acq_fname or os.path.abspath('combined_acqparams.txt')

    with open(out_topup_acq_fname, 'w') as f_out:
        for fname in [acq_epi_fname, acq_alt_epi_fname]:
            with open(fname, 'r') as f_in:
                f_out.write(f_in.read())
            f_out.write("\n")

    images = []
    header = None
    for fname in [unweighted_epi_fname, unweighted_alt_epi_fname]:
        nib_container = nib.load(fname)
        header = header or nib_container.get_header()
        images.append(nib_container.get_data())

    combined_image = np.concatenate(images, axis=3)
    nib.Nifti1Image(combined_image, None, header).to_filename(out_topup_image_fname)

    return [out_topup_image_fname, out_topup_acq_fname]


def create_acq_params_file(epi_list, out_file=None):
    """From the given list with paths to epi files create the acquisition parameters file.

    This is the file topup needs to process the unweighted volumes.

    Note that this will only write lines for the unweighted volumes in the epi volumes.

    Args:
        epi_list (list of str): The list with paths to the image, bval and bvec files
        out_file (str): The path to the out file

    Returns:
        str: the path to the out file
    """
    import os
    from mri_tools.dwi_pre_processing.common import read_epi_params, find_unweighted_indices, compute_readout_times

    out_file = out_file or os.path.abspath('acqparams.txt')

    phase_enc_dirs_translate = {'AP': '0 -1 0', 'PA': '0 1 0',
                                'LR': '-1 0 0', 'RL': '1 0 0', 'DS': '1 0 0', 'SD': '-1 0 0',
                                'SI': '0 0 -1', 'IS': '0 0 1', 'HF': '0 0 -1', 'FH': '0 0 1'}

    lines = []
    for item_path in epi_list:
        unweighted_ind = find_unweighted_indices(item_path + '.bvec', item_path + '.bval')
        epi_params = read_epi_params(item_path)
        read_out_time = compute_readout_times(epi_params['epi_factor'],
                                              epi_params['acc_factor'],
                                              epi_params['echo_spacing'])

        line = phase_enc_dirs_translate[epi_params['phase_enc_dir']] + ' ' + repr(read_out_time)
        for e in unweighted_ind:
            lines.append(line)

    with open(out_file, 'w') as f:
        f.write("\n".join(lines))

    return out_file


def create_mean_volumes(unweighted_volumes_fname, out_file=None):
    """Write the mean of the volumes of the given input file to the output file.

    Args:
        unweighted_volumes_fname (str): The input file from which to calculate the mean of all the volumes
        out_file (str): The path to the output file to which to write the mean of all the volumes

    Returns:
        str: the path to the out file
    """
    import subprocess
    import os

    out_file = out_file or os.path.abspath('mean_unweighted.nii.gz')
    bash_command = 'fslmaths {0} -Tmean {1}'.format(unweighted_volumes_fname, out_file)
    subprocess.call(bash_command.split())
    return out_file


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


def join_unweighted(dwi_fname, bvec_fname, bval_fname, out_fname=None):
    """Write an image with all the the unweighted volumes from the given dwi file.

    Args:
        dwi_fname (str): Full filename of the dwi file to get the unweighted volumes from
        bvec_fname (str): Full filename of the bvec file.
        bval_fname (str): Full filename of the bval file.
        out_fname (str): Full filename of the output file with the unweighted images.

    Returns:
        str: the path to the output filename
    """
    import nibabel as nib
    import os
    from mri_tools.dwi_pre_processing.common import find_unweighted_indices

    out_fname = out_fname or os.path.abspath('all_unweighted.nii.gz')

    image_nifti = nib.load(dwi_fname)
    image = image_nifti.get_data()
    header = image_nifti.get_header()

    unweighted_indices = find_unweighted_indices(bvec_fname, bval_fname)
    unweighted_volume = image[..., unweighted_indices]

    nib.Nifti1Image(unweighted_volume, None, header).to_filename(out_fname)

    return out_fname


def read_epi_params(item_path):
    """Read all the necessary EPI parameters from files with the given path.

    Args:
        path (str): The path for all the files we need to read.
            In particular the following files should exist:
                - path + '.acc_fac_pe.txt' (with the acceleration factor)
                - path + '.epi_factor.txt' (with the epi factor)
                - path + '.echo_spacing.txt' (with the echo spacing in seconds)
                - path + '.phase_enc_dir.txt' (with the phase encode direction, like AP, PA, LR, RL, SI, IS, ...)

    Returns:
        dict: A dictionary for use in the nipype workflow 'all_peb_pipeline'. It contains the keys:
            - echo_spacing (the echo spacing)
            - acc_factor (the acceleration factor)
            - phase_enc_dir (the phase encode direction)
            - epi_factor (the epi factor)
    """
    with open(item_path + '.acc_fac_pe.txt', 'r') as f:
        acc_factor = float(f.read())

    with open(item_path + '.epi_factor.txt', 'r') as f:
        epi_factor = float(f.read())

    with open(item_path + '.echo_spacing.txt', 'r') as f:
        echo_spacing = float(f.read())

    with open(item_path + '.phase_enc_dir.txt', 'r') as f:
        phase_encoding = f.read()

    return {'echo_spacing': echo_spacing,
            'acc_factor': acc_factor,
            'phase_enc_dir': phase_encoding,
            'epi_factor': epi_factor}


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


def compute_readout_times(epi_factor, acc_factor, echo_spacing):
    """Compute the readout times from the EPI parameters.

    Args:
        epi_factor (float): the epi factor
        acc_factor (float): the acceleration factor
        echo_spacing (float): the echo spacing in seconds

    Returns:
        float: the read out times, calculated with (1/acc_factor) * epi_factor * echo_spacing. With epi_factor-=1 if
            epi_factor is greater than 1.
    """
    #todo readout time may be wrong
    # from matteo: base-resolution ( * phase oversampling, if used) * (partial Fourier Factor / GRAPPAFactor) - 1
    # should it not be readOutTime = echoSpacing * ((matrixLines*partialFourier/accelerationFactor)-1)

    if epi_factor > 1:
        epi_factor -= 1
    return (1.0 / acc_factor) * epi_factor * echo_spacing