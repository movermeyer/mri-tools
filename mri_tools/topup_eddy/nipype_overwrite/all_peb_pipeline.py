from nipype.interfaces.ants.base import ANTSCommandInputSpec, ANTSCommand
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrectionOutputSpec
from nipype.interfaces.base import (File, traits, isdefined)
from nipype.utils.filemanip import split_filename
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.fsl as fsl
import os
from nipype.workflows.dmri.fsl.artifacts import _xfm_jacobian, _checkrnum
from nipype.workflows.dmri.fsl.utils import b0_average, apply_all_corrections, insert_mat, \
    rotate_bvecs, vsm2warp, extract_bval, recompose_xfm, recompose_dwi, _checkinitxfm, enhance


__author__ = 'Robbert Harms'
__date__ = "2015-05-08"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


"""This module overwrites some parts of Nipype since they did not work correctly.

The idea is that at a low level two functions did not work correctly. To enable nipype to use the fixed versions of
these functions we have to copy the entire chain to make it work.

Also, the original implementation calculated the read out times from the EPI parameters. This implementation requires
you to predefine the read out times.
"""


def all_peb_pipeline(name='hmc_sdc_ecc',
                     epi_params={'read_out_times': None, 'enc_dir': 'y-'},
                     altepi_params={'read_out_times': None, 'enc_dir': 'y'}):
    """
    Builds a pipeline including three artifact corrections: head-motion
    correction (HMC), susceptibility-derived distortion correction (SDC),
    and Eddy currents-derived distortion correction (ECC).

    .. warning:: this workflow rotates the gradients table (*b*-vectors)
      [Leemans09]_.


    Examples
    --------

    >>> from nipype.workflows.dmri.fsl.artifacts import all_peb_pipeline
    >>> allcorr = all_peb_pipeline()
    >>> allcorr.inputs.inputnode.in_file = 'epi.nii'
    >>> allcorr.inputs.inputnode.alt_file = 'epi_rev.nii'
    >>> allcorr.inputs.inputnode.in_bval = 'diffusion.bval'
    >>> allcorr.inputs.inputnode.in_bvec = 'diffusion.bvec'
    >>> allcorr.run() # doctest: +SKIP

    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_bvec',
                        'in_bval', 'alt_file']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_mask',
                         'out_bvec']), name='outputnode')

    avg_b0_0 = pe.Node(niu.Function(input_names=['in_dwi', 'in_bval'],
                       output_names=['out_file'], function=b0_average),
                       name='b0_avg_pre')
    avg_b0_1 = pe.Node(niu.Function(input_names=['in_dwi', 'in_bval'],
                       output_names=['out_file'], function=b0_average),
                       name='b0_avg_post')
    bet_dwi0 = pe.Node(fsl.BET(frac=0.3, mask=True, robust=True),
                       name='bet_dwi_pre')
    bet_dwi1 = pe.Node(fsl.BET(frac=0.3, mask=True, robust=True),
                       name='bet_dwi_post')

    hmc = hmc_pipeline()
    sdc = sdc_peb(epi_params=epi_params, altepi_params=altepi_params)
    ecc = ecc_pipeline()

    unwarp = apply_all_corrections()

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, hmc,        [('in_file', 'inputnode.in_file'),
                                 ('in_bvec', 'inputnode.in_bvec'),
                                 ('in_bval', 'inputnode.in_bval')]),
        (inputnode, avg_b0_0,   [('in_file', 'in_dwi'),
                                 ('in_bval', 'in_bval')]),
        (avg_b0_0,  bet_dwi0,   [('out_file', 'in_file')]),
        (bet_dwi0,  hmc,        [('mask_file', 'inputnode.in_mask')]),
        (hmc,       sdc,        [('outputnode.out_file', 'inputnode.in_file')]),
        (bet_dwi0,  sdc,        [('mask_file', 'inputnode.in_mask')]),
        (inputnode, sdc,        [('in_bval', 'inputnode.in_bval'),
                                 ('alt_file', 'inputnode.alt_file')]),
        (inputnode, ecc,        [('in_file', 'inputnode.in_file'),
                                 ('in_bval', 'inputnode.in_bval')]),
        (bet_dwi0,  ecc,        [('mask_file', 'inputnode.in_mask')]),
        (hmc,       ecc,        [('outputnode.out_xfms', 'inputnode.in_xfms')]),
        (ecc,       avg_b0_1,   [('outputnode.out_file', 'in_dwi')]),
        (inputnode, avg_b0_1,   [('in_bval', 'in_bval')]),
        (avg_b0_1,  bet_dwi1,   [('out_file', 'in_file')]),
        (inputnode, unwarp,     [('in_file', 'inputnode.in_dwi')]),
        (hmc,       unwarp,     [('outputnode.out_xfms', 'inputnode.in_hmc')]),
        (ecc,       unwarp,     [('outputnode.out_xfms', 'inputnode.in_ecc')]),
        (sdc,       unwarp,     [('outputnode.out_warp', 'inputnode.in_sdc')]),
        (hmc,       outputnode, [('outputnode.out_bvec', 'out_bvec')]),
        (unwarp,    outputnode, [('outputnode.out_file', 'out_file')]),
        (bet_dwi1,  outputnode, [('mask_file', 'out_mask')])
    ])
    return wf


def hmc_pipeline(name='motion_correct'):
    """
    HMC stands for head-motion correction.

    Creates a pipeline that corrects for head motion artifacts in dMRI
    sequences.
    It takes a series of diffusion weighted images and rigidly co-registers
    them to one reference image. Finally, the `b`-matrix is rotated accordingly
    [Leemans09]_ making use of the rotation matrix obtained by FLIRT.

    Search angles have been limited to 4 degrees, based on results in
    [Yendiki13]_.

    A list of rigid transformation matrices is provided, so that transforms
    can be chained.
    This is useful to correct for artifacts with only one interpolation process
    (as previously discussed `here
    <https://github.com/nipy/nipype/pull/530#issuecomment-14505042>`_),
    and also to compute nuisance regressors as proposed by [Yendiki13]_.

    .. warning:: This workflow rotates the `b`-vectors, so please be advised
      that not all the dicom converters ensure the consistency between the
      resulting nifti orientation and the gradients table (e.g. dcm2nii
      checks it).

    .. admonition:: References

      .. [Leemans09] Leemans A, and Jones DK, `The B-matrix must be rotated
        when correcting for subject motion in DTI data
        <http://dx.doi.org/10.1002/mrm.21890>`_,
        Magn Reson Med. 61(6):1336-49. 2009. doi: 10.1002/mrm.21890.

      .. [Yendiki13] Yendiki A et al., `Spurious group differences due to head
        motion in a diffusion MRI study
        <http://dx.doi.org/10.1016/j.neuroimage.2013.11.027>`_.
        Neuroimage. 21(88C):79-90. 2013. doi: 10.1016/j.neuroimage.2013.11.027

    Example
    -------

    >>> from nipype.workflows.dmri.fsl.artifacts import hmc_pipeline
    >>> hmc = hmc_pipeline()
    >>> hmc.inputs.inputnode.in_file = 'diffusion.nii'
    >>> hmc.inputs.inputnode.in_bvec = 'diffusion.bvec'
    >>> hmc.inputs.inputnode.in_bval = 'diffusion.bval'
    >>> hmc.inputs.inputnode.in_mask = 'mask.nii'
    >>> hmc.run() # doctest: +SKIP

    Inputs::

        inputnode.in_file - input dwi file
        inputnode.in_mask - weights mask of reference image (a file with data \
range in [0.0, 1.0], indicating the weight of each voxel when computing the \
metric.
        inputnode.in_bvec - gradients file (b-vectors)
        inputnode.ref_num (optional, default=0) index of the b0 volume that \
should be taken as reference

    Outputs::

        outputnode.out_file - corrected dwi file
        outputnode.out_bvec - rotated gradient vectors table
        outputnode.out_xfms - list of transformation matrices

    """
    from nipype.workflows.data import get_flirt_schedule

    params = dict(dof=6, bgvalue=0, save_log=True, no_search=True,
                  # cost='mutualinfo', cost_func='mutualinfo', bins=64,
                  schedule=get_flirt_schedule('hmc'))

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ref_num',
                        'in_bvec', 'in_bval', 'in_mask']), name='inputnode')
    split = pe.Node(niu.Function(function=hmc_split,
                    input_names=['in_file', 'in_bval', 'ref_num'],
                    output_names=['out_ref', 'out_mov', 'out_bval', 'volid']),
                    name='SplitDWI')
    flirt = dwi_flirt(flirt_param=params)
    insmat = pe.Node(niu.Function(input_names=['inlist', 'volid'],
                     output_names=['out'], function=insert_mat),
                     name='InsertRefmat')
    rot_bvec = pe.Node(niu.Function(input_names=['in_bvec', 'in_matrix'],
                       output_names=['out_file'], function=rotate_bvecs),
                       name='Rotate_Bvec')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file',
                         'out_bvec', 'out_xfms']),
                         name='outputnode')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,     split,   [('in_file', 'in_file'),
                                  ('in_bval', 'in_bval'),
                                  ('ref_num', 'ref_num')]),
        (inputnode,  flirt,      [('in_mask', 'inputnode.ref_mask')]),
        (split,      flirt,      [('out_ref', 'inputnode.reference'),
                                  ('out_mov', 'inputnode.in_file'),
                                  ('out_bval', 'inputnode.in_bval')]),
        (flirt,      insmat,     [('outputnode.out_xfms', 'inlist')]),
        (split,      insmat,     [('volid', 'volid')]),
        (inputnode,  rot_bvec,   [('in_bvec', 'in_bvec')]),
        (insmat,     rot_bvec,   [('out', 'in_matrix')]),
        (rot_bvec,   outputnode, [('out_file', 'out_bvec')]),
        (flirt,      outputnode, [('outputnode.out_file', 'out_file')]),
        (insmat,     outputnode, [('out', 'out_xfms')])
    ])
    return wf


def ecc_pipeline(name='eddy_correct'):
    """
    ECC stands for Eddy currents correction.

    Creates a pipeline that corrects for artifacts induced by Eddy currents in
    dMRI sequences.
    It takes a series of diffusion weighted images and linearly co-registers
    them to one reference image (the average of all b0s in the dataset).

    DWIs are also modulated by the determinant of the Jacobian as indicated by
    [Jones10]_ and [Rohde04]_.

    A list of rigid transformation matrices can be provided, sourcing from a
    :func:`.hmc_pipeline` workflow, to initialize registrations in a *motion
    free* framework.

    A list of affine transformation matrices is available as output, so that
    transforms can be chained (discussion
    `here <https://github.com/nipy/nipype/pull/530#issuecomment-14505042>`_).

    .. admonition:: References

      .. [Jones10] Jones DK, `The signal intensity must be modulated by the
        determinant of the Jacobian when correcting for eddy currents in
        diffusion MRI
        <http://cds.ismrm.org/protected/10MProceedings/files/1644_129.pdf>`_,
        Proc. ISMRM 18th Annual Meeting, (2010).

      .. [Rohde04] Rohde et al., `Comprehensive Approach for Correction of
        Motion and Distortion in Diffusion-Weighted MRI
        <http://stbb.nichd.nih.gov/pdf/com_app_cor_mri04.pdf>`_, MRM
        51:103-114 (2004).

    Example
    -------

    >>> from nipype.workflows.dmri.fsl.artifacts import ecc_pipeline
    >>> ecc = ecc_pipeline()
    >>> ecc.inputs.inputnode.in_file = 'diffusion.nii'
    >>> ecc.inputs.inputnode.in_bval = 'diffusion.bval'
    >>> ecc.inputs.inputnode.in_mask = 'mask.nii'
    >>> ecc.run() # doctest: +SKIP

    Inputs::

        inputnode.in_file - input dwi file
        inputnode.in_mask - weights mask of reference image (a file with data \
range sin [0.0, 1.0], indicating the weight of each voxel when computing the \
metric.
        inputnode.in_bval - b-values table
        inputnode.in_xfms - list of matrices to initialize registration (from \
head-motion correction)

    Outputs::

        outputnode.out_file - corrected dwi file
        outputnode.out_xfms - list of transformation matrices
    """

    from nipype.workflows.data import get_flirt_schedule
    params = dict(dof=12, no_search=True, interp='spline', bgvalue=0,
                  schedule=get_flirt_schedule('ecc'))
    # cost='normmi', cost_func='normmi', bins=64,

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_bval',
                        'in_mask', 'in_xfms']), name='inputnode')
    avg_b0 = pe.Node(niu.Function(input_names=['in_dwi', 'in_bval'],
                     output_names=['out_file'], function=b0_average),
                     name='b0_avg')
    pick_dws = pe.Node(niu.Function(input_names=['in_dwi', 'in_bval', 'b'],
                       output_names=['out_file'], function=extract_bval),
                       name='ExtractDWI')
    pick_dws.inputs.b = 'diff'

    flirt = dwi_flirt(flirt_param=params, excl_nodiff=True)

    mult = pe.MapNode(fsl.BinaryMaths(operation='mul'), name='ModulateDWIs',
                      iterfield=['in_file', 'operand_value'])
    thres = pe.MapNode(fsl.Threshold(thresh=0.0), iterfield=['in_file'],
                       name='RemoveNegative')

    split = pe.Node(fsl.Split(dimension='t'), name='SplitDWIs')
    get_mat = pe.Node(niu.Function(input_names=['in_bval', 'in_xfms'],
                      output_names=['out_files'], function=recompose_xfm),
                      name='GatherMatrices')
    merge = pe.Node(niu.Function(input_names=['in_dwi', 'in_bval', 'in_corrected'],
                    output_names=['out_file'], function=recompose_dwi), name='MergeDWIs')

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_xfms']),
                         name='outputnode')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,  avg_b0,     [('in_file', 'in_dwi'),
                                  ('in_bval', 'in_bval')]),
        (inputnode,  pick_dws,   [('in_file', 'in_dwi'),
                                  ('in_bval', 'in_bval')]),
        (inputnode,  merge,      [('in_file', 'in_dwi'),
                                  ('in_bval', 'in_bval')]),
        (inputnode,  flirt,      [('in_mask', 'inputnode.ref_mask'),
                                  ('in_xfms', 'inputnode.in_xfms'),
                                  ('in_bval', 'inputnode.in_bval')]),
        (inputnode,  get_mat,    [('in_bval', 'in_bval')]),
        (avg_b0,     flirt,      [('out_file', 'inputnode.reference')]),
        (pick_dws,   flirt,      [('out_file', 'inputnode.in_file')]),
        (flirt,      get_mat,    [('outputnode.out_xfms', 'in_xfms')]),
        (flirt,      mult,       [(('outputnode.out_xfms', _xfm_jacobian),
                                  'operand_value')]),
        (flirt,      split,      [('outputnode.out_file', 'in_file')]),
        (split,      mult,       [('out_files', 'in_file')]),
        (mult,       thres,      [('out_file', 'in_file')]),
        (thres,      merge,      [('out_file', 'in_corrected')]),
        (get_mat,    outputnode, [('out_files', 'out_xfms')]),
        (merge,      outputnode, [('out_file', 'out_file')])
    ])
    return wf


def sdc_peb(name='peb_correction',
            epi_params={'read_out_times': None, 'enc_dir': 'y-'},
            altepi_params={'read_out_times': None, 'enc_dir': 'y'}):
    """
    SDC stands for susceptibility distortion correction. PEB stands for
    phase-encoding-based.

    The phase-encoding-based (PEB) method implements SDC by acquiring
    diffusion images with two different enconding directions [Andersson2003]_.
    The most typical case is acquiring with opposed phase-gradient blips
    (e.g. *A>>>P* and *P>>>A*, or equivalently, *-y* and *y*)
    as in [Chiou2000]_, but it is also possible to use orthogonal
    configurations [Cordes2000]_ (e.g. *A>>>P* and *L>>>R*,
    or equivalently *-y* and *x*).
    This workflow uses the implementation of FSL
    (`TOPUP <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP>`_).

    Example
    -------

    >>> from nipype.workflows.dmri.fsl.artifacts import sdc_peb
    >>> peb = sdc_peb()
    >>> peb.inputs.inputnode.in_file = 'epi.nii'
    >>> peb.inputs.inputnode.alt_file = 'epi_rev.nii'
    >>> peb.inputs.inputnode.in_bval = 'diffusion.bval'
    >>> peb.inputs.inputnode.in_mask = 'mask.nii'
    >>> peb.run() # doctest: +SKIP

    .. admonition:: References

      .. [Andersson2003] Andersson JL et al., `How to correct susceptibility
        distortions in spin-echo echo-planar images: application to diffusion
        tensor imaging <http://dx.doi.org/10.1016/S1053-8119(03)00336-7>`_.
        Neuroimage. 2003 Oct;20(2):870-88. doi: 10.1016/S1053-8119(03)00336-7

      .. [Cordes2000] Cordes D et al., Geometric distortion correction in EPI
        using two images with orthogonal phase-encoding directions, in Proc.
        ISMRM (8), p.1712, Denver, US, 2000.

      .. [Chiou2000] Chiou JY, and Nalcioglu O, A simple method to correct
        off-resonance related distortion in echo planar imaging, in Proc.
        ISMRM (8), p.1712, Denver, US, 2000.

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

    readout = epi_params['read_out_time']
    topup.inputs.readout_times = [readout,
                                  altepi_params['read_out_time']]

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


def hmc_split(in_file, in_bval, ref_num=0, lowbval=25.0):
    """
    Selects the reference and moving volumes from a dwi dataset
    for the purpose of HMC.
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    from nipype.interfaces.base import isdefined

    im = nb.load(in_file)
    data = im.get_data()
    hdr = im.get_header().copy()
    bval = np.loadtxt(in_bval)

    lowbs = np.where(bval <= lowbval)[0]

    volid = lowbs[0]
    if (isdefined(ref_num) and (ref_num < len(lowbs))):
        volid = [ref_num]

    # todo add next two lines in Nipype git
    if len(volid) == 1:
        volid = volid[0]

    if volid == 0:
        data = data[..., 1:]
        bval = bval[1:]
    elif volid == (data.shape[-1] - 1):
        data = data[..., :-1]
        bval = bval[:-1]
    else:
        data = np.concatenate((data[..., :volid], data[..., (volid + 1):]),
                              axis=3)
        bval = np.hstack((bval[:volid], bval[(volid + 1):]))

    out_ref = op.abspath('hmc_ref.nii.gz')
    out_mov = op.abspath('hmc_mov.nii.gz')
    out_bval = op.abspath('bval_split.txt')

    refdata = data[..., volid]
    hdr.set_data_shape(refdata.shape)
    nb.Nifti1Image(refdata, im.get_affine(), hdr).to_filename(out_ref)

    hdr.set_data_shape(data.shape)
    nb.Nifti1Image(data, im.get_affine(), hdr).to_filename(out_mov)
    np.savetxt(out_bval, bval)
    return [out_ref, out_mov, out_bval, volid]


class N4BiasFieldCorrectionInputSpec(ANTSCommandInputSpec):
    # todo dimensionality in Nipype git
    dimension = traits.Enum(3, 2, argstr='--image-dimensionality %d',
                            usedefault=True,
                            desc='image dimension (2 or 3)')
    input_image = File(argstr='--input-image %s', mandatory=True,
                       desc=('image to apply transformation to (generally a '
                             'coregistered functional)'))
    mask_image = File(argstr='--mask-image %s')
    weight_image = File(argstr='--weight-image %s')
    output_image = traits.Str(argstr='--output %s',
                              desc='output file name', genfile=True,
                              hash_files=False)
    bspline_fitting_distance = traits.Float(argstr="--bspline-fitting %s")
    bspline_order = traits.Int(requires=['bspline_fitting_distance'])
    shrink_factor = traits.Int(argstr="--shrink-factor %d")
    n_iterations = traits.List(traits.Int(), argstr="--convergence %s",
                               requires=['convergence_threshold'])
    convergence_threshold = traits.Float(requires=['n_iterations'])
    save_bias = traits.Bool(False, mandatory=True, usedefault=True,
                            desc=('True if the estimated bias should be saved'
                                  ' to file.'), xor=['bias_image'])
    bias_image = File(desc='Filename for the estimated bias.',
                      hash_files=False)


class N4BiasFieldCorrection(ANTSCommand):
    """N4 is a variant of the popular N3 (nonparameteric nonuniform normalization)
    retrospective bias correction algorithm. Based on the assumption that the
    corruption of the low frequency bias field can be modeled as a convolution of
    the intensity histogram by a Gaussian, the basic algorithmic protocol is to
    iterate between deconvolving the intensity histogram by a Gaussian, remapping
    the intensities, and then spatially smoothing this result by a B-spline modeling
    of the bias field itself. The modifications from and improvements obtained over
    the original N3 algorithm are described in [Tustison2010]_.

    .. [Tustison2010] N. Tustison et al.,
      N4ITK: Improved N3 Bias Correction, IEEE Transactions on Medical Imaging,
      29(6):1310-1320, June 2010.

    Examples
    --------

	>>> import copy
    >>> from nipype.interfaces.ants import N4BiasFieldCorrection
    >>> n4 = N4BiasFieldCorrection()
    >>> n4.inputs.dimension = 3
    >>> n4.inputs.input_image = 'structural.nii'
    >>> n4.inputs.bspline_fitting_distance = 300
    >>> n4.inputs.shrink_factor = 3
    >>> n4.inputs.n_iterations = [50,50,30,20]
    >>> n4.inputs.convergence_threshold = 1e-6
    >>> n4.cmdline
    'N4BiasFieldCorrection --bspline-fitting [ 300 ] \
--image-dimension 3 --input-image structural.nii \
--convergence [ 50x50x30x20, 1e-06 ] --output structural_corrected.nii \
--shrink-factor 3'

	>>> n4_2 = copy.deepcopy(n4)
    >>> n4_2.inputs.bspline_order = 5
    >>> n4_2.cmdline
    'N4BiasFieldCorrection --bspline-fitting [ 300, 5 ] \
--image-dimension 3 --input-image structural.nii \
--convergence [ 50x50x30x20, 1e-06 ] --output structural_corrected.nii \
--shrink-factor 3'

    >>> n4_3 = N4BiasFieldCorrection()
    >>> n4_3.inputs.input_image = 'structural.nii'
    >>> n4_3.inputs.save_bias = True
    >>> n4_3.cmdline
    'N4BiasFieldCorrection --image-dimension 3 --input-image structural.nii \
--output [ structural_corrected.nii, structural_bias.nii ]'
    """

    _cmd = 'N4BiasFieldCorrection'
    input_spec = N4BiasFieldCorrectionInputSpec
    output_spec = N4BiasFieldCorrectionOutputSpec

    def _gen_filename(self, name):
        if name == 'output_image':
            output = self.inputs.output_image
            if not isdefined(output):
                _, name, ext = split_filename(self.inputs.input_image)
                output = name + '_corrected' + ext
            return output

        if name == 'bias_image':
            output = self.inputs.bias_image
            if not isdefined(output):
                _, name, ext = split_filename(self.inputs.input_image)
                output = name + '_bias' + ext
            return output
        return None

    def _format_arg(self, name, trait_spec, value):
        if ((name == 'output_image') and
           (self.inputs.save_bias or isdefined(self.inputs.bias_image))):
            bias_image = self._gen_filename('bias_image')
            output = self._gen_filename('output_image')
            newval = '[ %s, %s ]' % (output, bias_image)
            return trait_spec.argstr % newval

        if name == 'bspline_fitting_distance':
            if isdefined(self.inputs.bspline_order):
                newval = '[ %g, %d ]' % (value, self.inputs.bspline_order)
            else:
                newval = '[ %g ]' % value
            return trait_spec.argstr % newval

        if ((name == 'n_iterations') and
           (isdefined(self.inputs.convergence_threshold))):
            newval = '[ %s, %g ]' % ('x'.join([str(elt) for elt in value]),
                                     self.inputs.convergence_threshold)
            return trait_spec.argstr % newval

        return super(N4BiasFieldCorrection,
                     self)._format_arg(name, trait_spec, value)

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        skip += ['save_bias', 'bias_image']
        return super(N4BiasFieldCorrection, self)._parse_inputs(skip=skip)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = os.path.abspath(self._gen_filename('output_image'))

        if self.inputs.save_bias or isdefined(self.inputs.bias_image):
            outputs['bias_image'] = os.path.abspath(self._gen_filename('bias_image'))
        return outputs


# todo remove this if N4BiasFieldCorrection works again
def dwi_flirt(name='DWICoregistration', excl_nodiff=False,
              flirt_param={}):
    """
    Generates a workflow for linear registration of dwi volumes
    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['reference',
                        'in_file', 'ref_mask', 'in_xfms', 'in_bval']),
                        name='inputnode')

    initmat = pe.Node(niu.Function(input_names=['in_bval', 'in_xfms',
                      'excl_nodiff'], output_names=['init_xfms'],
                                   function=_checkinitxfm), name='InitXforms')
    initmat.inputs.excl_nodiff = excl_nodiff
    dilate = pe.Node(fsl.maths.MathsCommand(nan2zeros=True,
                     args='-kernel sphere 5 -dilM'), name='MskDilate')
    split = pe.Node(fsl.Split(dimension='t'), name='SplitDWIs')
    pick_ref = pe.Node(niu.Select(), name='Pick_b0')
    n4 = pe.Node(N4BiasFieldCorrection(dimension=3), name='Bias')
    enhb0 = pe.Node(niu.Function(input_names=['in_file', 'in_mask',
                    'clip_limit'], output_names=['out_file'],
                                 function=enhance), name='B0Equalize')
    enhb0.inputs.clip_limit = 0.015
    enhdw = pe.MapNode(niu.Function(input_names=['in_file', 'in_mask'],
                       output_names=['out_file'], function=enhance),
                       name='DWEqualize', iterfield=['in_file'])
    flirt = pe.MapNode(fsl.FLIRT(**flirt_param), name='CoRegistration',
                       iterfield=['in_file', 'in_matrix_file'])
    thres = pe.MapNode(fsl.Threshold(thresh=0.0), iterfield=['in_file'],
                       name='RemoveNegative')
    merge = pe.Node(fsl.Merge(dimension='t'), name='MergeDWIs')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file',
                         'out_xfms']), name='outputnode')
    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,  split,      [('in_file', 'in_file')]),
        (inputnode,  dilate,     [('ref_mask', 'in_file')]),
        (inputnode,  enhb0,      [('ref_mask', 'in_mask')]),
        (inputnode,  initmat,    [('in_xfms', 'in_xfms'),
                                  ('in_bval', 'in_bval')]),
        (inputnode,  n4,         [('reference', 'input_image'),
                                  ('ref_mask', 'mask_image')]),
        (dilate,     flirt,      [('out_file', 'ref_weight'),
                                  ('out_file', 'in_weight')]),
        (n4,         enhb0,      [('output_image', 'in_file')]),
        (split,      enhdw,      [('out_files', 'in_file')]),
        (dilate,     enhdw,      [('out_file', 'in_mask')]),
        (enhb0,      flirt,      [('out_file', 'reference')]),
        (enhdw,      flirt,      [('out_file', 'in_file')]),
        (initmat,    flirt,      [('init_xfms', 'in_matrix_file')]),
        (flirt,      thres,      [('out_file', 'in_file')]),
        (thres,      merge,      [('out_file', 'in_files')]),
        (merge,      outputnode, [('merged_file', 'out_file')]),
        (flirt,      outputnode, [('out_matrix_file', 'out_xfms')])
    ])
    return wf