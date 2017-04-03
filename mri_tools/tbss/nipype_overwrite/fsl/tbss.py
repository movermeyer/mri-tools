from nipype import Node
from nipype.algorithms.misc import Gunzip
from nipype.workflows.dmri.fsl import create_tbss_1_preproc
from nipype.workflows.dmri.fsl import create_tbss_2_reg
from nipype.workflows.dmri.fsl import create_tbss_4_prestats
from nipype.interfaces.utility import Function
from warnings import warn

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util
from nipype.interfaces import fsl as fsl


"""
Mergin volumes on the fourth dimension. Signature is made similar to fsl.Bet nipype function.
"""
def merge_volumes_4d(in_files=None, dimension=None, tr=None, merged_file=None):
    import nibabel as nib
    import numpy as np

    if not merged_file:
        merged_file = in_files[0]
        if merged_file.endswith('.gz'):
            merged_file = merged_file[:-3]
        if merged_file.endswith('.nii'):
            merged_file = merged_file[:-4]

        merged_file += '_merged.nii.gz'

    header = None
    combined_image = None

    for volume in in_files:
        nib_container = nib.load(volume)
        header = header or nib_container.get_header()
        image_data = nib_container.get_data()

        if len(image_data.shape) < 4:
            image_data = np.expand_dims(image_data, axis=3)

        if combined_image is None:
            combined_image = image_data
        else:
            combined_image = np.concatenate([combined_image, image_data], axis=3)

    nib.Nifti1Image(combined_image, None, header).to_filename(merged_file)
    return merged_file


"""
This implementation overwrites the ``fsl.Merge`` function node with a custom one.

The original implementation crashed when using fsl.Merge with a large number of files. This was due to the generated
command line string being to long for the OS.

Also, in the groupmask node, the flag -Tmin was replaced with -Tmedian to generate a better mask file.
"""
def create_tbss_3_postreg(name='tbss_3_postreg', estimate_skeleton=True):
    """Post-registration processing: derive mean_FA and mean_FA_skeleton from
    mean of all subjects in study. Target is assumed to be FMRIB58_FA_1mm.
    A pipeline that does the same as 'tbss_3_postreg -S' script from FSL
    Setting 'estimate_skeleton to False will use precomputed FMRIB58_FA-skeleton_1mm
    skeleton (same as 'tbss_3_postreg -T').

    Example
    -------

    >>> from nipype.workflows.dmri.fsl import tbss
    >>> tbss3 = tbss.create_tbss_3_postreg()
    >>> tbss3.inputs.inputnode.fa_list = ['s1_wrapped_FA.nii', 's2_wrapped_FA.nii', 's3_wrapped_FA.nii']

    Inputs::

        inputnode.field_list
        inputnode.fa_list

    Outputs::

        outputnode.groupmask
        outputnode.skeleton_file
        outputnode.meanfa_file
        outputnode.mergefa_file

    """

    # Create the inputnode
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['field_list',
                                                                 'fa_list']),
                        name='inputnode')

    # Apply the warpfield to the masked FA image
    applywarp = pe.MapNode(interface=fsl.ApplyWarp(),
                           iterfield=['in_file', 'field_file'],
                           name="applywarp")
    if fsl.no_fsl():
        warn('NO FSL found')
    else:
        applywarp.inputs.ref_file = fsl.Info.standard_image("FMRIB58_FA_1mm.nii.gz")

    # Merge the FA files into a 4D file
    mergefa = pe.Node(# fsl.Merge(dimension="t"),
        interface=Function(input_names=['in_files', 'dimension', 'tr', 'merged_file'],
                           output_names=['merged_file'],
                           function=merge_volumes_4d),
                      name="mergefa")

    # Get a group mask
    groupmask = pe.Node(fsl.ImageMaths(op_string="-max 0 -Tmedian -bin",
                                       out_data_type="char",
                                       suffix="_mask"),
                        name="groupmask")

    maskgroup = pe.Node(fsl.ImageMaths(op_string="-mas",
                                       suffix="_masked"),
                        name="maskgroup")

    tbss3 = pe.Workflow(name=name)
    tbss3.connect([
        (inputnode, applywarp, [("fa_list", "in_file"),
                                ("field_list", "field_file")]),
        (applywarp, mergefa, [("out_file", "in_files")]),
        (mergefa, groupmask, [("merged_file", "in_file")]),
        (mergefa, maskgroup, [("merged_file", "in_file")]),
        (groupmask, maskgroup, [("out_file", "in_file2")]),
    ])

    # Create outputnode
    outputnode = pe.Node(interface=util.IdentityInterface(fields=['groupmask',
                                                                  'skeleton_file',
                                                                  'meanfa_file',
                                                                  'mergefa_file']),
                         name='outputnode')

    if estimate_skeleton:
        # Take the mean over the fourth dimension
        meanfa = pe.Node(fsl.ImageMaths(op_string="-Tmean",
                                        suffix="_mean"),
                         name="meanfa")

        # Use the mean FA volume to generate a tract skeleton
        makeskeleton = pe.Node(fsl.TractSkeleton(skeleton_file=True),
                               name="makeskeleton")
        tbss3.connect([
            (maskgroup, meanfa, [("out_file", "in_file")]),
            (meanfa, makeskeleton, [("out_file", "in_file")]),
            (groupmask, outputnode, [('out_file', 'groupmask')]),
            (makeskeleton, outputnode, [('skeleton_file', 'skeleton_file')]),
            (meanfa, outputnode, [('out_file', 'meanfa_file')]),
            (maskgroup, outputnode, [('out_file', 'mergefa_file')])
        ])
    else:
        # $FSLDIR/bin/fslmaths $FSLDIR/data/standard/FMRIB58_FA_1mm -mas mean_FA_mask mean_FA
        maskstd = pe.Node(fsl.ImageMaths(op_string="-mas",
                                         suffix="_masked"),
                          name="maskstd")
        maskstd.inputs.in_file = fsl.Info.standard_image("FMRIB58_FA_1mm.nii.gz")

        # $FSLDIR/bin/fslmaths mean_FA -bin mean_FA_mask
        binmaskstd = pe.Node(fsl.ImageMaths(op_string="-bin"),
                             name="binmaskstd")

        # $FSLDIR/bin/fslmaths all_FA -mas mean_FA_mask all_FA
        maskgroup2 = pe.Node(fsl.ImageMaths(op_string="-mas",
                                            suffix="_masked"),
                             name="maskgroup2")

        tbss3.connect([
            (groupmask, maskstd, [("out_file", "in_file2")]),
            (maskstd, binmaskstd, [("out_file", "in_file")]),
            (maskgroup, maskgroup2, [("out_file", "in_file")]),
            (binmaskstd, maskgroup2, [("out_file", "in_file2")])
        ])

        outputnode.inputs.skeleton_file = fsl.Info.standard_image("FMRIB58_FA-skeleton_1mm.nii.gz")
        tbss3.connect([
            (binmaskstd, outputnode, [('out_file', 'groupmask')]),
            (maskstd, outputnode, [('out_file', 'meanfa_file')]),
            (maskgroup2, outputnode, [('out_file', 'mergefa_file')])
        ])
    return tbss3



"""
This implementation overwrites the ``create_tbss_3_postreg`` function with the one found above.
"""
def create_tbss_all(name='tbss_all', estimate_skeleton=True):
    """Create a pipeline that combines create_tbss_* pipelines

    Example
    -------

    >>> from nipype.workflows.dmri.fsl import tbss
    >>> tbss = tbss.create_tbss_all('tbss')
    >>> tbss.inputs.inputnode.skeleton_thresh = 0.2

    Inputs::

        inputnode.fa_list
        inputnode.skeleton_thresh

    Outputs::

        outputnode.meanfa_file
        outputnode.projectedfa_file
        outputnode.skeleton_file
        outputnode.skeleton_mask

    """

    # Define the inputnode
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['fa_list',
                                                                 'skeleton_thresh']),
                        name='inputnode')

    tbss1 = create_tbss_1_preproc(name='tbss1')
    tbss2 = create_tbss_2_reg(name='tbss2')
    if fsl.no_fsl():
        warn('NO FSL found')
    else:
        tbss2.inputs.inputnode.target = fsl.Info.standard_image("FMRIB58_FA_1mm.nii.gz")
    tbss3 = create_tbss_3_postreg(name='tbss3', estimate_skeleton=estimate_skeleton)
    tbss4 = create_tbss_4_prestats(name='tbss4')

    tbss_all = pe.Workflow(name=name)
    tbss_all.connect([
        (inputnode, tbss1, [('fa_list', 'inputnode.fa_list')]),
        (inputnode, tbss4, [('skeleton_thresh', 'inputnode.skeleton_thresh')]),

        (tbss1, tbss2, [('outputnode.fa_list', 'inputnode.fa_list'),
                        ('outputnode.mask_list', 'inputnode.mask_list')]),
        (tbss1, tbss3, [('outputnode.fa_list', 'inputnode.fa_list')]),
        (tbss2, tbss3, [('outputnode.field_list', 'inputnode.field_list')]),
        (tbss3, tbss4, [
            ('outputnode.groupmask', 'inputnode.groupmask'),
            ('outputnode.skeleton_file', 'inputnode.skeleton_file'),
            ('outputnode.meanfa_file', 'inputnode.meanfa_file'),
            ('outputnode.mergefa_file', 'inputnode.mergefa_file')
        ])
    ])

    # Define the outputnode
    outputnode = pe.Node(interface=util.IdentityInterface(fields=['groupmask',
                                                                  'skeleton_file3',
                                                                  'meanfa_file',
                                                                  'mergefa_file',
                                                                  'projectedfa_file',
                                                                  'skeleton_file4',
                                                                  'skeleton_mask',
                                                                  'distance_map']),
                         name='outputnode')
    outputall_node = pe.Node(interface=util.IdentityInterface(
        fields=['fa_list1',
                'mask_list1',
                'field_list2',
                'groupmask3',
                'skeleton_file3',
                'meanfa_file3',
                'mergefa_file3',
                'projectedfa_file4',
                'skeleton_mask4',
                'distance_map4']),
        name='outputall_node')

    tbss_all.connect([
        (tbss3, outputnode, [('outputnode.meanfa_file', 'meanfa_file'),
                             ('outputnode.mergefa_file', 'mergefa_file'),
                             ('outputnode.groupmask', 'groupmask'),
                             ('outputnode.skeleton_file', 'skeleton_file3'),
                             ]),
        (tbss4, outputnode, [('outputnode.projectedfa_file', 'projectedfa_file'),
                             ('outputnode.skeleton_file', 'skeleton_file4'),
                             ('outputnode.skeleton_mask', 'skeleton_mask'),
                             ('outputnode.distance_map', 'distance_map'),
                             ]),

        (tbss1, outputall_node, [('outputnode.fa_list', 'fa_list1'),
                                 ('outputnode.mask_list', 'mask_list1'),
                                 ]),
        (tbss2, outputall_node, [('outputnode.field_list', 'field_list2'),
                                 ]),
        (tbss3, outputall_node, [
            ('outputnode.meanfa_file', 'meanfa_file3'),
            ('outputnode.mergefa_file', 'mergefa_file3'),
            ('outputnode.groupmask', 'groupmask3'),
            ('outputnode.skeleton_file', 'skeleton_file3'),
        ]),
        (tbss4, outputall_node, [
            ('outputnode.projectedfa_file', 'projectedfa_file4'),
            ('outputnode.skeleton_mask', 'skeleton_mask4'),
            ('outputnode.distance_map', 'distance_map4'),
        ]),
    ])
    return tbss_all


"""
This implementation overwrites the ``fsl.Merge`` function node with a custom one.

The original implementation crashed when using fsl.Merge with a large number of files. This was due to the generated
command line string being to long for the OS.

Also, it adds an output file specifier for the output file.
"""
def create_tbss_non_FA(name='tbss_non_FA', output_file=None):
    """
    A pipeline that implement tbss_non_FA in FSL

    Example
    -------

    >>> from nipype.workflows.dmri.fsl import tbss
    >>> tbss_MD = tbss.create_tbss_non_FA()
    >>> tbss_MD.inputs.inputnode.file_list = []
    >>> tbss_MD.inputs.inputnode.field_list = []
    >>> tbss_MD.inputs.inputnode.skeleton_thresh = 0.2
    >>> tbss_MD.inputs.inputnode.groupmask = './xxx'
    >>> tbss_MD.inputs.inputnode.meanfa_file = './xxx'
    >>> tbss_MD.inputs.inputnode.distance_map = []
    >>> tbss_MD.inputs.inputnode.all_FA_file = './xxx'

    Inputs::

        inputnode.file_list
        inputnode.field_list
        inputnode.skeleton_thresh
        inputnode.groupmask
        inputnode.meanfa_file
        inputnode.distance_map
        inputnode.all_FA_file

    Outputs::

        outputnode.projected_nonFA_file

    """

    # Define the inputnode
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['file_list',
                                                                 'field_list',
                                                                 'skeleton_thresh',
                                                                 'groupmask',
                                                                 'meanfa_file',
                                                                 'distance_map',
                                                                 'all_FA_file']),
                        name='inputnode')

    # Apply the warpfield to the non FA image
    applywarp = pe.MapNode(interface=fsl.ApplyWarp(),
                           iterfield=['in_file', 'field_file'],
                           name="applywarp")
    if fsl.no_fsl():
        warn('NO FSL found')
    else:
        applywarp.inputs.ref_file = fsl.Info.standard_image("FMRIB58_FA_1mm.nii.gz")
    # Merge the non FA files into a 4D file
    merge = pe.Node(# fsl.Merge(dimension="t"),
                    interface=Function(input_names=['in_files', 'dimension', 'tr', 'merged_file'],
                                       output_names=['merged_file'],
                                       function=merge_volumes_4d),
                    name="merge")


    # merged_file="all_FA.nii.gz"
    maskgroup = pe.Node(fsl.ImageMaths(op_string="-mas",
                                       suffix="_masked"),
                        name="maskgroup")

    gunzip = Node(Gunzip(), name="gunzip")

    projectfa = pe.Node(fsl.TractSkeleton(project_data=True,
                                          projected_data=output_file,
                                          use_cingulum_mask=True
                                          ),
                        name="projectfa")

    tbss_non_FA = pe.Workflow(name=name)
    tbss_non_FA.connect([
        (inputnode, applywarp, [('file_list', 'in_file'),
                                ('field_list', 'field_file'),
                                ]),
        (applywarp, merge, [("out_file", "in_files")]),

        (merge, maskgroup, [("merged_file", "in_file")]),

        (inputnode, maskgroup, [('groupmask', 'in_file2')]),

        (maskgroup, gunzip, [('out_file', 'in_file')]),
        (gunzip, projectfa, [('out_file', 'alt_data_file')]),

        (inputnode, projectfa, [('skeleton_thresh', 'threshold'),
                                ("meanfa_file", "in_file"),
                                ("distance_map", "distance_map"),
                                ("all_FA_file", 'data_file')
                                ]),
    ])

    # Define the outputnode
    outputnode = pe.Node(interface=util.IdentityInterface(
        fields=['projected_nonFA_file']),
        name='outputnode')
    tbss_non_FA.connect([
        (projectfa, outputnode, [('projected_data', 'projected_nonFA_file'),
                                 ]),
    ])
    return tbss_non_FA
