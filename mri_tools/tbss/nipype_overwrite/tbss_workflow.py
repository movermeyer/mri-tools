from warnings import warn
from nipype.interfaces import fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
from nipype.workflows.dmri.fsl import create_tbss_2_reg, create_tbss_3_postreg, create_tbss_4_prestats


__author__ = 'Robbert Harms'
__date__ = "2015-02-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


# the line 'dimtup = dimtup[0:3]' is the reason this and the following few workflows are here
def tbss1_op_string(in_files):
    import nibabel as nib
    op_strings = []
    for infile in in_files:
        img = nib.load(infile)
        dimtup = tuple([d - 2 for d in img.get_shape()])
        dimtup = dimtup[0:3]
        op_str = '-min 1 -ero -roi 1 %d 1 %d 1 %d 0 1' % dimtup
        op_strings.append(op_str)
    return op_strings


def create_tbss_1_preproc(name='tbss_1_preproc'):
    """Preprocess FA data for TBSS: erodes a little and zero end slicers and
    creates masks(for use in FLIRT & FNIRT from FSL).
    A pipeline that does the same as tbss_1_preproc script in FSL

    Example
    -------

    >>> from nipype.workflows.dmri.fsl import tbss
    >>> tbss1 = tbss.create_tbss_1_preproc()
    >>> tbss1.inputs.inputnode.fa_list = ['s1_FA.nii', 's2_FA.nii', 's3_FA.nii']

    Inputs::

        inputnode.fa_list

    Outputs::

        outputnode.fa_list
        outputnode.mask_list
        outputnode.slices

    """

    # Define the inputnode
    inputnode = pe.Node(interface=util.IdentityInterface(fields=["fa_list"]),
                        name="inputnode")

    # Prep the FA images
    prepfa = pe.MapNode(fsl.ImageMaths(suffix="_prep"),
                                    name="prepfa",
                                    iterfield=['in_file', 'op_string'])

    # Slicer
    slicer = pe.MapNode(fsl.Slicer(all_axial=True, image_width=1280),
                        name='slicer',
                        iterfield=['in_file'])

    # Create a mask
    getmask1 = pe.MapNode(fsl.ImageMaths(op_string="-bin", suffix="_mask"),
                        name="getmask1",
                        iterfield=['in_file'])
    getmask2 = pe.MapNode(fsl.MultiImageMaths(op_string="-dilD -dilD -sub 1 -abs -add %s"),
                        name="getmask2",
                        iterfield=['in_file', 'operand_files'])

#    $FSLDIR/bin/fslmaths FA/${f}_FA_mask -dilD -dilD -sub 1 -abs -add FA/${f}_FA_mask FA/${f}_FA_mask -odt char
    # Define the tbss1 workflow
    tbss1 = pe.Workflow(name=name)
    tbss1.connect([
        (inputnode, prepfa, [("fa_list", "in_file")]),
        (inputnode, prepfa, [(("fa_list", tbss1_op_string), "op_string")]),
        (prepfa, getmask1, [("out_file", "in_file")]),
        (getmask1, getmask2, [("out_file", "in_file"),
                              ("out_file", "operand_files")]),
        (prepfa, slicer, [('out_file', 'in_file')]),
        ])

    # Define the outputnode
    outputnode = pe.Node(interface=util.IdentityInterface(fields=["fa_list",
                                                                "mask_list",
                                                                "slices"]),
                        name="outputnode")
    tbss1.connect([
                (prepfa, outputnode, [("out_file", "fa_list")]),
                (getmask2, outputnode, [("out_file", "mask_list")]),
                (slicer, outputnode, [('out_file', 'slices')])
                ])
    return tbss1


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

# Fixes bug https://github.com/nipy/nipype/issues/1030, this is fixed in the latest release (2015-08-05)
def create_tbss_non_FA(name='tbss_non_FA'):
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
    merge = pe.Node(fsl.Merge(dimension="t"), name="merge")
    #merged_file="all_FA.nii.gz"
    maskgroup = pe.Node(fsl.ImageMaths(op_string="-mas",
                                       suffix="_masked"),
                        name="maskgroup")
    projectfa = pe.Node(fsl.TractSkeleton(project_data=True,
                                        #projected_data = 'test.nii.gz',
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

                    (maskgroup, projectfa, [('out_file', 'alt_data_file')]),
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
