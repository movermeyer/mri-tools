import glob
import os
import shutil
from mri_tools.tbss.nipype_overwrite import tbss_workflow
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from nipype import Workflow
from mri_tools.tbss.nipype_overwrite.tbss_workflow import create_tbss_non_FA

__author__ = 'Robbert Harms'
__date__ = "2015-08-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def run_tbss(fa_maps, subjects_list, output_dir, recalculate=True):
    """Run TBSS on the given subjects.

    Args:
        fa_maps (dict): mapping subjects to fa map file names
        subjects_list (list of str): the list of subjects names. We provide these to make sure the images are
            all analyzed in the correct order.
        output_dir (str): the output directory
        recalculate (boolean): if we recalculate if the output already exists. Set this to False to easily get the
            results dictionary.

    Returns:
        dict: dictionary mapping subjects to output files. This can be used as input to run_tbss_non_FA.
            The dictionary contains:
                - distance_map: single nii.gz
                - group_mask: single nii.gz
                - mean_fa: single nii.gz
                - merge_fa: single nii.gz
                - projected_fa: single nii.gz
                - skeleton_mask: single nii.gz
                - field_list: dict with per subject field list (the warp file)
    """
    work_dir = os.path.join(output_dir, '_nipype_work_dir')

    if not recalculate:
        try:
            return get_tbss_info_dict(fa_maps, subjects_list, output_dir)
        except RuntimeError:
            pass

    fa_list = [fa_maps[subject] for subject in subjects_list]

    tbss_all = tbss_workflow.create_tbss_all(estimate_skeleton=False)
    tbss_all.base_dir = work_dir
    tbss_all.inputs.inputnode.fa_list = fa_list
    tbss_all.inputs.inputnode.skeleton_thresh = 0.2

    data_sink = pe.Node(nio.DataSink(), 'DataSink')
    data_sink.inputs.base_directory = output_dir
    data_sink.inputs.parameterization = True
    data_sink.inputs.substitutions = ('_nipype_work_dir/', '')

    wf = Workflow(name='tbss_all_wf', base_dir=work_dir)
    wf.connect([(tbss_all, data_sink, [('outputnode.meanfa_file', 'mean_fa'),
                                       ('outputnode.mergefa_file', 'merge_fa'),
                                       ('outputnode.groupmask', 'group_mask'),
                                       ('outputnode.projectedfa_file', 'projected_fa'),
                                       ('outputnode.skeleton_mask', 'skeleton_mask'),
                                       ('outputnode.distance_map', 'distance_map'),
                                       ('outputall_node.field_list2', 'field_list2')
                                       ])])
    wf.run(plugin='MultiProc')

    return get_tbss_info_dict(fa_maps, subjects_list, output_dir)


def get_tbss_info_dict(fa_maps, subjects_list, tbss_output_dir):
    """Get the output from the TBSS calculations.

    Args:
        fa_maps (dict): mapping subjects to fa map file names
        subjects_list (list of str): the list of subjects names. We provide these to make sure the images are
            all analyzed in the correct order.
        output_dir (str): the output directory

    Returns:
        dict: dictionary mapping subjects to output files. This can be used as input to run_tbss_non_FA.
            The dictionary contains:
                - distance_map: single nii.gz
                - group_mask: single nii.gz
                - mean_fa: single nii.gz
                - merge_fa: single nii.gz
                - projected_fa: single nii.gz
                - skeleton_mask: single nii.gz
                - field_list: dict with per subject field list (the warp file)

    Raises:
        RuntimeError: if one of the output images could not be found.
    """
    single_output_items = ['distance_map', 'group_mask', 'mean_fa', 'merge_fa', 'projected_fa', 'skeleton_mask']
    info_dict = {m: _first_img_in_dir(os.path.join(tbss_output_dir, m)) for m in single_output_items}

    info_dict.update(
        {'field_list': {subject: _first_img_in_dir(os.path.join(tbss_output_dir, 'field_list2', '_fnirt' + repr(i)))
                        for i, subject in enumerate(subjects_list)}}
    )

    return info_dict


def run_tbss_non_FA(tbss_info_dict, maps, subjects_list, output_dir, output_name='non_fa', recalculate=True):
    """Run TBSS non FA on the given subjects.

    Args:
        tbss_info_dict (dict): the information dict from 'run_tbss()'.
        maps (dict): mapping subjects to filenames containing the map to register to the FA skeleton.
        subjects_list (list of str): the list of subjects names. We provide these to make sure the images are
            all analyzed in the correct order.
        output_dir (str): the output directory
        output_name (str): the name of the output file (without extension)

    Returns:
        The full path the output file containing the tracts.
    """
    work_dir = os.path.join(output_dir, '_nipype_work_dir')
    output_file = os.path.join(output_dir, output_name + '.nii.gz')

    if not recalculate and os.path.isfile(output_file):
        return output_file

    maps_list = [maps[subject] for subject in subjects_list]
    field_list = [tbss_info_dict['field_list'][subject] for subject in subjects_list]

    tbss_non_fa = create_tbss_non_FA()
    tbss_non_fa.base_dir = work_dir
    tbss_non_fa.inputs.inputnode.file_list = maps_list
    tbss_non_fa.inputs.inputnode.field_list = field_list
    tbss_non_fa.inputs.inputnode.skeleton_thresh = 0.2
    tbss_non_fa.inputs.inputnode.groupmask = tbss_info_dict['group_mask']
    tbss_non_fa.inputs.inputnode.meanfa_file = tbss_info_dict['mean_fa']
    tbss_non_fa.inputs.inputnode.distance_map = tbss_info_dict['distance_map']
    tbss_non_fa.inputs.inputnode.all_FA_file = tbss_info_dict['merge_fa']

    data_sink = pe.Node(nio.DataSink(), 'DataSink')
    data_sink.inputs.base_directory = output_dir
    data_sink.inputs.parameterization = True
    data_sink.inputs.substitutions = ('_nipype_work_dir/', '')

    wf = Workflow(name='tbss_non_fa_wf', base_dir=work_dir)
    wf.connect([(tbss_non_fa, data_sink, [('outputnode.projected_nonFA_file', '_nonFA_file')])])

    wf.run(plugin='MultiProc')

    shutil.move(_first_img_in_dir(os.path.join(output_dir, '_nonFA_file')),
                os.path.join(output_dir, output_name + '.nii.gz'))
    shutil.rmtree(os.path.join(output_dir, '_nonFA_file'))

    return output_file


def _first_img_in_dir(directory):
    niftis = glob.glob(os.path.join(directory, '*.nii.gz'))

    if not len(niftis):
        raise RuntimeError('No nifti image found in the given directory.')

    return niftis[0]


