import os
from nipype.interfaces import fsl
import nipype.pipeline.engine as pe

__author__ = 'Robbert Harms'
__date__ = "2015-08-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def apply_warp(in_file, ref_file, field_file, output_dir, recalculate=False, **kwargs):
    """Wrapper for the Nipype FSL applywarp wrapper.

    Please remember to set the kwarg interp='nn' if you are warping a file with ROI's.

    Args:
        in_file: image to be warped
        ref_file: the reference image
        field_file: file containing warp field (this is the --warp parameter from FSL applywarp)
        output_dir: the output directory
        recalculate (boolean): if we recalculate if the output already exists. Set this to False to easily get the
            output easily.
        **kwargs: extra arguments for the Nipype FSL applywarp

    Returns:
        str: the path to the output image.
    """
    output_file = os.path.join(output_dir, 'warped_' + os.path.basename(in_file))

    if not recalculate and os.path.isfile(output_file):
        return output_file

    aw = fsl.ApplyWarp(**kwargs)
    aw.inputs.in_file = in_file
    aw.inputs.ref_file = ref_file
    aw.inputs.field_file = field_file
    aw.inputs.out_file = output_file

    applywarp = pe.Node(aw, name='applywarp')
    applywarp.base_dir = os.path.join(output_dir, '_nipype_work_dir')
    applywarp.run()

    return output_file
