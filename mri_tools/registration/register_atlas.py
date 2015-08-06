import os
from nipype.interfaces import fsl
import nipype.pipeline.engine as pe

__author__ = 'Robbert Harms'
__date__ = "2015-08-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def register_atlas(atlas_file, reference_file, output_dir, recalculate=True):
    """Registers the atlas to your reference image.

    This uses FNIRT to register the atlas to your reference image (for example the mean FA of the TBSS results).

    Args:
        atlas_file (str): the location of the atlas file we will register to the reference image.
        reference_file (str): the location of the reference image.
        output_dir (str): where to write the output
        recalculate (boolean): if we recalculate if the output already exists. Set this to False to easily get the
            results dictionary.

    Returns:
        dict: a dictionary with the filenames of the results. All files are in the output dir. Dir content:
            fieldcoeff_file: the path to the field coefficients file. To be used input to the 'warp' parameter of
                'applywarp'.
            warped_image: the warped image.
    """
    fieldcoeff_file = os.path.join(output_dir, 'fieldcoeff.nii.gz')
    warped_image = os.path.join(output_dir, 'warped.nii.gz')
    results_dict = {'fieldcoeff_file': fieldcoeff_file, 'warped_image': warped_image}

    if not recalculate:
        if all(os.path.isfile(f) for f in [fieldcoeff_file, warped_image]):
            return results_dict

    fnirt = pe.Node(fsl.FNIRT(ref_file=reference_file, in_file=atlas_file, fieldcoeff_file=fieldcoeff_file,
                              warped_file=warped_image),
                    name='fnirt')
    fnirt.base_dir = os.path.join(output_dir, '_nipype_work_dir')
    fnirt.run()

    return results_dict
