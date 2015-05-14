__author__ = 'Robbert Harms'
__date__ = "2015-05-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def coregister(items, input_dir, tmp_dir, output_dir, reference_img=None):
    """Coregisters several corrected dMRI datasets to each other.

    This will output all the volumes corrected separately and one with all items combined.

    Args:
        items (list of str): The list of base names of the items, should contain at least an image, bval and bvec.
        input_dir (str): the input directory
        tmp_dir (str): the temporary files location
        output_dir (str): the location of the output files
        reference_img (str): Which one of the given items to use as the reference image for the co-registration.
    """
    # get the b0's out of all the images
    # calculate the mean b0's
    # create a brain mask using those b0's with options frac=0.35, mask=True
    # coregister the b0's one for one to the reference image
    # apply the transformation to the original volume
    # combine the volumes
    # write output
