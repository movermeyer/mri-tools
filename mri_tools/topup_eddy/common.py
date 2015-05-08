import os
import re
import numpy as np
import nibabel as nib
import subprocess
import shutil
from mri_tools.shell_utils import get_fsl_command

__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TopupEddy(object):

    def __init__(self, images_info, output_name, output_dir=None, _joined_output=True):
        """Apply topup on the main image by using the extra images.

        Args:
            images_info (list of InputImageInformation): The list with images information objects
            output_name (str): The basename for all the output files
            output_dir (str): The output folder. If not set, the location of the image of the first image info is used.
            joined_output (boolean): If we want to join the input files to produce one result, or not.
                If True the output is the concatenation of the input images and b-vectors.
                If False the output is only the image and b-vectors of the first image.
        """
        self._topup_command = get_fsl_command('topup')

        self._images_info = images_info
        self._output_dir = output_dir
        self._output_name = output_name
        self._joined_output = _joined_output
        self._output_basename = os.path.join(self._output_dir, self._output_name)

        if self._output_dir is None:
            self._output_dir = os.path.dirname(self._images_info[0].image_fname)

        self._unweighted_vols_fname = self._output_basename + '_unweighted.nii.gz'
        self._acq_params_fname = self._output_basename + '_acqparams.txt'

        self._topup_output_fname = self._output_basename + '_topup'
        self._topup_iout_fname = self._output_basename + '_unweighted_corrected.nii.gz'

        self._mean_unweighted_fname = self._output_basename + '_unweighted_corrected_mean.nii.gz'
        self._mask_fname = self._output_basename + '_mask.nii.gz'

    def prepare_topup(self):
        """Prepare the volumes for running topup. This will create the files with the unweighted images."""
        combine_unweighted_volumes(self._images_info, self._unweighted_vols_fname)
        create_acq_params_file(self._images_info, self._acq_params_fname)

    def run_topup(self):
        """Run topup on the unweighted volumes."""
        bash_command = self._topup_command + \
                       " --imain=" + self._unweighted_vols_fname + \
                       " --datain=" + self._acq_params_fname + \
                       " --out=" + self._topup_output_fname + \
                       " --iout=" + self._topup_iout_fname + \
                       " --config=b02b0.cnf"
        subprocess.call(bash_command.split())

    def prepare_eddy(self):
        """Prepares the input for the eddy routine.

        This creates the brain mask for eddy and create the index file indexing the acq_params file.
        Next, it joins the input images and bvecs/bvals if joined_output was set in the constructor.
        """
        # bash_command = 'fslmaths {0} -Tmean {1}'.format(self._topup_iout_fname, self._mean_unweighted_fname)
        # subprocess.call(bash_command.split())
        #
        # create_binary_mask(self._mean_unweighted_fname, self._mask_fname)

        if self._joined_output:
            combine_write_image_info(self._images_info, self._output_basename)
        else:
            shutil.copyfile(self._images_info[0].image_fname, self._output_basename + '.nii.gz')
            shutil.copyfile(self._images_info[0].bval_fname, self._output_basename + '.bval')
            shutil.copyfile(self._images_info[0].bvec_fname, self._output_basename + '.bvec')

        #todo  next step is to create the index file

        pass

    def run_eddy(self):
        pass

    def get_output_image_information(self):
        """Get the output image information object."""
        pass


def create_input_image_information(basedir, basenames):
    """Create and return image information objects for all images in the given list of basenames.

    Args:
        basedir (str): The directory containing all the images, bvals, bvecs etc.
        basenames (list of str): List with basenames for files to construct ImageInformation objects out of.

    Returns:
        list of ImageInformation: List with the image information objects.
    """
    image_create = lambda e: InputImageInformation.create_from_files(os.path.join(basedir, e))
    return [image_create(item) for item in basenames]


class InputImageInformation(object):

    def __init__(self, name, image_fname, bval_fname, bvec_fname, phase_enc_dir, read_out_time):
        """Create a structure that holds the information for a single image.

        Args:
            name (str): The name of this image, used to identify this image
            image_fname (str): The image filename
            bval_fname (str): The bval filename
            bvec_fname (str): The bvec filename
            phase_enc_dir (str): A single word indicating the read out direction. Can be one of
                ['AP', 'PA', 'LR', 'RL', 'DS', 'SD', 'SI', 'IS', 'HF', 'FH']
                where (A: Anterior, P: Posterior), (L: Left, R: Right), (D: Dexter, S: Sinister),
                (S: Superior, I: Inferior), (H: Head, F: Feet)
            read_out_time (double): The read out time for this image.

        Attributes:
            name (str): The name of this image, used to identify this image
            image_fname (str): The image filename
            bval_fname (str): The bval filename
            bvec_fname (str): The bvec filename
            phase_enc_dir (str): A single word indicating the read out direction. Can be one of
                ['AP', 'PA', 'LR', 'RL', 'DS', 'SD', 'SI', 'IS', 'HF', 'FH']
                where (A: Anterior, P: Posterior), (L: Left, R: Right), (D: Dexter, S: Sinister),
                (S: Superior, I: Inferior), (H: Head, F: Feet)
            read_out_time (double): The read out time for this image.
        """
        self.name = name
        self.image_fname = image_fname
        self.bval_fname = bval_fname
        self.bvec_fname = bvec_fname
        self.phase_enc_dir = phase_enc_dir
        self.read_out_time = read_out_time

    @classmethod
    def create_from_files(cls, base_path):
        """Create an ImageInformation container from a single base path.

        Args:
            base_path (str): The directory and basename of all the necessary information files.
                The base_path should look like for example: '/experiments/first/base_name'
                This path is searched for the following files:
                    <base_name>.nii.gz: the image
                    <base_name>.bval: the bvals
                    <base_name>.bvec: the bvecs
                    <base_name>.phase_enc_dir.txt: file with the phase enc. direction in it (single word)
                    <base_name>.read_out_time.txt: file witht the read out time in it (single value)

        Returns:
            ImageInformation: A new image information with all the paths setup correctly.

        Raises:
            ValueError: If the base path does not contain all the necessary files.
        """
        for extension in ['nii.gz', 'bval', 'bvec', 'phase_enc_dir.txt', 'read_out_time.txt']:
            if not os.path.isfile(base_path + '.' + extension):
                raise ValueError('For this basename the file with extension {} could not be found.'.format(extension))

        image_fname = base_path + '.nii.gz'
        bval_fname = base_path + '.bval'
        bvec_fname = base_path + '.bvec'
        name = os.path.basename(base_path)

        with open(base_path + '.phase_enc_dir.txt', 'r') as f:
            phase_enc_dir = f.read()

        with open(base_path + '.read_out_time.txt', 'r') as f:
            read_out_time = float(re.sub(r'[^0-9.]', '', f.read()))

        return InputImageInformation(name, image_fname, bval_fname, bvec_fname, phase_enc_dir, read_out_time)


def find_unweighted_indices(bvec_file, bval_file, column_based='auto', bval_scale='auto', unweighted_threshold=25e6):
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
        bval_scale (float): The amount by which we want to scale (multiply) the b-values. The default is auto,
            this checks if the b-val is lower then 1e4 and if so multiplies it by 1e6.
            (sets bval_scale to 1e6 and multiplies), else multiplies by 1.
        unweighted_threshold (double): The threshold under which we call a direction unweighted (in m^2)

    Returns:
        list of int: A list of indices to the unweighted volumes.
    """
    bvec = np.genfromtxt(bvec_file)
    bval = np.expand_dims(np.genfromtxt(bval_file), axis=1)

    if bval_scale == 'auto' and bval[0, 0] < 1e4:
        bval *= 1e6
    else:
        bval *= bval_scale

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


def combine_unweighted_volumes(image_info_lists, output_fname):
    """Combine all the unweighted volumes of the given volumes.

    This will not return any images, it will write the image to the given file name.

    Args:
        image_info_list (list of ImageInformation): The list with the image information files
        output_fname (str): The output filename
    """
    unweighted_volume = None
    header = None

    for image_info in image_info_lists:
        unweighted_ind = find_unweighted_indices(image_info.bvec_fname, image_info.bval_fname)
        image_nifti = nib.load(image_info.image_fname)
        image = image_nifti.get_data()
        header = image_nifti.get_header()

        image_unweighted = image[..., unweighted_ind]

        if unweighted_volume is None:
            unweighted_volume = image_unweighted
        else:
            unweighted_volume = np.concatenate([unweighted_volume, image_unweighted], axis=3)

    nib.Nifti1Image(unweighted_volume, None, header).to_filename(output_fname)


def create_acq_params_file(image_info_lists, output_fname):
    """Create the acquisition parameters file.

    This is the file topup needs to process the unweighted volumes.

    Args:
        image_info_list (list of ImageInformation): The list with the image information files
        output_fname (str): The output filename
    """
    phase_enc_dirs_translate = {'AP': '0 -1 0', 'PA': '0 1 0',
                                'LR': '-1 0 0', 'RL': '1 0 0', 'DS': '1 0 0', 'SD': '-1 0 0',
                                'SI': '0 0 -1', 'IS': '0 0 1', 'HF': '0 0 -1', 'FH': '0 0 1'}

    lines = []
    for image_info in image_info_lists:
        unweighted_ind = find_unweighted_indices(image_info.bvec_fname, image_info.bval_fname)
        line = phase_enc_dirs_translate[image_info.phase_enc_dir] + ' ' + repr(image_info.read_out_time)
        for e in unweighted_ind:
            lines.append(line)

    with open(output_fname, 'w') as f:
        f.write("\n".join(lines))


def create_binary_mask(input_fname, output_fname):
    """Create a binary mask out of the input image and store in the output image.

    Args:
        input_fname (str): The input filename
        output_fname (str): The output filename
    """
    if input_fname[-len('.nii.gz'):] != '.nii.gz':
        input_fname += '.nii.gz'

    if output_fname[-len('.nii.gz'):] != '.nii.gz':
        output_fname += '.nii.gz'

    bash_command = 'bet {0} {1} -m'.format(input_fname, output_fname)
    subprocess.call(bash_command.split())

    os.remove(output_fname)
    os.rename(output_fname[0:-len('.nii.gz')] + '_mask.nii.gz', output_fname)


def combine_write_image_info(images_info, output_basename):
    """Combine the images, bvals and bvecs of the given input image informations.

    Args:
        images_info (list of InputImageInformation): The list with images to concatenate
        output_basename (str): The basename of the output files. Something like: /tmp/experiment/35dir
            This will then write /tmp/experiment/35dir.bval, /tmp/experiment/35dir.bvec
            and /tmp/experiment/35dir.nii.gz
    """
    bvals = []
    bvecs = []
    images = []
    header = None

    for image_info in images_info:
        bvecs.append(np.genfromtxt(image_info.bvec_fname))
        bvals.append(np.genfromtxt(image_info.bval_fname))

        nib_container = nib.load(image_info.image_fname)
        if header is None:
            header = nib_container.get_header()
        images.append(nib_container.get_data())

    np.savetxt(output_basename + '.bvec', np.concatenate(bvecs, axis=1))
    np.savetxt(output_basename + '.bval', np.expand_dims(np.concatenate(bvals), axis=1).transpose())

    combined_image = np.concatenate(images, axis=3)
    nib.Nifti1Image(combined_image, None, header).to_filename(output_basename + '.nii.gz')