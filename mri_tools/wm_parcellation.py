import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib
import yaml

__author__ = 'Robbert Harms'
__date__ = "2015-08-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def mean_and_std(values):
    """A function that returns the mean and standard deviation of the input values.

    Args:
        values (ndarray): a numpy array from which to calculate the mean and standard deviation.

    Returns:
        list: the mean and standard deviation in that order
    """
    return [np.mean(values), np.std(values)]


def apply_func_to_roi_subjects(csv_region_files, func, output_dir, recalculate=True):
    """Maps a function to every roi and every subject.

    The callback function should accept as input a single ndarray representing all the voxel values for a
    single ROI of a single person. The output of the callback function should be a ndarray with values.

    Args:
        csv_region_files (list): the roi files to apply the function to, should contain the key 'data' per list item.
        func (python function): the function to apply to the rois per subject.
        recalculate (boolean): if False we return if all the output files exist
    """
    data_fnames = [os.path.join(output_dir, str(ind) + '.csv') for ind in range(len(csv_region_files))]

    if not recalculate and all(map(os.path.exists, data_fnames)):
        return data_fnames

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        map(os.remove, glob.glob(os.path.join(output_dir, '*')))

    for ind, roi in enumerate(csv_region_files):
        data = np.genfromtxt(roi, delimiter=',')
        output = [func(data[i]) for i in range(data.shape[0])]
        np.savetxt(data_fnames[ind], output, delimiter=',')

    return data_fnames


def extract_regions(input_image, regions, output_dir, recalculate=True):
    """Extract the voxel information for all the subjects for all the regions.

    This will write a series of csv files for every ROI.

    Args:
        input_image (str): the location of the input image. This is assumed to be a 4d file containing
            per subject (4th dimension) a 3d matrix (first three dimensions) with subject data.
        regions (list): list with regions information. Contains per region a dictionary with
            at least the keys 'voxels', 'label' and 'region_id'.
        output_dir (str): output folder
        recalculate (boolean): if False we return if all the files exist.

    Returns:
        tuple: two lists, one for the filenames of the data files, the other for the filenames of the header files.
    """
    data = nib.load(input_image).get_data()

    data_fnames = [os.path.join(output_dir, str(ind) + '_data.csv') for ind in range(len(regions))]
    header_fnames = [os.path.join(output_dir, str(ind) + '_header.yml') for ind in range(len(regions))]

    if not recalculate and all(map(os.path.exists, data_fnames)) and all(map(os.path.exists, header_fnames)):
        return data_fnames, header_fnames

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        map(os.remove, glob.glob(os.path.join(output_dir, '*')))

    for ind, region_info in enumerate(regions):
        rois_per_subject = np.array([data[..., subject_ind][region_info['voxels']]
                                     for subject_ind in range(data.shape[3])])

        np.savetxt(data_fnames[ind], rois_per_subject, delimiter=",")
        with open(header_fnames[ind], 'w') as outfile:
            outfile.write(yaml.dump({k: region_info[k] for k in ['label', 'region_id']}))

    return data_fnames, header_fnames


def get_regions_info(wmpm_image, labels_file=None, ignore_regions=(0,)):
    """Get information about the white matter parcellation regions.

    Args:
        wmpm_image (str): the path the to the image file containing the regions. This is expected to be a nifti file
            with regions identified by an integer value.
        labels_file (str): a file containing the labels. Current support is only for XML files from FSL.
        ignore_regions (list of int): list of regions id we wish to ignore. Standard set to exclude the region with
            id '0'. This is general masked data.

    Returns:
        list of dict: list of dictionary with information per region.
            The information per region contains:
                - region_id: the original region id
                - label: the label of the region
                - voxels: the locations of the voxels in the region
    """
    data = nib.load(wmpm_image).get_data().astype(np.int32)
    regions = np.unique(data)

    labels_reader = labels_file_reader_factory(labels_file)

    regions_info = []
    for ind, region in enumerate(regions):
        if region not in ignore_regions:
            info = {'region_id': int(region),
                    'label': labels_reader.get_label(region),
                    'voxels': np.where(data == region)}
            regions_info.append(info)

    return regions_info


def labels_file_reader_factory(labels_file):
    extension = os.path.splitext(labels_file)[1].lower()[1:]

    if extension == 'xml':
        return XMLLabelsFileReader(labels_file)
    else:
        raise ValueError('Could not identify the file type of the given settings file.')


class LabelsFileReader(object):

    def get_label(self, region_id):
        """Get the label for the region with the given id

        Args:
            region_id (int): the region identifier

        Returns:
            str: the label of the given region
        """


class XMLLabelsFileReader(LabelsFileReader):

    def __init__(self, labels_file):
        """Reads labels from a FSL XML file.

        This expects to find a number of <label index="{region_id}"> tags.

        Args:
            labels_file (str): the XML file containing the labels
        """
        tree = ET.parse(labels_file)
        root = tree.getroot()
        labels = root.findall("data/label")
        self._labels = {int(l.attrib['index']): l.text for l in labels}

    def get_label(self, region_id):
        return self._labels[region_id]