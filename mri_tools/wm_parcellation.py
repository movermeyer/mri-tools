import csv
import glob
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
import os

from mri_tools.common import ensure_unzipped

__author__ = 'Robbert Harms'
__date__ = "2015-08-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ROIAggregate(object):

    def aggregate(self, values):
        """Get and return an aggregate of the given values.

        Args:
            values (ndarray): a numpy array from which to calculate the mean and standard deviation.

        Returns:
            list: a list of aggregates
        """

    def get_column_names(self):
        """Get a list of column names with the names of the aggregate columns.

        Returns:
            list: the names of the aggragate columns
        """


class MeanAndStdaggregate(ROIAggregate):

    def aggregate(self, values):
        """A function that returns the mean and standard deviation of the input values.

        Args:
            values (ndarray): a numpy array from which to calculate the mean and standard deviation.

        Returns:
            list: the mean and standard deviation in that order
        """
        return [np.mean(values), np.std(values)]

    def get_column_names(self):
        return ['mean', 'std']


class RegionsInfo(object):

    def __init__(self, wmpm_image, labels_file=None, ignore_regions=(0,)):
        """Information class for the white matter parcellation regions.

        Args:
            wmpm_image (str): the path the to the image file containing the regions. This is expected to be a nifti file
                with regions identified by an integer value.
            labels_file (str): a file containing the labels. Current support is only for XML files from FSL.
            ignore_regions (list of int): list of regions id we wish to ignore. Standard set to exclude the region with
                id '0'. This is general masked data.
        """
        self._ignore_regions = ignore_regions
        self._data = nib.load(wmpm_image).get_data().astype(np.int32)
        self._regions = [r for r in np.unique(self._data) if r not in self._ignore_regions]
        self._labels_reader = labels_file_reader_factory(labels_file)
        self._labels_dict = {region: self._labels_reader.get_label(region) for region in self._regions}
        self._labels_list = [(region, self._labels_reader.get_label(region)) for region in self._regions]

    def get_labels_dict(self):
        """Get the labels per region.

        Returns:
            dict: per region the white matter label
        """
        return self._labels_dict

    def get_labels_region_listing(self):
        """Get the ordered list of (region, label) tuples, ordered by region index

        Returns:
            list: the ordered list of (region, label) tuples
        """
        return self._labels_list

    def get_number_of_regions(self):
        """Get the number of regions.

        Returns:
            int: the number of regions.
        """
        return len(self._regions)

    def get_voxel_indices(self, region_id):
        """Get the voxel indices for the given region.

        Args:
            region_id (int): the region id

        Returns:
            ndarray: the indices of the voxels in that region
        """
        return np.where(self._data == region_id)


def apply_aggregate_to_roi_subjects(csv_region_files, roi_aggregate, output_dir, recalculate=True):
    """Maps a function to every roi and every subject.

    The callback function should accept as input a single ndarray representing all the voxel values for a
    single ROI of a single person. The output of the callback function should be a ndarray with values.

    Args:
        csv_region_files (list): the roi files to apply the function to, should contain the key 'data' per list item.
        roi_aggregate (python function): the function to apply to the rois per subject.
        output_dir (str): the output directory where the csv files are stored
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
        # read comment
        with open(roi, 'r') as f:
            comment = next(f)
            if comment[0] != '#':
                comment = ''

        # read data
        data = np.genfromtxt(roi, delimiter=',')
        data = data[:, 1:]

        # read subject ids. There is probably a more optimal way here. It is late though.
        subjects_list = []
        with open(roi, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                if row[0][0] != '#':
                    subjects_list.append(row[0])

        with open(data_fnames[ind], 'wb') as f:
            f.write(str(comment).encode('latin1'))
            f.write(str('# "subject_id, ' + ', '.join(roi_aggregate.get_column_names()) + '"\n').encode('latin1'))

            output = [roi_aggregate.aggregate(data[i]) for i in range(data.shape[0])]

            for subject_ind, subject_id in enumerate(subjects_list):
                f.write(str('"' + str(subject_id) + '",').encode('latin1'))
                np.savetxt(f, np.array(output[subject_ind])[None], delimiter=",")

    return data_fnames


def write_regions(input_image, subjects_list, wm_regions_info, output_dir, recalculate=True):
    """Extract the voxel information for all the subjects for all the regions.

    This will write a series of csv files for every ROI.

    Args:
        input_image (str): the location of the input image. This is assumed to be a 4d file containing
            per subject (the 4th dimension) a 3d matrix (the first three dimensions) with subject data.
        subjects_list (list of str): the list with the ids of the subjects
        wm_regions_info (RegionsInfo): the regions info class
        output_dir (str): output folder
        recalculate (boolean): if False we return if all the files exist.

    Returns:
        list: the filenames of the data files
    """
    data_fnames = [os.path.join(output_dir, str(ind) + '_data.csv')
                   for ind in range(wm_regions_info.get_number_of_regions())]

    if not recalculate and all(map(os.path.exists, data_fnames)):
        return data_fnames

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        map(os.remove, glob.glob(os.path.join(output_dir, '*')))

    with ensure_unzipped(input_image) as unzipped_image:
        data = nib.load(unzipped_image).get_data()

        for ind, (region_id, label) in enumerate(wm_regions_info.get_labels_dict().items()):
            voxel_indices = wm_regions_info.get_voxel_indices(region_id)
            rois_per_subject = np.array([data[..., subject_ind][voxel_indices] for subject_ind in range(data.shape[3])])

            with open(data_fnames[ind], 'wb') as f:
                f.write(str('# "Region id: ' + str(region_id) + ', label: ' + label + '"' + "\n").encode('latin1'))

                for subject_ind, subject_id in enumerate(subjects_list):
                    f.write(str('"' + str(subject_id) + '",').encode('latin1'))
                    np.savetxt(f, rois_per_subject[subject_ind, :][None], delimiter=",")

    return data_fnames


def labels_file_reader_factory(labels_file):
    """A factory for generating file readers that can read label files.

    Args:
        labels_file (str): the file with the labels of the regions

    Returns:
        LabelsFileReader: the reader for the given file
    """
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
