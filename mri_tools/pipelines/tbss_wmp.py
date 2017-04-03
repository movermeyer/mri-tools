from subprocess import call

import os
import numpy as np
import shutil
import six

from mri_tools.common import multiply_volumes
from mri_tools.registration.common import apply_warp
from mri_tools.registration.register_atlas import register_atlas
from mri_tools.shell_utils import get_fsl_path
from mri_tools.tbss.tbss import run_tbss, get_tbss_info_dict, run_tbss_non_FA
from mri_tools.wm_parcellation import write_regions, apply_aggregate_to_roi_subjects, \
    MeanAndStdaggregate, RegionsInfo

__author__ = 'Robbert Harms'
__date__ = "2015-08-11"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TBSS_WMP(object):

    def __init__(self, fa_maps, output_dir, work_dir, additional_maps, wm_atlas_info=None, region_statistic=None):
        """This class first runs TBSS and then performs White matter parcellation on the white matter tracts.

        The final output is a CSV file with for all the maps the mean and std of each white matter region in the given
        atlas.

        Args:
            fa_maps (dict): {subjects: map file names} dictionary.
                Example: {'subj001': '/output/001/Tensor/FA.nii',
                          'subj002': '/output/002/Tensor/FA.nii',
                          ...
                         }
            output_dir (str): the output directory
            work_dir (str): the work directory
            additional_maps (dict): dictionary mapping map names to dictionaries containing per subject a map name.
                Example: {'Charmed_FR': {'subj001': '/output/001/Charmed/FR.nii',
                                         'subj002': '/output/002/Charmed/FR.nii',
                                         ...
                                        },
                          'Tensor_MD': {'subj001': '/output/001/Tensor/MD.nii',
                                        'subj002': '/output/002/Tensor/MD.nii',
                                        ...
                                       },
                          ...
                         }
            wm_atlas_info (dict): Information about the white matter atlas to use. Requires the keys:
                'fa': the atlas FA map
                'wmpm': the white matter parcellation map
                'labels': the white matter parcellation map labels

                If not given we default to:
                    fa: /fsl/path/to/JHU-ICBM-FA-1mm.nii.gz
                    wmpm: /fsl/path/to/JHU-ICBM-labels-1mm.nii.gz
                    labels: /fsl/path/to/JHU-labels.xml
            region_statistic (ROIAggregate): the aggregate object we will apply on each of the ROIs of each of
                the subjects. If None we calculate the mean and std. of each ROI.
        """
        self._fa_maps = fa_maps
        self._subjects_list = sorted(fa_maps.keys())
        self._output_dir = output_dir
        self._work_dir = work_dir
        self._recalculate = True
        self._additional_maps = additional_maps
        self._region_statistic = region_statistic or MeanAndStdaggregate()

        self._csv_output_dir = os.path.join(self._output_dir, 'csv')
        self._skeleton_maps_dir = os.path.join(self._output_dir, 'skeletons')

        if not os.path.isdir(self._csv_output_dir):
            os.makedirs(self._csv_output_dir)

        if not os.path.isdir(self._skeleton_maps_dir):
            os.makedirs(self._skeleton_maps_dir)

        if wm_atlas_info is None:
            self._wm_atlas_info = {
                'fa': os.path.join(get_fsl_path(), 'data', 'atlases', 'JHU', 'JHU-ICBM-FA-1mm.nii.gz'),
                'wmpm': os.path.join(get_fsl_path(), 'data', 'atlases', 'JHU', 'JHU-ICBM-labels-1mm.nii.gz'),
                'labels': os.path.join(get_fsl_path(), 'data', 'atlases', 'JHU-labels.xml')
            }
        else:
            self._wm_atlas_info = wm_atlas_info

        self._tmp_output_dirs = {
            'tbss': os.path.join(self._work_dir, '1_tbss'),
            'tbss_non_fa': os.path.join(self._work_dir, '2_tbss_non_fa'),
            'atlas_registration': os.path.join(self._work_dir, '3_registered_atlas'),
            'wmpm_registration': os.path.join(self._work_dir, '4_warped_wmpm'),
            # for nmr. 5 see output file below
            'region_csv': os.path.join(self._work_dir, '6_rois'),
            'region_aggregates': os.path.join(self._work_dir, '7_rois_aggregates')
        }

        self._tmp_output_files = {
            'wmpm_skeleton': os.path.join(self._work_dir, '5_wmpm_skeleton_intersection', 'wmpm_skeleton.nii.gz')
        }

    def run_all(self, recalculate=True):
        """Start the pipeline.

        Args:
            recalculate (boolean): if we want to recalculate in the case of existing work output.
        """
        self._recalculate = recalculate

        tbss_info_dict = self._run_tbss()
        tbss_info_dict = self._extract_niftis(tbss_info_dict)
        additional_map_results = self._run_tbss_non_FA(tbss_info_dict)
        # atlas_info_dict = self._register_atlas(tbss_info_dict)
        # warped_wmpm = self._warp_wmpm(tbss_info_dict, atlas_info_dict)
        # wmpm_skeleton = self._create_wmpm_skeleton(tbss_info_dict, warped_wmpm)
        # wm_regions_info = RegionsInfo(wmpm_skeleton, labels_file=self._wm_atlas_info['labels'])
        # skeletons = self._get_wm_skeleton_files(tbss_info_dict, additional_map_results)
        # regions_files_per_map = self._write_region_files(skeletons, wm_regions_info)
        # self._write_output_csv(regions_files_per_map, wm_regions_info)
        # self._copy_skeleton_files(skeletons)

    def _extract_niftis(self, tbss_info_dict):
        """Extract some of the niftis in the tbss info dict and return a dict with the unzipped filenames.

        Since this pipeline uses very large files, having zipped files requires a lot of memory to keep the files in
        memory. Unzipping them before continuing lowers the memory usage.

        Args:
            tbss_info_dict (dict): the TBSS output information dictionary

        Returns:
            dict: the TBSS output information dictionary pointing to the unzipped files
        """
        updated_dict = {}
        for key, value in tbss_info_dict.items():
            if isinstance(value, six.string_types):
                if value.endswith('.nii.gz'):
                    new_filename = value[:-len('.nii.gz')] + '_unzipped.nii'

                    if not os.path.isfile(new_filename):
                        if not os.path.isfile(value[:-3]):
                            call(['gunzip', '-f', '--keep', value])
                        shutil.move(value[:-3], new_filename)

                    updated_dict[key] = new_filename
                else:
                    updated_dict[key] = value
            else:
                updated_dict[key] = value
        return updated_dict

    def _run_tbss(self):
        """Runs TBSS on the FA maps.

        Returns:
            dict: the dictionary with the location of the TBSS results.
        """
        run_tbss(self._fa_maps, self._subjects_list, self._tmp_output_dirs['tbss'], recalculate=self._recalculate)
        return get_tbss_info_dict(self._fa_maps, self._subjects_list, self._tmp_output_dirs['tbss'])

    def _run_tbss_non_FA(self, tbss_info_dict):
        """Run TBSS on non FA maps. Uses the TBSS results.

        Args:
            tbss_info_dict (dict): the results from the function _run_tbss of this class

        Returns:
            dict: mapping the map names of the additional_maps to the skeleton files.
        """
        additional_map_results = {}
        for map_name, subjects in self._additional_maps.items():
            skeleton_fname = run_tbss_non_FA(tbss_info_dict, subjects, self._subjects_list,
                                             self._tmp_output_dirs['tbss_non_fa'],
                                             output_name=map_name, recalculate=self._recalculate)
            additional_map_results.update({map_name: skeleton_fname})

        return additional_map_results

    def _register_atlas(self, tbss_info_dict):
        """Register an atlas to the mean FA of the TBSS results.

        Args:
            tbss_info_dict (dict): the results from the function _run_tbss of this class

        Returns:
            dict: field coefficient and warped image locations
        """
        return register_atlas(self._wm_atlas_info['fa'], tbss_info_dict['mean_fa'],
                              self._tmp_output_dirs['atlas_registration'], recalculate=False)

    def _warp_wmpm(self, tbss_info_dict, atlas_info_dict):
        """Warp the white matter parcellation map using the transformation from the atlas registration.

        Args:
            tbss_info_dict (dict): the results from the function _run_tbss of this class
            atlas_info_dict (dict): the results from the function _register_atlas of this class

        Returns:
            str: the location of the output warped WMPM
        """
        info_dict = apply_warp(self._wm_atlas_info['wmpm'], tbss_info_dict['mean_fa'],
                               atlas_info_dict['fieldcoeff_file'], self._tmp_output_dirs['wmpm_registration'],
                               recalculate=False, interp='nn')
        return info_dict['warped_image']

    def _create_wmpm_skeleton(self, tbss_info_dict, warped_wmpm_file):
        """Get the intersection of the white matter TBSS skeleton and the warped WMPM.

        This multiplies the binary skeleton mask of the TBSS results with the warped white matter parcellation map.

        Args:
            tbss_info_dict (dict): the results from the function _run_tbss of this class
            warped_wmpm_file (str): the results from the function _warp_wmpm of this class

        Returns:
            str: the location of the resulting intersection file.
        """
        multiply_volumes([tbss_info_dict['skeleton_mask'], warped_wmpm_file], self._tmp_output_files['wmpm_skeleton'],
                         recalculate=False)
        return self._tmp_output_files['wmpm_skeleton']

    def _get_wm_skeleton_files(self, tbss_info_dict, additional_map_results):
        """Get a dictionary with the names and the locations of all the skeleton files.

        Returns:
            dict: mapping skeleton names to skeleton files.
        """
        skeletons = {'FA': tbss_info_dict['projected_fa']}
        skeletons.update(additional_map_results)
        return skeletons

    def _write_region_files(self, skeleton_files, wm_regions_info):
        """Write the CSV files with the regions information for all of the skeletons.

        Args:
            skeleton_files (dict):  a dictionary with the names and the locations of all the skeleton files.
            wm_regions_info (RegionsInfo): the regions info class

        Returns:
            dict: mapping map names to list of CSV region files
        """
        regions_files_per_map = {}
        for map_name, skeleton_file in skeleton_files.items():
            output_dir = os.path.join(self._tmp_output_dirs['region_csv'], map_name)
            csv_data_files = write_regions(skeleton_file, self._subjects_list, wm_regions_info, output_dir,
                                           recalculate=False)

            regions_files_per_map.update({map_name: csv_data_files})

        return regions_files_per_map

    def _write_output_csv(self, regions_files_per_map, wm_regions_info):
        """Apply the callback function to the ROIs to calculate the statistics and write the results.

        Args:
            skeleton_files (dict): a dictionary with the names and the locations of all the skeleton files.
            wm_regions_info (RegionsInfo): the regions info class
        """
        for map_name, region_files in regions_files_per_map.items():
            aggregates_csv = apply_aggregate_to_roi_subjects(
                region_files, self._region_statistic,
                os.path.join(self._tmp_output_dirs['region_aggregates'], map_name),
                recalculate=False)

            aggregates_csv_fname = os.path.join(self._csv_output_dir, map_name + '.csv')
            regions = np.hstack([np.genfromtxt(roi, delimiter=',')[:, 1:] for roi in aggregates_csv])

            if not os.path.isfile(aggregates_csv_fname):
                with open(aggregates_csv_fname, 'wb') as f:
                    for subject_ind, subject_id in enumerate(self._subjects_list):
                        f.write(str('"' + str(subject_id) + '",').encode('latin1'))
                        np.savetxt(f, regions[subject_ind][None], delimiter=',')

        self._write_output_labels(wm_regions_info, self._csv_output_dir, recalculate=self._recalculate)

    def _write_output_labels(self, wm_regions_info, output_dir, recalculate=True):
        """Write the file with the regions names.

        Args:
            output_dir (str): the directory to write the column info file to
            wm_regions_info (RegionsInfo): the regions info class
            recalculate (boolean): if you want to recreate this file if it already exists
        """
        output_file = os.path.join(output_dir, 'column_info.txt')

        if not recalculate and os.path.isfile(output_file):
            return output_file

        regions_labels = wm_regions_info.get_labels_region_listing()
        aggregate_names = self._region_statistic.get_column_names()

        rows = ['# column_index,region_id,label,aggregrate_column' + "\n",
                '0,,subject id,\n']

        ind = 1
        for region, label in regions_labels:
            for ag_name in aggregate_names:
                rows.append(str(ind) + ',' + str(region) + ',"' + label + '","' + ag_name + '"' + "\n")
                ind += 1

        with open(output_file, 'w') as f:
            f.writelines(rows)

        return output_file

    def _copy_skeleton_files(self, skeleton_files):
        """Copy all the skeleton files to a output directory.

        Args:
            skeleton_files (dict):  a dictionary with the names and the locations of all the skeleton files.
        """
        for map_name, skeleton_file in skeleton_files.items():
            output_file = os.path.join(self._skeleton_maps_dir, map_name + '.nii.gz')

            if not os.path.exists(output_file) or self._recalculate:
                shutil.copy(skeleton_file, output_file)
