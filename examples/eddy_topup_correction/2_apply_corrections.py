import os
import nipype
from mri_tools.topup_eddy.common import TopupEddy, create_input_image_information
import multiprocessing
from mri_tools.topup_eddy.nipype.artifacts import all_peb_pipeline, hmc_split
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio

__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

uncorrected_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/1_uncorrected/'
output_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/2_corrected_tmp/'

# items = ((('mbdiffb1k35dirs2mmAPMB2G2', 'mbdiffb1k35dirs2mmPAMB2G2'), '35dir'),
#          (('mbdiffb2k53dirs2mmAPMB2G2', 'mbdiffb2k53dirs2mmPAMB2G2'), '53dir'),
#          (('mbdiffb2k53dirs2mmAPMB2G2TEmax', 'mbdiffb2k53dirs2mmPAMB2G2TEmax'), '53dir_TEmax'),
#          (('mbdiffb1k35dirs2mmAPMB2G2TEmax', 'mbdiffb1k35dirs2mmPAMB2G2TEmax'), '35dir_TEmax'))

#

epi_bname = os.path.join(uncorrected_dir, 'mbdiffb1k35dirs2mmAPMB2G2')
alt_epi_bname = os.path.join(uncorrected_dir, 'mbdiffb1k35dirs2mmPAMB2G2')


def read_epi_params(item_basename):
    with open(item_basename + '.acc_fac_pe.txt', 'r') as f:
        acc_factor = float(f.read())

    with open(item_basename + '.epi_factor.txt', 'r') as f:
        epi_factor = float(f.read())

    with open(item_basename + '.echo_spacing.txt', 'r') as f:
        echo_spacing = float(f.read())

    with open(item_basename + '.phase_enc_dir.txt', 'r') as f:
        phase_encoding = f.read()

    phase_enc_dirs_translate = {'AP': 'y-', 'PA': 'y',
                                'LR': 'x-', 'RL': 'x', 'SD': 'x-', 'DS': 'x',
                                'SI': 'z-', 'IS': 'z', 'HF': 'z-', 'FH': 'z'}

    return {'echospacing': echo_spacing,
            'acc_factor': acc_factor,
            'enc_dir': phase_enc_dirs_translate[phase_encoding],
            'epi_factor': epi_factor}


wf = pe.Workflow(name='correction_workflow', base_dir=output_dir)

correction_wf = all_peb_pipeline(epi_params=read_epi_params(epi_bname), altepi_params=read_epi_params(alt_epi_bname))
correction_wf.inputs.inputnode.in_file = epi_bname + '.nii.gz'
correction_wf.inputs.inputnode.alt_file = alt_epi_bname + '.nii.gz'
correction_wf.inputs.inputnode.in_bval = epi_bname + '.bval'
correction_wf.inputs.inputnode.in_bvec = epi_bname + '.bvec'
correction_wf.base_dir = output_dir

datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = '/tmp/'
wf.connect([(correction_wf, datasink, [('outputnode.out_bvec', 'bvec'),
                                       ('outputnode.out_file', 'image'),
                                       ('outputnode.out_mask', 'mask')])])
wf.run()


#
# uncorrected_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/1_uncorrected/'
# output_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/2_corrected_tmp/'
#
# items = ((('mbdiffb1k35dirs2mmAPMB2G2', 'mbdiffb1k35dirs2mmPAMB2G2'), '35dir'),
#          (('mbdiffb2k53dirs2mmAPMB2G2', 'mbdiffb2k53dirs2mmPAMB2G2'), '53dir'),
#          (('mbdiffb2k53dirs2mmAPMB2G2TEmax', 'mbdiffb2k53dirs2mmPAMB2G2TEmax'), '53dir_TEmax'),
#          (('mbdiffb1k35dirs2mmAPMB2G2TEmax', 'mbdiffb1k35dirs2mmPAMB2G2TEmax'), '35dir_TEmax'))
#
# input_list = [[create_input_image_information(uncorrected_dir, item[0]), item[1]] for item in items]
#
#
# def multi_process_func(args):
#     processor = TopupEddy(*args, output_dir=output_dir)
#     # processor.prepare_topup()
#     # processor.run_topup()
#     processor.prepare_eddy()
#
#     #todo what next? Running eddy or running eddy_correct -dof6 rotating bvecs and then running eddy?
#     # apply_topup_eddy(*to_correct, output_dir=output_dir)
#
# # pool = multiprocessing.Pool()
# # pool.map(multi_process_func, input_list)
# map(multi_process_func, input_list)