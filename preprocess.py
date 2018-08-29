from __future__ import print_function

import argparse
import os

from preprocessing import csv_utils

PROJECT_ID = 'lepton-maps-207611'
GS_BUCKET = 'gs://lepton'


# def crowdai_preprocess(args):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--cloud', action='store_true')
#     pargs, rem_args = parser.parse_known_args(args)
#     if pargs.cloud:
#         pipeline_args = ('--project {project} '
#                          '--runner DataFlowRunner '
#                          '--staging_location {bucket}/staging '
#                          '--temp_location {bucket}/temp '
#                          '--working {bucket}/data/mapping_challenge '
#                          '--setup_file ./setup.py ').format(project=PROJECT_ID, bucket='gs://lepton').split()
#     else:
#         pipeline_args = ('--project {project} '
#                          '--runner DirectRunner '
#                          '--staging_location {bucket}/staging '
#                          '--temp_location {bucket}/temp '
#                          '--working {bucket}/data/mapping_challenge ').format(project=PROJECT_ID, bucket='.').split()
#     print()
#     print("crowdai_preprocess " + ' '.join(pipeline_args))
#     crowdai.run(pipeline_args)


def combine_csvs(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+')
    parser.add_argument('--main', required=True)
    pargs, _ = parser.parse_known_args(args)
    columns = ("image_id", "file_path_image", "file_path_mask")
    frame = csv_utils.combine_csvs(columns, pargs.files)
    csv_utils.save_to_csv(os.path.join('data', pargs.main), frame, columns, index=False)
    print()
    print("combine csvs")


def data_split(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+')
    parser.add_argument('--folder', required=True)
    # parser.add_argument('-d', '--dev-mode', action='store_true')
    pargs, _ = parser.parse_known_args(args)
    # portion = 0.001 if pargs.dev_mode else 1.0
    try:
        os.makedirs(os.path.join('data', pargs.folder))
    except:
        pass
    csv_utils.train_val_split(pargs.files,
                                   1082,
                                   os.path.join('data', pargs.folder, 'train.csv'),
                                   os.path.join('data', pargs.folder, 'val.csv'))
    print()
    print("train-valid split")


def localize(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+')
    pargs, _ = parser.parse_known_args(args)
    print()
    print("localizing...")
    for filename in pargs.files:
        try:
            csv_utils.create_local_version(filename)
        except:
            print('{} failed'.format(filename))

def datafy(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+')
    pargs, _ = parser.parse_known_args(args)
    print()
    print("datafying...")
    for filename in pargs.files:
        csv_utils.datafy(filename)



PROCESSES = {
    # 'crowdai': crowdai_preprocess,
    'split': data_split,
    'localize': localize,
    'combine': combine_csvs,
    'datafy': datafy
}

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('-p', '--process', required=True, help="Name of process to run")
    main_args, process_args = main_parser.parse_known_args()
    print("Running preprocess for {}".format(main_args.process.upper()))
    PROCESSES[main_args.process](process_args)

