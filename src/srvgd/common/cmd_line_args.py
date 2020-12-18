"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 17, 2020

PURPOSE: The type is a place to put reuseable argparse defs.

NOTES:

TODO:
"""
import argparse


def get_cmd_line_args_for_datasets():
    parser = argparse.ArgumentParser()

    parser.add_argument('--multiple_scaling', action='store_true',
                        help='Scale each y-value with all y-values that '
                             'share the same x-value. Thus, y-values '
                             'at different x-values may scale differently.'
                             'If both multiple_scaling and consistent_scaling '
                             'are false, no scaling is performed.')
    parser.add_argument('--consistent_scaling', action='store_true',
                        help='Scale each y-value with the same scale. If both '
                             'multiple_scaling and consistent_scaling are false, '
                             'no scaling is performed.')

    args = parser.parse_args()
    assert not (args.multiple_scaling and args.consistent_scaling), 'cannot use --multiple_scaling and --consistent_scaling simultaneously'

    # get dataset_name from args
    if args.multiple_scaling:
        dataset_name = 'multiple_scaling'
    elif args.consistent_scaling:
        dataset_name = 'consistent_scaling'
    else:
        dataset_name = 'no_scaling'

    return args, dataset_name
