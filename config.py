#################################################################
#Config
hs_image_dir='/home/shawn/data/nist/ECODSEdataset/RSdata/hs/'
chm_image_dir='/home/shawn/data/nist/ECODSEdataset/RSdata/chm/'
rgb_image_dir='/home/shawn/data/nist/ECODSEdataset/RSdata/camera/'

training_polygons_dir = '/home/shawn/data/nist/ECODSEdataset/Task1/ITC/'

prediction_polygons_dir = '/home/shawn/data/nist/ECODSEdataset/Task1/predictions/'

image_types_to_load=['hs','chm']

train_plots = ['OSBS_001',
                'OSBS_003',
                'OSBS_006', # missing rgb for this one
                'OSBS_007',
                'OSBS_008',
                'OSBS_009',
                'OSBS_010',
                'OSBS_011',
                'OSBS_014',
                'OSBS_015',
                'OSBS_016',
                'OSBS_017',
                'OSBS_018',
                'OSBS_019',
                'OSBS_025',
                'OSBS_026',
                'OSBS_029',
                'OSBS_030',
                'OSBS_032',
                'OSBS_033',
                'OSBS_034',
                'OSBS_035',
                'OSBS_036',
                'OSBS_037',
                'OSBS_038',
                'OSBS_042',
#                'OSBS_043', # cutoff chm image
#                'OSBS_044', # cutoff chm image
                'OSBS_048',
                'OSBS_051']

test_plots = ['OSBS_002',
            'OSBS_004',
            'OSBS_005',
#            'OSBS_013', # cutoff chm image
            'OSBS_020',
            'OSBS_021',
            'OSBS_027',
            'OSBS_028',
            'OSBS_031',
            'OSBS_039',
            'OSBS_040',
            'OSBS_041',
            'OSBS_050']
