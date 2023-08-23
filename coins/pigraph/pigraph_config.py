import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
from configuration.config import *

NUM_JOINTS = 55  # number of joints for each skeleton

# dict for interaction sequences of each type, key is defined as verb-noun pair or pairs
# value is a list of triplets of format (recording name, start frame, end frame), frame number starts from 1, both start and end frame included
interaction_sequences = {}
interaction_sequences['sit-bed'] = [
                                    # ('MPH8_00034_01', 1000, 1025),
                                    # ('MPH8_00034_01', 1130, 1291),
                                    ('MPH16_00157_01', 1405, 1650),
                                    ('MPH112_00150_01', 213, 507),
                                    ]
interaction_sequences['sit-chair'] = [('MPH1Library_00034_01', 200, 610),
                                      ('MPH1Library_00034_01', 1056, 1220),
                                      ('MPH1Library_00034_01', 1480, 1700),
                                      # ('MPH8_00034_01', 1820, 1960),
                                      # ('MPH8_00034_01', 2770, 2800),
                                      ('MPH11_00034_01', 357, 430),
                                      ('MPH16_00157_01', 366, 420),
                                      ('N3Library_03301_02', 453, 500),
                                      ('N3Office_00034_01', 733, 980),
                                      ('N3Office_00034_01', 1573, 1815),
                                      ('N3OpenArea_00157_01', 208, 240),
                                      ('Werkraum_03403_01', 520, 605),
                                      # ('Werkraum_03403_01', 928, 980),
                                      ('Werkraum_03516_01', 596, 810),
                                      ('Werkraum_03516_01', 1740, 1990),
                                      ]
interaction_sequences['lie-sofa'] = [
                                    # ('BasementSittingBooth_00145_01', 1020, 1192),
                                    #  ('BasementSittingBooth_00145_01', 1318, 1500),
                                    #  ('MPH8_00034_01', 2190, 2290),
                                    #  ('MPH11_00034_01', 1620, 1785), # the input from siwei of this sequence is bad
                                    #  ('N0Sofa_00034_01', 545, 565),
                                     ('N0Sofa_00034_01', 822, 855),
                                     ]
interaction_sequences['stand-table'] = [
    # ('BasementSittingBooth_00145_01', 1695, 1780),
('MPH11_00034_01', 700, 750),
                                       ]
