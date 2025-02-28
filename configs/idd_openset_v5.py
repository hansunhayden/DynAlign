CONFIG = {
    "mapillary_openset": {
        "road": {'road'},
        # "road": {'road', 'drivable fallback', 'non-drivable fallback', 'obs-str-bar-fallback'},
        "sidewalk": {'sidewalk',
                     'curb',
                     # 'obs-str-bar-fallback',
                     'drivable fallback', 'non-drivable fallback'
                     },
        "building": {'building', 'billboard'},
        "wall": {'wall', 'obs-str-bar-fallback'},
        "fence": {'fence', 'guard rail', 'obs-str-bar-fallback'},
        # "pole": {'pole', 'obs-str-bar-fallback'},
        "pole": {'pole'},
        "traffic light": {'traffic light'},
        "traffic sign": {'traffic sign', 'billboard'},
        "vegetation": {'vegetation',
                        'obs-str-bar-fallback'
                       # 'non-drivable fallback', "not drivable terrain"
                       },
        "terrain": {'non-drivable fallback',
                    'obs-str-bar-fallback',
                    'drivable fallback',
                    # 'vegetation'
                    },
        "sky": {'sky'},
        "person": {'person', },
        "rider": {'rider', },
        "car": {'car', "autorickshaw"},
        "truck": {'truck'},
        "bus": {'bus'},
        "train": {'train',
                  "vehicle fallback"},  # not exist
        "motorcycle": {'motorcycle', },
        "bicycle": {'bicycle'},
        # "unlabeled": {"bridge", 'billboard'}
        "unlabeled": {"bridge"}

    }
}
