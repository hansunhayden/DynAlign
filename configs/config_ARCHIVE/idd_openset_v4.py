CONFIG = {
    "mapillary_openset": {
        # "road": {'road', 'drivable fallback', 'obstacle objects'},
        "road": {'urban road',
                 'rural path', 'drivable terrain',
                 'obstacle objects', 'barriers',
                 "not drivable terrain"},
        "sidewalk": {'sidewalk',
                     'curb', 'sidewalk edge',
                     'obstacle objects', 'barriers',
                     "not drivable terrain",
                     'drivable terrain'},
        "building": {'building', 'billboard'},
        "wall": {'wall'},
        "fence": {'fence', 'guard rail', 'obstacle objects', 'barriers'},
        # "pole": {'pole', 'obs-str-bar-fallback'},
        "pole": {'pole', 'obstacle objects', 'barriers'},
        "traffic light": {'traffic light'},
        "traffic sign": {'traffic sign', 'billboard'},
        "vegetation": {'vegetation',
                       # 'non-drivable fallback', "not drivable terrain"
                       },
        "terrain": {'non-drivable fallback', "not drivable terrain",
                    'obstacle objects', 'barriers',
                    'drivable terrain',
                    'vegetation'},
        "sky": {'sky'},
        "person": {'person', },
        "rider": {'rider', },
        "car": {'car', "autorickshaw"},
        "truck": {'truck'},
        "bus": {'bus'},
        "train": {'train'},  # not exist
        "motorcycle": {'motorcycle', },
        "bicycle": {'bicycle'},
        # "unlabeled": {"bridge", 'billboard'}
        "unlabeled": {"bridge"}

    }
}
