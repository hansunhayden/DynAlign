CONFIG = {
    "mapillary_openset": {
        "road": {'road', 'drivable fallback', 'obstacle objects'},
        "sidewalk": {'sidewalk', 'curb', 'obstacle objects'},
        "building": {'building'},
        "wall": {'wall'},
        "fence": {'fence', 'guard rail'},
        # "pole": {'pole', 'obs-str-bar-fallback'},
        "pole": {'pole', 'obstacle objects'},
        "traffic light": {'traffic light'},
        "traffic sign": {'traffic sign',},
        "vegetation": {'vegetation'},
        "terrain": {'non-drivable fallback', 'obstacle objects'},
        "sky": {'sky'},
        "person": {'person', },
        "rider": {'rider', },
        "car": {'autorickshaw', 'car'},
        "truck": {'truck'},
        "bus": {'bus'},
        "train": {'train'},  # not exist
        "motorcycle": {'motorcycle', },
        "bicycle": {'bicycle'},
        "unlabeled": {"bridge", 'billboard'}
    }
}
