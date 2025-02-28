CONFIG = {
    "mapillary_openset": {
        "road": {'road', 'sidewalk'},
        # "road": {'road', 'drivable fallback', 'non-drivable fallback', 'obs-str-bar-fallback'},
        "sidewalk": {'sidewalk'},
        "building": {'building', 'wall'},
        "wall": {'wall'},
        "fence": {'fence'},
        # "pole": {'pole', 'obs-str-bar-fallback'},
        "pole": {'pole'},
        "traffic light": {'traffic light'},
        "traffic sign": {'traffic sign'},
        "vegetation": {'vegetation',
                       'terrain'
                       },
        "terrain": {'terrain',
                    'vegetation'
                    },
        "sky": {'sky'},
        "person": {'person', },
        "rider": {'rider', },
        "car": {'car', },
        "truck": {'truck'},
        "bus": {'bus'},
        "train": {'train',},  # not exist
        "motorcycle": {'motorcycle', },
        "bicycle": {'bicycle'},
        "unlabeled": {'unlabeled', }

    }
}
