CONFIG = {
    "mapillary_openset": {
        "road": {'road', 'sidewalk', 'terrain',},
        # "road": {'road', 'drivable fallback', 'non-drivable fallback', 'obs-str-bar-fallback'},
        "sidewalk": {'sidewalk', 'road', 'terrain',},
        "building": {'building'},
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
        "car": {'car', "truck", "train"},
        "truck": {'truck'},
        "bus": {'bus',  "truck", "train"},
        "train": {'train',},  # not exist
        "motorcycle": {'motorcycle', },
        "bicycle": {'bicycle'},

    }
}
