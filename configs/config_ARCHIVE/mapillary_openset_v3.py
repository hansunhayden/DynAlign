CONFIG = {
    "mapillary_openset": {
        "road": {
                "Road",
                # "Bridge", "Snow",
                "Catch Basin",
                "Manhole",
                "Bike Lane",
                # "Crosswalk - Plain",  # too difficult
                "Parking",
                "Rail Track",
                # "Road",
                # "Service Lane",  # lane beside the main road on highway
                # "Lane Marking - Crosswalk", "Lane Marking - General",
                "Crosswalk Lane Marking",
                "Lane Marking On The Road", # lane marking tend to cover the whole road
                "Pothole",
        },
        "sidewalk": {"Sidewalk", "Curb",
                     # "Curb Cut",  # too detailed
                     "Pedestrian Area",  # what's the difference.....
                     },
        "building": {"Building",
                     # "Phone Booth", # tend to be chuncks of building, exclude currently for now
                     },
        "wall": {"Wall", },
        "fence": {"Fence", "Guard Rail",
                  # "Barrier",  # can be arbitrary type
                  },
        "pole": {"Pole", "Utility Pole", },
        "traffic light": {"Traffic Light", },
        "traffic sign": {"Traffic Sign (Front)", "Traffic Sign Frame", "Traffic Sign (Back)"},
        "vegetation": {"Vegetation"},
        "terrain": {"Mountain",
                    # "Sand",  # never seen before, and maybe on road or other stuff
                    "Terrain"},
        "sky": {"Sky", },
        "person": {"Person", },
        "rider": {"Bicyclist", "Motorcyclist", "Other Rider", },
        "car": {"Car",
                # "Other Vehicle",  # too vague
                "Trailer", "Car Mount",
                # "Ego Vehicle"  # indicate the car which is taking the pictures
                },
        "truck": {"Truck", "Caravan"},
        "bus": {"Bus"},
        "train": {"On Rails", },
        "motorcycle": {"Motorcycle", },
        "bicycle": {"Bicycle"},
        "unlabeled": {"Bird", "Ground Animal", "Bike Rack", "CCTV Camera", "Fire Hydrant", "Junction Box", "Mailbox",
                      "Trash Can", "Boat", "Banner", "Bench", "Billboard", "Street Light", "Wheeled Slow",
                      "Tunnel", "Water", "Unlabeled"}

    }
}
