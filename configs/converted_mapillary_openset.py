CONFIG = {
    "mapillary_openset": {
        "Road": {
                # "Bridge", "Snow",
                "Catch Basin", "Manhole", "Bike Lane",
                # "Crosswalk - Plain",  # too difficult
                "Parking", "Rail Track",
                "Road",
                # "Service Lane",  # lane beside the main road on highway
                # "Lane Marking - Crosswalk", "Lane Marking - General",
                "Crosswalk Lane Marking", "Lane Marking On The Road", # lane marking tend to cover the whole road
                "Pothole",
        },
        "Sidewalk": {"Curb",
                     # "Curb Cut",  # too detailed
                     "Pedestrian Area",  # what's the difference.....
                     "Sidewalk", },
        "Building": {"Building",
                     # "Phone Booth", # tend to be chuncks of building, exclude currently for now
                     },
        "Wall": {"Wall", },
        "Fence": {"Guard Rail",
                  # "Barrier",  # can be arbitrary type
                  "Fence", },
        "Pole": {"Pole", "Utility Pole", },
        "Traffic Light": {"Traffic Light", },
        "Traffic Sign (Front)": {"Traffic Sign Frame", "Traffic Sign (Back)", "Traffic Sign (Front)", },
        "Vegetation": {"Vegetation"},
        "Terrain": {"Mountain",
                    # "Sand",  # never seen before, and maybe on road or other stuff
                    "Terrain"},
        "Sky": {"Sky", },
        "Person": {"Person", },
        "Bicyclist": {"Bicyclist", "Motorcyclist", "Other Rider", },
        "Car": {"Car",
                # "Other Vehicle",  # too vague
                "Trailer", "Car Mount",
                # "Ego Vehicle"  # indicate the car which is taking the pictures
                },
        "Truck": {"Caravan", "Truck"},
        "Bus": {"Bus"},
        "On Rails": {"On Rails", },
        "Motorcycle": {"Motorcycle", },
        "Bicycle": {"Bicycle"},
        "unlabeled": {"Bird", "Ground Animal", "Bike Rack", "CCTV Camera", "Fire Hydrant", "Junction Box", "Mailbox",
                      "Trash Can", "Boat", "Banner", "Bench", "Billboard", "Street Light", "Wheeled Slow",
                      "Tunnel", "Water", "Unlabeled"}

    }
}
