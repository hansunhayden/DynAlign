CONFIG = {
    "mapillary_openset": {
        "road": {
                # "Road",
                # "Main Road",
                # "Driving Lane",
                "Road",
                "Snow",
                "Sand",

                "Catch Basin",
                "Manhole",
                "Pothole",
                # "Bike Lane",
                # "Crosswalk - Plain",  # too difficult
                # "Parking",  # too hard and area too small
                "Rail Track",

                # "Service Lane",  # lane beside the main road on highway

                "Lane Marking - Crosswalk",
                "Lane Marking - General",
                # "Crosswalk Lane Marking",
                # "Road Lane Marking",  # lane marking tend to cover the whole road
                # "General Lane Marking On The Road",
                # "Lane Marking",
                # "Lane Marking On The Road",

                "Water",
        },
        "sidewalk": {"Sidewalk",
                     "Curb",
                     # "Curb Cut",  # too detailed
                     "Pedestrian Area",  # what's the difference.....
                     },
        "building": {"Building",
                     # "Road Bridge",
                     "Bridge",
                     "Phone Booth", # tend to be chuncks of building, exclude currently for now
                     # "Traffic Sign Frame",
                     # "Trash Can",
                     # "Banner",
                     "Billboard",
                     "Tunnel",
                     },
        "wall": {"Wall",
                 "Traffic Sign Frame",
                 # "Road Bridge",
                 "Bridge",
                 "Trash Can",
                 "Banner",
                 "Billboard",
                 "Tunnel",
                 },
        "fence": {"Fence",
                  "Guard Rail",
                  # "Barrier",  # can be arbitrary type
                  },
        "pole": {"Pole",
                 # "Telegraph Pole",
                 "Utility Pole",
                 "Trash Can",
                 "Banner",
                 "Street Light",

                 "Front Side Of Traffic Sign",
                 "Back Side Of Traffic Sign",
                 "Billboard",

                 },
        "traffic light": {"Traffic Light",
                          "Street Light"},

        "traffic sign": {"Front Side Of Traffic Sign",
                         "Back Side Of Traffic Sign",
                         "Billboard",
                         # "Traffic Sign (Front)",
                         # "Traffic Sign Frame",
                         # "Traffic Sign (Back)"
                        },
        "vegetation": {"Vegetation",
                       # "Mountain",
                       "Snow",
                       # "Sand",  # never seen before, and maybe on road or other stuff
                       # "Water",
                       },
        "terrain": {
                    "Terrain",
                    "Mountain",
                    "Snow",
                    "Sand",  # never seen before, and maybe on road or other stuff
                    "Water"},
        "sky": {"Sky", },
        "person": {"Person", },
        "rider": {"Bicyclist", "Motorcyclist",
                  # "Other Rider",
                  },
        "car": {"Car",
                # "Other Vehicle",  # too vague
                "Trailer",
                # "Car Mount",
                # "Ego Vehicle"  # indicate the car which is taking the pictures
                "Boat"
                },
        "truck": {"Truck", "Caravan"},
        "bus": {"Bus"},
        "train": {"On Rails", },
        "motorcycle": {"Motorcycle", },
        "bicycle": {"Bicycle"},
        "unlabeled": {
                      # "Bird", "Ground Animal",
                      # "Bike Rack",  # too small and difficult
                      # "CCTV Camera",  # tooooooo small
                      # "Fire Hydrant",  # tooooooo small
                      # "Junction Box",  # tooooooo small
                      # "Mailbox",  # tooooooo small
                      "Bench",
                      # "Wheeled Slow",
                      # "Unlabeled"
        }

    }
}
