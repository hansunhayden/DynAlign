CONFIG = {
    "mapillary_context": {
        # "road": {
        "Road": {"road", "main road", "driving lane", "paved road", "highway", "residential street", "arterial road", "rural road", "city road", "thoroughfare"},
        "Snow": {"snow", "snow pile", "street snow", "roadside snow", "accumulated snow", "snowbank", "plowed snow", "urban snow", "compacted snow", "snow drift", "snow on pavement"},
        "Sand": {"sand", "sand pile", "street sand", "roadside sand", "piled sand", "sandbank", "accumulated sand", "urban sand", "sand on pavement", "construction sand", "loose sand"},

        "Catch Basin": {"catch basin", "road catch basin", "street catch basin", "roadside catch basin", "storm drain", "drainage basin", "sewer catch basin", "street drain", "gutter catch basin", "road drain", "stormwater basin"},
        "Manhole": {"manhole", "road manhole", "street manhole", "sewer manhole", "manhole cover", "utility manhole", "drainage manhole", "storm drain manhole", "roadside manhole", "underground access", "inspection manhole"},
        "Pothole": {"pothole", "road pothole", "street pothole", "asphalt pothole", "pavement pothole", "highway pothole", "surface pothole", "pothole damage", "roadway pothole", "pothole crater", "pothole on pavement"},
        "Bike Lane": {"bike lane", "marked bike lane", "roadside bike lane", "main road bike lane", "dedicated bike lane", "paved bike lane", "urban bike lane", "bike path", "protected bike lane", "street bike lane", "lane-marked bike lane"},
        # "Crosswalk - Plain",  # too difficult
        # "Parking",  # too hard and area too small
        "Rail Track": {"rail track", "tram rail track", "train rail track", "street rail track", "road rail track", "urban rail track", "tramway track", "railroad track", "commuter rail track", "embedded rail track", "rail track on pavement"},

        # "Service Lane",  # lane beside the main road on highway

        "Lane Marking - Crosswalk": {"crosswalk lane marking", "street crosswalk marking", "pedestrian crosswalk marking", "zebra crossing marking", "road crosswalk marking", "intersection crosswalk marking", "crosswalk lane marking", "painted crosswalk", "crosswalk lines", "crosswalk road marking", "sidewalk crosswalk marking"},
        "Lane Marking - General": {"general lane marking", "road lane marking", "street lane marking", "highway lane marking", "pavement lane marking", "lane divider marking", "traffic lane marking", "lane line marking", "roadway lane marking", "lane boundary marking", "asphalt lane marking"},
        # "Crosswalk Lane Marking",
        # "Road Lane Marking",  # lane marking tend to cover the whole road
        # "General Lane Marking On The Road",
        # "Lane Marking",
        # "Lane Marking On The Road",
        "Water": {"water", "urban water", "river water", "lake water", "city river", "roadside pond", "street water", "urban pond", "city lake", "small urban river", "stormwater"},

        # Sidewalk
        "Sidewalk": {"sidewalk", "pavement", "footpath", "walkway", "pedestrian path", "side path", "sidewalk pavement", "urban sidewalk", "street sidewalk", "sidewalk lane", "sidewalk area"},
        "Curb": {"curb", "road curb", "sidewalk curb", "curbside", "street curb", "pavement curb", "curb edge", "curb line", "curb boundary", "urban curb", "curb strip"},        # "Curb Cut",  # too detailed
        "Pedestrian Area": {"pedestrian area", "street pedestrian area", "pedestrian zone", "pedestrian walkway", "pedestrian street", "urban pedestrian area", "pedestrian plaza", "pedestrian path", "sidewalk pedestrian area", "pedestrian crossing area", "designated pedestrian area"},  # what's the difference.....

        # "building"
        "Building": {"building", "structure", "edifice", "construction", "residential building", "commercial building", "office building", "apartment building", "skyscraper", "public building", "urban building"},
        # "Road Bridge",
        "Bridge": {"road bridge", "footbridge", "pedestrian bridge", "walking bridge", "footpath bridge", "foot crossing", "small bridge", "pedestrian crossing", "walkway bridge", "urban footbridge", "trail bridge"},
        "Phone Booth": {"phone booth", "telephone booth", "public phone booth", "phone kiosk", "payphone booth", "call box", "public telephone", "street phone booth", "outdoor phone booth", "urban phone booth", "phone stand"}, # tend to be chuncks of building, exclude currently for now
        # "Traffic Sign Frame",
        # "Trash Can",
        # "Banner",
        "Billboard": {"billboard", "advertising billboard", "roadside billboard", "digital billboard", "outdoor billboard", "highway billboard", "commercial billboard", "urban billboard", "street billboard", "electronic billboard", "large billboard"},
        "Tunnel": {"tunnel", "road tunnel", "tunnel entrance", "highway tunnel", "urban tunnel", "vehicle tunnel", "tunnel passage", "tunnel opening", "subway tunnel", "underground tunnel", "traffic tunnel"},

        # "wall": {
        "Wall": {"wall", "barrier wall", "protective wall", "retaining wall", "boundary wall", "perimeter wall", "dividing wall", "sound barrier wall", "security wall", "freestanding wall", "partition wall"},
        "Traffic Sign Frame": {"traffic sign frame", "signpost frame", "traffic sign holder", "sign frame", "sign support frame", "road sign frame", "traffic sign structure", "sign mounting frame", "sign frame support", "traffic sign bracket"},
        # "Road Bridge",
        # "Bridge",
        "Trash Can": {"trash can", "street trash can", "public trash can", "roadside trash can", "outdoor trash can", "urban trash can", "sidewalk trash can", "street garbage can", "public waste bin", "street litter bin", "municipal trash can"},
        "Banner": {"banner", "advertising banner", "promotional banner", "street banner", "event banner", "hanging banner", "outdoor banner", "banner sign", "vertical banner", "display banner", "publicity banner"},
        # "Billboard",
        # "Tunnel",

        # "fence":
        "Fence": {"fence", "building fence", "road fence", "vehicle separation fence", "pedestrian fence", "safety fence", "boundary fence", "traffic fence", "divider fence", "protective fence", "barrier fence"},
        "Guard Rail": {"guard rail", "road guard rail", "highway guard rail", "safety guard rail", "traffic guard rail", "barrier guard rail", "roadside guard rail", "protective guard rail", "metal guard rail", "crash barrier", "median guard rail"},
        # "Barrier",  # can be arbitrary type

        # "pole":
        "Pole": {"pole", "street pole", "lamp pole", "traffic pole", "sign pole", "light pole", "support pole", "signal pole", "flag pole", "decorative pole", "banner pole"},
         # "Telegraph Pole",
        "Utility Pole": {"utility pole", "electric pole", "telephone pole", "power pole", "transmission pole", "cable pole", "utility line pole", "utility post", "service pole", "communication pole", "distribution pole"},
        # "Trash Can",
        # "Banner",
        "Street Light": {"street light", "street lamp", "road light", "streetlight", "lamp post", "street lighting", "urban street light", "sidewalk light", "public street light", "street lantern", "street illumination"},

        "Front Side Of Traffic Sign": {"front side of traffic sign", "traffic sign front", "front face of traffic sign", "sign front", "traffic sign face", "front panel of traffic sign", "signboard front", "traffic sign display", "front view of traffic sign", "sign front side", "traffic sign surface"},
        "Back Side Of Traffic Sign": {"back side of traffic sign", "traffic sign back", "back face of traffic sign", "sign back", "rear of traffic sign", "signboard back", "traffic sign reverse", "sign back panel", "back side of sign", "traffic sign rear view", "reverse side of traffic sign"},
        # "Billboard",


        # "traffic light": {
        "Traffic Light": {"traffic light", "traffic signal", "stoplight", "traffic control light", "intersection signal", "traffic lamp", "signal light", "road signal", "street light", "traffic signal light", "traffic control signal"},
        # "Street Light",

        # "traffic sign": {"Front Side Of Traffic Sign",
        #                  "Back Side Of Traffic Sign",
        #                  "Billboard",
        #                  # "Traffic Sign (Front)",
        #                  # "Traffic Sign Frame",
        #                  # "Traffic Sign (Back)"
        #                 },

        # "vegetation": {
        "Vegetation": {"vegetation", "urban vegetation", "city greenery", "street vegetation", "roadside vegetation", "urban plants", "city foliage", "urban flora", "street greenery", "public vegetation", "cityscape vegetation"},
                       # "Mountain",
                       # "Snow",
                       # "Sand",  # never seen before, and maybe on road or other stuff
                       # "Water",
        # "terrain": {
        "Terrain": {"terrain", "urban terrain", "city landscape", "street terrain", "roadside terrain", "urban ground", "cityscape terrain", "urban land", "urban surface", "city terrain", "urban topography"},
        "Mountain": {"mountain", "mountain peak", "mountain range", "mountain slope", "rocky mountain", "highland mountain", "mountain summit", "alpine mountain", "mountain ridge", "forest mountain", "mountain terrain"},
        # "Snow",
        # "Sand",  # never seen before, and maybe on road or other stuff
        # "Water",
        # "sky": {
        "Sky": {"sky"},
        "Person": {"person", },
        # "rider": {
        "Bicyclist": {"bicyclist", "bike rider", "cyclist", "bicycle rider", "bicycle commuter", "mountain biker", "road cyclist",},
        "Motorcyclist": {"motorcyclist", "motorcycle rider", "motorcycle diver", "motorbike rider", "motorcycle commuter", "road motorcyclist", },

        # "car": {
        "Car": {"car", "sedan", "hatchback", "coupe", "convertible", "SUV", "sports car", "station wagon", "compact car", "electric car", "luxury car"},
        # "Other Vehicle",  # too vague
        "Trailer": {"trailer", "utility trailer", "travel trailer", "cargo trailer", "flatbed trailer", "camper trailer", "enclosed trailer", "livestock trailer", "dump trailer"},
        # "Car Mount",
        # "Ego Vehicle"  # indicate the car which is taking the pictures
        "Boat": {"boat", "sailboat", "motorboat", "fishing boat", "speedboat", "yacht", "canoe", "kayak", "pontoon boat", "dinghy", "houseboat"},
        # "truck": {
        "Truck": {"truck", "pickup truck", "semi-truck", "delivery truck", "dump truck", "fire truck", "tow truck", "box truck", "flatbed truck", "garbage truck", "tanker truck"},
        "Caravan": {"caravan", "travel caravan", "camper caravan", "motorhome", "touring caravan", "RV (recreational vehicle)", "fifth-wheel caravan", "pop-up caravan", "teardrop caravan", "static caravan", "off-road caravan"},
        

        "Bus": {"bus"},
        "On Rails": {"on rails"},
        "Motorcycle": {"motorcycle", },
        "Bicycle": {"bicycle"},
        # "unlabeled": {
        # "Bird", "Ground Animal",
        # "Bike Rack",  # too small and difficult
        # "CCTV Camera",  # tooooooo small
        # "Fire Hydrant",  # tooooooo small
        # "Junction Box",  # tooooooo small
        # "Mailbox",  # tooooooo small
        "Bench": {"bench", "street bench", "public bench", "park bench", "sidewalk bench", "outdoor bench", "urban bench", "pavement bench", "city bench", "public seating bench", "roadside bench"},
        # "Wheeled Slow",
        # "Unlabeled"

    }
}
