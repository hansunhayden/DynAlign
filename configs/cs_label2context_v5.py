CONFIG = {
    "mapillary_context": {
        "road": {"road", "main road", "driving lane", "paved road", "highway", "residential street", "arterial road", "rural road", "city road", "thoroughfare"},
        'drivable fallback': {"drivable terrain", "traffic lane", "vehicle lane", "driveable path", "car lane", "driveable street", "urban roadway", "paved path", "driveable surface", "roadway"},
        "sidewalk": {"sidewalk", "pavement", "footpath", "walkway", "pedestrian path", "side path", "sidewalk pavement", "urban sidewalk", "street sidewalk", "sidewalk lane", "sidewalk area"},
        'terrain': {"non-drivable terrain", "pedestrian area", "park path", "garden path", "bike lane", "footpath", "public plaza", "grass area", "green space", "pedestrian walkway", "non-driveable zone"},
        'person': {"person"},
        'rider': {"rider"},
        'motorcycle': {"motorcycle"},
        'bicycle': {"bycycle"},
        'autorickshaw': {"autorickshaw", "three-wheeler", "tuk-tuk", "auto-rickshaw", "motorized rickshaw", "auto taxi", "rickshaw", "three-wheeled taxi", "auto", "motor tricycle", "auto rickshaw"},
        "car": {"car", "sedan", "hatchback", "coupe", "convertible", "SUV", "sports car", "station wagon", "compact car", "electric car", "luxury car"},
        "truck": {"truck", "pickup truck", "semi-truck", "delivery truck", "dump truck", "fire truck", "tow truck", "box truck", "flatbed truck", "garbage truck", "tanker truck"},
        "bus": {"bus"},
        "train": {"train"},
        "vehicle fallback": {"other vehicles", "train", "tram", "metro", "trolleybus", "light rail", "cable car"},
        "curb": {"curb", "road curb", "sidewalk curb", "curbside", "street curb", "pavement curb", "curb edge", "curb line", "curb boundary", "urban curb", "curb strip"},        # "Curb Cut",  # too detailed
        "wall": {"wall", "barrier wall", "protective wall", "retaining wall", "boundary wall", "perimeter wall", "dividing wall", "sound barrier wall", "security wall", "freestanding wall", "partition wall"},
        "fence": {"fence", "building fence", "road fence", "vehicle separation fence", "pedestrian fence", "safety fence", "boundary fence", "traffic fence", "divider fence", "protective fence", "barrier fence"},
        "guard rail": {"guard rail", "road guard rail", "highway guard rail", "safety guard rail", "traffic guard rail", "barrier guard rail", "roadside guard rail", "protective guard rail", "metal guard rail", "crash barrier", "median guard rail"},
        "billboard": {"billboard", "advertising billboard", "roadside billboard", "digital billboard", "outdoor billboard", "highway billboard", "commercial billboard", "urban billboard", "street billboard", "electronic billboard", "large billboard"},
        'traffic sign': {"traffic sign", "road sign", "highway sign", "street sign", "regulatory sign", "warning sign", "directional sign", "informational sign", "traffic control sign", "signpost", "traffic marker"},
        "traffic light": {"traffic light", "traffic signal", "stoplight", "traffic control light", "intersection signal", "traffic lamp", "signal light", "road signal", "street light", "traffic signal light", "traffic control signal"},
        "pole": {"pole", "street pole", "lamp pole", "traffic pole", "sign pole", "light pole", "support pole", "signal pole", "flag pole", "decorative pole", "banner pole"},
        'obs-str-bar-fallback': {"obstructive structures and barriers", "construction barrier", "roadblock", "traffic cone", "temporary fence", "safety barrier", "barricade", "obstruction", "traffic barricade", "road barrier", "construction zone marker"},
        "building": {"building", "structure", "edifice", "construction", "residential building", "commercial building", "office building", "apartment building", "skyscraper", "public building", "urban building"},
        "bridge": {"road bridge", "footbridge", "pedestrian bridge", "walking bridge", "footpath bridge", "foot crossing", "small bridge", "pedestrian crossing", "walkway bridge", "urban footbridge", "trail bridge"},
        'vegetation': {"vegetation", "urban vegetation", "city greenery", "roadside plants", "street vegetation", "urban foliage", "city flora", "park vegetation", "public greenery", "urban plants", "green space"},
        'sky': {"sky"},
        "unlabeled": {"unlabeled"}
    }
}