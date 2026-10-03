# Automatically extracted empirical priors for Pokemon Champions Regulation M-C VGC
# Sourced strictly from tournament team files in teams/ without any SV VGC assumptions.

import re

CHAMPIONS_MOVE_PRIORS = {
    "absol": {
        "nightslash": 1.0,
        "willowisp": 1.0,
        "closecombat": 1.0,
        "protect": 1.0,
    },
    "aerodactyl": {
        "rockslide": 1.0,
        "dualwingbeat": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
    },
    "altaria": {
        "icebeam": 1.0,
        "willowisp": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
    },
    "annihilape": {
        "shadowclaw": 1.0,
        "closecombat": 1.0,
        "seedbomb": 1.0,
        "icepunch": 1.0,
    },
    "arcaninehisui": {
        "flareblitz": 1.0,
        "protect": 1.0,
        "headsmash": 0.9474,
        "extremespeed": 0.8947,
        "rockslide": 0.1579,
    },
    "archaludon": {
        "protect": 1.0,
        "electroshot": 0.9231,
        "dragonpulse": 0.9231,
        "flashcannon": 0.6154,
        "aurasphere": 0.2308,
        "snarl": 0.2308,
        "thunderbolt": 0.0769,
    },
    "armarouge": {
        "armorcannon": 0.75,
        "expandingforce": 0.75,
        "protect": 0.75,
        "trickroom": 0.75,
        "heatwave": 0.5,
        "psychic": 0.25,
        "calmmind": 0.25,
    },
    "basculegion": {
        "wavecrash": 1.0,
        "lastrespects": 1.0,
        "aquajet": 1.0,
        "protect": 0.6,
        "flipturn": 0.4,
    },
    "blastoisemega": {
        "protect": 1.0,
        "waterspout": 1.0,
        "terrainpulse": 1.0,
        "shellsmash": 1.0,
    },
    "camerupt": {
        "protect": 1.0,
        "heatwave": 1.0,
        "ancientpower": 1.0,
        "earthpower": 1.0,
    },
    "ceruledge": {
        "bitterblade": 1.0,
        "shadowsneak": 0.6667,
        "protect": 0.6667,
        "swordsdance": 0.6667,
        "bulkup": 0.3333,
        "poltergeist": 0.3333,
        "closecombat": 0.3333,
    },
    "charizard": {
        "heatwave": 1.0,
        "weatherball": 1.0,
        "protect": 1.0,
        "ancientpower": 0.6364,
        "hurricane": 0.1818,
        "solarbeam": 0.1818,
    },
    "corviknight": {
        "bravebird": 1.0,
        "powertrip": 1.0,
        "bulkup": 1.0,
        "roost": 1.0,
    },
    "delphox": {
        "heatwave": 1.0,
        "psychic": 1.0,
        "nastyplot": 1.0,
        "protect": 1.0,
    },
    "dragapult": {
        "dracometeor": 1.0,
        "shadowball": 1.0,
        "protect": 1.0,
        "willowisp": 1.0,
    },
    "dragonite": {
        "protect": 0.6,
        "dragonpulse": 0.6,
        "heatwave": 0.6,
        "tailwind": 0.6,
        "extremespeed": 0.4,
        "dragonclaw": 0.2,
        "lowkick": 0.2,
        "scaleshot": 0.2,
        "stompingtantrum": 0.2,
        "rockslide": 0.2,
        "dragondance": 0.2,
    },
    "excadrill": {
        "highhorsepower": 1.0,
        "ironhead": 1.0,
        "protect": 1.0,
        "rockslide": 0.8,
        "swordsdance": 0.2,
    },
    "farigiraf": {
        "trickroom": 1.0,
        "psychic": 0.875,
        "helpinghand": 0.625,
        "protect": 0.625,
        "thunderbolt": 0.375,
        "hypervoice": 0.25,
        "twinbeam": 0.125,
        "grassknot": 0.125,
    },
    "floetteeternal": {
        "dazzlinggleam": 1.0,
        "protect": 1.0,
        "moonblast": 0.8,
        "calmmind": 0.7,
        "lightofruin": 0.3,
        "drainingkiss": 0.2,
    },
    "floettemega": {
        "protect": 1.0,
        "dazzlinggleam": 1.0,
        "drainingkiss": 1.0,
        "calmmind": 1.0,
    },
    "froslass": {
        "blizzard": 1.0,
        "auroraveil": 1.0,
        "protect": 1.0,
        "shadowball": 0.8333,
        "raindance": 0.1667,
    },
    "garchomp": {
        "protect": 0.6667,
        "dragonclaw": 0.5,
        "stompingtantrum": 0.5,
        "powergem": 0.4167,
        "flamethrower": 0.4167,
        "dracometeor": 0.3333,
        "earthquake": 0.3333,
        "rocktomb": 0.25,
        "rockslide": 0.25,
        "dragonpulse": 0.1667,
        "earthpower": 0.1667,
    },
    "gardevoir": {
        "hypervoice": 1.0,
        "expandingforce": 1.0,
        "protect": 1.0,
        "trickroom": 0.6667,
        "calmmind": 0.3333,
    },
    "gardevoirmega": {
        "hypervoice": 1.0,
        "expandingforce": 1.0,
        "calmmind": 1.0,
        "protect": 1.0,
    },
    "gengar": {
        "perishsong": 1.0,
        "shadowball": 1.0,
        "sludgebomb": 1.0,
        "protect": 1.0,
    },
    "gengarmega": {
        "shadowball": 1.0,
        "sludgebomb": 1.0,
        "perishsong": 1.0,
        "protect": 1.0,
    },
    "gholdengo": {
        "makeitrain": 1.0,
        "shadowball": 1.0,
        "protect": 1.0,
        "nastyplot": 0.973,
        "focusblast": 0.027,
    },
    "glimmora": {
        "powergem": 1.0,
        "sludgebomb": 1.0,
        "earthpower": 1.0,
        "spikyshield": 1.0,
    },
    "golisopod": {
        "leechlife": 0.9167,
        "ironhead": 0.8333,
        "protect": 0.5833,
        "suckerpunch": 0.5833,
        "swordsdance": 0.5833,
        "closecombat": 0.1667,
        "liquidation": 0.1667,
        "firstimpression": 0.0833,
        "drillrun": 0.0833,
    },
    "golisopodmega": {
        "leechlife": 1.0,
        "ironhead": 1.0,
        "swordsdance": 1.0,
        "protect": 1.0,
    },
    "grimmsnarl": {
        "partingshot": 1.0,
        "spiritbreak": 1.0,
        "reflect": 1.0,
        "lightscreen": 1.0,
    },
    "hatterene": {
        "expandingforce": 1.0,
        "trickroom": 1.0,
        "dazzlinggleam": 1.0,
        "protect": 0.5,
        "gigadrain": 0.5,
    },
    "hippowdon": {
        "protect": 1.0,
        "highhorsepower": 1.0,
        "yawn": 1.0,
        "icefang": 1.0,
    },
    "hydreigon": {
        "dracometeor": 1.0,
        "darkpulse": 1.0,
        "snarl": 1.0,
        "uturn": 1.0,
    },
    "incineroar": {
        "partingshot": 1.0,
        "fakeout": 1.0,
        "flareblitz": 0.9143,
        "throatchop": 0.4,
        "darkestlariat": 0.2286,
        "protect": 0.1714,
        "snarl": 0.1429,
        "helpinghand": 0.0857,
        "taunt": 0.0286,
        "assurance": 0.0286,
    },
    "indeedee": {
        "expandingforce": 1.0,
        "protect": 1.0,
        "helpinghand": 0.5,
        "trickroom": 0.5,
        "mysticalfire": 0.5,
        "trick": 0.5,
    },
    "indeedeef": {
        "followme": 1.0,
        "helpinghand": 0.9167,
        "psychic": 0.8333,
        "trickroom": 0.6667,
        "protect": 0.3333,
        "terrainpulse": 0.0833,
        "fakeout": 0.0833,
        "mysticalfire": 0.0833,
    },
    "kingambit": {
        "kowtowcleave": 1.0,
        "suckerpunch": 1.0,
        "ironhead": 0.8125,
        "lowkick": 0.625,
        "protect": 0.375,
        "swordsdance": 0.1875,
    },
    "kommoo": {
        "clangingscales": 1.0,
        "protect": 1.0,
        "aurasphere": 0.75,
        "clangoroussoul": 0.75,
        "flamethrower": 0.5,
    },
    "lopunny": {
        "closecombat": 1.0,
        "tripleaxel": 1.0,
        "thunderpunch": 1.0,
        "fakeout": 1.0,
    },
    "lucario": {
        "aurasphere": 1.0,
        "flashcannon": 1.0,
        "detect": 1.0,
        "calmmind": 0.5,
        "darkpulse": 0.25,
        "steelbeam": 0.25,
    },
    "mausholdfour": {
        "populationbomb": 1.0,
        "bite": 1.0,
        "followme": 1.0,
        "protect": 1.0,
    },
    "mawile": {
        "protect": 1.0,
        "ironhead": 1.0,
        "playrough": 1.0,
        "suckerpunch": 1.0,
    },
    "metagross": {
        "psychicfangs": 1.0,
        "protect": 1.0,
        "ironhead": 0.5,
        "stompingtantrum": 0.5,
        "bodypress": 0.5,
        "psychup": 0.5,
    },
    "metagrossmega": {
        "psychicfangs": 1.0,
        "bodypress": 1.0,
        "psychup": 1.0,
        "protect": 1.0,
    },
    "milotic": {
        "protect": 0.75,
        "scald": 0.5,
        "icebeam": 0.5,
        "muddywater": 0.5,
        "hypnosis": 0.5,
        "coil": 0.5,
        "icywind": 0.4167,
        "recover": 0.25,
        "lifedew": 0.0833,
    },
    "ninetalesalola": {
        "blizzard": 1.0,
        "encore": 1.0,
        "icywind": 1.0,
        "freezedry": 1.0,
    },
    "noivern": {
        "dracometeor": 1.0,
        "superfang": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
    },
    "pawmot": {
        "fakeout": 1.0,
        "doubleshock": 1.0,
        "closecombat": 1.0,
        "revivalblessing": 0.75,
        "protect": 0.25,
    },
    "pelipper": {
        "weatherball": 1.0,
        "hurricane": 1.0,
        "tailwind": 0.8182,
        "protect": 0.6364,
        "wideguard": 0.5455,
    },
    "politoed": {
        "weatherball": 1.0,
        "protect": 1.0,
        "perishsong": 0.75,
        "encore": 0.75,
        "muddywater": 0.25,
        "icebeam": 0.25,
    },
    "primarina": {
        "hypervoice": 1.0,
        "moonblast": 1.0,
        "protect": 1.0,
        "calmmind": 0.75,
        "dazzlinggleam": 0.25,
    },
    "pyroarmega": {
        "heatwave": 1.0,
        "overheat": 1.0,
        "scorchingsands": 1.0,
        "protect": 1.0,
    },
    "raichu": {
        "zapcannon": 1.0,
        "focusblast": 1.0,
        "protect": 0.9565,
        "fakeout": 0.7826,
        "encore": 0.2609,
    },
    "rillaboom": {
        "grassyglide": 1.0,
        "fakeout": 1.0,
        "woodhammer": 0.8909,
        "highhorsepower": 0.6182,
        "uturn": 0.3455,
        "protect": 0.1091,
        "superpower": 0.0182,
        "drumbeating": 0.0182,
    },
    "salamence": {
        "hypervoice": 1.0,
        "protect": 1.0,
        "tailwind": 0.9375,
        "dracometeor": 0.6875,
        "flamethrower": 0.1562,
        "doubleedge": 0.125,
        "hyperbeam": 0.0312,
        "dragonpulse": 0.0312,
        "fireblast": 0.0312,
    },
    "salamencemega": {
        "hypervoice": 1.0,
        "protect": 1.0,
        "dragonpulse": 0.5,
        "flamethrower": 0.5,
        "doubleedge": 0.5,
        "tailwind": 0.5,
    },
    "scrafty": {
        "closecombat": 1.0,
        "knockoff": 1.0,
        "fakeout": 1.0,
        "protect": 1.0,
    },
    "sinistcha": {
        "matchagotcha": 1.0,
        "ragepowder": 1.0,
        "trickroom": 1.0,
        "protect": 0.6667,
        "scald": 0.3333,
    },
    "sinistchamasterpiece": {
        "matchagotcha": 1.0,
        "ragepowder": 1.0,
        "lifedew": 1.0,
        "protect": 1.0,
    },
    "sirfetchd": {
        "leafblade": 1.0,
        "detect": 1.0,
        "meteorassault": 1.0,
        "closecombat": 1.0,
    },
    "sneasler": {
        "closecombat": 1.0,
        "protect": 0.8788,
        "direclaw": 0.8485,
        "rockslide": 0.3333,
        "fakeout": 0.3333,
        "rocktomb": 0.2424,
        "throatchop": 0.1212,
        "poisonjab": 0.0909,
        "coaching": 0.0606,
        "gunkshot": 0.0606,
        "quickguard": 0.0303,
    },
    "snorlax": {
        "yawn": 1.0,
        "encore": 1.0,
        "fissure": 1.0,
        "protect": 1.0,
    },
    "squawkabilly": {
        "bravebird": 1.0,
        "partingshot": 1.0,
        "uturn": 1.0,
        "finalgambit": 1.0,
    },
    "staraptor": {
        "closecombat": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
        "bravebird": 0.8571,
        "dualwingbeat": 0.1429,
    },
    "swampert": {
        "wavecrash": 1.0,
        "icepunch": 1.0,
        "earthquake": 1.0,
        "protect": 1.0,
    },
    "swampertmega": {
        "wavecrash": 1.0,
        "icepunch": 1.0,
        "protect": 1.0,
        "earthquake": 0.5,
        "highhorsepower": 0.5,
    },
    "sylveon": {
        "hypervoice": 1.0,
        "hyperbeam": 1.0,
        "quickattack": 1.0,
        "detect": 0.8,
        "protect": 0.2,
    },
    "talonflame": {
        "feint": 1.0,
        "tailwind": 1.0,
        "bravebird": 1.0,
        "flareblitz": 1.0,
    },
    "torkoal": {
        "eruption": 1.0,
        "earthpower": 1.0,
        "heatwave": 0.6667,
        "protect": 0.6667,
        "solarbeam": 0.3333,
        "overheat": 0.3333,
    },
    "toxapex": {
        "banefulbunker": 1.0,
        "toxic": 1.0,
        "infestation": 1.0,
        "wideguard": 1.0,
    },
    "tyranitar": {
        "rockslide": 1.0,
        "knockoff": 1.0,
        "protect": 1.0,
        "lowkick": 0.8333,
        "dragondance": 0.1667,
    },
    "tyranitarmega": {
        "rockslide": 1.0,
        "knockoff": 1.0,
        "lowkick": 1.0,
        "protect": 1.0,
    },
    "venusaur": {
        "sludgebomb": 1.0,
        "leafstorm": 1.0,
        "sleeppowder": 1.0,
        "protect": 1.0,
    },
    "volcarona": {
        "protect": 1.0,
        "flamethrower": 0.5,
        "gigadrain": 0.5,
        "quiverdance": 0.5,
        "overheat": 0.5,
        "strugglebug": 0.5,
        "ragepowder": 0.5,
    },
    "whimsicott": {
        "moonblast": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
        "encore": 0.6667,
        "mistyterrain": 0.3333,
    },
}

CHAMPIONS_ABILITY_PRIORS = {
    "absol": {
        "pressure": 1.0,
    },
    "aerodactyl": {
        "unnerve": 1.0,
    },
    "altaria": {
        "cloudnine": 1.0,
    },
    "annihilape": {
        "defiant": 1.0,
    },
    "arcaninehisui": {
        "rockhead": 1.0,
    },
    "archaludon": {
        "stamina": 1.0,
    },
    "armarouge": {
        "flashfire": 1.0,
    },
    "basculegion": {
        "adaptability": 1.0,
    },
    "blastoisemega": {
        "megalauncher": 1.0,
    },
    "camerupt": {
        "solidrock": 1.0,
    },
    "ceruledge": {
        "flashfire": 1.0,
    },
    "charizard": {
        "blaze": 1.0,
    },
    "corviknight": {
        "mirrorarmor": 1.0,
    },
    "delphox": {
        "blaze": 1.0,
    },
    "dragapult": {
        "infiltrator": 1.0,
    },
    "dragonite": {
        "innerfocus": 0.8,
        "multiscale": 0.2,
    },
    "excadrill": {
        "sandrush": 1.0,
    },
    "farigiraf": {
        "armortail": 1.0,
    },
    "floetteeternal": {
        "flowerveil": 1.0,
    },
    "floettemega": {
        "flowerveil": 0.6667,
        "fairyaura": 0.3333,
    },
    "froslass": {
        "cursedbody": 0.8333,
        "snowcloak": 0.1667,
    },
    "garchomp": {
        "roughskin": 1.0,
    },
    "gardevoir": {
        "trace": 0.6667,
        "synchronize": 0.3333,
    },
    "gardevoirmega": {
        "synchronize": 1.0,
    },
    "gengar": {
        "cursedbody": 1.0,
    },
    "gengarmega": {
        "shadowtag": 1.0,
    },
    "gholdengo": {
        "goodasgold": 1.0,
    },
    "glimmora": {
        "toxicdebris": 1.0,
    },
    "golisopod": {
        "emergencyexit": 1.0,
    },
    "golisopodmega": {
        "emergencyexit": 1.0,
    },
    "grimmsnarl": {
        "prankster": 1.0,
    },
    "hatterene": {
        "magicbounce": 1.0,
    },
    "hippowdon": {
        "sandstream": 1.0,
    },
    "hydreigon": {
        "levitate": 1.0,
    },
    "incineroar": {
        "intimidate": 1.0,
    },
    "indeedee": {
        "psychicsurge": 1.0,
    },
    "indeedeef": {
        "psychicsurge": 0.9167,
        "synchronize": 0.0833,
    },
    "kingambit": {
        "defiant": 1.0,
    },
    "kommoo": {
        "soundproof": 0.75,
        "bulletproof": 0.25,
    },
    "lopunny": {
        "limber": 1.0,
    },
    "lucario": {
        "innerfocus": 1.0,
    },
    "mausholdfour": {
        "technician": 1.0,
    },
    "mawile": {
        "hypercutter": 1.0,
    },
    "metagross": {
        "clearbody": 1.0,
    },
    "metagrossmega": {
        "clearbody": 1.0,
    },
    "milotic": {
        "competitive": 1.0,
    },
    "ninetalesalola": {
        "snowwarning": 1.0,
    },
    "noivern": {
        "infiltrator": 1.0,
    },
    "pawmot": {
        "ironfist": 1.0,
    },
    "pelipper": {
        "drizzle": 1.0,
    },
    "politoed": {
        "drizzle": 1.0,
    },
    "primarina": {
        "liquidvoice": 1.0,
    },
    "pyroarmega": {
        "unnerve": 1.0,
    },
    "raichu": {
        "lightningrod": 1.0,
    },
    "rillaboom": {
        "grassysurge": 1.0,
    },
    "salamence": {
        "intimidate": 1.0,
    },
    "salamencemega": {
        "intimidate": 0.5,
        "aerilate": 0.5,
    },
    "scrafty": {
        "intimidate": 1.0,
    },
    "sinistcha": {
        "hospitality": 1.0,
    },
    "sinistchamasterpiece": {
        "hospitality": 1.0,
    },
    "sirfetchd": {
        "scrappy": 1.0,
    },
    "sneasler": {
        "unburden": 0.9394,
        "poisontouch": 0.0606,
    },
    "snorlax": {
        "thickfat": 1.0,
    },
    "squawkabilly": {
        "intimidate": 1.0,
    },
    "staraptor": {
        "intimidate": 1.0,
    },
    "swampert": {
        "damp": 0.5,
        "torrent": 0.5,
    },
    "swampertmega": {
        "swiftswim": 0.5,
        "torrent": 0.5,
    },
    "sylveon": {
        "pixilate": 1.0,
    },
    "talonflame": {
        "galewings": 1.0,
    },
    "torkoal": {
        "drought": 1.0,
    },
    "toxapex": {
        "regenerator": 1.0,
    },
    "tyranitar": {
        "sandstream": 1.0,
    },
    "tyranitarmega": {
        "sandstream": 1.0,
    },
    "venusaur": {
        "chlorophyll": 1.0,
    },
    "volcarona": {
        "flamebody": 1.0,
    },
    "whimsicott": {
        "prankster": 1.0,
    },
}

CHAMPIONS_ITEM_PRIORS = {
    "absol": {
        "absolitez": 1.0,
    },
    "aerodactyl": {
        "focussash": 1.0,
    },
    "altaria": {
        "habanberry": 1.0,
    },
    "annihilape": {
        "choicescarf": 1.0,
    },
    "arcaninehisui": {
        "focussash": 1.0,
    },
    "archaludon": {
        "leftovers": 1.0,
    },
    "armarouge": {
        "lifeorb": 0.5,
        "leftovers": 0.25,
        "twistedspoon": 0.25,
    },
    "basculegion": {
        "lifeorb": 0.4,
        "choicescarf": 0.4,
        "mysticwater": 0.2,
    },
    "blastoisemega": {
        "blastoisinite": 1.0,
    },
    "camerupt": {
        "cameruptite": 1.0,
    },
    "ceruledge": {
        "colburberry": 0.3333,
        "focussash": 0.3333,
        "grassyseed": 0.3333,
    },
    "charizard": {
        "charizarditey": 1.0,
    },
    "corviknight": {
        "psychicseed": 1.0,
    },
    "delphox": {
        "delphoxite": 1.0,
    },
    "dragapult": {
        "lifeorb": 1.0,
    },
    "dragonite": {
        "dragoninite": 0.6,
        "lifeorb": 0.4,
    },
    "excadrill": {
        "focussash": 1.0,
    },
    "farigiraf": {
        "sitrusberry": 0.375,
        "colburberry": 0.25,
        "grassyseed": 0.125,
        "twistedspoon": 0.125,
        "lifeorb": 0.125,
    },
    "floetteeternal": {
        "floettite": 1.0,
    },
    "floettemega": {
        "floettite": 1.0,
    },
    "froslass": {
        "froslassite": 1.0,
    },
    "garchomp": {
        "garchompitez": 0.5,
        "choicescarf": 0.3333,
        "lifeorb": 0.1667,
    },
    "gardevoir": {
        "gardevoirite": 1.0,
    },
    "gardevoirmega": {
        "gardevoirite": 1.0,
    },
    "gengar": {
        "gengarite": 1.0,
    },
    "gengarmega": {
        "gengarite": 1.0,
    },
    "gholdengo": {
        "lifeorb": 0.9459,
        "whiteherb": 0.027,
        "leftovers": 0.027,
    },
    "glimmora": {
        "glimmoranite": 0.6667,
        "focussash": 0.3333,
    },
    "golisopod": {
        "golisopite": 1.0,
    },
    "golisopodmega": {
        "golisopite": 1.0,
    },
    "grimmsnarl": {
        "lightclay": 1.0,
    },
    "hatterene": {
        "lifeorb": 1.0,
    },
    "hippowdon": {
        "leftovers": 1.0,
    },
    "hydreigon": {
        "focussash": 1.0,
    },
    "incineroar": {
        "sitrusberry": 0.7714,
        "passhoberry": 0.0857,
        "rockyhelmet": 0.0857,
        "chopleberry": 0.0571,
    },
    "indeedee": {
        "focussash": 0.5,
        "choicescarf": 0.5,
    },
    "indeedeef": {
        "rockyhelmet": 0.5833,
        "colburberry": 0.3333,
        "focussash": 0.0833,
    },
    "kingambit": {
        "chopleberry": 0.5625,
        "lifeorb": 0.1875,
        "focussash": 0.125,
        "blackglasses": 0.125,
    },
    "kommoo": {
        "leftovers": 0.75,
        "lifeorb": 0.25,
    },
    "lopunny": {
        "lopunnite": 1.0,
    },
    "lucario": {
        "lucarionitez": 1.0,
    },
    "mausholdfour": {
        "widelens": 1.0,
    },
    "mawile": {
        "mawilite": 1.0,
    },
    "metagross": {
        "metagrossite": 1.0,
    },
    "metagrossmega": {
        "metagrossite": 1.0,
    },
    "milotic": {
        "sitrusberry": 0.5,
        "leftovers": 0.3333,
        "psychicseed": 0.1667,
    },
    "ninetalesalola": {
        "choicescarf": 1.0,
    },
    "noivern": {
        "focussash": 1.0,
    },
    "pawmot": {
        "focussash": 1.0,
    },
    "pelipper": {
        "focussash": 0.5455,
        "sitrusberry": 0.3636,
        "lifeorb": 0.0909,
    },
    "politoed": {
        "sitrusberry": 0.5,
        "mysticwater": 0.25,
        "ejectbutton": 0.25,
    },
    "primarina": {
        "grassyseed": 0.5,
        "leftovers": 0.25,
        "lifeorb": 0.25,
    },
    "pyroarmega": {
        "pyroarite": 1.0,
    },
    "raichu": {
        "raichunitey": 1.0,
    },
    "rillaboom": {
        "miracleseed": 0.7455,
        "sitrusberry": 0.0909,
        "ejectbutton": 0.0727,
        "expertbelt": 0.0364,
        "occaberry": 0.0364,
        "lifeorb": 0.0182,
    },
    "salamence": {
        "salamencite": 1.0,
    },
    "salamencemega": {
        "salamencite": 1.0,
    },
    "scrafty": {
        "focussash": 1.0,
    },
    "sinistcha": {
        "sitrusberry": 0.6667,
        "rockyhelmet": 0.3333,
    },
    "sinistchamasterpiece": {
        "colburberry": 1.0,
    },
    "sirfetchd": {
        "expertbelt": 1.0,
    },
    "sneasler": {
        "grassyseed": 0.4848,
        "whiteherb": 0.303,
        "psychicseed": 0.1515,
        "focussash": 0.0606,
    },
    "snorlax": {
        "leftovers": 1.0,
    },
    "squawkabilly": {
        "choicescarf": 1.0,
    },
    "staraptor": {
        "staraptite": 1.0,
    },
    "swampert": {
        "swampertite": 1.0,
    },
    "swampertmega": {
        "swampertite": 1.0,
    },
    "sylveon": {
        "fairyfeather": 1.0,
    },
    "talonflame": {
        "sharpbeak": 1.0,
    },
    "torkoal": {
        "charcoal": 0.6667,
        "lifeorb": 0.3333,
    },
    "toxapex": {
        "leftovers": 1.0,
    },
    "tyranitar": {
        "tyranitarite": 0.8333,
        "chopleberry": 0.1667,
    },
    "tyranitarmega": {
        "tyranitarite": 1.0,
    },
    "venusaur": {
        "focussash": 0.6667,
        "widelens": 0.3333,
    },
    "volcarona": {
        "grassyseed": 0.5,
        "rockyhelmet": 0.5,
    },
    "whimsicott": {
        "focussash": 1.0,
    },
}

CHAMPIONS_COMMON_SPREADS = {
    "absol": {"nature": "jolly", "evs": {"hp": 11, "atk": 19, "def": 4, "spe": 32}},
    "aerodactyl": {"nature": "jolly", "evs": {"hp": 2, "atk": 32, "spe": 32}},
    "altaria": {"nature": "timid", "evs": {"hp": 32, "def": 5, "spe": 29}},
    "annihilape": {"nature": "adamant", "evs": {"hp": 2, "atk": 32, "spe": 32}},
    "arcaninehisui": {"nature": "jolly", "evs": {"hp": 2, "atk": 32, "spe": 32, "spd": 1, "def": 1}},
    "archaludon": {"nature": "modest", "evs": {"hp": 30, "spa": 4, "spd": 24, "spe": 8, "def": 4}},
    "armarouge": {"nature": "modest", "evs": {"hp": 32, "def": 11, "spa": 28, "spd": 2}},
    "basculegion": {"nature": "adamant", "evs": {"hp": 6, "atk": 25, "spd": 10, "spe": 25, "def": 12}},
    "blastoisemega": {"nature": "modest", "evs": {"hp": 22, "spa": 25, "spe": 19}},
    "camerupt": {"nature": "quiet", "evs": {"hp": 32, "def": 2, "spa": 32}},
    "ceruledge": {"nature": "adamant", "evs": {"hp": 32, "atk": 6, "def": 23, "spe": 4, "spd": 3}},
    "charizard": {"nature": "modest", "evs": {"hp": 27, "def": 16, "spa": 4, "spe": 20}},
    "corviknight": {"nature": "careful", "evs": {"hp": 32, "atk": 18, "def": 2, "spd": 5, "spe": 9}},
    "delphox": {"nature": "timid", "evs": {"hp": 23, "spa": 11, "spe": 32}},
    "dragapult": {"nature": "modest", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "dragonite": {"nature": "modest", "evs": {"hp": 18, "spa": 30, "spe": 26, "spd": 2}},
    "excadrill": {"nature": "adamant", "evs": {"hp": 2, "atk": 32, "spe": 32, "def": 2}},
    "farigiraf": {"nature": "bold", "evs": {"hp": 30, "def": 18, "spd": 18}},
    "floetteeternal": {"nature": "modest", "evs": {"hp": 24, "def": 9, "spa": 18, "spe": 20}},
    "floettemega": {"nature": "modest", "evs": {"hp": 22, "def": 13, "spa": 5, "spe": 28}},
    "froslass": {"nature": "timid", "evs": {"hp": 17, "def": 15, "spe": 26, "spa": 18}},
    "garchomp": {"nature": "modest", "evs": {"hp": 5, "def": 4, "spa": 28, "spd": 3, "spe": 32}},
    "gardevoir": {"nature": "modest", "evs": {"hp": 31, "def": 10, "spa": 24, "spe": 6, "spd": 3}},
    "gardevoirmega": {"nature": "timid", "evs": {"hp": 17, "def": 1, "spa": 23, "spd": 1, "spe": 24}},
    "gengar": {"nature": "modest", "evs": {"hp": 31, "def": 6, "spd": 14, "spe": 15, "spa": 1}},
    "gengarmega": {"nature": "modest", "evs": {"hp": 30, "def": 8, "spa": 5, "spd": 16, "spe": 7}},
    "gholdengo": {"nature": "timid", "evs": {"hp": 11, "def": 5, "spa": 22, "spe": 31, "spd": 2}},
    "glimmora": {"nature": "modest", "evs": {"def": 1, "spa": 31, "spd": 2, "spe": 32, "hp": 2}},
    "golisopod": {"nature": "adamant", "evs": {"hp": 31, "atk": 27, "spd": 5, "spe": 8, "def": 1}},
    "golisopodmega": {"nature": "adamant", "evs": {"hp": 32, "atk": 32, "spd": 2}},
    "grimmsnarl": {"nature": "sassy", "evs": {"hp": 32, "def": 20, "spd": 15}},
    "hatterene": {"nature": "quiet", "evs": {"hp": 32, "def": 2, "spa": 32}},
    "hippowdon": {"nature": "brave", "evs": {"hp": 25, "atk": 8, "def": 1, "spd": 32}},
    "hydreigon": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "incineroar": {"nature": "careful", "evs": {"hp": 32, "def": 12, "spd": 13, "spe": 12, "atk": 3}},
    "indeedee": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "indeedeef": {"nature": "bold", "evs": {"hp": 32, "def": 31, "spd": 2, "spe": 8}},
    "kingambit": {"nature": "adamant", "evs": {"hp": 26, "atk": 25, "spe": 10, "def": 5, "spd": 7}},
    "kommoo": {"nature": "modest", "evs": {"hp": 6, "spa": 30, "spe": 30}},
    "lopunny": {"nature": "jolly", "evs": {"hp": 2, "atk": 32, "spe": 32}},
    "lucario": {"nature": "timid", "evs": {"hp": 27, "spa": 16, "spe": 16, "def": 4, "spd": 7}},
    "mausholdfour": {"nature": "jolly", "evs": {"atk": 32, "spd": 2, "spe": 32}},
    "mawile": {"nature": "brave", "evs": {"hp": 32, "atk": 32, "spd": 2}},
    "metagross": {"nature": "adamant", "evs": {"hp": 24, "atk": 15, "spe": 27}},
    "metagrossmega": {"nature": "jolly", "evs": {"hp": 12, "atk": 15, "def": 11, "spe": 28}},
    "milotic": {"nature": "calm", "evs": {"hp": 32, "def": 17, "spd": 10, "spe": 6, "spa": 17}},
    "ninetalesalola": {"nature": "modest", "evs": {"hp": 1, "def": 1, "spa": 32, "spe": 32}},
    "noivern": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "pawmot": {"nature": "jolly", "evs": {"hp": 2, "atk": 32, "spe": 32}},
    "pelipper": {"nature": "modest", "evs": {"hp": 17, "spa": 24, "spd": 7, "spe": 25, "def": 2}},
    "politoed": {"nature": "calm", "evs": {"hp": 32, "def": 21, "spd": 12, "spe": 3, "spa": 3}},
    "primarina": {"nature": "modest", "evs": {"hp": 25, "def": 14, "spa": 14, "spe": 13}},
    "pyroarmega": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "raichu": {"nature": "timid", "evs": {"hp": 27, "def": 12, "spe": 27, "spd": 1, "spa": 2}},
    "rillaboom": {"nature": "adamant", "evs": {"hp": 25, "atk": 19, "spd": 13, "spe": 10, "def": 3}},
    "salamence": {"nature": "timid", "evs": {"hp": 4, "spa": 28, "spe": 32, "def": 1, "spd": 3}},
    "salamencemega": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "scrafty": {"nature": "impish", "evs": {"hp": 32, "def": 4, "spd": 4, "spe": 26}},
    "sinistcha": {"nature": "relaxed", "evs": {"hp": 32, "def": 4, "spd": 30}},
    "sinistchamasterpiece": {"nature": "bold", "evs": {"hp": 32, "def": 14, "spd": 19, "spe": 1}},
    "sirfetchd": {"nature": "brave", "evs": {"hp": 32, "atk": 32, "spd": 2}},
    "sneasler": {"nature": "adamant", "evs": {"hp": 15, "atk": 24, "spd": 3, "spe": 27, "def": 4}},
    "snorlax": {"nature": "relaxed", "evs": {"hp": 21, "def": 31, "spd": 14}},
    "squawkabilly": {"nature": "jolly", "evs": {"hp": 32, "atk": 2, "spe": 32}},
    "staraptor": {"nature": "jolly", "evs": {"hp": 23, "atk": 12, "spe": 32, "def": 2, "spd": 2}},
    "swampert": {"nature": "adamant", "evs": {"hp": 12, "atk": 30, "spe": 24}},
    "swampertmega": {"nature": "adamant", "evs": {"hp": 9, "atk": 31, "def": 2, "spd": 3, "spe": 24}},
    "sylveon": {"nature": "modest", "evs": {"hp": 19, "def": 12, "spa": 22, "spe": 13}},
    "talonflame": {"nature": "adamant", "evs": {"hp": 7, "atk": 32, "spe": 27}},
    "torkoal": {"nature": "quiet", "evs": {"hp": 32, "spa": 32, "spd": 2, "def": 1}},
    "toxapex": {"nature": "bold", "evs": {"hp": 32, "def": 18, "spd": 16}},
    "tyranitar": {"nature": "adamant", "evs": {"hp": 18, "atk": 20, "def": 1, "spe": 27, "spd": 1}},
    "tyranitarmega": {"nature": "jolly", "evs": {"hp": 17, "atk": 17, "spe": 32}},
    "venusaur": {"nature": "modest", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "volcarona": {"nature": "modest", "evs": {"hp": 21, "def": 15, "spa": 5, "spd": 1, "spe": 24}},
    "whimsicott": {"nature": "timid", "evs": {"hp": 2, "def": 1, "spa": 32, "spe": 32}},
}


def get_champions_move_priors(species: str) -> dict[str, float]:
    """Return move priors for a given species in Champions Regulation M-C."""
    species_clean = re.sub(r"[^a-zA-Z0-9]", "", species).lower()
    return CHAMPIONS_MOVE_PRIORS.get(species_clean, {})

def get_champions_ability_priors(species: str) -> dict[str, float]:
    """Return ability priors for a given species in Champions Regulation M-C."""
    species_clean = re.sub(r"[^a-zA-Z0-9]", "", species).lower()
    return CHAMPIONS_ABILITY_PRIORS.get(species_clean, {})

def get_champions_item_priors(species: str) -> dict[str, float]:
    """Return item priors for a given species in Champions Regulation M-C."""
    species_clean = re.sub(r"[^a-zA-Z0-9]", "", species).lower()
    return CHAMPIONS_ITEM_PRIORS.get(species_clean, {})

def get_champions_common_spread(species: str) -> dict:
    """Return typical nature and EV spread for a given species in Champions Regulation M-C."""
    species_clean = re.sub(r"[^a-zA-Z0-9]", "", species).lower()
    return CHAMPIONS_COMMON_SPREADS.get(species_clean, {"nature": "hardy", "evs": {}})
