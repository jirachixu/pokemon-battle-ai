# Automatically extracted empirical priors for Pokemon Champions Regulation M-C VGC
import re

CHAMPIONS_MOVE_PRIORS = {
    "aerodactyl": {
        "rockslide": 1.0,
        "dualwingbeat": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
    },
    "alcremie": {
        "decorate": 1.0,
        "helpinghand": 1.0,
        "dazzlinggleam": 1.0,
        "mysticalfire": 1.0,
    },
    "altaria": {
        "icebeam": 1.0,
        "willowisp": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
    },
    "annihilape": {
        "closecombat": 1.0,
        "uturn": 1.0,
        "shadowclaw": 0.6667,
        "icepunch": 0.6667,
        "rockslide": 0.3333,
        "phantomforce": 0.3333,
    },
    "arcaninehisui": {
        "flareblitz": 1.0,
        "protect": 1.0,
        "headsmash": 0.9333,
        "extremespeed": 0.8,
        "rockslide": 0.2667,
    },
    "archaludon": {
        "electroshot": 1.0,
        "dragonpulse": 1.0,
        "protect": 1.0,
        "flashcannon": 0.48,
        "snarl": 0.32,
        "aurasphere": 0.2,
    },
    "armarouge": {
        "expandingforce": 1.0,
        "armorcannon": 1.0,
        "protect": 1.0,
        "trickroom": 0.6667,
        "heatwave": 0.3333,
    },
    "basculegion": {
        "lastrespects": 1.0,
        "aquajet": 0.9333,
        "wavecrash": 0.8667,
        "protect": 0.5333,
        "flipturn": 0.4,
        "liquidation": 0.1333,
        "icywind": 0.0667,
        "soak": 0.0667,
    },
    "basculegionf": {
        "muddywater": 1.0,
        "shadowball": 1.0,
        "icebeam": 1.0,
        "protect": 1.0,
    },
    "baxcalibur": {
        "glaiverush": 1.0,
        "iceshard": 1.0,
        "swordsdance": 1.0,
        "protect": 1.0,
    },
    "camerupt": {
        "protect": 1.0,
        "heatwave": 1.0,
        "ancientpower": 1.0,
        "earthpower": 1.0,
    },
    "cameruptmega": {
        "heatwave": 1.0,
        "fireblast": 1.0,
        "earthpower": 1.0,
        "ancientpower": 1.0,
    },
    "ceruledge": {
        "bitterblade": 1.0,
        "swordsdance": 1.0,
        "shadowsneak": 0.6667,
        "protect": 0.6667,
        "poltergeist": 0.3333,
        "closecombat": 0.3333,
    },
    "charizard": {
        "protect": 1.0,
        "heatwave": 0.9412,
        "weatherball": 0.9412,
        "solarbeam": 0.3529,
        "ancientpower": 0.3529,
        "hurricane": 0.2941,
        "heatcrash": 0.0588,
        "focusblast": 0.0588,
    },
    "charizardmegay": {
        "weatherball": 1.0,
        "heatwave": 1.0,
        "protect": 1.0,
        "hurricane": 0.5,
        "ancientpower": 0.25,
        "solarbeam": 0.25,
    },
    "delphox": {
        "heatwave": 1.0,
        "protect": 1.0,
        "nastyplot": 0.8571,
        "psychic": 0.5714,
        "psyshock": 0.4286,
        "calmmind": 0.1429,
    },
    "dragapult": {
        "dracometeor": 1.0,
        "shadowball": 1.0,
        "protect": 1.0,
        "willowisp": 1.0,
    },
    "dragonite": {
        "dragonpulse": 0.8333,
        "tailwind": 0.8333,
        "protect": 0.8333,
        "heatwave": 0.6667,
        "scaleshot": 0.1667,
        "stompingtantrum": 0.1667,
        "rockslide": 0.1667,
        "dragondance": 0.1667,
        "thunderbolt": 0.1667,
    },
    "dragonitemega": {
        "dragonpulse": 1.0,
        "hurricane": 1.0,
        "heatwave": 1.0,
        "protect": 1.0,
    },
    "excadrill": {
        "highhorsepower": 1.0,
        "ironhead": 1.0,
        "swordsdance": 0.5,
        "protect": 0.5,
        "earthquake": 0.5,
        "rockslide": 0.5,
    },
    "farigiraf": {
        "trickroom": 1.0,
        "psychic": 0.8824,
        "protect": 0.7059,
        "helpinghand": 0.6471,
        "hypervoice": 0.2353,
        "thunderbolt": 0.1765,
        "grassknot": 0.0588,
        "raindance": 0.0588,
        "twinbeam": 0.0588,
        "expandingforce": 0.0588,
        "sunnyday": 0.0588,
        "imprison": 0.0588,
    },
    "floetteeternal": {
        "dazzlinggleam": 1.0,
        "protect": 1.0,
        "calmmind": 0.75,
        "moonblast": 0.625,
        "drainingkiss": 0.375,
        "lightofruin": 0.25,
    },
    "floettemega": {
        "protect": 1.0,
        "drainingkiss": 1.0,
        "dazzlinggleam": 1.0,
        "calmmind": 1.0,
    },
    "froslass": {
        "blizzard": 1.0,
        "shadowball": 1.0,
        "auroraveil": 1.0,
        "protect": 1.0,
    },
    "garchomp": {
        "protect": 0.7826,
        "dragonclaw": 0.4783,
        "stompingtantrum": 0.4348,
        "powergem": 0.4348,
        "dracometeor": 0.3478,
        "earthpower": 0.3478,
        "earthquake": 0.3478,
        "rockslide": 0.2609,
        "flamethrower": 0.1739,
        "rocktomb": 0.1739,
        "dragonpulse": 0.1304,
        "scaleshot": 0.0435,
        "bulldoze": 0.0435,
    },
    "garchompmegaz": {
        "dracometeor": 1.0,
        "earthpower": 1.0,
        "powergem": 1.0,
        "protect": 1.0,
    },
    "gardevoir": {
        "hypervoice": 1.0,
        "expandingforce": 1.0,
        "protect": 1.0,
        "trickroom": 0.5,
        "calmmind": 0.5,
    },
    "gardevoirmega": {
        "hypervoice": 1.0,
        "expandingforce": 1.0,
        "calmmind": 1.0,
        "protect": 0.5,
        "trickroom": 0.5,
    },
    "gengar": {
        "protect": 1.0,
        "shadowball": 1.0,
        "sludgebomb": 1.0,
        "perishsong": 0.8,
        "icywind": 0.2,
    },
    "gholdengo": {
        "makeitrain": 1.0,
        "shadowball": 1.0,
        "protect": 1.0,
        "nastyplot": 0.875,
        "focusblast": 0.0625,
        "powergem": 0.0625,
    },
    "golisopod": {
        "leechlife": 1.0,
        "ironhead": 0.9333,
        "protect": 0.6667,
        "suckerpunch": 0.6,
        "swordsdance": 0.4,
        "drillrun": 0.2667,
        "liquidation": 0.0667,
        "closecombat": 0.0667,
    },
    "golisopodmega": {
        "leechlife": 1.0,
        "ironhead": 1.0,
        "swordsdance": 1.0,
        "protect": 1.0,
    },
    "grapploct": {
        "stormthrow": 1.0,
        "stompingtantrum": 1.0,
        "machpunch": 1.0,
        "detect": 1.0,
    },
    "grimmsnarl": {
        "spiritbreak": 1.0,
        "lightscreen": 1.0,
        "reflect": 1.0,
        "partingshot": 1.0,
    },
    "hatterene": {
        "protect": 1.0,
        "expandingforce": 1.0,
        "trickroom": 1.0,
        "dazzlinggleam": 1.0,
    },
    "hydreigon": {
        "dracometeor": 1.0,
        "darkpulse": 1.0,
        "snarl": 1.0,
        "uturn": 1.0,
    },
    "incineroar": {
        "fakeout": 1.0,
        "partingshot": 1.0,
        "flareblitz": 0.8857,
        "throatchop": 0.4286,
        "darkestlariat": 0.3143,
        "snarl": 0.1143,
        "helpinghand": 0.1143,
        "protect": 0.1143,
        "assurance": 0.0286,
    },
    "indeedee": {
        "expandingforce": 1.0,
        "imprison": 0.75,
        "trickroom": 0.75,
        "protect": 0.5,
        "hypervoice": 0.25,
        "energyball": 0.25,
        "helpinghand": 0.25,
        "mysticalfire": 0.25,
    },
    "indeedeef": {
        "followme": 1.0,
        "psychic": 0.9091,
        "helpinghand": 0.9091,
        "trickroom": 0.6364,
        "protect": 0.3636,
        "fakeout": 0.0909,
        "mysticalfire": 0.0909,
    },
    "kangaskhan": {
        "doubleedge": 1.0,
        "hammerarm": 1.0,
        "crunch": 1.0,
        "fakeout": 1.0,
    },
    "kingambit": {
        "kowtowcleave": 1.0,
        "suckerpunch": 1.0,
        "protect": 0.6538,
        "ironhead": 0.5769,
        "lowkick": 0.4615,
        "swordsdance": 0.3077,
    },
    "kommoo": {
        "protect": 1.0,
        "clangingscales": 1.0,
        "aurasphere": 0.8571,
        "clangoroussoul": 0.8571,
        "flamethrower": 0.2857,
    },
    "lopunny": {
        "closecombat": 1.0,
        "tripleaxel": 1.0,
        "thunderpunch": 1.0,
        "fakeout": 1.0,
    },
    "lucario": {
        "detect": 1.0,
        "aurasphere": 1.0,
        "flashcannon": 1.0,
        "calmmind": 0.6667,
        "steelbeam": 0.3333,
    },
    "mawile": {
        "protect": 1.0,
        "ironhead": 1.0,
        "playrough": 1.0,
        "suckerpunch": 1.0,
    },
    "meganiummega": {
        "solarbeam": 1.0,
        "weatherball": 1.0,
        "dazzlinggleam": 1.0,
        "protect": 1.0,
    },
    "meowsticf": {
        "expandingforce": 1.0,
        "thunderbolt": 1.0,
        "fakeout": 1.0,
        "protect": 1.0,
    },
    "metagross": {
        "psychicfangs": 1.0,
        "protect": 1.0,
        "bodypress": 0.5,
        "psychup": 0.5,
        "stompingtantrum": 0.5,
        "steelroller": 0.5,
    },
    "metagrossmega": {
        "psychicfangs": 1.0,
        "bodypress": 1.0,
        "psychup": 1.0,
        "protect": 1.0,
    },
    "milotic": {
        "muddywater": 0.6667,
        "icebeam": 0.5556,
        "protect": 0.5556,
        "coil": 0.5556,
        "hypnosis": 0.5556,
        "scald": 0.3333,
        "recover": 0.3333,
        "icywind": 0.2222,
        "haze": 0.1111,
        "lightscreen": 0.1111,
    },
    "ninetales": {
        "weatherball": 1.0,
        "snarl": 1.0,
        "helpinghand": 1.0,
        "protect": 1.0,
    },
    "noivern": {
        "dracometeor": 1.0,
        "superfang": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
    },
    "pawmot": {
        "doubleshock": 1.0,
        "closecombat": 1.0,
        "fakeout": 1.0,
        "revivalblessing": 0.5,
        "protect": 0.5,
    },
    "pelipper": {
        "hurricane": 1.0,
        "weatherball": 0.95,
        "tailwind": 0.8,
        "wideguard": 0.6,
        "protect": 0.5,
        "helpinghand": 0.15,
    },
    "politoed": {
        "protect": 1.0,
        "weatherball": 0.875,
        "muddywater": 0.625,
        "encore": 0.5,
        "perishsong": 0.5,
        "icebeam": 0.5,
    },
    "primarina": {
        "protect": 1.0,
        "hypervoice": 1.0,
        "moonblast": 1.0,
        "calmmind": 0.8,
        "dazzlinggleam": 0.2,
    },
    "pyroar": {
        "heatwave": 1.0,
        "overheat": 1.0,
        "protect": 1.0,
        "scorchingsands": 0.5,
        "solarbeam": 0.5,
    },
    "pyroarmega": {
        "heatwave": 1.0,
        "overheat": 1.0,
        "scorchingsands": 1.0,
        "protect": 1.0,
    },
    "raichu": {
        "focusblast": 0.96,
        "zapcannon": 0.96,
        "protect": 0.96,
        "fakeout": 0.76,
        "encore": 0.28,
        "volttackle": 0.04,
        "voltswitch": 0.04,
    },
    "raichumegax": {
        "volttackle": 1.0,
        "voltswitch": 1.0,
        "feint": 1.0,
        "fakeout": 1.0,
    },
    "rillaboom": {
        "grassyglide": 1.0,
        "fakeout": 1.0,
        "woodhammer": 0.9091,
        "highhorsepower": 0.8182,
        "uturn": 0.1591,
        "protect": 0.0909,
        "drumbeating": 0.0227,
    },
    "rotomheat": {
        "discharge": 1.0,
        "overheat": 1.0,
        "trick": 1.0,
        "electroweb": 1.0,
    },
    "salamence": {
        "hypervoice": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
        "dracometeor": 0.7368,
        "flamethrower": 0.1579,
        "doubleedge": 0.1053,
    },
    "sceptile": {
        "dragonpulse": 1.0,
        "energyball": 1.0,
        "earthpower": 1.0,
        "detect": 1.0,
    },
    "sinistcha": {
        "matchagotcha": 1.0,
        "ragepowder": 1.0,
        "trickroom": 1.0,
        "protect": 0.8,
        "scald": 0.2,
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
        "direclaw": 0.8293,
        "protect": 0.7317,
        "fakeout": 0.3902,
        "rockslide": 0.3659,
        "rocktomb": 0.1951,
        "poisonjab": 0.122,
        "quickguard": 0.122,
        "throatchop": 0.0976,
        "gunkshot": 0.0488,
        "coaching": 0.0488,
        "acrobatics": 0.0244,
        "swordsdance": 0.0244,
    },
    "staraptor": {
        "bravebird": 1.0,
        "closecombat": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
    },
    "starmie": {
        "liquidation": 1.0,
        "expandingforce": 1.0,
        "icebeam": 1.0,
        "protect": 1.0,
    },
    "swampert": {
        "wavecrash": 1.0,
        "earthquake": 1.0,
        "icepunch": 1.0,
        "protect": 1.0,
    },
    "swampertmega": {
        "wavecrash": 1.0,
        "highhorsepower": 1.0,
        "icepunch": 1.0,
        "protect": 1.0,
    },
    "sylveon": {
        "hypervoice": 1.0,
        "detect": 1.0,
        "hyperbeam": 0.8333,
        "quickattack": 0.8333,
        "yawn": 0.3333,
    },
    "talonflame": {
        "tailwind": 1.0,
        "quickguard": 1.0,
        "dualwingbeat": 1.0,
        "protect": 1.0,
    },
    "torkoal": {
        "eruption": 1.0,
        "protect": 1.0,
        "earthpower": 0.75,
        "weatherball": 0.5,
        "overheat": 0.25,
        "helpinghand": 0.25,
        "heatwave": 0.25,
    },
    "toxapex": {
        "banefulbunker": 1.0,
        "wideguard": 1.0,
        "toxic": 1.0,
        "infestation": 1.0,
    },
    "tyranitar": {
        "rockslide": 1.0,
        "protect": 1.0,
        "knockoff": 1.0,
        "dragondance": 0.5,
        "lowkick": 0.5,
    },
    "venusaur": {
        "sludgebomb": 1.0,
        "protect": 1.0,
        "sleeppowder": 0.875,
        "leafstorm": 0.75,
        "earthpower": 0.25,
        "energyball": 0.125,
    },
    "vivillon": {
        "hurricane": 1.0,
        "sleeppowder": 1.0,
        "ragepowder": 1.0,
        "protect": 1.0,
    },
    "volcarona": {
        "ragepowder": 0.8,
        "protect": 0.8,
        "tailwind": 0.6,
        "overheat": 0.6,
        "heatwave": 0.4,
        "strugglebug": 0.4,
        "gigadrain": 0.2,
        "quiverdance": 0.2,
    },
    "watchog": {
        "icepunch": 1.0,
        "detect": 1.0,
        "lowkick": 1.0,
        "thunderwave": 1.0,
    },
    "whimsicott": {
        "moonblast": 1.0,
        "tailwind": 1.0,
        "protect": 1.0,
        "encore": 0.75,
        "sunnyday": 0.25,
    },
}

CHAMPIONS_ABILITY_PRIORS = {
    "aerodactyl": {
        "unnerve": 1.0,
    },
    "alcremie": {
        "sweetveil": 1.0,
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
    "basculegionf": {
        "swiftswim": 1.0,
    },
    "baxcalibur": {
        "thermalexchange": 1.0,
    },
    "camerupt": {
        "solidrock": 1.0,
    },
    "cameruptmega": {
        "solidrock": 1.0,
    },
    "ceruledge": {
        "flashfire": 1.0,
    },
    "charizard": {
        "blaze": 1.0,
    },
    "charizardmegay": {
        "blaze": 0.5,
        "drought": 0.5,
    },
    "delphox": {
        "blaze": 1.0,
    },
    "dragapult": {
        "infiltrator": 0.75,
        "clearbody": 0.25,
    },
    "dragonite": {
        "innerfocus": 0.6667,
        "multiscale": 0.3333,
    },
    "dragonitemega": {
        "multiscale": 1.0,
    },
    "excadrill": {
        "sandrush": 0.5,
        "moldbreaker": 0.5,
    },
    "farigiraf": {
        "armortail": 1.0,
    },
    "floetteeternal": {
        "flowerveil": 1.0,
    },
    "floettemega": {
        "flowerveil": 1.0,
    },
    "froslass": {
        "cursedbody": 0.8,
        "snowcloak": 0.2,
    },
    "garchomp": {
        "roughskin": 1.0,
    },
    "garchompmegaz": {
        "roughskin": 1.0,
    },
    "gardevoir": {
        "synchronize": 0.5,
        "trace": 0.5,
    },
    "gardevoirmega": {
        "synchronize": 0.5,
        "pixilate": 0.5,
    },
    "gengar": {
        "cursedbody": 1.0,
    },
    "gholdengo": {
        "goodasgold": 1.0,
    },
    "golisopod": {
        "emergencyexit": 1.0,
    },
    "golisopodmega": {
        "emergencyexit": 1.0,
    },
    "grapploct": {
        "technician": 1.0,
    },
    "grimmsnarl": {
        "prankster": 1.0,
    },
    "hatterene": {
        "magicbounce": 1.0,
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
        "psychicsurge": 0.9091,
        "synchronize": 0.0909,
    },
    "kangaskhan": {
        "scrappy": 1.0,
    },
    "kingambit": {
        "defiant": 1.0,
    },
    "kommoo": {
        "soundproof": 0.7143,
        "bulletproof": 0.2857,
    },
    "lopunny": {
        "limber": 1.0,
    },
    "lucario": {
        "innerfocus": 1.0,
    },
    "mawile": {
        "hypercutter": 1.0,
    },
    "meganiummega": {
        "megasol": 1.0,
    },
    "meowsticf": {
        "competitive": 1.0,
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
    "ninetales": {
        "drought": 1.0,
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
    "pyroar": {
        "unnerve": 1.0,
    },
    "pyroarmega": {
        "unnerve": 1.0,
    },
    "raichu": {
        "lightningrod": 1.0,
    },
    "raichumegax": {
        "lightningrod": 1.0,
    },
    "rillaboom": {
        "grassysurge": 1.0,
    },
    "rotomheat": {
        "levitate": 1.0,
    },
    "salamence": {
        "intimidate": 1.0,
    },
    "sceptile": {
        "overgrow": 1.0,
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
        "unburden": 0.8537,
        "poisontouch": 0.1463,
    },
    "staraptor": {
        "intimidate": 1.0,
    },
    "starmie": {
        "naturalcure": 1.0,
    },
    "swampert": {
        "torrent": 1.0,
    },
    "swampertmega": {
        "torrent": 1.0,
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
    "venusaur": {
        "chlorophyll": 1.0,
    },
    "vivillon": {
        "compoundeyes": 1.0,
    },
    "volcarona": {
        "flamebody": 1.0,
    },
    "watchog": {
        "analytic": 1.0,
    },
    "whimsicott": {
        "prankster": 1.0,
    },
}

CHAMPIONS_ITEM_PRIORS = {
    "aerodactyl": {
        "focussash": 0.5,
        "aerodactylite": 0.5,
    },
    "alcremie": {
        "rockyhelmet": 1.0,
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
        "lifeorb": 0.6667,
        "twistedspoon": 0.3333,
    },
    "basculegion": {
        "choicescarf": 0.4667,
        "lifeorb": 0.2,
        "mysticwater": 0.2,
        "focussash": 0.1333,
    },
    "basculegionf": {
        "focussash": 1.0,
    },
    "baxcalibur": {
        "baxcalibrite": 1.0,
    },
    "camerupt": {
        "cameruptite": 1.0,
    },
    "cameruptmega": {
        "cameruptite": 1.0,
    },
    "ceruledge": {
        "grassyseed": 0.6667,
        "focussash": 0.3333,
    },
    "charizard": {
        "charizarditey": 1.0,
    },
    "charizardmegay": {
        "charizarditey": 1.0,
    },
    "delphox": {
        "delphoxite": 1.0,
    },
    "dragapult": {
        "lifeorb": 1.0,
    },
    "dragonite": {
        "dragoninite": 0.8333,
        "lifeorb": 0.1667,
    },
    "dragonitemega": {
        "dragoninite": 1.0,
    },
    "excadrill": {
        "focussash": 0.5,
        "choicescarf": 0.5,
    },
    "farigiraf": {
        "sitrusberry": 0.6471,
        "colburberry": 0.1176,
        "twistedspoon": 0.0588,
        "lifeorb": 0.0588,
        "wiseglasses": 0.0588,
        "silkscarf": 0.0588,
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
        "garchompitez": 0.4783,
        "choicescarf": 0.2174,
        "lifeorb": 0.1739,
        "garchompite": 0.0435,
        "dragonfang": 0.0435,
        "habanberry": 0.0435,
    },
    "garchompmegaz": {
        "garchompitez": 1.0,
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
    "gholdengo": {
        "lifeorb": 0.9375,
        "whiteherb": 0.0312,
        "leftovers": 0.0312,
    },
    "golisopod": {
        "golisopite": 1.0,
    },
    "golisopodmega": {
        "golisopite": 1.0,
    },
    "grapploct": {
        "lifeorb": 1.0,
    },
    "grimmsnarl": {
        "lightclay": 1.0,
    },
    "hatterene": {
        "lifeorb": 1.0,
    },
    "hydreigon": {
        "focussash": 1.0,
    },
    "incineroar": {
        "sitrusberry": 0.7714,
        "passhoberry": 0.1143,
        "chopleberry": 0.1143,
    },
    "indeedee": {
        "focussash": 1.0,
    },
    "indeedeef": {
        "rockyhelmet": 0.8182,
        "colburberry": 0.1818,
    },
    "kangaskhan": {
        "kangaskhanite": 1.0,
    },
    "kingambit": {
        "chopleberry": 0.3077,
        "lifeorb": 0.3077,
        "focussash": 0.2692,
        "blackglasses": 0.0769,
        "occaberry": 0.0385,
    },
    "kommoo": {
        "leftovers": 0.8571,
        "lifeorb": 0.1429,
    },
    "lopunny": {
        "lopunnite": 1.0,
    },
    "lucario": {
        "lucarionitez": 1.0,
    },
    "mawile": {
        "mawilite": 1.0,
    },
    "meganiummega": {
        "meganiumite": 1.0,
    },
    "meowsticf": {
        "meowsticite": 1.0,
    },
    "metagross": {
        "metagrossite": 1.0,
    },
    "metagrossmega": {
        "metagrossite": 1.0,
    },
    "milotic": {
        "sitrusberry": 0.3333,
        "leftovers": 0.3333,
        "psychicseed": 0.2222,
        "grassyseed": 0.1111,
    },
    "ninetales": {
        "focussash": 1.0,
    },
    "noivern": {
        "focussash": 1.0,
    },
    "pawmot": {
        "focussash": 1.0,
    },
    "pelipper": {
        "sitrusberry": 0.4,
        "focussash": 0.35,
        "damprock": 0.2,
        "lifeorb": 0.05,
    },
    "politoed": {
        "mysticwater": 0.5,
        "sitrusberry": 0.375,
        "ejectbutton": 0.125,
    },
    "primarina": {
        "leftovers": 0.8,
        "lifeorb": 0.2,
    },
    "pyroar": {
        "pyroarite": 1.0,
    },
    "pyroarmega": {
        "pyroarite": 1.0,
    },
    "raichu": {
        "raichunitey": 0.96,
        "raichunitex": 0.04,
    },
    "raichumegax": {
        "raichunitex": 1.0,
    },
    "rillaboom": {
        "miracleseed": 0.8409,
        "expertbelt": 0.0455,
        "ejectbutton": 0.0455,
        "sitrusberry": 0.0227,
        "occaberry": 0.0227,
        "lifeorb": 0.0227,
    },
    "rotomheat": {
        "choicescarf": 1.0,
    },
    "salamence": {
        "salamencite": 1.0,
    },
    "sceptile": {
        "sceptilite": 1.0,
    },
    "sinistcha": {
        "colburberry": 0.4,
        "rockyhelmet": 0.2,
        "sitrusberry": 0.2,
        "occaberry": 0.2,
    },
    "sinistchamasterpiece": {
        "colburberry": 1.0,
    },
    "sirfetchd": {
        "expertbelt": 1.0,
    },
    "sneasler": {
        "whiteherb": 0.3659,
        "grassyseed": 0.2195,
        "psychicseed": 0.2195,
        "focussash": 0.1707,
        "electricseed": 0.0244,
    },
    "staraptor": {
        "staraptite": 1.0,
    },
    "starmie": {
        "starminite": 1.0,
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
        "charcoal": 0.75,
        "lifeorb": 0.25,
    },
    "toxapex": {
        "leftovers": 1.0,
    },
    "tyranitar": {
        "tyranitarite": 1.0,
    },
    "venusaur": {
        "focussash": 0.75,
        "widelens": 0.125,
        "lifeorb": 0.125,
    },
    "vivillon": {
        "focussash": 1.0,
    },
    "volcarona": {
        "rockyhelmet": 0.6,
        "sitrusberry": 0.2,
        "leftovers": 0.2,
    },
    "watchog": {
        "chopleberry": 1.0,
    },
    "whimsicott": {
        "focussash": 1.0,
    },
}

CHAMPIONS_COMMON_SPREADS = {
    "aerodactyl": {"nature": "jolly", "evs": {"hp": 5, "atk": 27, "spe": 32, "spd": 8}},
    "alcremie": {"nature": "bold", "evs": {"hp": 32, "def": 32, "spd": 2}},
    "altaria": {"nature": "timid", "evs": {"hp": 32, "def": 5, "spe": 29}},
    "annihilape": {"nature": "jolly", "evs": {"hp": 2, "atk": 32, "spe": 32, "def": 1}},
    "arcaninehisui": {"nature": "jolly", "evs": {"hp": 2, "atk": 32, "spd": 1, "spe": 32, "def": 1}},
    "archaludon": {"nature": "modest", "evs": {"hp": 31, "def": 2, "spd": 24, "spe": 7, "spa": 5}},
    "armarouge": {"nature": "modest", "evs": {"hp": 23, "spa": 32, "spd": 2, "spe": 30}},
    "basculegion": {"nature": "adamant", "evs": {"hp": 13, "atk": 29, "def": 11, "spe": 27, "spd": 2}},
    "basculegionf": {"nature": "modest", "evs": {"hp": 2, "spa": 32, "spe": 32, "def": 1, "spd": 1}},
    "baxcalibur": {"nature": "adamant", "evs": {"hp": 21, "atk": 16, "def": 3, "spe": 26}},
    "camerupt": {"nature": "quiet", "evs": {"hp": 32, "def": 2, "spa": 32}},
    "cameruptmega": {"nature": "quiet", "evs": {"hp": 32, "spa": 32, "spd": 2}},
    "ceruledge": {"nature": "adamant", "evs": {"hp": 29, "atk": 16, "def": 14, "spd": 3, "spe": 5}},
    "charizard": {"nature": "modest", "evs": {"hp": 26, "def": 12, "spa": 11, "spe": 20, "spd": 11}},
    "charizardmegay": {"nature": "modest", "evs": {"hp": 28, "def": 20, "spa": 1, "spd": 19, "spe": 11}},
    "delphox": {"nature": "timid", "evs": {"hp": 10, "spa": 15, "spe": 32, "def": 12}},
    "dragapult": {"nature": "modest", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "dragonite": {"nature": "modest", "evs": {"hp": 7, "spa": 27, "spe": 32, "spd": 2}},
    "dragonitemega": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "excadrill": {"nature": "adamant", "evs": {"hp": 2, "atk": 32, "spe": 32}},
    "farigiraf": {"nature": "bold", "evs": {"hp": 29, "def": 22, "spd": 15}},
    "floetteeternal": {"nature": "modest", "evs": {"def": 13, "spa": 19, "spe": 19, "hp": 19, "spd": 1}},
    "floettemega": {"nature": "modest", "evs": {"hp": 24, "def": 10, "spe": 32}},
    "froslass": {"nature": "timid", "evs": {"hp": 8, "spa": 23, "spe": 29, "def": 16}},
    "garchomp": {"nature": "modest", "evs": {"spa": 30, "spd": 3, "spe": 32, "hp": 3, "def": 2}},
    "garchompmegaz": {"nature": "modest", "evs": {"hp": 1, "def": 1, "spa": 32, "spe": 32}},
    "gardevoir": {"nature": "modest", "evs": {"hp": 32, "def": 10, "spa": 20, "spe": 7, "spd": 3}},
    "gardevoirmega": {"nature": "timid", "evs": {"hp": 17, "def": 1, "spa": 23, "spd": 1, "spe": 24}},
    "gengar": {"nature": "modest", "evs": {"hp": 31, "def": 2, "spa": 1, "spd": 20, "spe": 14}},
    "gholdengo": {"nature": "timid", "evs": {"hp": 10, "spa": 22, "spe": 31, "def": 5, "spd": 2}},
    "golisopod": {"nature": "adamant", "evs": {"hp": 32, "atk": 27, "spd": 5, "spe": 4, "def": 1}},
    "golisopodmega": {"nature": "adamant", "evs": {"hp": 32, "atk": 32, "spd": 2}},
    "grapploct": {"nature": "adamant", "evs": {"hp": 32, "atk": 32, "spd": 2}},
    "grimmsnarl": {"nature": "sassy", "evs": {"hp": 31, "def": 19, "spd": 16}},
    "hatterene": {"nature": "quiet", "evs": {"hp": 32, "def": 2, "spa": 32}},
    "hydreigon": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "incineroar": {"nature": "careful", "evs": {"hp": 31, "def": 12, "spd": 13, "spe": 15, "atk": 4}},
    "indeedee": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "indeedeef": {"nature": "bold", "evs": {"hp": 32, "def": 31, "spd": 2, "spe": 8}},
    "kangaskhan": {"nature": "adamant", "evs": {"hp": 14, "atk": 32, "def": 12, "spd": 8}},
    "kingambit": {"nature": "adamant", "evs": {"hp": 23, "atk": 27, "def": 8, "spe": 12, "spd": 3}},
    "kommoo": {"nature": "modest", "evs": {"hp": 8, "spa": 24, "spe": 32, "def": 1, "spd": 2}},
    "lopunny": {"nature": "jolly", "evs": {"hp": 2, "atk": 32, "spe": 32}},
    "lucario": {"nature": "timid", "evs": {"hp": 32, "def": 4, "spa": 8, "spd": 7, "spe": 16}},
    "mawile": {"nature": "brave", "evs": {"hp": 32, "atk": 32, "spd": 2}},
    "meganiummega": {"nature": "modest", "evs": {"hp": 27, "spa": 27, "spe": 12}},
    "meowsticf": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "metagross": {"nature": "jolly", "evs": {"hp": 7, "atk": 24, "def": 11, "spe": 30}},
    "metagrossmega": {"nature": "jolly", "evs": {"hp": 12, "atk": 15, "def": 11, "spe": 28}},
    "milotic": {"nature": "calm", "evs": {"hp": 32, "def": 17, "spd": 9, "spe": 10}},
    "ninetales": {"nature": "timid", "evs": {"def": 2, "spa": 32, "spe": 32}},
    "noivern": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "pawmot": {"nature": "jolly", "evs": {"hp": 2, "atk": 32, "spe": 32}},
    "pelipper": {"nature": "modest", "evs": {"hp": 17, "spa": 25, "spe": 26, "spd": 6, "def": 9}},
    "politoed": {"nature": "calm", "evs": {"hp": 32, "def": 20, "spa": 3, "spd": 13, "spe": 2}},
    "primarina": {"nature": "modest", "evs": {"hp": 22, "def": 19, "spa": 8, "spe": 17}},
    "pyroar": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "pyroarmega": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "raichu": {"nature": "timid", "evs": {"hp": 26, "def": 11, "spe": 28, "spa": 4, "spd": 1}},
    "raichumegax": {"nature": "jolly", "evs": {"hp": 30, "def": 4, "spe": 32}},
    "rillaboom": {"nature": "adamant", "evs": {"hp": 25, "atk": 21, "def": 3, "spd": 12, "spe": 11}},
    "rotomheat": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "salamence": {"nature": "timid", "evs": {"hp": 5, "spa": 27, "spe": 32, "spd": 5, "def": 1}},
    "sceptile": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
    "sinistcha": {"nature": "bold", "evs": {"hp": 32, "def": 16, "spd": 17, "spe": 1}},
    "sinistchamasterpiece": {"nature": "bold", "evs": {"hp": 32, "def": 14, "spd": 19, "spe": 1}},
    "sirfetchd": {"nature": "brave", "evs": {"hp": 32, "atk": 32, "spd": 2}},
    "sneasler": {"nature": "adamant", "evs": {"hp": 14, "atk": 25, "spe": 26, "def": 6, "spd": 4}},
    "staraptor": {"nature": "jolly", "evs": {"hp": 25, "atk": 8, "spe": 32, "spd": 2, "def": 1}},
    "starmie": {"nature": "hasty", "evs": {"atk": 32, "spa": 16, "spe": 18}},
    "swampert": {"nature": "adamant", "evs": {"hp": 14, "atk": 30, "spe": 22}},
    "swampertmega": {"nature": "adamant", "evs": {"hp": 11, "atk": 32, "spe": 23}},
    "sylveon": {"nature": "modest", "evs": {"hp": 22, "def": 16, "spa": 21, "spe": 11, "spd": 2}},
    "talonflame": {"nature": "adamant", "evs": {"hp": 2, "atk": 32, "spe": 32}},
    "torkoal": {"nature": "quiet", "evs": {"hp": 30, "spa": 31, "spd": 4}},
    "toxapex": {"nature": "relaxed", "evs": {"hp": 32, "def": 18, "spd": 16}},
    "tyranitar": {"nature": "jolly", "evs": {"hp": 9, "atk": 25, "spe": 32}},
    "venusaur": {"nature": "modest", "evs": {"hp": 2, "spa": 32, "spe": 32, "spd": 1}},
    "vivillon": {"nature": "timid", "evs": {"def": 2, "spa": 32, "spe": 32}},
    "volcarona": {"nature": "calm", "evs": {"hp": 30, "def": 18, "spd": 18, "spe": 1}},
    "watchog": {"nature": "impish", "evs": {"hp": 32, "atk": 2, "def": 32}},
    "whimsicott": {"nature": "timid", "evs": {"hp": 2, "spa": 32, "spe": 32}},
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
