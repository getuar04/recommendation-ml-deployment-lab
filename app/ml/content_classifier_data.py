"""Deterministic, hand-authored, explicitly-SYNTHETIC bootstrap dataset for
`app.ml.content_classifier`. This is training data for a statistical text classifier, not a
runtime keyword lookup: `app.ml.content_classifier.classify_text` never inspects this module,
it only calls the fitted model produced by training on it.

NOT real labeled production data. No real Content Service integration exists yet (see the
event-driven architecture audit) and this repository has never captured a real, human-labeled
(title/hashtags -> category) dataset. This bootstrap set exists solely to prove the
classification pipeline (text normalization -> TF-IDF -> LogisticRegression -> confidence
threshold -> creator-prior blending) works end-to-end and produces a real, non-hardcoded
statistical model over CATEGORY_TAXONOMY_VERSION's 10 categories. It must be replaced or
supplemented with real labeled content before any accuracy measured on it is presented as
production evidence -- see `app.ml.content_classifier.train_classifier`'s own
`datasetSource` metadata, which carries this same caveat into the trained artifact itself.

The 10-category taxonomy (`app.ml.content_classifier.CATEGORIES`) is reused verbatim from
the only concrete category vocabulary that already exists anywhere in this repository
(`scripts/generate_synthetic_data.py` / `app.experiments.definitions.DEFAULT_CATEGORIES`) --
itself a synthetic-data artifact, not a declared product taxonomy. See
`app.ml.content_classifier`'s module docstring for why this must still be confirmed as a
real product/domain decision before being treated as final.
"""
from __future__ import annotations

import pandas as pd

# (title, hashtags, category) -- deliberately plain, readable, auditable rows rather than a
# random template generator, so a reviewer can see exactly what this dataset does and does
# not contain. ~15-18 rows per category, varied phrasing/hashtags within each.
_ROWS: list[tuple[str, list[str], str]] = [
    # FOOD
    ("Best pasta recipe you will ever try", ["cooking", "recipe", "pasta", "foodie"], "FOOD"),
    ("How to make sushi at home", ["sushi", "cooking", "japanese", "foodie"], "FOOD"),
    ("Street food tour in Bangkok", ["streetfood", "travel", "foodie", "thailand"], "FOOD"),
    ("Easy chocolate cake recipe", ["baking", "dessert", "recipe", "cake"], "FOOD"),
    ("Grilling the perfect steak", ["bbq", "grilling", "steak", "cooking"], "FOOD"),
    ("Vegan meal prep for the week", ["vegan", "mealprep", "healthy", "recipe"], "FOOD"),
    ("Tasting the spiciest ramen in town", ["ramen", "spicy", "foodreview", "noodles"], "FOOD"),
    ("Homemade pizza dough from scratch", ["pizza", "baking", "recipe", "italian"], "FOOD"),
    ("Top 10 breakfast ideas", ["breakfast", "recipe", "foodie", "morning"], "FOOD"),
    ("Michelin star restaurant review", ["finedining", "restaurant", "review", "foodie"], "FOOD"),
    ("Quick and easy stir fry dinner", ["stirfry", "dinner", "recipe", "asianfood"], "FOOD"),
    ("Baking sourdough bread for beginners", ["sourdough", "baking", "bread", "recipe"], "FOOD"),
    ("Best tacos in Mexico City", ["tacos", "mexicanfood", "streetfood", "foodie"], "FOOD"),
    ("Coffee brewing methods explained", ["coffee", "brewing", "espresso", "cafe"], "FOOD"),
    ("Farmers market haul and cooking", ["farmersmarket", "cooking", "freshfood", "recipe"], "FOOD"),

    # SPORT
    ("Real Madrid vs Barcelona highlights", ["football", "laliga", "realmadrid", "barcelona"], "SPORT"),
    ("Champions League goals compilation", ["ucl", "football", "soccer", "goals"], "SPORT"),
    ("NBA finals full recap", ["nba", "basketball", "finals", "sports"], "SPORT"),
    ("Premier League matchday review", ["premierleague", "football", "soccer", "epl"], "SPORT"),
    ("Tennis grand slam final highlights", ["tennis", "grandslam", "sports", "wimbledon"], "SPORT"),
    ("Marathon training tips for beginners", ["marathon", "running", "training", "sports"], "SPORT"),
    ("Boxing match knockout compilation", ["boxing", "knockout", "sports", "combat"], "SPORT"),
    ("World Cup qualifiers analysis", ["worldcup", "football", "soccer", "fifa"], "SPORT"),
    ("Olympic swimming records broken", ["olympics", "swimming", "sports", "records"], "SPORT"),
    ("Formula 1 race weekend recap", ["f1", "formula1", "racing", "motorsport"], "SPORT"),
    ("College basketball tournament highlights", ["basketball", "ncaa", "sports", "tournament"], "SPORT"),
    ("Cricket world cup final highlights", ["cricket", "worldcup", "sports", "t20"], "SPORT"),
    ("Rugby championship best tries", ["rugby", "sports", "championship", "tries"], "SPORT"),
    ("Golf tournament final round highlights", ["golf", "pga", "sports", "tournament"], "SPORT"),
    ("Volleyball championship match highlights", ["volleyball", "sports", "championship", "match"], "SPORT"),

    # MUSIC
    ("New album review and reaction", ["album", "musicreview", "newmusic", "reaction"], "MUSIC"),
    ("Live concert highlights from the tour", ["concert", "livemusic", "tour", "music"], "MUSIC"),
    ("Guitar tutorial for beginners", ["guitar", "tutorial", "musiclessons", "learntoplay"], "MUSIC"),
    ("Top 10 pop songs this year", ["pop", "musiccharts", "topsongs", "music"], "MUSIC"),
    ("Behind the scenes of the music video", ["musicvideo", "bts", "artist", "music"], "MUSIC"),
    ("Piano cover of a classical piece", ["piano", "classicalmusic", "cover", "music"], "MUSIC"),
    ("Hip hop freestyle session", ["hiphop", "freestyle", "rap", "music"], "MUSIC"),
    ("DJ set from the festival", ["dj", "edm", "festival", "music"], "MUSIC"),
    ("Songwriting tips for beginners", ["songwriting", "musictips", "producer", "music"], "MUSIC"),
    ("Reaction to the new music video", ["reaction", "musicvideo", "newmusic", "music"], "MUSIC"),
    ("Band rehearsal before the big show", ["band", "rehearsal", "livemusic", "music"], "MUSIC"),
    ("Vinyl record collection tour", ["vinyl", "records", "musiccollector", "music"], "MUSIC"),
    ("Singing competition audition highlights", ["singing", "competition", "audition", "music"], "MUSIC"),
    ("Music theory basics explained", ["musictheory", "tutorial", "learnmusic", "music"], "MUSIC"),
    ("Acoustic cover of a popular song", ["acoustic", "cover", "singer", "music"], "MUSIC"),

    # TECH
    ("iPhone 17 full review", ["iphone", "tech", "review", "smartphone"], "TECH"),
    ("Best laptops to buy this year", ["laptops", "tech", "buyersguide", "review"], "TECH"),
    ("How AI is changing everything", ["ai", "artificialintelligence", "tech", "future"], "TECH"),
    ("Unboxing the newest smartwatch", ["smartwatch", "tech", "unboxing", "wearable"], "TECH"),
    ("Building a custom gaming PC", ["pcbuild", "tech", "hardware", "diy"], "TECH"),
    ("Top programming languages to learn", ["programming", "coding", "tech", "developer"], "TECH"),
    ("Review of the latest graphics card", ["gpu", "tech", "hardware", "review"], "TECH"),
    ("Cloud computing explained simply", ["cloudcomputing", "tech", "explainer", "aws"], "TECH"),
    ("Smart home setup and automation", ["smarthome", "tech", "automation", "iot"], "TECH"),
    ("New camera drone review", ["drone", "tech", "review", "camera"], "TECH"),
    ("Cybersecurity tips everyone should know", ["cybersecurity", "tech", "privacy", "security"], "TECH"),
    ("Comparing the newest flagship phones", ["smartphone", "tech", "comparison", "review"], "TECH"),
    ("Electric car technology explained", ["ev", "tech", "electriccar", "innovation"], "TECH"),
    ("Software development best practices", ["softwaredev", "coding", "tech", "programming"], "TECH"),
    ("Virtual reality headset review", ["vr", "tech", "review", "virtualreality"], "TECH"),

    # GAMING
    ("Minecraft survival episode one", ["minecraft", "gaming", "survival", "letsplay"], "GAMING"),
    ("Best gaming moments of the year", ["gaming", "highlights", "esports", "gameplay"], "GAMING"),
    ("Fortnite victory royale highlights", ["fortnite", "gaming", "battleroyale", "victory"], "GAMING"),
    ("Speedrunning a classic platformer", ["speedrun", "gaming", "retrogaming", "platformer"], "GAMING"),
    ("New RPG game full review", ["rpg", "gaming", "review", "videogames"], "GAMING"),
    ("Esports championship finals recap", ["esports", "gaming", "championship", "tournament"], "GAMING"),
    ("Building a base in a survival game", ["survivalgame", "gaming", "letsplay", "gameplay"], "GAMING"),
    ("Ranked matches in the new shooter", ["fps", "gaming", "ranked", "shooter"], "GAMING"),
    ("Indie game hidden gem review", ["indiegame", "gaming", "review", "videogames"], "GAMING"),
    ("Speedrun world record attempt", ["speedrun", "gaming", "worldrecord", "retrogaming"], "GAMING"),
    ("Open world exploration gameplay", ["openworld", "gaming", "gameplay", "exploration"], "GAMING"),
    ("Boss fight walkthrough guide", ["walkthrough", "gaming", "bossfight", "guide"], "GAMING"),
    ("Retro console collecting haul", ["retrogaming", "gaming", "collector", "consoles"], "GAMING"),
    ("Multiplayer co-op gameplay session", ["coop", "gaming", "multiplayer", "gameplay"], "GAMING"),
    ("Game development devlog update", ["gamedev", "gaming", "devlog", "indiedev"], "GAMING"),

    # TRAVEL
    ("Backpacking through Europe on a budget", ["backpacking", "travel", "europe", "budgettravel"], "TRAVEL"),
    ("Best beaches in Thailand", ["thailand", "travel", "beaches", "vacation"], "TRAVEL"),
    ("Hidden gems in Japan you must visit", ["japan", "travel", "hiddengems", "vacation"], "TRAVEL"),
    ("Road trip through the national parks", ["roadtrip", "travel", "nationalparks", "adventure"], "TRAVEL"),
    ("Solo travel tips for first timers", ["solotravel", "travel", "tips", "adventure"], "TRAVEL"),
    ("Exploring the streets of Paris", ["paris", "travel", "europe", "citytour"], "TRAVEL"),
    ("Island hopping in the Philippines", ["philippines", "travel", "islandhopping", "vacation"], "TRAVEL"),
    ("Best hiking trails in the mountains", ["hiking", "travel", "adventure", "mountains"], "TRAVEL"),
    ("Cheap flights and travel hacks", ["traveltips", "travel", "cheapflights", "budgettravel"], "TRAVEL"),
    ("Cultural festival in Southeast Asia", ["travel", "festival", "culture", "southeastasia"], "TRAVEL"),
    ("Van life on the open road", ["vanlife", "travel", "roadtrip", "adventure"], "TRAVEL"),
    ("Safari adventure in Africa", ["safari", "travel", "africa", "wildlife"], "TRAVEL"),
    ("City guide to New York", ["newyork", "travel", "cityguide", "usa"], "TRAVEL"),
    ("Scuba diving in a coral reef", ["scubadiving", "travel", "coralreef", "adventure"], "TRAVEL"),
    ("Winter trip to the Alps", ["alps", "travel", "winter", "skiing"], "TRAVEL"),

    # COMEDY
    ("Funniest fails compilation", ["funny", "fails", "comedy", "compilation"], "COMEDY"),
    ("Stand up comedy special highlights", ["standupcomedy", "comedy", "funny", "special"], "COMEDY"),
    ("Try not to laugh challenge", ["challenge", "funny", "comedy", "laugh"], "COMEDY"),
    ("Best memes of the week", ["memes", "funny", "comedy", "internet"], "COMEDY"),
    ("Sketch comedy skit compilation", ["sketchcomedy", "comedy", "funny", "skit"], "COMEDY"),
    ("Prank gone wrong compilation", ["prank", "funny", "comedy", "fails"], "COMEDY"),
    ("Improv comedy show highlights", ["improv", "comedy", "funny", "show"], "COMEDY"),
    ("Hilarious animal fails compilation", ["animals", "funny", "comedy", "fails"], "COMEDY"),
    ("Roast battle best moments", ["roast", "comedy", "funny", "battle"], "COMEDY"),
    ("Parody video of a popular movie", ["parody", "comedy", "funny", "movie"], "COMEDY"),
    ("Funny reaction compilation", ["reaction", "funny", "comedy", "compilation"], "COMEDY"),
    ("Comedy sketch about everyday life", ["comedy", "sketch", "funny", "relatable"], "COMEDY"),
    ("Blooper reel from the show", ["bloopers", "funny", "comedy", "behindthescenes"], "COMEDY"),
    ("Satirical news segment", ["satire", "comedy", "funny", "news"], "COMEDY"),
    ("Funniest game show moments", ["gameshow", "funny", "comedy", "moments"], "COMEDY"),

    # NEWS
    ("Breaking news update today", ["news", "breaking", "update", "worldnews"], "NEWS"),
    ("World news roundup this week", ["news", "worldnews", "roundup", "currentevents"], "NEWS"),
    ("Election results analysis", ["news", "election", "politics", "analysis"], "NEWS"),
    ("Weather alert for the region", ["news", "weather", "alert", "update"], "NEWS"),
    ("Economic outlook and market news", ["news", "economy", "markets", "finance"], "NEWS"),
    ("Local news coverage of the event", ["news", "local", "coverage", "community"], "NEWS"),
    ("Government policy announcement", ["news", "politics", "policy", "government"], "NEWS"),
    ("International summit coverage", ["news", "summit", "worldnews", "diplomacy"], "NEWS"),
    ("Press conference full coverage", ["news", "pressconference", "update", "politics"], "NEWS"),
    ("Investigative report on the story", ["news", "investigation", "report", "journalism"], "NEWS"),
    ("Live news broadcast highlights", ["news", "livebroadcast", "update", "breaking"], "NEWS"),
    ("Court ruling and legal analysis", ["news", "legal", "court", "analysis"], "NEWS"),
    ("Natural disaster relief update", ["news", "disaster", "relief", "update"], "NEWS"),
    ("Public health announcement", ["news", "health", "publichealth", "announcement"], "NEWS"),
    ("Business merger news report", ["news", "business", "merger", "finance"], "NEWS"),

    # FASHION
    ("Fall fashion trends this year", ["fashion", "trends", "style", "ootd"], "FASHION"),
    ("How to style a denim jacket", ["fashion", "style", "denim", "ootd"], "FASHION"),
    ("Runway highlights from fashion week", ["fashion", "runway", "fashionweek", "style"], "FASHION"),
    ("Thrift store fashion haul", ["fashion", "thrift", "haul", "style"], "FASHION"),
    ("Sustainable fashion brands to know", ["fashion", "sustainable", "ecofriendly", "style"], "FASHION"),
    ("Makeup tutorial for a night out", ["makeup", "fashion", "beauty", "tutorial"], "FASHION"),
    ("Streetwear outfit inspiration", ["streetwear", "fashion", "outfit", "style"], "FASHION"),
    ("Designer bag review and unboxing", ["fashion", "designer", "unboxing", "luxury"], "FASHION"),
    ("Skincare routine for glowing skin", ["skincare", "beauty", "fashion", "routine"], "FASHION"),
    ("Vintage clothing collection tour", ["vintage", "fashion", "collection", "style"], "FASHION"),
    ("Capsule wardrobe essentials", ["fashion", "capsulewardrobe", "style", "minimalist"], "FASHION"),
    ("Red carpet fashion recap", ["fashion", "redcarpet", "style", "celebrity"], "FASHION"),
    ("Shoe collection and styling tips", ["fashion", "shoes", "style", "sneakers"], "FASHION"),
    ("Accessorizing your everyday outfit", ["fashion", "accessories", "style", "ootd"], "FASHION"),
    ("Seasonal wardrobe transition guide", ["fashion", "wardrobe", "style", "seasonal"], "FASHION"),

    # FITNESS
    ("30 minute home workout routine", ["fitness", "workout", "homeworkout", "exercise"], "FITNESS"),
    ("How to build muscle fast", ["fitness", "muscle", "gym", "bodybuilding"], "FITNESS"),
    ("Yoga for beginners full session", ["yoga", "fitness", "beginners", "wellness"], "FITNESS"),
    ("HIIT workout for fat loss", ["hiit", "fitness", "workout", "fatloss"], "FITNESS"),
    ("Gym leg day workout routine", ["gym", "fitness", "legday", "workout"], "FITNESS"),
    ("Stretching routine for flexibility", ["stretching", "fitness", "flexibility", "mobility"], "FITNESS"),
    ("Running training plan for a 10k", ["running", "fitness", "training", "10k"], "FITNESS"),
    ("Bodyweight workout no equipment needed", ["bodyweight", "fitness", "workout", "noequipment"], "FITNESS"),
    ("Nutrition tips for muscle gain", ["nutrition", "fitness", "musclegain", "diet"], "FITNESS"),
    ("CrossFit workout of the day", ["crossfit", "fitness", "workout", "wod"], "FITNESS"),
    ("Pilates core strengthening routine", ["pilates", "fitness", "core", "wellness"], "FITNESS"),
    ("Weightlifting form tips for beginners", ["weightlifting", "fitness", "gym", "form"], "FITNESS"),
    ("Cardio workout for endurance", ["cardio", "fitness", "endurance", "workout"], "FITNESS"),
    ("Recovery and rest day routine", ["recovery", "fitness", "restday", "wellness"], "FITNESS"),
    ("Full body workout challenge", ["fullbody", "fitness", "challenge", "workout"], "FITNESS"),
]


def build_dataframe() -> pd.DataFrame:
    """Returns columns: text (already-featurized via app.ml.content_classifier.feature_text),
    category. Imported lazily by the caller to avoid a hard import cycle with
    app.ml.content_classifier (which imports CATEGORIES from nowhere else -- this module has
    no dependency on it, kept that way deliberately)."""
    from app.ml.content_classifier import feature_text

    rows = [
        {"text": feature_text(title, hashtags), "category": category}
        for title, hashtags, category in _ROWS
    ]
    return pd.DataFrame(rows)
