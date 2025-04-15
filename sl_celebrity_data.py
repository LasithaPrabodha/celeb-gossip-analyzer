# Enhanced Sri Lankan Celebrity Data Structure
# --------------------------------------------

# Sri Lankan celebrity names - greatly expanded and categorized
SL_CELEBRITIES = [
    # Film & TV Actors - Veterans
    "Jackson Anthony", "Sangeetha Weeraratne", "Malini Fonseka", "Ravindra Randeniya",
    "Sriyani Amarasena", "Vijaya Kumaratunga", "Swarna Mallawarachchi", "Joe Abeywickrama",
    "Kamal Addaraarachchi", "Vasanthi Chathurani", "Dilhani Ekanayake", "Ranjan Ramanayake",
    
    # Film & TV Actors - Mid-career
    "Roshan Ranawana", "Saranga Disasekara", "Dinakshie Priyasad", "Shanudrie Priyasad",
    "Umaria Sinhawansa", "Sheshadri Priyasad", "Nehara Peiris", "Menaka Rajapaksa",
    "Pubudu Chathuranga", "Semini Iddamalgoda", "Amila Abeysekara", "Nirosha Perera",
    "Chulakshi Ranathunga", "Uddika Premarathna", "Chathurika Peiris", "Jayalath Manoratne",
    
    # Film & TV Actors - Newcomers
    "Kaveesha Ayeshani", "Tehan Perera", "Rithu Akarsha", "Yureni Noshika",
    "Samanalee Fonseka", "Hemal Ranasinghe", "Shalani Tharaka", "Ashan Dias",
    "Isuru Lokuhettiarachchi", "Devnaka Porage", "Thumindu Dodantenne", "Himali Sayurangi",
    
    # Musicians
    "Bathiya and Santhush", "Yohani", "Santhush Weeraman", "Bhathiya Jayakody",
    "Iraj", "Umaria Sinhawansa", "Lahiru Perera", "Randhir Witana",
    "Ashanthi", "Hiran Thenuwara", "Chiranthi Rajapakse", "Romesh Sugathapala",
    "Ruwan Hettiarachchi", "Nadeeka Guruge", "Pasan Liyanage", "Sanuka Wickramasinghe",
    "Charitha Attalage", "Ridma Weerawardena", "Dinesh Gamage", "Kasun Kalhara",
    
    # Sports Figures - Cricket
    "Kumar Sangakkara", "Lasith Malinga", "Mahela Jayawardene", "Angelo Mathews", 
    "Upul Tharanga", "Roshan Mahanama", "Sanath Jayasuriya", "Chamari Athapaththu",
    "Wanindu Hasaranga", "Charith Asalanka", "Bhanuka Rajapaksa", "Avishka Fernando",
    "Dasun Shanaka", "Dhananjaya de Silva", "Dimuth Karunaratne", "Dinesh Chandimal",
    
    # Sports Figures - Other Sports
    "Suraj Randiv", "Niluka Karunaratne", "Julian Bolling", "Susanthika Jayasinghe",
    "Damayanthi Dharsha", "Mangala Samarakoon", "Thilini Priyamali", "Manjula Kumara",
    "Nadeeka Lakmali", "Sriyani Kulawansa", "Matthew Abeysinghe", "Niluka Geethani",
    
    # Politicians and Families
    "Mahinda Rajapaksa", "Gotabaya Rajapaksa", "Namal Rajapaksa", "Yoshitha Rajapaksa", 
    "Basil Rajapaksa", "Chamal Rajapaksa", "Sajith Premadasa", "Hirunika Premachandra", 
    "Ranil Wickremesinghe", "Maithripala Sirisena", "Chandrika Kumaratunga",
    "Anura Kumara Dissanayake", "Ranjan Ramanayake", "Dayasiri Jayasekara",
    
    # Social Media Personalities
    "Piumi Hansamali", "Sachini Nipunsala", "Nadeesha Hemamali", "Samanalee Fonseka",
    "Daisy Achchi", "Yasas Iddamalgoda", "Natasha Rathnayake", "Stephanie Siriwardhana",
    "Vinu Udani Siriwardana", "Rithu Akarsha", "Hashini Samuel", "Shenelle Rodrigo", 
    "Falan Andrea", "Dinakshi Priyasad", "Sheshadri Priyasad", "Thinuli Wickramanayake",
    "Yureni Noshika", "Shanudri Priyasad", "Kavindya Adikari", "Dennis Mala",
    
    # International Crossover Celebrities
    "Jacqueline Fernandez", "Pooja Umashankar", "M.I.A.", "Alston Koch",
    "Roshini Singarayar", "Thusitha Jayasundera", "Nimmi Harasgama", "Ruwanthi Mangala",
    "Tony Ranasinghe", "Akalanka Ganegama", "Roshan Seth", "Nikini Botheju",
    
    # TV Presenters and Hosts
    "Rozanne Diasz", "Danu Innasithamby", "Gayathri Dias", "Suraj Mapa",
    "Prasanna Puwanarajah", "Peshala Manoj", "Kanchana Mendis", "Buddhika Jayaratne",
    "Saumya Liyanage", "Samantha Epasinghe", "Siyatha Dilrukshi", "Wishwa Kodikara",
    "Upeksha Swarnamali", "Shashi Weerawansa", "Udari Warnakulasooriya", "Ryan Van Rooyen"
]

# Common nicknames (to complement the main list)
SL_CELEBRITY_NICKNAMES = {
    "Sanga": "Kumar Sangakkara",
    "Mali": "Lasith Malinga",
    "Mahela": "Mahela Jayawardene",
    "BnS": "Bathiya and Santhush",
    "Jackie": "Jacqueline Fernandez",
    "Chandi": "Dinesh Chandimal",
    "Gota": "Gotabaya Rajapaksa",
    "Sanath": "Sanath Jayasuriya",
    "MR": "Mahinda Rajapaksa",
    "Dasun": "Dasun Shanaka",
    "Pooja": "Pooja Umashankar",
    "Yohani": "Yohani de Silva",
    "Ranjan": "Ranjan Ramanayake",
    "Peshala": "Peshala Manoj",
    "Sudu Aiya": "Ryan Van Rooyen",
    "Chooty": "Dinakshie Priyasad"
}

# Sri Lankan media companies, studios, production houses
SL_COMPANIES = [
    # TV Channels
    "Derana", "Sirasa", "Hiru", "Swarnavahini", "Rupavahini", "ITN", "TV Derana",
    "Sirasa TV", "Hiru TV", "Swarnavahini TV", "Rupavahini TV", "Shakthi TV",
    "Nethra TV", "Channel Eye", "TNL TV", "MTV Sports", "Channel C", "TV1",
    
    # Production Companies
    "EAP Films", "Torana Video", "Sarasaviya", "Maharaja Organization", 
    "Laxapana Studios", "Ceylon Theatres", "EAP Holdings", "Lanka Films", 
    "Ranmihitenna Film Studio", "Unlimited Entertainment", "Power House", 
    "Red Hen Productions", "Exceptional Entertainment", "Lyca Productions",
    "Ama Creations", "Swarnavahini Productions", "Derana Films", "Hiru Films",
    
    # Media Organizations
    "Wijeya Newspapers", "Daily Mirror", "Sunday Observer", "Sunday Times",
    "Lakehouse", "Daily News", "Lankadeepa", "Rivira Media", "Mawbima",
    "Ada Derana", "Hiru News", "News First", "Dinamina", "Silumina",
    
    # Digital Media
    "Roar Media", "Readme.lk", "Pulse.lk", "Yamu.lk", "Gossip Lanka",
    "Hiru Gossip", "Lanka Hot Gossip", "Lankadeepa Online", "Newsfirst.lk",
    "Ada.lk", "Lankadiva", "Gossip-Lanka News", "GossipC.com", "Hirufm.lk",
    
    # Telecom Companies
    "Dialog", "SLT Mobitel", "Etisalat", "Airtel", "Hutch", "Lanka Bell",
    
    # Popular YouTube Channels
    "Wasthi Productions", "Ratta Channel", "Isuru Lifestyle", "Derana Official",
    "Hiru TV Official", "Sirasa TV Sri Lanka", "Ahas Maliga", "Tharu Niwadu",
    "CineTv Lanka", "BnS Official", "Iraj Official", "Yohani Music", 
    
    # Talent Agencies
    "Derana Dream Star", "Sirasa Superstar", "Hiru Star", "Model Agency Lanka",
    "Champi Model Management", "Colombo Fashion Week", "Ramani Fernando Salons",
    "Soul Sounds Academy", "Wootz Agency", "OMG Lanka"
]

# Sri Lankan locations and venues
SL_LOCATIONS = [
    # Major Cities
    "Colombo", "Kandy", "Galle", "Negombo", "Jaffna", "Anuradhapura", "Batticaloa",
    "Trincomalee", "Nuwara Eliya", "Matara", "Kalmunai", "Vavuniya", "Ratnapura",
    "Badulla", "Puttalam", "Kotte", "Dehiwala", "Moratuwa", "Kolonnawa", "Kurunegala",
    
    # Hotels & Luxury Venues
    "Cinnamon Grand", "Taj Samudra", "Kingsbury", "Shangri-La", "Hilton Colombo", 
    "Mount Lavinia Hotel", "Galle Face Hotel", "Galadari Hotel", "Jetwing Hotels",
    "Amari Galle", "Heritance Kandalama", "Cinnamon Lakeside", "Mövenpick Hotel",
    "The Fortress Resort & Spa", "Amanwella", "Anantara Kalutara", "Weligama Bay Marriott",
    
    # Shopping & Entertainment Venues
    "Odel", "Majestic City", "Liberty Cinema", "Colombo City Centre", "Crescat Boulevard", 
    "Marino Mall", "One Galle Face", "Liberty Plaza", "Colombo Gold Center", 
    "Arcade Independence Square", "Dutch Hospital", "Excel World", "Savoy Cinema",
    "CCC", "Platinum One", "Colombo Fashion Week", "Fashion Market", "Fashion Bug",
    
    # Cultural & Performance Venues
    "Nelum Pokuna", "BMICH", "Viharamahadevi Park", "Independence Square", 
    "Race Course", "Waters Edge", "Lionel Wendt Theatre", "Navarangahala", 
    "John de Silva Theatre", "Tower Hall", "National Museum", "National Art Gallery",
    "Galle Literary Festival", "Colombo Music Festival", "Sri Lanka Exhibition & Convention Centre",
    
    # Beach Areas
    "Mount Lavinia Beach", "Unawatuna Beach", "Bentota Beach", "Hikkaduwa Beach",
    "Mirissa Beach", "Arugam Bay", "Pasikuda Beach", "Nilaveli Beach", "Uppuveli",
    "Tangalle", "Dickwella", "Weligama Bay", "Batticaloa Lagoon", "Panama Beach",
    
    # Suburban Areas
    "Dehiwala", "Rajagiriya", "Nugegoda", "Maharagama", "Battaramulla", "Malabe",
    "Nawala", "Kollupitiya", "Bambalapitiya", "Wellawatte", "Kirulapone", "Ratmalana",
    "Borella", "Pettah", "Fort", "Ethul Kotte", "Piliyandala", "Kotahena", "Dematagoda"
]

# Sensationalist language specific to Sri Lankan gossip
SENSATIONALIST_TERMS = [
    # Common English Sensationalist Terms
    "shocking", "bombshell", "scandal", "exclusive", "secret affair", "caught in the act",
    "breaking news", "exposed", "you won't believe", "leaked", "insider information",
    "drama", "controversy", "disaster", "disgrace", "shame", "infamous", "rumors",
    "sources close to", "anonymous source", "unnamed insider", "relationship drama",
    "caught red-handed", "intimate moment", "private life", "hot gossip", "secret meeting",
    "behind the scenes", "forbidden love", "special relationship", "stunning revelation",
    "unconfirmed reports", "alleged relationship", "inside sources claim", "whispers in the industry",
    
    # Sri Lankan Colloquial Terms (transliterated)
    "keliya", "rasthiyadu", "hoda show", "hada", "pissu", "bayanak", "kunu harapa",
    "kohulan maththa", "jalaya", "baduwa", "katakaranawa", "asammanai", "kellam", 
    "kella", "bomba scene", "rasa katha", "vishakas", "goda", "pissu kellam", "hot news",
    "patta kella", "haduwa", "lamayo", "mekada karanne", "choon eka", "kaputa kiyala",
    "hariyata hot", "malli bayaune", "kata kata", "kiyawida", "ahanna kamathi", "vihilu",
    
    # Mixed Sinhala-English Terms
    "hot kella", "big scene", "love affair eka", "kata show", "real story eka", 
    "scene eka", "photo leak", "party scene eka", "video leak una", "secret place ekata",
    "romance ekak", "big reveal", "patta hot", "vishakas gossip", "romance pawula", 
    "leak una photos", "reveal kala", "truth eka", "hot romance", "leak una", "scene dala"
]

# Specific categories of gossip based on requirements
GOSSIP_CATEGORIES = {
    'secret_affair': [
        # Basic Relationship Terms
        "affair", "secret relationship", "romance", "dating", "seeing each other", 
        "together", "intimate", "lovers", "cheating", "two-timing", "unfaithful",
        
        # Descriptive Phrases
        "behind closed doors", "private meetings", "secret rendezvous", "hidden relationship",
        "meeting secretly", "sneaking around", "secret lover", "affair with", "caught together",
        "spotted together", "spending time together", "close relationship", "more than friends",
        "midnight meeting", "cozy dinner", "hotel room", "private dinner", "late night", 
        
        # Sri Lankan Context
        "kaluwara karmaraya", "rahas sambandhaya", "hora sambandhaya", "penema",
        "rahas hamuwima", "rahas adaraya", "adara sambandhaya", "pemin bandhaya",
        "aei samaga", "samaga hamuwuna", "ratriyata hamuwuna", "pravrutti naththe",
        "ratriye navatina", "horen hamuwena", "pavul getaluwa"
    ],
    
    'sexting': [
        # Message Content Types
        "sexting", "explicit messages", "nude", "private photos", "intimate pictures", 
        "inappropriate texts", "leaked messages", "personal chat", "explicit content",
        "private conversation", "compromising photos", "revealing photos", "dirty messages", 
        "flirty texts", "suggestive messages", "bedroom photos", "shower pictures", 
        "inappropriate content", "private video call", "video scandal",
        
        # Platforms Mentioned
        "WhatsApp leak", "Instagram DM", "private DM", "messenger chat", "TikTok messages",
        "Snapchat", "Signal messages", "Telegram chat", "viral message", "screenshot leaked",
        "chat history", "direct message", "chat record", "phone message", "SMS leak",
        
        # Sri Lankan Context
        "patturuwana", "pathuru", "lansu", "photo dala", "patta photo", "horen gatta",
        "photo leak", "photo hedda", "photo danna", "pin photo", "kaluwara pin", "hora pin",
        "chat eka", "chat leak", "chat hedda", "message eka", "WhatsApp eke", "photo gif",
        "screenshot karala", "chat eka wattala", "message warga", "DM karanawa", "msg"
    ],
    
    'boss_employee': [
        # Hierarchical Relationships
        "boss", "director", "producer", "manager", "supervisor", "senior", "junior",
        "assistant", "staff", "employee", "workplace relationship", "office romance",
        "professional relationship", "working together", "professional boundaries",
        
        # Power Dynamics
        "power imbalance", "casting couch", "favoritism", "special treatment", 
        "promotion", "career advancement", "mentor", "protege", "contract", "work relationship",
        "getting roles", "special opportunity", "career boost", "recommendation", "preference",
        
        # Industry Context
        "film set", "production", "studio", "company", "agency", "talent agency",
        "casting", "audition", "screen test", "upcoming project", "new movie",
        "new series", "new production", "music video", "commercial", "advertisement",
        
        # Sri Lankan Context
        "lokka", "boss samaga", "director samaga", "film karanna", "chance ekak", 
        "role ekak", "kanistaya", "jestaya", "weda polawe", "office ekata", 
        "karyalayata", "himikaru", "ayathanadhipathi", "bada ginna", "rasawe natuwa", 
        "wage natuwa", "rasawa thamai", "rasawak thiyenawa", "rakiyawa", "rassa"
    ],
    
    'celebrity_fan': [
        # Fan Relationship Terms
        "fan", "follower", "admirer", "supporter", "devotee", "fan club", "fan meeting",
        "meet and greet", "selfie", "autograph", "photo with fan", "fan encounter",
        "fan interaction", "obsessed fan", "stalker", "fan mail", "message from fan",
        
        # Fan Behavior
        "fan gift", "meeting fan", "private meeting", "special fan", "loyal fan",
        "dedicated fan", "fan relationship", "social media fan", "fan pages",
        "following everywhere", "waiting outside", "showing up", "tracking movements",
        "at the airport", "outside hotel", "concert crowd", "backstage pass",
        
        # Sri Lankan Context
        "rasika", "lokku fan", "rasika samagama", "loku fan kenekwa", "selfi ganna",
        "athi rasika", "fan mail", "selfi ekak", "rasikayo", "rasikawo", "love karanawa",
        "adare karanawa", "follow karanawa", "page eka", "fan page", "fan club eka",
        "autograph ganna", "pissu fan", "rasika hamuwa", "fan meet", "godak kamathi"
    ],
    
    'financial_scandal': [
        # Financial Issues
        "tax evasion", "money laundering", "undisclosed income", "hidden assets",
        "offshore accounts", "bankruptcy", "debt", "unpaid loans", "financial trouble",
        "evading taxes", "illegal transactions", "undeclared wealth", "ill-gotten gains",
        
        # Legal Language
        "under investigation", "charges filed", "legal trouble", "court appearance", 
        "case filed", "summoned by", "questioned by authorities", "tax authority",
        "regulatory investigation", "financial crimes", "economic offenses",
        
        # Sri Lankan Context
        "salli prashnaya", "salli walata", "badu gevilla", "tax gevilla", "salli hora",
        "bank ginung", "loan eka", "naya", "naya gewanna", "badu gewanna be", 
        "adhikaranaya", "police", "cid", "parikshana", "badu hora", "black money",
        "hora salli", "hora ginum", "hora passport", "hora deywal", "wanchawa"
    ],
    
    'family_drama': [
        # Family Relationships
        "divorce", "separation", "custody battle", "family feud", "in-laws", "relatives",
        "inheritance", "family property", "family business", "family name", "dynasty",
        "marriage troubles", "family conflict", "estranged", "reconciliation",
        
        # Emotional Language
        "family crisis", "broken home", "domestic dispute", "household tension",
        "family breakdown", "bitter fight", "emotional distress", "turbulent relationship",
        "troubled marriage", "walked out", "kicked out", "thrown out", "living separately",
        
        # Sri Lankan Context
        "pawul prashnaya", "vivahaya", "dinaseda", "lamay baaraganima", "guti kama",
        "pavul depola", "pavul vyaparaya", "udasawa", "vivaha getaluwa", "gey dora",
        "ammata appata", "nanda mama", "nagi masina", "loku amma", "pavul arauwa",
        "gedara getaluva", "vivaha bandanaya", "kadapali", "balanna beri",
        "kalakireema", "andamandaya", "kadakappal"
    ]
}

# Privacy related terms
PRIVACY_TERMS = [
    # General Privacy Terms
    "private", "personal", "confidential", "intimate", "secret", "leaked", "exposed",
    "hacked", "unauthorized", "not meant to be shared", "hidden", "sensitive",
    "not for public", "behind closed doors", "off the record", "anonymous",
    "private account", "personal device", "personal chat", "locked", "password protected",
    
    # Digital Privacy
    "screenshot", "private message", "DM", "personal email", "private number",
    "private social media", "WhatsApp chat", "Instagram messages", "direct message",
    "personal data", "browsing history", "location data", "contact list", "private gallery",
    
    # Physical Privacy
    "bedroom", "bathroom", "changing room", "private residence", "home invasion",
    "trespassing", "spying", "stalking", "surveillance", "hidden camera", "paparazzi",
    "intrusion", "privacy breach", "personal space", "unwanted attention",
    
    # Sri Lankan Context
    "paudgalika", "kopya", "rahase", "rahasia", "ekama kamara", "private kamara",
    "horagena", "horen balana", "rahasia video", "rahasia pattura", "ekama deyval",
    "paudgalika deyval", "wenas kelin", "beraganna ba", "horen gatta", "leak una",
    "screenshot karala", "kata karanna ba", "kiyana deyval", "horen ahana"
]

# Additional data structures to enhance model performance

# Relationship proximity terms (used to detect relationship discussion)
RELATIONSHIP_PROXIMITY_TERMS = [
    "together with", "close to", "intimate with", "friendly with", "spotted with",
    "seen with", "caught with", "linked to", "romantically linked", "associated with",
    "paired with", "coupled with", "alongside", "accompanied by", "in the company of",
    "meeting with", "spending time with", "vacationing with", "traveling with",
    "dining with", "dating", "in a relationship with", "having an affair with",
    
    # Sri Lankan context
    "ekka innawa", "ekka giya", "samaga sitinawa", "samanga", "langata", "langa",
    "ekta", "ha samaga", "ekama thane", "ekama hotele", "ekama kamara", "eksath", 
    "sambandhayak", "pavul sambandha", "adara sambandha", "ekama giya", "ekama ena"
]

# Time and date patterns - for detecting specific temporal details
TIME_PATTERNS = [
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)",  # Standard time format
    r"\d{1,2}\s*o\'clock",  # O'clock format
    r"(?:early|late) (?:morning|afternoon|evening|night)",  # Time of day
    r"(?:midnight|noon|dusk|dawn|sunrise|sunset)",  # Specific times
    r"(?:yesterday|today|tomorrow) (?:morning|afternoon|evening|night)",  # Relative days
    
    # Sri Lankan context
    r"(?:udey|dawale|hawasa|rate|rata welawa)",  # Time periods in Sinhala
    r"(?:paandara|dahawala|sandewa|rathriya|re)",  # More time periods
    r"paya \d{1,2}",  # Hour in Sinhala (e.g., paya 7)
    r"paya \d{1,2}\.?\d{0,2}",  # Decimal hour in Sinhala
]

DATE_PATTERNS = [
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # Standard date formats
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",  # Year first format
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?", # Month day
    r"\d{1,2}(?:st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*", # Day month
    r"(?:last|this|next) (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",  # Relative weekdays
    r"(?:last|this|next) (?:week|month|year)",  # Relative time periods
    
    # Sri Lankan context
    r"(?:ada|eheta|pereda|oheta|lamana satiye|giya satiye)",  # Relative days in Sinhala
    r"(?:giya masaya|me masaya|lamana masaya)",  # Months in Sinhala
    r"\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December)",
    r"pasu giya",  # Past tense reference
]

# Credibility indicators (to help model evaluate firsthand vs hearsay)
CREDIBILITY_INDICATORS = {
    'firsthand': [
        "I saw", "I witnessed", "I was there", "I heard", "I know for a fact",
        "I have evidence", "I have proof", "I recorded", "I have photos",
        "I was present", "I attended", "I participated", "I observed",
        "mama duththa", "mama daheka", "mata therennava", "mama danne", 
        "mama hithanne", "mata hondata dhanaganna puluvan", "mata danwa",
        "eye witness", "with my own eyes", "personally confirmed"
    ],
    
    'hearsay': [
        "someone told me", "I heard that", "people are saying", "rumor has it",
        "according to sources", "allegedly", "supposedly", "apparently", 
        "word is", "the gossip is", "they say", "everyone knows", "it's going around",
        "kata kata", "kellem", "armuthu kata", "wachala", "kata katha", 
        "minissu kiyanne", "minissu laga ahanne", "kiyana vidiyata", 
        "anonymous source", "unnamed insider", "close friend revealed"
    ],
    
    'speculation': [
        "might be", "could be", "possibly", "perhaps", "probably", "seems like",
        "looks like", "appears to be", "I suspect", "I'm guessing", "I think",
        "I believe", "more likely", "chances are", "there's a possibility",
        "wenna puluwan", "weyi", "wagey", "wage thiyenne", "wage inne",
        "nethnam", "nemeyi", "thamai", "wage kiyanne", "balanna one"
    ]
}

# Messaging platforms (for detecting sexting/private message allegations)
MESSAGING_PLATFORMS = [
    "WhatsApp", "Messenger", "Instagram", "DM", "text message", "SMS", "iMessage",
    "Telegram", "Signal", "Snapchat", "TikTok", "Viber", "WeChat", "Line",
    "Facebook Messenger", "Twitter DM", "Discord", "IMO", "Google Chat", "Skype",
    "WhatsApp eke", "message eke", "chat eke", "DM eke", "Instagram eke",
    "message ekak", "chat ekak", "messiah"
]

RELATIONSHIP_PATTERNS = [
    # Two entities with relationship terms between
    r"(\w+(?:\s\w+){0,3})\s+(?:and|with|dating|seeing|together|alongside)\s+(\w+(?:\s\w+){0,3})",
    
    # Specific relationship mentions
    r"(\w+(?:\s\w+){0,3})\s+(?:has a|in a|having an|started a)\s+(?:relationship|affair|romance|connection)\s+(?:with|and)\s+(\w+(?:\s\w+){0,3})",
    
    # Dating patterns
    r"(\w+(?:\s\w+){0,3})\s+(?:is|has been|was|started)\s+dating\s+(\w+(?:\s\w+){0,3})",
    
    # Spotted together patterns
    r"(\w+(?:\s\w+){0,3})\s+(?:and)\s+(\w+(?:\s\w+){0,3})\s+(?:were|was|have been|has been)\s+(?:spotted|seen|caught|photographed)",
    
    # Sri Lankan context
    r"(\w+(?:\s\w+){0,3})\s+(?:ekka|samaga|ha|saha)\s+(\w+(?:\s\w+){0,3})",
    r"(\w+(?:\s\w+){0,3})\s+(?:adare karanawa|love karanawa)\s+(\w+(?:\s\w+){0,3})",
    r"(\w+(?:\s\w+){0,3})\s+(?:sambandayak|sambandhaya)\s+(\w+(?:\s\w+){0,3})",
    r"(\w+(?:\s\w+){0,3})\s+(?:ekka hitiye|ekka inne|samaga sitinne)\s+(\w+(?:\s\w+){0,3})"
]

# Location patterns - to identify where celebrity sightings occurred
LOCATION_PATTERNS = [
    # At specific venues
    r"at\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
    r"in\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
    r"visited\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
    
    # Sri Lankan context
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+(?:wala|wala di|waladi)",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+(?:hotel eke|hotale)",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+(?:ekata giya|ekata awa|ekata aawa)"
]

# Sentiment analysis dictionaries for gossip tone detection
SENTIMENT_WORDS = {
    'positive': [
        # General positive terms
        "happy", "excited", "delighted", "joyful", "pleased", "content", "satisfied",
        "thrilled", "enthusiastic", "elated", "overjoyed", "ecstatic", "jubilant",
        "glowing", "radiant", "beautiful", "handsome", "stunning", "gorgeous", "amazing",
        
        # Sri Lankan context 
        "santhose", "sathutui", "hondai", "alankara", "lassanai", "sundara", 
        "patta", "hoda", "supirii", "rasai", "lokui", "ganu", "masth", "cool eka"
    ],
    
    'negative': [
        # General negative terms
        "sad", "disappointed", "upset", "angry", "furious", "enraged", "bitter",
        "jealous", "resentful", "heartbroken", "devastated", "crushed", "miserable",
        "depressed", "distressed", "troubled", "distraught", "annoyed", "irritated",
        
        # Sri Lankan context
        "duka", "kansiya", "amaaru", "amaarui", "duk", "karadara", "asahanaya",
        "thada", "epa", "naraka", "kala kanni", "payanna", "bayanak", "pissu", "huththo",
        "wenna kara", "hukana", "hukapan", "karala", "wadak na", "hutta", "paka"
    ],
    
    'neutral': [
        # General neutral terms
        "okay", "fine", "average", "neutral", "indifferent", "impartial", "fair",
        "objective", "balanced", "moderate", "standard", "typical", "common", "usual",
        
        # Sri Lankan context
        "hodatama hodayi", "tharamai", "samanya", "madyastha", "api balamu",
        "eyata kamak na", "kamak na", "ehema thama", "harida", "ok", "man danne na"
    ]
}

# Event types for categorizing celebrity activities
EVENT_TYPES = [
    # Professional Events
    "award ceremony", "film premiere", "movie launch", "press conference", 
    "promotion event", "interview", "photo shoot", "TV show", "radio show", 
    "brand endorsement", "advertisement shoot", "magazine cover", "product launch",
    
    # Personal Events
    "birthday party", "wedding", "engagement", "anniversary", "baby shower",
    "housewarming", "vacation", "trip abroad", "beach outing", "shopping spree",
    "dinner date", "night out", "hospital visit", "funeral", "private party",
    
    # Sri Lankan Context
    "Avurudu celebration", "poya day event", "poruwa ceremony", "homecoming",
    "pirith ceremony", "dansala", "vesak celebration", "charity event", "almsgiving",
    "political rally", "beach party", "musical show", "reality show", "Colombo fashion week",
    "cricket match", "award nite", "sinhala film fare", "derana awards", "hiru golden awards"
]
