# window_presets.py

# Voreinstellungen für verschiedene CT-Gewebetypen
WINDOW_PRESETS = {
    "soft_tissue": {"center": 40, "width": 400},
    "lung": {"center": -600, "width": 1500},
    "bone": {"center": 500, "width": 2000},
    "brain": {"center": 40, "width": 80},
    "liver": {"center": 60, "width": 150},
    "abdomen": {"center": 60, "width": 400},
    "default": {"center": 40, "width": 400},  # fallback
}
