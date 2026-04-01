"""
Foursquare NYC Check-in Dataset for Sequential Venue Choice.

This module provides the Foursquare NYC check-in dataset (Yang et al., 2015),
modeling sequential venue/activity choice as a dynamic discrete choice problem.

Each user's check-in sequence is treated as a trajectory where:
- State: (current_venue_category, time_of_day_bin)
- Action: next venue category visited
- Transitions: empirical category-to-category patterns

Reference:
    Yang, D., Zhang, D., Zheng, V.W., & Yu, Z. (2015). "Modeling User Activity
    Preference by Leveraging User Spatial Temporal Characteristics in LBSNs."
    IEEE Trans. Systems, Man, and Cybernetics: Systems, 45(1), 129-142.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 10 super-categories mapping the 252 Foursquare categories
SUPER_CATEGORIES = {
    "Home": ["Home (private)", "Residential Building (Apartment / Condo)"],
    "Work": ["Office", "Coworking Space", "Tech Startup", "Conference Room"],
    "Social": ["Bar", "Nightclub", "Lounge", "Brewery", "Wine Bar", "Pub",
               "Hotel", "Hotel Bar", "Cocktail Bar", "Dive Bar", "Sports Bar",
               "Hookah Bar", "Gay Bar", "Beer Garden", "Whisky Bar"],
    "Transport": ["Subway", "Train Station", "Bus Station", "Bus Stop",
                  "Light Rail Station", "Airport", "Airport Terminal",
                  "Airport Gate", "Taxi", "Boat or Ferry", "Platform"],
    "Food": ["Coffee Shop", "Food & Drink Shop", "Deli / Bodega",
             "American Restaurant", "Italian Restaurant", "Chinese Restaurant",
             "Japanese Restaurant", "Mexican Restaurant", "Pizza Place",
             "Thai Restaurant", "Indian Restaurant", "French Restaurant",
             "Sushi Restaurant", "Burger Joint", "Bakery", "Café",
             "Fast Food Restaurant", "Food Truck", "Sandwich Place",
             "Seafood Restaurant", "Steakhouse", "Vegetarian / Vegan Restaurant",
             "Greek Restaurant", "Korean Restaurant", "Ramen Restaurant",
             "Vietnamese Restaurant", "BBQ Joint", "Taco Place", "Noodle House",
             "Salad Place", "Bagel Shop", "Donut Shop", "Ice Cream Shop",
             "Frozen Yogurt Shop", "Juice Bar", "Smoothie Shop",
             "Restaurant", "Asian Restaurant", "Latin American Restaurant",
             "Mediterranean Restaurant", "Middle Eastern Restaurant",
             "Spanish Restaurant", "Breakfast Spot", "Gastropub", "Diner",
             "Wings Joint", "Falafel Restaurant", "Food Court"],
    "Fitness": ["Gym / Fitness Center", "Gym", "Yoga Studio", "Gym Pool",
                "Martial Arts Dojo", "Boxing Gym", "Pilates Studio",
                "Cycle Studio", "Athletics & Sports", "Recreation Center",
                "Rock Climbing Spot"],
    "Shopping": ["Grocery Store", "Clothing Store", "Pharmacy", "Supermarket",
                 "Department Store", "Shopping Mall", "Bookstore",
                 "Electronics Store", "Hardware Store", "Pet Store",
                 "Convenience Store", "Liquor Store", "Cosmetics Shop",
                 "Boutique", "Thrift / Vintage Store", "Gift Shop",
                 "Shoe Store", "Jewelry Store", "Market", "Farmers Market",
                 "Flea Market", "Wine Shop", "Record Shop", "Toy / Game Store",
                 "Furniture / Home Store", "Sporting Goods Shop",
                 "Mobile Phone Shop", "Music Store", "Art Supply Store"],
    "Entertainment": ["Movie Theater", "Theater", "Music Venue", "Comedy Club",
                      "Art Gallery", "Museum", "Performing Arts Venue",
                      "Concert Hall", "Karaoke Bar", "Arcade", "Casino",
                      "Bowling Alley", "Pool Hall", "Board Shop",
                      "Multiplex", "Indie Movie Theater", "Sculpture Garden",
                      "Stadium", "Basketball Stadium", "Baseball Stadium",
                      "Soccer Stadium", "Hockey Arena", "Tennis Stadium"],
    "Outdoors": ["Park", "Other Great Outdoors", "Neighborhood", "Beach",
                 "Plaza", "Trail", "Garden", "Playground", "Dog Run",
                 "River", "Lake", "Harbor / Marina", "Scenic Lookout",
                 "Bridge", "Pier", "Waterfront", "Campground", "Field",
                 "Mountain", "National Park", "State / Provincial Park",
                 "Roof Deck", "Courtyard"],
    "Services": [],  # Catch-all for everything else
}

N_SUPER_CATEGORIES = 10
N_TIME_BINS = 4  # night(0-6), morning(6-12), afternoon(12-18), evening(18-24)
CATEGORY_NAMES = list(SUPER_CATEGORIES.keys())


def _build_category_map(df: pd.DataFrame) -> dict:
    """Build mapping from raw Foursquare category → super-category index."""
    cat_map = {}
    for idx, (super_cat, members) in enumerate(SUPER_CATEGORIES.items()):
        for member in members:
            cat_map[member] = idx

    # Map remaining categories by keyword matching
    all_cats = df["venueCategory"].unique()
    for cat in all_cats:
        if cat not in cat_map:
            cat_lower = cat.lower()
            if any(w in cat_lower for w in ["restaurant", "food", "eat", "cook", "bistro"]):
                cat_map[cat] = CATEGORY_NAMES.index("Food")
            elif any(w in cat_lower for w in ["bar", "pub", "club", "lounge"]):
                cat_map[cat] = CATEGORY_NAMES.index("Social")
            elif any(w in cat_lower for w in ["shop", "store", "market", "mall"]):
                cat_map[cat] = CATEGORY_NAMES.index("Shopping")
            elif any(w in cat_lower for w in ["park", "beach", "outdoor", "garden", "trail"]):
                cat_map[cat] = CATEGORY_NAMES.index("Outdoors")
            elif any(w in cat_lower for w in ["gym", "fitness", "sport", "yoga", "pool"]):
                cat_map[cat] = CATEGORY_NAMES.index("Fitness")
            elif any(w in cat_lower for w in ["theater", "museum", "gallery", "cinema", "stadium"]):
                cat_map[cat] = CATEGORY_NAMES.index("Entertainment")
            elif any(w in cat_lower for w in ["station", "airport", "bus", "train", "subway", "ferry"]):
                cat_map[cat] = CATEGORY_NAMES.index("Transport")
            elif any(w in cat_lower for w in ["office", "work", "cowork"]):
                cat_map[cat] = CATEGORY_NAMES.index("Work")
            elif any(w in cat_lower for w in ["home", "residen", "apartment"]):
                cat_map[cat] = CATEGORY_NAMES.index("Home")
            else:
                cat_map[cat] = CATEGORY_NAMES.index("Services")
    return cat_map


def load_foursquare(
    as_panel: bool = False,
    min_checkins: int = 50,
    n_time_bins: int = N_TIME_BINS,
) -> pd.DataFrame:
    """
    Load the Foursquare NYC check-in dataset as a sequential venue choice problem.

    Each user's check-in sequence is converted to (state, action, next_state) tuples
    where state encodes the current venue category and time of day, and action is
    the next venue category visited.

    Args:
        as_panel: If True, return as Panel object for econirl estimators.
        min_checkins: Minimum check-ins per user to include (default 50).
        n_time_bins: Number of time-of-day bins (default 4).

    Returns:
        DataFrame with columns: user_id, period, state, action, next_state,
        super_category, next_category, time_bin, hour, is_weekend
    """
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "foursquare" / "dataset_TSMC2014_NYC.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Foursquare data not found at {data_path}. "
            "Download from: https://github.com/ruslansco/Foursquare-Data-Analysis"
        )

    df = pd.read_csv(data_path)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["utcTimestamp"], format="mixed")
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
    df["time_bin"] = df["hour"] // (24 // n_time_bins)
    df["time_bin"] = df["time_bin"].clip(upper=n_time_bins - 1)

    # Map to super-categories
    cat_map = _build_category_map(df)
    df["super_category"] = df["venueCategory"].map(cat_map)

    # Sort by user and time
    df = df.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    # Filter users with enough check-ins
    user_counts = df["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_checkins].index
    df = df[df["userId"].isin(valid_users)].copy()

    # Build sequential transitions
    records = []
    for user_id, user_df in df.groupby("userId"):
        user_df = user_df.sort_values("timestamp")
        cats = user_df["super_category"].values
        time_bins = user_df["time_bin"].values
        hours = user_df["hour"].values
        weekends = user_df["is_weekend"].values

        for t in range(len(cats) - 1):
            state = int(cats[t]) * n_time_bins + int(time_bins[t])
            action = int(cats[t + 1])  # next category is the "choice"
            next_time_bin = int(time_bins[t + 1])
            next_state = int(cats[t + 1]) * n_time_bins + next_time_bin

            records.append({
                "user_id": user_id,
                "period": t,
                "state": state,
                "action": action,
                "next_state": next_state,
                "super_category": CATEGORY_NAMES[cats[t]],
                "next_category": CATEGORY_NAMES[cats[t + 1]],
                "time_bin": int(time_bins[t]),
                "hour": int(hours[t]),
                "is_weekend": int(weekends[t]),
            })

    result = pd.DataFrame(records)

    if as_panel:
        from econirl.core.types import Panel, Trajectory
        import jax.numpy as jnp

        trajectories = []
        for user_id in result["user_id"].unique():
            user_data = result[result["user_id"] == user_id].sort_values("period")
            traj = Trajectory(
                states=jnp.array(user_data["state"].values, dtype=jnp.int32),
                actions=jnp.array(user_data["action"].values, dtype=jnp.int32),
                next_states=jnp.array(user_data["next_state"].values, dtype=jnp.int32),
                individual_id=int(user_id),
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    return result


def get_foursquare_info() -> dict:
    """Return metadata about the Foursquare NYC dataset."""
    return {
        "name": "Foursquare NYC Check-ins",
        "description": "Sequential venue choice from 1,084 NYC users over 6 months",
        "source": "Yang et al. (2015), IEEE Trans. SMC",
        "url": "https://github.com/ruslansco/Foursquare-Data-Analysis",
        "n_states": N_SUPER_CATEGORIES * N_TIME_BINS,  # 40
        "n_actions": N_SUPER_CATEGORIES,  # 10
        "n_individuals": 1084,
        "n_observations": "~226K transitions",
        "state_description": "(venue_super_category, time_of_day_bin)",
        "action_description": "next venue super-category",
        "categories": CATEGORY_NAMES,
        "time_bins": ["night(0-6h)", "morning(6-12h)", "afternoon(12-18h)", "evening(18-24h)"],
    }
