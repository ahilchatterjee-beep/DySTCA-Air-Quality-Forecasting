# ==========================================================
# check_graph_similarity.py
#
# PURPOSE:
# Inspect the raw station-to-station similarity scores
# BEFORE the threshold is applied.
#
# NO TRAINING
# NO MODEL CHANGES
# NO GAT
# NO HYPER-HEURISTIC SEARCH
#
# We are only checking:
#
#     raw similarity S_ij
#             ↓
#     threshold = 0.0961
#             ↓
#     graph edges
# ==========================================================

import math
import numpy as np
import torch

from preprocessing.pre_updated import prepare_model_data


# ==========================================================
# GRAPH PARAMETERS
# ==========================================================

K = 0.8933

THRESHOLD = 0.0961

SAMPLE_INDEX = 0


# ==========================================================
# LOAD DATA
# ==========================================================

print("\n" + "=" * 70)
print("RAW GRAPH SIMILARITY DIAGNOSTIC")
print("=" * 70)

print("\nLoading dataset...")

data = prepare_model_data()

X_test = data["X_test"]

station_records_test = (
    data["station_records_test"]
)


print(
    "X_test:",
    X_test.shape
)


# ==========================================================
# SELECT SAMPLE
# ==========================================================

X_sample = X_test[
    SAMPLE_INDEX
]

sample_records = (
    station_records_test[
        SAMPLE_INDEX
    ]
)


print("\n" + "=" * 70)
print("SELECTED SAMPLE")
print("=" * 70)

print(
    "Sample:",
    SAMPLE_INDEX
)

print(
    "Shape:",
    X_sample.shape
)

print(
    "Expected:",
    "[24, 10, 14]"
)


# ==========================================================
# GRAPH PARAMETERS
# ==========================================================

print("\n" + "=" * 70)
print("GRAPH PARAMETERS")
print("=" * 70)

print(
    f"k         = {K}"
)

print(
    f"threshold = {THRESHOLD}"
)


# ==========================================================
# HELPERS
# ==========================================================

def normalize_station_timestep(
    timestep
):
    """
    Convert station-record representation into
    a list of station dictionaries.
    """

    if isinstance(
        timestep,
        dict
    ):

        return list(
            timestep.values()
        )

    if isinstance(
        timestep,
        list
    ):

        return timestep

    raise TypeError(
        "Unsupported timestep structure: "
        f"{type(timestep)}"
    )


# ==========================================================
# GET VALUE FROM STATION RECORD
# ==========================================================

def get_value(
    station,
    possible_keys,
    default=None
):

    for key in possible_keys:

        if key in station:

            value = station[key]

            if value is not None:

                try:

                    value = float(
                        value
                    )

                    if math.isfinite(
                        value
                    ):

                        return value

                except (
                    TypeError,
                    ValueError
                ):

                    pass

    return default


# ==========================================================
# GET STATION ID
# ==========================================================

def get_station_id(
    station,
    fallback
):

    possible_keys = [

        "station_id",

        "Station_ID",

        "StationID",

        "id",

        "ID"

    ]

    for key in possible_keys:

        if key in station:

            return station[key]

    return fallback


# ==========================================================
# GET LATITUDE
# ==========================================================

def get_latitude(
    station
):

    return get_value(
        station,
        [
            "Latitude",
            "latitude",
            "lat"
        ]
    )


# ==========================================================
# GET LONGITUDE
# ==========================================================

def get_longitude(
    station
):

    return get_value(
        station,
        [
            "Longitude",
            "longitude",
            "lon"
        ]
    )


# ==========================================================
# GET WIND U
# ==========================================================

def get_wind_u(
    station
):

    return get_value(
        station,
        [
            "Wind_U",
            "wind_u",
            "WindU",
            "u"
        ],
        default=0.0
    )


# ==========================================================
# GET WIND V
# ==========================================================

def get_wind_v(
    station
):

    return get_value(
        station,
        [
            "Wind_V",
            "wind_v",
            "WindV",
            "v"
        ],
        default=0.0
    )


# ==========================================================
# DISTANCE
# ==========================================================

def haversine_distance_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    radius = 6371.0

    lat1 = math.radians(
        lat1
    )

    lon1 = math.radians(
        lon1
    )

    lat2 = math.radians(
        lat2
    )

    lon2 = math.radians(
        lon2
    )

    dlat = (
        lat2 - lat1
    )

    dlon = (
        lon2 - lon1
    )

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2) ** 2
    )

    c = (
        2
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(
                max(
                    0.0,
                    1.0 - a
                )
            )
        )
    )

    return radius * c


# ==========================================================
# WIND SIMILARITY
# ==========================================================

def cosine_similarity(
    u1,
    v1,
    u2,
    v2
):

    vector_1 = np.array(
        [
            u1,
            v1
        ],
        dtype=np.float64
    )

    vector_2 = np.array(
        [
            u2,
            v2
        ],
        dtype=np.float64
    )


    norm_1 = np.linalg.norm(
        vector_1
    )

    norm_2 = np.linalg.norm(
        vector_2
    )


    if (
        norm_1 == 0
        or
        norm_2 == 0
    ):

        return 0.0


    return float(
        np.dot(
            vector_1,
            vector_2
        )
        /
        (
            norm_1
            *
            norm_2
        )
    )


# ==========================================================
# BUILD RAW SIMILARITY
# ==========================================================

def calculate_similarity(
    source,
    target
):

    # ------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------

    source_lat = get_latitude(
        source
    )

    source_lon = get_longitude(
        source
    )

    target_lat = get_latitude(
        target
    )

    target_lon = get_longitude(
        target
    )


    if (
        source_lat is None
        or
        source_lon is None
        or
        target_lat is None
        or
        target_lon is None
    ):

        return None


    # ------------------------------------------------------
    # Wind
    # ------------------------------------------------------

    source_u = get_wind_u(
        source
    )

    source_v = get_wind_v(
        source
    )

    target_u = get_wind_u(
        target
    )

    target_v = get_wind_v(
        target
    )


    # ------------------------------------------------------
    # Wind direction similarity
    # ------------------------------------------------------

    cosine = cosine_similarity(

        source_u,
        source_v,

        target_u,
        target_v
    )


    positive_cosine = max(
        0.0,
        cosine
    )


    # ------------------------------------------------------
    # Wind magnitude
    #
    # V_i = magnitude of SOURCE
    # ------------------------------------------------------

    source_speed = math.sqrt(
        source_u ** 2
        +
        source_v ** 2
    )


    # ------------------------------------------------------
    # Distance
    # ------------------------------------------------------

    distance = (
        haversine_distance_km(

            source_lat,
            source_lon,

            target_lat,
            target_lon
        )
    )


    # ------------------------------------------------------
    # Distance decay
    # ------------------------------------------------------

    decay = math.exp(
        -K * distance
    )


    # ------------------------------------------------------
    # FINAL RAW SIMILARITY
    #
    # S_ij =
    # V_i *
    # max(0, cosine) *
    # exp(-k*d_ij)
    # ------------------------------------------------------

    similarity = (
        source_speed
        *
        positive_cosine
        *
        decay
    )


    return {

        "similarity":
            similarity,

        "cosine":
            cosine,

        "positive_cosine":
            positive_cosine,

        "source_speed":
            source_speed,

        "distance_km":
            distance,

        "decay":
            decay
    }


# ==========================================================
# ANALYZE ONE TIMESTEP
# ==========================================================

def analyze_timestep(
    timestep,
    timestep_index
):

    stations = (
        normalize_station_timestep(
            timestep
        )
    )


    results = []


    for i, source in enumerate(
        stations
    ):

        source_id = (
            get_station_id(
                source,
                i
            )
        )


        for j, target in enumerate(
            stations
        ):

            # ------------------------------------------------
            # Skip self
            # ------------------------------------------------

            if i == j:

                continue


            target_id = (
                get_station_id(
                    target,
                    j
                )
            )


            values = calculate_similarity(
                source,
                target
            )


            if values is None:

                continue


            results.append({

                "source_index":
                    i,

                "target_index":
                    j,

                "source_id":
                    source_id,

                "target_id":
                    target_id,

                **values
            })


    return results


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

all_results = []

timestep_statistics = []


for t in range(
    24
):

    print("\n" + "=" * 70)

    print(
        f"TIMESTEP {t + 1:02d}"
    )

    print("=" * 70)


    results = analyze_timestep(

        sample_records[t],

        t
    )


    all_results.extend(
        results
    )


    # ------------------------------------------------------
    # Sort by similarity
    # ------------------------------------------------------

    sorted_results = sorted(

        results,

        key=lambda x:
            x["similarity"],

        reverse=True
    )


    # ------------------------------------------------------
    # Count threshold crossings
    # ------------------------------------------------------

    above_threshold = [

        r

        for r in results

        if r["similarity"]
        >=
        THRESHOLD
    ]


    below_threshold = [

        r

        for r in results

        if r["similarity"]
        <
        THRESHOLD
    ]


    # ------------------------------------------------------
    # Scores near threshold
    #
    # ±0.02 around threshold
    # ------------------------------------------------------

    near_threshold = [

        r

        for r in results

        if abs(
            r["similarity"]
            -
            THRESHOLD
        )
        <=
        0.02
    ]


    timestep_statistics.append({

        "timestep":
            t + 1,

        "total_pairs":
            len(results),

        "above":
            len(above_threshold),

        "below":
            len(below_threshold),

        "near":
            len(near_threshold)
    })


    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    print(
        f"Total station pairs : "
        f"{len(results)}"
    )

    print(
        f"Above threshold     : "
        f"{len(above_threshold)}"
    )

    print(
        f"Below threshold     : "
        f"{len(below_threshold)}"
    )

    print(
        f"Within ±0.02        : "
        f"{len(near_threshold)}"
    )


    if len(
        sorted_results
    ) > 0:

        similarities = np.array(

            [
                r["similarity"]

                for r in results
            ]
        )


        print(
            f"Maximum similarity  : "
            f"{np.max(similarities):.8f}"
        )

        print(
            f"Mean similarity     : "
            f"{np.mean(similarities):.8f}"
        )

        print(
            f"Median similarity   : "
            f"{np.median(similarities):.8f}"
        )


        # --------------------------------------------------
        # TOP 10
        # --------------------------------------------------

        print(
            "\nTop similarity scores:"
        )


        for rank, result in enumerate(

            sorted_results[:10],

            start=1

        ):

            print(

                f"  {rank:02d}. "
                f"{result['source_id']} → "
                f"{result['target_id']} | "
                f"S={result['similarity']:.8f} | "
                f"cos={result['cosine']:.4f} | "
                f"V={result['source_speed']:.4f} | "
                f"d={result['distance_km']:.2f} km | "
                f"decay={result['decay']:.6f}"
            )


        # --------------------------------------------------
        # SCORES CLOSE TO THRESHOLD
        # --------------------------------------------------

        if len(
            near_threshold
        ) > 0:

            near_sorted = sorted(

                near_threshold,

                key=lambda x:
                    abs(
                        x["similarity"]
                        -
                        THRESHOLD
                    )
            )


            print(
                "\nScores close to threshold:"
            )


            for result in (
                near_sorted[:10]
            ):

                print(

                    f"  {result['source_id']} → "
                    f"{result['target_id']} | "
                    f"S={result['similarity']:.8f} | "
                    f"distance="
                    f"{result['distance_km']:.2f} km"
                )


# ==========================================================
# GLOBAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)

print(
    "GLOBAL SIMILARITY SUMMARY"
)

print("=" * 70)


all_similarities = np.array(

    [
        r["similarity"]

        for r in all_results
    ]
)


if len(
    all_similarities
) > 0:

    print(
        "Total directed pairs checked:",
        len(all_similarities)
    )


    print(
        f"Mean similarity    : "
        f"{np.mean(all_similarities):.8f}"
    )


    print(
        f"Median similarity  : "
        f"{np.median(all_similarities):.8f}"
    )


    print(
        f"Std similarity     : "
        f"{np.std(all_similarities):.8f}"
    )


    print(
        f"Minimum similarity : "
        f"{np.min(all_similarities):.8f}"
    )


    print(
        f"Maximum similarity : "
        f"{np.max(all_similarities):.8f}"
    )


    # ------------------------------------------------------
    # Threshold statistics
    # ------------------------------------------------------

    above_count = np.sum(
        all_similarities
        >=
        THRESHOLD
    )


    below_count = np.sum(
        all_similarities
        <
        THRESHOLD
    )


    print(
        "\nThreshold:",
        THRESHOLD
    )


    print(
        "Pairs above threshold:",
        int(above_count)
    )


    print(
        "Pairs below threshold:",
        int(below_count)
    )


    print(
        "Percentage above:",
        f"{100 * above_count / len(all_similarities):.2f}%"
    )


    # ------------------------------------------------------
    # Near threshold
    # ------------------------------------------------------

    near_count = np.sum(

        np.abs(
            all_similarities
            -
            THRESHOLD
        )
        <=
        0.02
    )


    print(
        "\nPairs within ±0.02 of threshold:",
        int(near_count)
    )


# ==========================================================
# TIMESTEP SUMMARY TABLE
# ==========================================================

print("\n" + "=" * 70)

print(
    "TIMESTEP SUMMARY"
)

print("=" * 70)


print(
    f"{'Time':<8}"
    f"{'Pairs':<10}"
    f"{'Above':<10}"
    f"{'Below':<10}"
    f"{'Near':<10}"
)


for stats in (
    timestep_statistics
):

    print(

        f"{stats['timestep']:<8}"

        f"{stats['total_pairs']:<10}"

        f"{stats['above']:<10}"

        f"{stats['below']:<10}"

        f"{stats['near']:<10}"
    )


# ==========================================================
# FINAL INTERPRETATION
# ==========================================================

print("\n" + "=" * 70)

print(
    "INTERPRETATION"
)

print("=" * 70)


if len(
    all_similarities
) == 0:

    print(
        "No similarity scores were computed."
    )

else:

    percentage_above = (
        100.0
        *
        np.mean(
            all_similarities
            >=
            THRESHOLD
        )
    )


    if percentage_above < 5:

        print(
            "VERY SPARSE:"
        )

        print(
            f"Only {percentage_above:.2f}% "
            "of station pairs exceed the threshold."
        )

        print(
            "The threshold/formulation may be "
            "strongly restricting the graph."
        )


    elif percentage_above < 15:

        print(
            "SPARSE:"
        )

        print(
            f"{percentage_above:.2f}% "
            "of station pairs exceed the threshold."
        )

        print(
            "This is consistent with the sparse "
            "graphs observed earlier."
        )


    else:

        print(
            "MODERATELY CONNECTED:"
        )

        print(
            f"{percentage_above:.2f}% "
            "of station pairs exceed the threshold."
        )


print("\n" + "=" * 70)

print(
    "RAW SIMILARITY DIAGNOSTIC COMPLETE"
)

print("=" * 70)